"""Hidden-Markov-Modell auf einem Straßengitter, numpy-only: Filter, Glätter (Vorwärts-Rückwärts) und Viterbi.

Zustände sind Straßenstücke („Zellen“) von 10 m. Die Messung ist die GPS-Position (Gauß um die Zellenmitte), der Übergang bevorzugt eine
Weiterfahrt um etwa eine Zelle je Sekunde entlang des Netzes (Newson/Krumm-artig, exponentiell in der Abweichung der Schrittlänge).
Die Übergangsmatrix ist dünn besetzt (nur Zellen innerhalb weniger Schritte): sie wird als Nachbarliste gespeichert, die dichte Matrix gibt es
nur zur Gegenprobe. Jede Funktion ist eine Einheit mit expliziten Ein- und Ausgaben.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Network:
    coords: np.ndarray    # (N, 2) Zellenmitten in m
    adj: np.ndarray       # (N, N) bool: Nachbarzellen (ein Schritt)
    hops: np.ndarray      # (N, N) uint8: Zahl der Schritte (0 .. max_hops), max_hops + 1 = ferner
    cell: float


@lru_cache(maxsize=8)
def build_network(nodes, spacing, cell=10.0, max_hops=4):
    """Gitter mit nodes x nodes Kreuzungen; jede Straße zwischen zwei Kreuzungen besteht aus spacing/cell Zellen.

    Zellen an derselben Kreuzung sind miteinander verbunden. Schrittzahlen bis max_hops werden durch wiederholte Nachbarschaftsprodukte gerechnet.
    """
    n_cells = int(round(spacing / cell))
    coords, ends = [], {}
    links = []
    for i in range(nodes):
        for j in range(nodes):
            if i + 1 < nodes:
                links.append(((i, j), (i + 1, j)))
            if j + 1 < nodes:
                links.append(((i, j), (i, j + 1)))
    edge_starts = []
    for a, b in links:
        base = len(coords)
        pa, pb = np.array(a, float) * spacing, np.array(b, float) * spacing
        for k in range(n_cells):
            coords.append(pa + (k + 0.5) / n_cells * (pb - pa))
        edge_starts.append(base)
        ends.setdefault(a, []).append(base)
        ends.setdefault(b, []).append(base + n_cells - 1)
    N = len(coords)
    adj = np.zeros((N, N), bool)
    for base in edge_starts:
        for k in range(n_cells - 1):
            adj[base + k, base + k + 1] = adj[base + k + 1, base + k] = True
    for cells in ends.values():
        for u in cells:
            for v in cells:
                if u != v:
                    adj[u, v] = True
    hops = np.full((N, N), max_hops + 1, np.uint8)
    reach = np.eye(N, dtype=bool)
    hops[reach] = 0
    A = adj.astype(np.float32)
    frontier = reach.astype(np.float32)
    seen = reach.copy()
    for h in range(1, max_hops + 1):
        frontier = (frontier @ A) > 0
        new = frontier & ~seen
        hops[new] = h
        seen |= frontier
        frontier = frontier.astype(np.float32)
    return Network(np.array(coords), adj, hops, cell)


@dataclass(frozen=True)
class Model:
    """Dünn besetzte Übergangsstruktur: Gewichte W[i, j] (symmetrisch) und Zeilensummen r[i]; A[i, j] = W[i, j] / r[i]."""
    idx: np.ndarray       # (N, K) Nachbarn j von i (aufgefüllt mit i selbst)
    w: np.ndarray         # (N, K) Gewichte W[i, j] (0 bei Auffüllung)
    r: np.ndarray         # (N,) Zeilensummen
    log_w: np.ndarray     # (N, K) log W, -inf bei Auffüllung


def build_model(net, beta, max_step_hops=3):
    """Übergänge: aus Zelle i nach Zelle j mit Gewicht exp(-|d_ij - cell| / beta) für d_ij <= max_step_hops * cell (inklusive Stehenbleiben)."""
    N = len(net.coords)
    lists = [np.flatnonzero(net.hops[i] <= max_step_hops) for i in range(N)]
    K = max(len(x) for x in lists)
    idx = np.tile(np.arange(N)[:, None], (1, K))
    w = np.zeros((N, K))
    for i, nb in enumerate(lists):
        idx[i, :len(nb)] = nb
        w[i, :len(nb)] = np.exp(-np.abs(net.hops[i, nb].astype(float) * net.cell - net.cell) / beta)
    r = w.sum(axis=1)
    with np.errstate(divide="ignore"):
        log_w = np.where(w > 0, np.log(np.where(w > 0, w, 1.0)), -np.inf)
    return Model(idx, w, r, log_w)


def dense_matrix(model):
    """Dichte Übergangsmatrix A[i, j] (nur zur Gegenprobe und für kleine Netze)."""
    N = len(model.r)
    A = np.zeros((N, N))
    for i in range(N):
        for k in range(model.idx.shape[1]):
            if model.w[i, k] > 0:
                A[i, model.idx[i, k]] += model.w[i, k] / model.r[i]
    return A


def emission(net, z, sigma):
    """Wahrscheinlichkeit der GPS-Messung z je Zelle (Gauß um die Zellenmitte, unnormiert in der Position; sigma in m)."""
    d2 = ((net.coords - z) ** 2).sum(axis=1)
    return np.exp(-d2 / (2.0 * sigma ** 2)) + 1e-300


def snap_path(net, Z):
    """Ohne Modell: je Messung die nächstgelegene Zelle."""
    return np.array([int(((net.coords - z) ** 2).sum(axis=1).argmin()) for z in Z])


def _push(model, a):
    """a @ A: Wahrscheinlichkeiten einen Schritt vorwärts. new[j] = sum_i a[i] W[i, j] / r[i] = sum_k W[j, k] (a / r)[idx[j, k]] (W symmetrisch)."""
    return (model.w * (a / model.r)[model.idx]).sum(axis=1)


def _pull(model, x):
    """A @ x: Rückwärtsschritt. out[i] = sum_j A[i, j] x[j]."""
    return (model.w * x[model.idx]).sum(axis=1) / model.r


def forward_filter(net, model, Z, sigma):
    """Vorwärts-Filter: Wahrscheinlichkeit jeder Zelle gegeben alle Messungen BIS jetzt. Gibt (T, N) zurück (Zeilen normiert)."""
    T, N = len(Z), len(net.coords)
    al = np.zeros((T, N))
    a = emission(net, Z[0], sigma)
    al[0] = a / a.sum()
    for t in range(1, T):
        a = _push(model, al[t - 1]) * emission(net, Z[t], sigma)
        s = a.sum()
        al[t] = a / s if s > 0 else np.full(N, 1.0 / N)
    return al


def forward_backward(net, model, Z, sigma):
    """Glätter: Wahrscheinlichkeit jeder Zelle gegeben ALLE Messungen (Vorwärts-Rückwärts, skaliert). Gibt (T, N) zurück."""
    T, N = len(Z), len(net.coords)
    E = np.array([emission(net, z, sigma) for z in Z])
    al = np.zeros((T, N))
    c = np.zeros(T)
    al[0] = E[0] / N
    c[0] = al[0].sum()
    al[0] /= c[0]
    for t in range(1, T):
        al[t] = _push(model, al[t - 1]) * E[t]
        c[t] = al[t].sum()
        al[t] /= c[t]
    be = np.ones((T, N))
    for t in range(T - 2, -1, -1):
        be[t] = _pull(model, E[t + 1] * be[t + 1])
        be[t] /= c[t + 1]
    g = al * be
    return g / g.sum(axis=1, keepdims=True)


def viterbi(net, model, Z, sigma):
    """Wahrscheinlichster GEMEINSAMER Pfad (Viterbi): dynamische Programmierung im Trellis, Rückverfolgung. Gibt (T,) Zellennummern zurück.

    Trellis-Schritt: d_t[j] = max_i ( d_{t-1}[i] + log A[i, j] ) + log E_t[j]; mit A[i, j] = W[j, i] / r[i] nur über die Nachbarn von j.
    """
    N = len(net.coords)
    log_r = np.log(model.r)
    d = np.log(emission(net, Z[0], sigma))
    back = []
    for t in range(1, len(Z)):
        cand = (d - log_r)[model.idx] + model.log_w          # (N, K): Kandidaten i für jedes j
        k = cand.argmax(axis=1)
        back.append(model.idx[np.arange(N), k])
        d = cand[np.arange(N), k] + np.log(emission(net, Z[t], sigma))
    path = [int(d.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return np.array(path[::-1])
