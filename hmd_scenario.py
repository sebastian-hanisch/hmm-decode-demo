"""Szenario: ein Fahrzeug fährt eine Zelle je Sekunde durch das Straßengitter, GPS misst seine Position verrauscht."""

from dataclasses import dataclass

import numpy as np

import hmd_constants as C


@dataclass(frozen=True)
class Track:
    path: np.ndarray     # (T,) wahre Zellennummern
    pos: np.ndarray      # (T, 2) wahre Position (Zellenmitte)
    z: np.ndarray        # (T, 2) GPS-Messung


def random_walk(net, rng, T):
    """Zufällige Fahrt: Start in zufälliger Zelle, danach je Sekunde eine Nachbarzelle, nie sofort zurück (außer an Sackgassen)."""
    N = len(net.coords)
    neighbours = [np.flatnonzero(net.adj[i]) for i in range(N)]
    c = int(rng.integers(N))
    prev = -1
    path = [c]
    for _ in range(T - 1):
        cand = [n for n in neighbours[c] if n != prev]
        if not cand:
            cand = list(neighbours[c])
        nxt = int(cand[int(rng.integers(len(cand)))])
        prev, c = c, nxt
        path.append(c)
    return np.array(path)


def simulate(net, seed, sigma):
    """Wahre Fahrt und GPS-Messungen. Der Zufallsstrom ist fest: ein anderes sigma skaliert nur dasselbe Rauschen."""
    rng = np.random.default_rng(seed)
    path = random_walk(net, rng, C.T_STEPS)
    noise = rng.standard_normal((C.T_STEPS, 2))
    pos = net.coords[path]
    return Track(path, pos, pos + sigma * noise)
