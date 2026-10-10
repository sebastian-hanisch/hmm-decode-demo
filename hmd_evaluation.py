"""Kennzahlen, Verdikte und Sweeps über die festen Instanzen (SWEEP_SEEDS)."""

from dataclasses import dataclass, replace

import numpy as np

import hmd_algorithm as A
import hmd_constants as C
import hmd_scenario as S


@dataclass(frozen=True)
class Settings:
    sigma: float = C.DEFAULT_SIGMA          # wahres GPS-Rauschen in m
    spacing: int = C.DEFAULT_SPACING        # Straßenabstand in m
    assumed: float = C.DEFAULT_ASSUMED      # Annahme des Modells zum Rauschen als Vielfaches des wahren Werts
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    net: A.Network
    track: S.Track
    paths: dict             # Methode -> (T,) Zellennummern
    post_filter: np.ndarray   # (T, N)
    post_smooth: np.ndarray
    hit: dict               # Methode -> Trefferquote (Netzabstand zur wahren Zelle <= HIT_RADIUS)
    jumps: dict             # Methode -> Anteil unmöglicher Sprünge zwischen aufeinanderfolgenden Schätzungen
    conf_filter: float      # mittlere Sicherheit von Filter / Glätter: Wahrscheinlichkeit, dass die Wahrheit im Trefferradius der Schätzung liegt
    conf_smooth: float


@dataclass(frozen=True)
class Decoded:
    paths: dict
    post_filter: np.ndarray
    post_smooth: np.ndarray


def decode(net, model, Z, sigma_assumed):
    pf = A.forward_filter(net, model, Z, sigma_assumed)
    ps = A.forward_backward(net, model, Z, sigma_assumed)
    paths = {"snap": A.snap_path(net, Z), "filter": pf.argmax(axis=1), "smoother": ps.argmax(axis=1), "viterbi": A.viterbi(net, model, Z, sigma_assumed)}
    return Decoded(paths, pf, ps)


def hit_confidence(net, post):
    """Je Zeitpunkt: Wahrscheinlichkeitsmasse der Zellen im Trefferradius um die wahrscheinlichste Zelle = Modell-Sicherheit, dass die Schätzung „trifft“."""
    best = post.argmax(axis=1)
    near = net.hops[best] * net.cell <= C.HIT_RADIUS          # (T, N)
    return (post * near).sum(axis=1)


def hit_rate(net, est, truth):
    return float(np.mean(net.hops[est, truth] * net.cell <= C.HIT_RADIUS))


def jump_rate(net, est):
    """Anteil aufeinanderfolgender Schätzungen, deren Netzabstand über JUMP_DIST liegt („fern“ oder mehr als MAX_HOPS Schritte)."""
    return float(np.mean(net.hops[est[:-1], est[1:]] * net.cell > C.JUMP_DIST))


def analyse(settings):
    net = A.build_network(C.NODES, settings.spacing, C.CELL, C.MAX_HOPS)
    model = A.build_model(net, C.BETA)
    tr = S.simulate(net, settings.seed, settings.sigma)
    dec = decode(net, model, tr.z, settings.sigma * settings.assumed)
    hit = {m: hit_rate(net, p, tr.path) for m, p in dec.paths.items()}
    jumps = {m: jump_rate(net, p) for m, p in dec.paths.items()}
    return Analysis(settings, net, tr, dec.paths, dec.post_filter, dec.post_smooth, hit, jumps,
                    float(hit_confidence(net, dec.post_filter).mean()), float(hit_confidence(net, dec.post_smooth).mean()))


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------

def _traces(settings):
    """Dekodierte Fahrten über die festen Instanzen: Liste von (net, track, Decoded)."""
    net = A.build_network(C.NODES, settings.spacing, C.CELL, C.MAX_HOPS)
    model = A.build_model(net, C.BETA)
    out = []
    for s in C.SWEEP_SEEDS:
        tr = S.simulate(net, s, settings.sigma)
        out.append((net, tr, decode(net, model, tr.z, settings.sigma * settings.assumed)))
    return out


def _summary(settings):
    res = _traces(settings)
    hit = {m: float(np.mean([hit_rate(n, d.paths[m], t.path) for n, t, d in res])) for m in C.METHODS}
    jumps = {m: float(np.mean([jump_rate(n, d.paths[m]) for n, t, d in res])) for m in C.METHODS}
    return hit, jumps


def sweep_sigma(settings):
    """GPS-Rauschen variieren: Trefferquote und unmögliche Sprünge der vier Verfahren (Mittel über die festen Fahrten)."""
    rows = []
    for sg in C.SIGMA_SWEEP:
        hit, jumps = _summary(replace(settings, sigma=sg))
        rows.append({"sigma": sg, "hit": hit, "jumps": jumps})
    return rows


def sweep_spacing(settings):
    """Straßenabstand variieren (enges Netz = mehr Verwechslungsmöglichkeiten) bei dem Rauschen der Seitenleiste."""
    rows = []
    for sp in C.SPACING_CHOICES:
        hit, jumps = _summary(replace(settings, spacing=sp))
        rows.append({"spacing": sp, "hit": hit, "jumps": jumps})
    return rows


def _calibration_stats(conf, hit):
    return {"conf": float(conf.mean()), "hit": float(hit.mean())}


def sweep_calibration(settings):
    """Sind die Wahrscheinlichkeiten ehrlich? Modell-Sicherheit (Masse im Trefferradius um die Schätzung) gegen tatsächliche Trefferquote, für Filter und
    Glätter und für ein Modell, das dem GPS halb so viel, genau so viel oder doppelt so viel Rauschen zutraut."""
    out = []
    for fac in C.ASSUMED_CHOICES:
        res = _traces(replace(settings, assumed=fac))
        row = {"assumed": fac}
        for name, key in (("filter", "post_filter"), ("smoother", "post_smooth")):
            conf = np.concatenate([hit_confidence(n, getattr(d, key)) for n, t, d in res])
            hits = np.concatenate([n.hops[getattr(d, key).argmax(axis=1), t.path] * n.cell <= C.HIT_RADIUS for n, t, d in res])
            row[name] = _calibration_stats(conf, hits)
        out.append(row)
    return out


def sample_from_model(net, model, rng, T):
    """Eine Fahrt aus dem Übergangsmodell selbst (mit Stehenbleiben, Zurückfahren, Sprüngen über bis zu drei Zellen)."""
    c = int(rng.integers(len(net.coords)))
    path = [c]
    for _ in range(T - 1):
        p = model.w[c] / model.r[c]
        c = int(model.idx[c, rng.choice(len(p), p=p)])
        path.append(c)
    return np.array(path)


def self_calibration(settings, n_tracks=None):
    """Selbstprüfung: Daten aus dem Modell selbst erzeugt. Dort MUSS die Modell-Sicherheit gleich der Trefferquote sein (sonst wäre der Code falsch)."""
    net = A.build_network(C.NODES, settings.spacing, C.CELL, C.MAX_HOPS)
    model = A.build_model(net, C.BETA)
    rng = np.random.default_rng(424242)
    acc = {"filter": ([], []), "smoother": ([], [])}
    for _ in range(n_tracks or len(C.SWEEP_SEEDS)):
        path = sample_from_model(net, model, rng, C.T_STEPS)
        Z = net.coords[path] + settings.sigma * rng.standard_normal((C.T_STEPS, 2))
        posts = {"filter": A.forward_filter(net, model, Z, settings.sigma), "smoother": A.forward_backward(net, model, Z, settings.sigma)}
        for name, post in posts.items():
            acc[name][0].append(hit_confidence(net, post))
            acc[name][1].append(net.hops[post.argmax(axis=1), path] * net.cell <= C.HIT_RADIUS)
    return {name: _calibration_stats(np.concatenate(c), np.concatenate(h)) for name, (c, h) in acc.items()}
