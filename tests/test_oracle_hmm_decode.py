"""Orakel-Test hmm-decode-demo: Filter, Glätter und Viterbi gegen ANDERE Rechenwege.

1. Aufzählung aller Pfade auf dem Mini-Netz: Viterbi-Pfad, Glätter-Randverteilungen, Filterverteilungen,
2. Viterbi als kürzester Weg im Trellis (networkx, Dijkstra) mit Kosten -log A - log b,
3. die dünn besetzte Implementierung gegen eine eigenständige dichte Rechnung auf dem großen Netz,
4. Selbstkalibrierung: Fahrten aus dem Modell selbst, dort ist die Modell-Sicherheit gleich der Trefferquote,
5. die angezeigten Kennzahlen gegen unabhängige Neuberechnung.
"""

import itertools

import numpy as np
import pytest

import hmd_algorithm as A
import hmd_constants as C
import hmd_evaluation as ev
import hmd_scenario as S


def _dense_A(net, beta, max_step_hops=3):
    d = net.hops.astype(float) * net.cell
    W = np.where(net.hops <= max_step_hops, np.exp(-np.abs(d - net.cell) / beta), 0.0)
    return W / W.sum(axis=1, keepdims=True)


def _all_path_weights(net, A_mat, Z, sigma):
    """Alle Pfade x_1..x_T mit ihrem unnormierten Gewicht (gleichverteilter Start) · Produkt der Übergänge · Produkt der Messwahrscheinlichkeiten."""
    N, T = len(net.coords), len(Z)
    E = np.array([A.emission(net, z, sigma) for z in Z])
    paths = np.array(list(itertools.product(range(N), repeat=T)))
    w = E[0][paths[:, 0]]
    for t in range(1, T):
        w = w * A_mat[paths[:, t - 1], paths[:, t]] * E[t][paths[:, t]]
    return paths, w


def _noisy(net, path, sigma, seed):
    return net.coords[path] + np.random.default_rng(seed).normal(0, sigma, (len(path), 2))


def test_viterbi_and_marginals_against_enumeration_of_all_paths(mini_net, mini_model):
    A_mat = _dense_A(mini_net, 5.0)
    assert A.dense_matrix(mini_model) == pytest.approx(A_mat)
    for seed in range(8):
        path = np.array([0, 1, 5, 4, 3])[: 4]
        Z = _noisy(mini_net, path, 12.0, seed)
        paths, w = _all_path_weights(mini_net, A_mat, Z, 12.0)
        best = paths[int(np.argmax(w))]
        assert A.viterbi(mini_net, mini_model, Z, 12.0).tolist() == best.tolist()
        gamma = A.forward_backward(mini_net, mini_model, Z, 12.0)
        for t in range(len(Z)):
            marg = np.bincount(paths[:, t], weights=w, minlength=8)
            assert gamma[t] == pytest.approx(marg / marg.sum(), abs=1e-9)


def test_filter_distribution_equals_the_enumerated_prefix_marginal(mini_net, mini_model):
    A_mat = _dense_A(mini_net, 5.0)
    path = np.array([0, 1, 5, 4])
    Z = _noisy(mini_net, path, 12.0, 3)
    alpha = A.forward_filter(mini_net, mini_model, Z, 12.0)
    for t in range(len(Z)):
        paths, w = _all_path_weights(mini_net, A_mat, Z[: t + 1], 12.0)
        marg = np.bincount(paths[:, t], weights=w, minlength=8)
        assert alpha[t] == pytest.approx(marg / marg.sum(), abs=1e-9)


def test_viterbi_is_the_shortest_path_in_the_trellis():
    nx = pytest.importorskip("networkx")
    net = A.build_network(2, 20, 10.0, 4)
    model = A.build_model(net, 5.0)
    A_mat = _dense_A(net, 5.0)
    N = len(net.coords)
    for seed in range(5):
        Z = _noisy(net, np.array([0, 1, 5, 4, 3, 2]), 12.0, seed)
        T = len(Z)
        E = np.array([A.emission(net, z, 12.0) for z in Z])
        G = nx.DiGraph()
        for j in range(N):
            G.add_edge("s", (0, j), weight=-np.log(E[0, j]))
            G.add_edge((T - 1, j), "t", weight=0.0)
        for t in range(1, T):
            for i in range(N):
                for j in range(N):
                    if A_mat[i, j] > 0:
                        G.add_edge((t - 1, i), (t, j), weight=-np.log(A_mat[i, j]) - np.log(E[t, j]))
        sp = nx.shortest_path(G, "s", "t", weight="weight")
        ref = [node[1] for node in sp[1:-1]]
        assert A.viterbi(net, model, Z, 12.0).tolist() == ref


def _log_score(net, A_mat, Z, sigma, path):
    s = np.log(A.emission(net, Z[0], sigma)[path[0]])
    for t in range(1, len(Z)):
        s += np.log(A_mat[path[t - 1], path[t]] + 1e-300) + np.log(A.emission(net, Z[t], sigma)[path[t]])
    return float(s)


def test_sparse_implementation_matches_independent_dense_computation_on_the_big_network():
    net = A.build_network(C.NODES, 50, C.CELL, C.MAX_HOPS)
    model = A.build_model(net, C.BETA)
    A_mat = _dense_A(net, C.BETA)
    for seed in range(3):
        tr = S.simulate(net, 900 + seed, 25.0)
        Z, sigma = tr.z, 25.0
        # Filter und Glätter: dichte Vorwärts-Rückwärts-Rechnung
        E = np.array([A.emission(net, z, sigma) for z in Z])
        al = np.zeros((len(Z), len(net.coords)))
        al[0] = E[0] / E[0].sum()
        for t in range(1, len(Z)):
            a = (al[t - 1] @ A_mat) * E[t]
            al[t] = a / a.sum()
        assert A.forward_filter(net, model, Z, sigma) == pytest.approx(al, abs=1e-10)
        be = np.ones_like(al)
        for t in range(len(Z) - 2, -1, -1):
            b = A_mat @ (E[t + 1] * be[t + 1])
            be[t] = b / b.sum()
        g = al * be
        g /= g.sum(axis=1, keepdims=True)
        assert A.forward_backward(net, model, Z, sigma) == pytest.approx(g, abs=1e-9)
        # Viterbi: gleicher Pfadwert wie die dichte Rechnung (Pfad selbst nur bei Eindeutigkeit vergleichbar)
        v_sparse = A.viterbi(net, model, Z, sigma)
        d = np.log(E[0])
        logA = np.log(A_mat + 1e-300)
        back = []
        for t in range(1, len(Z)):
            cand = d[:, None] + logA
            back.append(cand.argmax(axis=0))
            d = cand.max(axis=0) + np.log(E[t])
        v_dense = [int(d.argmax())]
        for b in reversed(back):
            v_dense.append(int(b[v_dense[-1]]))
        v_dense = np.array(v_dense[::-1])
        assert _log_score(net, A_mat, Z, sigma, v_sparse) == pytest.approx(_log_score(net, A_mat, Z, sigma, v_dense), abs=1e-8)


def test_viterbi_path_is_locally_optimal_and_beats_the_smoother_path():
    net = A.build_network(C.NODES, 50, C.CELL, C.MAX_HOPS)
    model = A.build_model(net, C.BETA)
    A_mat = _dense_A(net, C.BETA)
    rng = np.random.default_rng(2)
    for seed in range(4):
        tr = S.simulate(net, 950 + seed, 30.0)
        v = A.viterbi(net, model, tr.z, 30.0)
        best = _log_score(net, A_mat, tr.z, 30.0, v)
        sm = A.forward_backward(net, model, tr.z, 30.0).argmax(axis=1)
        assert best >= _log_score(net, A_mat, tr.z, 30.0, sm) - 1e-9
        for _ in range(100):
            alt = v.copy()
            t = int(rng.integers(len(alt)))
            nb = np.flatnonzero(net.hops[alt[t]] <= 2)
            alt[t] = int(rng.choice(nb))
            assert best >= _log_score(net, A_mat, tr.z, 30.0, alt) - 1e-9


def test_self_calibration_on_data_drawn_from_the_model():
    sc = ev.self_calibration(ev.Settings(sigma=30.0, spacing=50), n_tracks=40)
    for name in ("filter", "smoother"):
        assert sc[name]["conf"] == pytest.approx(sc[name]["hit"], abs=0.045), name
    assert sc["smoother"]["conf"] > sc["filter"]["conf"]


def test_displayed_numbers_match_independent_recomputation():
    for seed in (3, 11):
        s = ev.Settings(sigma=25.0, spacing=50, seed=seed)
        a = ev.analyse(s)
        net = A.build_network(C.NODES, 50, C.CELL, C.MAX_HOPS)
        model = A.build_model(net, C.BETA)
        tr = S.simulate(net, seed, 25.0)
        v = A.viterbi(net, model, tr.z, 25.0)
        assert a.paths["viterbi"].tolist() == v.tolist()
        manual_hit = np.mean([net.hops[v[t], tr.path[t]] * net.cell <= C.HIT_RADIUS for t in range(len(v))])
        assert a.hit["viterbi"] == pytest.approx(manual_hit)
        manual_jump = np.mean([net.hops[v[t], v[t + 1]] * net.cell > C.JUMP_DIST for t in range(len(v) - 1)])
        assert a.jumps["viterbi"] == pytest.approx(manual_jump) and a.jumps["viterbi"] == 0.0
        post = A.forward_filter(net, model, tr.z, 25.0)
        best = post.argmax(axis=1)
        conf = np.mean([post[t][net.hops[best[t]] * net.cell <= C.HIT_RADIUS].sum() for t in range(len(best))])
        assert a.conf_filter == pytest.approx(conf)
