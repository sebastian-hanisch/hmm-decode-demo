"""Einheiten des HMM-Kerns gegen Handrechnung auf dem Mini-Netz (conftest.mini_net) und Randfälle."""

import numpy as np
import pytest

import hmd_algorithm as A
import hmd_constants as C
import hmd_scenario as S


def test_mini_net_size_and_symmetry(mini_net):
    assert len(mini_net.coords) == 8
    assert np.array_equal(mini_net.hops, mini_net.hops.T) and np.array_equal(mini_net.adj, mini_net.adj.T)
    assert np.all(np.diag(mini_net.hops) == 0)
    assert np.all(mini_net.hops[mini_net.adj] == 1)


def test_cell_midpoints_of_the_first_street(mini_net):
    # erste Straße: von Kreuzung (0,0) nach (1,0), Länge 20 m entlang x: Zellenmitten bei 5 m und 15 m
    assert mini_net.coords[0] == pytest.approx([5.0, 0.0]) and mini_net.coords[1] == pytest.approx([15.0, 0.0])
    assert mini_net.hops[0, 1] == 1


def test_standard_network_sizes():
    for spacing, cells in ((30, 252), (50, 420), (100, 840)):
        net = A.build_network(C.NODES, spacing, C.CELL, C.MAX_HOPS)
        assert len(net.coords) == cells


def test_transition_weights_and_rows(mini_net, mini_model):
    A_dense = A.dense_matrix(mini_model)
    assert A_dense.sum(axis=1) == pytest.approx(np.ones(8))
    i = 0
    row = mini_model.w[i]
    nb = mini_model.idx[i][row > 0]
    d = mini_net.hops[i, nb] * mini_net.cell
    assert row[row > 0] == pytest.approx(np.exp(-np.abs(d - 10.0) / 5.0))
    assert np.exp(-2.0) == pytest.approx(row[row > 0][list(nb).index(i)])       # Stehenbleiben: exp(-10/5)


def test_emission_peaks_at_the_nearest_cell_center(mini_net):
    z = mini_net.coords[3]
    e = A.emission(mini_net, z, 10.0)
    assert int(e.argmax()) == 3 and e[3] == pytest.approx(1.0)


def test_snap_picks_the_nearest_cell(mini_net):
    Z = mini_net.coords[[0, 5, 2]] + 0.1
    assert A.snap_path(mini_net, Z).tolist() == [0, 5, 2]


def _connected_path(net, start, steps):
    """Ein zusammenhängender Weg ohne sofortige Umkehr, aus der Nachbarschaft des Netzes gebaut."""
    path = [start]
    prev = -1
    for _ in range(steps):
        cur = path[-1]
        nxt = [int(n) for n in np.flatnonzero(net.adj[cur]) if n != prev]
        prev = cur
        path.append(nxt[0])
    return np.array(path)


def test_noise_free_measurements_are_decoded_exactly_by_all_methods(mini_net, mini_model):
    path = _connected_path(mini_net, 0, 4)
    for a, b in zip(path[:-1], path[1:]):
        assert mini_net.hops[a, b] == 1
    Z = mini_net.coords[path]
    sigma = 1.0
    assert A.forward_filter(mini_net, mini_model, Z, sigma).argmax(axis=1).tolist() == path.tolist()
    assert A.forward_backward(mini_net, mini_model, Z, sigma).argmax(axis=1).tolist() == path.tolist()
    assert A.viterbi(mini_net, mini_model, Z, sigma).tolist() == path.tolist()


def test_posteriors_are_probability_distributions(mini_net, mini_model):
    rng = np.random.default_rng(0)
    Z = mini_net.coords[[0, 1, 5, 4, 4]] + rng.normal(0, 8, (5, 2))
    for post in (A.forward_filter(mini_net, mini_model, Z, 8.0), A.forward_backward(mini_net, mini_model, Z, 8.0)):
        assert post.shape == (5, 8) and np.all(post >= 0)
        assert post.sum(axis=1) == pytest.approx(np.ones(5))


def test_last_filter_and_smoother_distributions_coincide(mini_net, mini_model):
    rng = np.random.default_rng(1)
    Z = mini_net.coords[[0, 1, 5, 4, 4, 6]] + rng.normal(0, 8, (6, 2))
    f = A.forward_filter(mini_net, mini_model, Z, 8.0)
    s = A.forward_backward(mini_net, mini_model, Z, 8.0)
    assert f[-1] == pytest.approx(s[-1])


def test_random_walk_moves_to_neighbours_and_does_not_reverse_on_streets():
    net = A.build_network(C.NODES, 100, C.CELL, C.MAX_HOPS)
    rng = np.random.default_rng(5)
    path = S.random_walk(net, rng, 200)
    assert all(net.adj[a, b] for a, b in zip(path[:-1], path[1:]))
    assert all(a != c for a, c in zip(path[:-2], path[2:]))      # nie sofort zurück (Sackgassen gibt es im Gitter nicht)


def test_simulation_is_reproducible_and_sigma_scales_the_same_noise():
    net = A.build_network(C.NODES, 100, C.CELL, C.MAX_HOPS)
    a = S.simulate(net, 7, 10.0)
    b = S.simulate(net, 7, 10.0)
    assert np.array_equal(a.path, b.path) and np.array_equal(a.z, b.z)
    c = S.simulate(net, 7, 20.0)
    assert np.array_equal(a.path, c.path)
    assert np.allclose(c.z - c.pos, 2 * (a.z - a.pos))


def test_constants_are_consistent():
    assert all(sp % int(C.CELL) == 0 for sp in C.SPACING_CHOICES) and C.DEFAULT_SPACING in C.SPACING_CHOICES
    assert C.DEFAULT_ASSUMED in C.ASSUMED_CHOICES and set(C.METHODS) == set(C.METHOD_LABELS) == set(C.COLORS)
    assert C.HIT_RADIUS >= C.CELL and C.JUMP_DIST < C.MAX_HOPS * C.CELL
