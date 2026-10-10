"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (30 feste Fahrten).

Plattformrobust: Bänder statt exakter Werte (CI installiert neueste numpy/scipy)."""

import pytest

import hmd_constants as C
import hmd_evaluation as ev


@pytest.fixture(scope="module")
def sigma_rows():
    return {r["sigma"]: r for r in ev.sweep_sigma(ev.Settings())}


@pytest.fixture(scope="module")
def spacing_rows():
    return {r["spacing"]: r for r in ev.sweep_spacing(ev.Settings(sigma=30.0))}


@pytest.fixture(scope="module")
def calibration():
    s = ev.Settings(sigma=30.0, spacing=50)
    return {r["assumed"]: r for r in ev.sweep_calibration(s)}, ev.self_calibration(s)


def test_the_future_is_what_helps_not_viterbi(sigma_rows):
    r = sigma_rows[20.0]["hit"]
    assert r["snap"] == pytest.approx(0.39, abs=0.07) and r["filter"] == pytest.approx(0.62, abs=0.08)
    assert r["smoother"] == pytest.approx(0.90, abs=0.06) and r["viterbi"] == pytest.approx(0.87, abs=0.06)
    for s in C.SIGMA_SWEEP:
        h = sigma_rows[s]["hit"]
        assert h["smoother"] > h["filter"] + 0.08 and h["filter"] > h["snap"]
        assert abs(h["smoother"] - h["viterbi"]) < 0.07


def test_filter_jumps_viterbi_never(sigma_rows):
    for s in C.SIGMA_SWEEP:
        j = sigma_rows[s]["jumps"]
        assert j["viterbi"] == 0.0 and j["smoother"] < 0.01
    assert sigma_rows[10.0]["jumps"]["filter"] < 0.015
    assert sigma_rows[30.0]["jumps"]["filter"] == pytest.approx(0.04, abs=0.025)
    assert sigma_rows[60.0]["jumps"]["filter"] == pytest.approx(0.10, abs=0.04)
    assert sigma_rows[10.0]["jumps"]["snap"] == pytest.approx(0.11, abs=0.05) and sigma_rows[60.0]["jumps"]["snap"] > 0.8


def test_dense_network_costs_accuracy(spacing_rows):
    tight, mid, wide = spacing_rows[30]["hit"], spacing_rows[50]["hit"], spacing_rows[100]["hit"]
    assert tight["smoother"] == pytest.approx(0.56, abs=0.08) and wide["smoother"] == pytest.approx(0.79, abs=0.07)
    assert tight["smoother"] < mid["smoother"] < wide["smoother"]
    assert tight["viterbi"] < wide["viterbi"] and tight["smoother"] > tight["viterbi"]


def test_self_calibration_filter_and_smoother_are_honest_on_model_data(calibration):
    _, sc = calibration
    assert sc["filter"]["conf"] == pytest.approx(sc["filter"]["hit"], abs=0.045)
    assert sc["smoother"]["conf"] == pytest.approx(sc["smoother"]["hit"], abs=0.045)
    assert sc["filter"]["conf"] == pytest.approx(0.54, abs=0.06) and sc["smoother"]["conf"] == pytest.approx(0.68, abs=0.06)


def test_misspecified_model_is_overconfident(calibration):
    rows, _ = calibration
    one, half = rows[1.0], rows[0.5]
    assert one["filter"]["conf"] - one["filter"]["hit"] == pytest.approx(0.12, abs=0.06)
    assert abs(one["smoother"]["conf"] - one["smoother"]["hit"]) < 0.08
    assert half["filter"]["conf"] - half["filter"]["hit"] == pytest.approx(0.30, abs=0.08)
    assert half["smoother"]["conf"] - half["smoother"]["hit"] == pytest.approx(0.24, abs=0.08)


def test_sweeps_use_the_fixed_instances():
    assert len(C.SWEEP_SEEDS) == 30 and C.SWEEP_SEEDS[0] == 100000
