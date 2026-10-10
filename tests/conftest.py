"""Gemeinsame Fixtures: das Mini-Netz, an dem jede Einheit von Hand und durch Aufzählung geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hmd_algorithm as A  # noqa: E402


@pytest.fixture(scope="session")
def mini_net():
    """2 x 2 Kreuzungen, Straßenabstand 20 m, Zellen von 10 m: 4 Straßen zu je 2 Zellen = 8 Zellen.

    Handrechnung: die beiden Zellen einer Straße sind benachbart (1 Schritt); an jeder Kreuzung treffen 2 Straßenenden aufeinander und sind miteinander benachbart.
    """
    return A.build_network(2, 20, 10.0, 4)


@pytest.fixture(scope="session")
def mini_model(mini_net):
    return A.build_model(mini_net, 5.0)
