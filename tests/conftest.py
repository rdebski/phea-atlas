"""
Shared fixtures for the gp-hea-screening test suite.

The GP + calibration predictor is expensive to load (torch model + sklearn
pipeline), so it is loaded once per session.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def predictor():
    from src.phea.predict import HEAPredictor
    return HEAPredictor.load()


@pytest.fixture(scope="session")
def gp():
    from src.gp.predict import GPPredictor
    return GPPredictor.load(str(ROOT / "models" / "gp_full_model.pt"))


@pytest.fixture(scope="session")
def feat(gp):
    from src.phea.features import MulticomponentFeatures
    return MulticomponentFeatures(gp)


# Canonical 44-element pool (single source of truth for tests)
POOL = [
    "Ag", "Al", "Au", "B",  "Ba", "Bi", "Ca", "Cd", "Ce", "Co",
    "Cr", "Cu", "Eu", "Fe", "Ga", "Gd", "Ge", "Hg", "In", "Ir",
    "Li", "Mg", "Mn", "Na", "Nb", "Ni", "Pb", "Pd", "Pt", "Rh",
    "Sb", "Sc", "Se", "Si", "Sn", "Sr", "Tb", "Te", "Ti", "Tl",
    "Y",  "Yb", "Zn", "Zr",
]

CANTOR = {"Co": 0.2, "Cr": 0.2, "Fe": 0.2, "Mn": 0.2, "Ni": 0.2}
COCRCUFENI = {"Co": 0.2, "Cr": 0.2, "Cu": 0.2, "Fe": 0.2, "Ni": 0.2}
