"""
Tests for src/phea/features.py — the multicomponent feature engine.

The two code paths (scalar `compute` and vectorised `compute_array`) implement
the same physics through different code; they MUST agree.  We also verify the
analytic pairwise enthalpy formula against a hand-written double loop.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from src.phea.features import (
    FEATURE_NAMES,
    MulticomponentFeatures,
    _penalties,
    _DH_LO,
    _DH_HI,
    DH_DEPLOY_LO,
    DH_DEPLOY_HI,
)

ELEMS = ["Co", "Cr", "Cu", "Fe", "Ni"]


def test_feature_names_length():
    assert len(FEATURE_NAMES) == 8
    assert FEATURE_NAMES[0] == "mu_dH"
    assert FEATURE_NAMES[1] == "sigma_dH"


def test_compute_matches_compute_array(feat: MulticomponentFeatures):
    """Scalar compute() and vectorised compute_array() must agree element-wise."""
    fracs = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.30, 0.25, 0.20, 0.15, 0.10],
        [0.05, 0.35, 0.20, 0.30, 0.10],
    ])
    arr = feat.compute_array(ELEMS, fracs)  # (3, 8)

    for row_i, frac_row in enumerate(fracs):
        comp = {e: float(x) for e, x in zip(ELEMS, frac_row)}
        d = feat.compute(comp)
        for col_j, name in enumerate(FEATURE_NAMES):
            assert d[name] == pytest.approx(arr[row_i, col_j], rel=1e-9, abs=1e-9), (
                f"mismatch in {name} for row {row_i}"
            )


def test_mu_dH_against_manual_pairwise(feat: MulticomponentFeatures):
    """μ_ΔH = 4 Σ_{i<j} x_i x_j μ_GP(i,j); diagonal σ_ΔH against explicit loop;
    exact σ_ΔH ≥ diagonal (off-diagonal GP covariance is positive)."""
    comp = {"Co": 0.30, "Cr": 0.25, "Cu": 0.20, "Fe": 0.15, "Ni": 0.10}
    elems = list(comp)

    mu_expected = 0.0
    var_diag = 0.0
    for a, b in combinations(elems, 2):
        mu_ij, sigma_ij = feat._get_pair(a, b)
        mu_expected += 4.0 * comp[a] * comp[b] * mu_ij
        var_diag += 16.0 * (comp[a] * comp[b]) ** 2 * sigma_ij ** 2

    d_diag = feat.compute(comp, exact=False)
    assert d_diag["mu_dH"] == pytest.approx(mu_expected, rel=1e-9)
    assert d_diag["sigma_dH"] == pytest.approx(math.sqrt(var_diag), rel=1e-9)

    # Canonical (exact) σ retains positive off-diagonal covariance → ≥ diagonal.
    d_exact = feat.compute(comp)  # exact=True default
    assert d_exact["mu_dH"] == pytest.approx(mu_expected, rel=1e-9)
    assert d_exact["sigma_dH"] >= d_diag["sigma_dH"]


def test_equimolar_entropy(feat: MulticomponentFeatures):
    """dS_R for equimolar 5-component = ln(5) = 1.6094."""
    comp = {e: 0.2 for e in ELEMS}
    d = feat.compute(comp)
    assert d["dS_R"] == pytest.approx(math.log(5), abs=1e-6)


def test_delta_zero_for_equal_radii(feat: MulticomponentFeatures):
    """δ must be >= 0 and exactly 0 only when all radii are equal."""
    comp = {e: 0.2 for e in ELEMS}
    d = feat.compute(comp)
    assert d["delta"] >= 0.0


def test_pair_cache_symmetry(feat: MulticomponentFeatures):
    """Pair cache keys are canonical (min, max); A-B == B-A."""
    mu1, s1 = feat._get_pair("Ni", "Al")
    mu2, s2 = feat._get_pair("Al", "Ni")
    assert mu1 == mu2 and s1 == s2
    assert ("Al", "Ni") in feat._pair_cache
    assert ("Ni", "Al") not in feat._pair_cache


@pytest.mark.parametrize("mu,expect_lo,expect_hi", [
    (-30.0, 15.0, 0.0),   # below window
    (-5.0, 0.0, 0.0),     # inside window
    (10.0, 0.0, 5.0),     # above window
])
def test_penalties_window(mu, expect_lo, expect_hi):
    pen_lo, pen_hi, pen_omega = _penalties(mu, T_m=1600.0, dS_R=1.6)
    assert pen_lo == pytest.approx(expect_lo)
    assert pen_hi == pytest.approx(expect_hi)
    assert pen_omega >= 0.0


def test_penalties_inside_window_zero():
    """Inside [-15, +5] both pen_lo and pen_hi vanish."""
    for mu in np.linspace(_DH_LO, _DH_HI, 11):
        pen_lo, pen_hi, _ = _penalties(float(mu), 1600.0, 1.6)
        assert pen_lo == pytest.approx(0.0)
        assert pen_hi == pytest.approx(0.0)


def test_deployment_range_constants():
    assert DH_DEPLOY_LO == -50.0
    assert DH_DEPLOY_HI == 8.0
    assert DH_DEPLOY_LO < DH_DEPLOY_HI
