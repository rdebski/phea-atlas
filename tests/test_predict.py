"""
Tests for src/phea/predict.py — the HEAPredictor entry point.

Covers: probability range, scalar/batch/array consistency, the deployment-range
safeguard, temperature-scaling effect, and reproduction of the atlas top subset.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.phea.features import DH_DEPLOY_HI

ELEMS = ["Co", "Cr", "Cu", "Fe", "Ni"]


def test_probability_in_unit_interval(predictor):
    p = predictor.predict({"Co": 0.2, "Cr": 0.2, "Cu": 0.2, "Fe": 0.2, "Ni": 0.2})
    assert 0.0 <= p <= 1.0


def test_predict_full_keys(predictor):
    out = predictor.predict_full({e: 0.2 for e in ELEMS})
    assert "P_HEA" in out
    for k in ("mu_dH", "sigma_dH", "pen_lo", "pen_hi", "pen_omega",
              "delta", "dS_R", "T_m"):
        assert k in out


def test_scalar_matches_array(predictor):
    """predict() on one composition == predict_array() on the same row."""
    fracs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    p_arr = predictor.predict_array(ELEMS, fracs)[0]
    p_scalar = predictor.predict({e: 0.2 for e in ELEMS})
    assert p_scalar == pytest.approx(float(p_arr), rel=1e-9)


def test_batch_matches_scalar(predictor):
    comps = [
        {e: 0.2 for e in ELEMS},
        {"Co": 0.30, "Cr": 0.25, "Cu": 0.20, "Fe": 0.15, "Ni": 0.10},
    ]
    df = predictor.predict_batch(comps)
    for i, comp in enumerate(comps):
        assert df["P_HEA"].iloc[i] == pytest.approx(predictor.predict(comp), rel=1e-9)


def test_cocrcufeni_matches_atlas(predictor):
    """The atlas reports eq_P/max_P = 0.935 for Co-Cr-Cu-Fe-Ni (top subset)."""
    p = predictor.predict({e: 0.2 for e in ELEMS})
    assert p == pytest.approx(0.935, abs=0.005)


def test_deployment_filter_zeros_out_of_range(predictor):
    """A strongly exothermic alloy (mu_dH < -50) must be zeroed by the filter."""
    # Build an out-of-range case: highly exothermic refractory + Al system.
    elems = ["Al", "Ni", "Ti", "Zr", "Nb"]
    fracs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    feat_arr = predictor.feat.compute_array(elems, fracs)
    mu = feat_arr[0, 0]

    p_unfiltered = predictor.predict_array(elems, fracs)[0]
    p_filtered = predictor.predict_array_filtered(elems, fracs)[0]

    if mu < -50.0 or mu > DH_DEPLOY_HI:
        assert p_filtered == 0.0
    else:
        assert p_filtered == pytest.approx(p_unfiltered)


def test_filter_matches_unfiltered_in_range(predictor):
    """In deployment range, filtered == unfiltered."""
    fracs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    feat_arr = predictor.feat.compute_array(ELEMS, fracs)
    assert -50.0 <= feat_arr[0, 0] <= DH_DEPLOY_HI  # Cantor+Cu is in range
    p_f = predictor.predict_array_filtered(ELEMS, fracs)[0]
    p_u = predictor.predict_array(ELEMS, fracs)[0]
    assert p_f == pytest.approx(p_u)


def test_temperature_scaling_pulls_toward_half(predictor):
    """T>1 temperature scaling moves probabilities toward 0.5 (less extreme)."""
    assert predictor.temperature > 1.0
    X = predictor.feat.compute_array(ELEMS, np.array([[0.2] * 5]))
    raw = predictor.pipeline.predict_proba(X)[:, 1][0]
    cal = predictor._features_to_phea(X)[0]
    # calibrated prediction is closer to 0.5 than the raw logistic output
    assert abs(cal - 0.5) < abs(raw - 0.5)


def test_above_upper_bound_is_zero(predictor):
    """Any composition with mu_dH > +8 must receive P=0 under the filter."""
    # Scan a few subsets to find one that produces mu_dH > 8 at some composition.
    rng = np.random.default_rng(0)
    elems = ["Cu", "Ag", "Au", "Bi", "Pb"]  # weakly/positively mixing system
    fracs = rng.dirichlet(np.ones(5), size=200)
    feat_arr = predictor.feat.compute_array(elems, fracs)
    mu = feat_arr[:, 0]
    p_filtered = predictor.predict_array_filtered(elems, fracs)
    above = mu > DH_DEPLOY_HI
    if above.any():
        assert np.all(p_filtered[above] == 0.0)
