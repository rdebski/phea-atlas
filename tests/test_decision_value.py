"""
Reproduce the paper's headline decision-value numbers from the saved LOO-CV
predictions (out/data/calibration_predictions.csv).

These are the numbers in the abstract / Table 4.3:
    P(HEA) ranking  Precision@20 = 100%
                    HEA@50       = 45/50
                    EF@50        = 1.571
Reproducing them from the raw predictions guarantees the claim is not a
hand-typed constant but follows from the model output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
PRED = ROOT / "out" / "data" / "calibration_predictions.csv"


@pytest.fixture(scope="module")
def ranking():
    from src.phea.features import DH_DEPLOY_LO, DH_DEPLOY_HI
    df = pd.read_csv(PRED)
    feat = pd.read_csv(ROOT / "out" / "data" / "calibration_features.csv")
    y = df["is_hea"].values.astype(int)
    p = df["P_HEA_loocv_cal"].values.copy()
    # Apply the deployment-range safeguard exactly as run/05 / the atlas do:
    # compositions with GP mu_dH outside [DH_DEPLOY_LO, DH_DEPLOY_HI] get P=0.
    mu = feat["mu_dH"].values
    p[(mu < DH_DEPLOY_LO) | (mu > DH_DEPLOY_HI)] = 0.0
    order = np.argsort(-p, kind="stable")   # descending P(HEA)
    return y, y[order]


def _ef(y_sorted, B, base):
    return (y_sorted[:B].sum() / B) / base


def test_base_rate(ranking):
    y, _ = ranking
    assert len(y) == 433
    assert y.mean() == pytest.approx(0.5727, abs=0.001)


def test_precision_at_20(ranking):
    _, ys = ranking
    assert ys[:20].sum() == 20            # every top-20 is a confirmed HEA
    assert ys[:20].mean() == pytest.approx(1.0)


def test_hea_at_50(ranking):
    _, ys = ranking
    assert ys[:50].sum() == 45
    assert ys[:50].mean() == pytest.approx(0.90)


def test_ef_at_50(ranking):
    y, ys = ranking
    base = y.mean()
    assert _ef(ys, 50, base) == pytest.approx(1.571, abs=0.01)


def test_ef_at_20(ranking):
    y, ys = ranking
    base = y.mean()
    assert _ef(ys, 20, base) == pytest.approx(1.746, abs=0.01)


def test_monotone_better_than_random(ranking):
    """At every budget the P(HEA) ranking beats the base rate (EF > 1)."""
    y, ys = ranking
    base = y.mean()
    for B in (20, 50, 100, 150):
        assert _ef(ys, B, base) > 1.0


def test_raw_and_calibrated_same_order(ranking):
    """Temperature scaling is monotone — it must not change the ranking."""
    df = pd.read_csv(PRED)
    order_raw = np.argsort(-df["P_HEA_loocv_raw"].values)
    order_cal = np.argsort(-df["P_HEA_loocv_cal"].values)
    assert np.array_equal(order_raw, order_cal)


# ---------------------------------------------------------------------------
# Baselines — produced by run/05_decision_value.py into decision_value.json.
# These lock the paper's Table 4.3 comparator values.
# ---------------------------------------------------------------------------
DV_JSON = ROOT / "out" / "data" / "decision_value.json"


@pytest.fixture(scope="module")
def dv():
    import json
    if not DV_JSON.exists():
        pytest.skip("decision_value.json absent — run run/05_decision_value.py")
    return json.loads(DV_JSON.read_text())


def test_phea_strategy_in_json(dv):
    s = dv["strategies"]["P(HEA)"]
    assert s["prec@20"] == pytest.approx(1.0)
    assert s["hea@50"] == 45
    assert s["ef@50"] == pytest.approx(1.571, abs=0.01)


def test_headline_miedema_omega_baseline(dv):
    """HEADLINE: Ω rule on PURE Miedema h05 (current practice): 40/50, EF@50≈1.40."""
    s = dv["strategies"]["Miedema+Omega"]
    assert s["hea@20"] == 15
    assert s["hea@50"] == 40
    assert s["ef@50"] == pytest.approx(1.40, abs=0.01)


def test_headline_window_centre_baseline(dv):
    """HEADLINE: window-centre on PURE Miedema h05: 41/50, EF@50≈1.43."""
    s = dv["strategies"]["Miedema window-centre"]
    assert s["hea@50"] == 41
    assert s["ef@50"] == pytest.approx(1.43, abs=0.01)


def test_conjunctive_screen_baseline(dv):
    """Actual deterministic practice (window AND Ω). The conjunction coincides with the
    window leg alone (every in-window comp. also satisfies Ω≥1.1), so it retrieves 40/50
    at B=50 -- matching the single-criterion baselines and well below P(HEA)'s 45/50."""
    s = dv["strategies"]["Miedema window AND Omega"]
    assert s["hea@20"] == 15
    assert s["hea@50"] == 40
    assert s["hea@100"] == 80
    assert s["ef@50"] == pytest.approx(1.40, abs=0.01)
    assert dv["strategies"]["P(HEA)"]["hea@50"] > s["hea@50"]


def test_honest_unordered_screen_baseline(dv):
    """Deterministic screen returns an UNORDERED accept set: scored honestly (B drawn at
    random from the 296-composition accept set) it gives EF≈1.11, precision 0.635 -- barely
    above the base rate and far below P(HEA). The Ω/window rankings are charitable upper
    bounds that impute a priority the screen does not possess (1.40-1.43 > 1.11)."""
    s = dv["strategies"]["Screen accept-set (random within)"]
    assert s["accept_set_size"] == 296
    assert s["accept_set_precision"] == pytest.approx(0.635, abs=0.005)
    assert s["ef@50"] == pytest.approx(1.11, abs=0.01)
    # honest screen < ranked deterministic baselines < P(HEA)
    assert s["ef@50"] < dv["strategies"]["Miedema+Omega"]["ef@50"]
    assert s["ef@50"] < dv["strategies"]["Miedema window-centre"]["ef@50"]
    assert s["ef@50"] < dv["strategies"]["P(HEA)"]["ef@50"]


def test_ablation_gp_mu_dH(dv):
    """ABLATION: same Ω rule on GP μ_ΔH still loses to P(HEA): 43/50 vs 45/50."""
    s = dv["ablation_gp_mu_dH"]["Omega rule (GP mu_dH)"]
    assert s["hea@50"] == 43
    assert s["ef@50"] == pytest.approx(1.50, abs=0.01)
    # P(HEA) beats even the stronger GP-μ_ΔH baseline
    assert dv["strategies"]["P(HEA)"]["hea@50"] > s["hea@50"]


def test_phea_beats_headline_baselines_at_every_budget(dv):
    phea = dv["strategies"]["P(HEA)"]
    for base in ("Miedema+Omega", "Miedema window-centre", "Miedema window AND Omega"):
        b = dv["strategies"][base]
        for B in (20, 50, 100):
            assert phea[f"hea@{B}"] >= b[f"hea@{B}"]
