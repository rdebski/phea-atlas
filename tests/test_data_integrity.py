"""
Integrity checks on the prepared datasets and saved calibration model.

These guard against silent corruption / accidental regeneration with different
filters: the paper quotes n=433, 248/185 split, and a specific feature order.
"""
from __future__ import annotations

import ast
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
OUT_DATA = ROOT / "out" / "data"

POOL = {
    "Ag", "Al", "Au", "B",  "Ba", "Bi", "Ca", "Cd", "Ce", "Co",
    "Cr", "Cu", "Eu", "Fe", "Ga", "Gd", "Ge", "Hg", "In", "Ir",
    "Li", "Mg", "Mn", "Na", "Nb", "Ni", "Pb", "Pd", "Pt", "Rh",
    "Sb", "Sc", "Se", "Si", "Sn", "Sr", "Tb", "Te", "Ti", "Tl",
    "Y",  "Yb", "Zn", "Zr",
}


@pytest.fixture(scope="module")
def exp_df():
    df = pd.read_csv(OUT_DATA / "exp_data_clean.csv")
    df["composition"] = df["composition"].apply(json.loads)
    df["elements"] = df["elements"].apply(ast.literal_eval)
    return df


def test_exp_dataset_size(exp_df):
    assert len(exp_df) == 433
    assert int((exp_df["is_hea"] == 1).sum()) == 248
    assert int((exp_df["is_hea"] == 0).sum()) == 185


def test_exp_all_k5_in_pool(exp_df):
    for els in exp_df["elements"]:
        assert len(els) == 5
        assert all(e in POOL for e in els)


def test_exp_composition_constraints(exp_df):
    for comp in exp_df["composition"]:
        assert abs(sum(comp.values()) - 1.0) < 1e-6
        assert min(comp.values()) >= 0.05 - 1e-9
        dS_R = -sum(v * math.log(v) for v in comp.values() if v > 0)
        assert dS_R >= 1.5 - 1e-9


def test_exp_labels_consistent(exp_df):
    """is_hea == (p_label >= 0.5); conflict == mixed reports."""
    for _, r in exp_df.iterrows():
        assert r["is_hea"] == int(r["p_label"] >= 0.5)
        expect_conflict = (r["n_positive"] > 0) and (r["n_positive"] < r["n_reports"])
        assert bool(r["conflict"]) == bool(expect_conflict)


def test_stats_json_matches_data(exp_df):
    stats = json.loads((OUT_DATA / "exp_data_stats.json").read_text())
    assert stats["unique_compositions"] == len(exp_df)
    assert stats["n_hea_1"] == int((exp_df["is_hea"] == 1).sum())
    assert stats["n_conflict"] == int(exp_df["conflict"].sum())


def test_calibration_model_structure():
    ckpt = pickle.loads((ROOT / "out" / "models" / "calibration_model.pkl").read_bytes())
    assert set(ckpt.keys()) == {"pipeline", "temperature"}
    assert ckpt["temperature"] > 1.0
    pipe = ckpt["pipeline"]
    assert pipe.named_steps["logreg"].coef_.shape[1] == 8  # 8 features


def test_metrics_json_reasonable():
    m = json.loads((OUT_DATA / "calibration_metrics.json").read_text())
    assert m["n_compositions"] == 433
    assert m["loocv_temperature_scaled"]["ece"] < 0.10  # well calibrated
    assert 0.70 <= m["loocv_raw"]["auc_roc"] <= 0.75
    # pen_hi coefficient is the documented "≈ 0" empirical finding
    assert abs(m["coefficients"]["pen_hi"]) < 0.05


def test_predictions_align_with_exp(exp_df):
    pred = pd.read_csv(OUT_DATA / "calibration_predictions.csv")
    assert len(pred) == len(exp_df)
    # comp_key ordering is preserved between the two files
    assert list(pred["comp_key"]) == list(exp_df["comp_key"])
    assert pred["P_HEA_loocv_cal"].between(0, 1).all()
