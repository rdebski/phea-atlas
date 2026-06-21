"""
Integrity checks on the precomputed atlas (out/data/atlas_phea.csv).

The atlas is the paper's headline decision-support artefact: 1,086,008 rows.
We verify row count, value ranges, internal consistency of aggregates, and the
top-subset claim (Co-Cr-Cu-Fe-Ni, mean_P=0.467).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
ATLAS = ROOT / "out" / "data" / "atlas_phea.csv"

C_44_5 = math.comb(44, 5)  # 1,086,008


@pytest.fixture(scope="module")
def atlas():
    return pd.read_csv(ATLAS)


def test_row_count(atlas):
    assert len(atlas) == C_44_5 == 1_086_008


def test_columns(atlas):
    """The user-facing atlas carries only the 15 decision-support columns.
    Independence-approximation values live in atlas_phea_independence.csv."""
    expected = {
        "subset_key", "el1", "el2", "el3", "el4", "el5",
        "mean_P", "max_P", "eq_P", "frac_05", "mean_sigma_dH",
        "n_valid", "n_oor", "best_comp", "best_mu_dH",
    }
    assert set(atlas.columns) == expected


def test_independence_atlas_separate(atlas):
    """The appendix independence atlas is a separate, equally-long file with its
    own *_indep columns (kept out of the main decision atlas)."""
    indep = pd.read_csv(ROOT / "out" / "data" / "atlas_phea_independence.csv")
    assert len(indep) == len(atlas)
    assert set(indep.columns) == {
        "subset_key", "el1", "el2", "el3", "el4", "el5",
        "mean_P_indep", "max_P_indep", "eq_P_indep", "mean_sigma_indep",
    }


def test_probabilities_in_range(atlas):
    for col in ("mean_P", "max_P", "eq_P", "frac_05"):
        s = atlas[col].dropna()
        assert s.min() >= 0.0
        assert s.max() <= 1.0


def test_mean_le_max(atlas):
    """mean_P <= max_P wherever both are defined."""
    sub = atlas.dropna(subset=["mean_P", "max_P"])
    assert (sub["mean_P"] <= sub["max_P"] + 1e-9).all()


def test_valid_plus_oor_constant(atlas):
    """n_valid + n_oor must equal 1451 (compositions per subset)."""
    assert ((atlas["n_valid"] + atlas["n_oor"]) == 1451).all()


def test_nan_iff_no_valid(atlas):
    """mean_P is NaN exactly when the subset has no in-range composition."""
    no_valid = atlas["n_valid"] == 0
    assert atlas.loc[no_valid, "mean_P"].isna().all()
    assert atlas.loc[~no_valid, "mean_P"].notna().all()


def test_top_subset_is_cocrcufeni(atlas):
    valid = atlas.dropna(subset=["mean_P"])
    top = valid.nlargest(1, "mean_P").iloc[0]
    assert set(top["subset_key"].split("-")) == {"Co", "Cr", "Cu", "Fe", "Ni"}
    assert top["mean_P"] == pytest.approx(0.467, abs=0.01)
    assert top["max_P"] == pytest.approx(0.935, abs=0.01)


def test_distribution_shares(atlas):
    """~95-97% of valid subsets have mean_P < 0.1; none above 0.5."""
    valid = atlas.dropna(subset=["mean_P"])
    frac_low = (valid["mean_P"] < 0.1).mean()
    assert 0.94 <= frac_low <= 0.98
    assert (valid["mean_P"] > 0.5).sum() == 0


def test_oor_subsets_count(atlas):
    """A non-trivial number of subsets are entirely out-of-range (all OOR)."""
    n_all_oor = (atlas["n_valid"] == 0).sum()
    assert 5_000 <= n_all_oor <= 12_000  # CLAUDE.md reports 8,068
