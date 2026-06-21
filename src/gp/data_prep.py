"""
Prepare data for the Miedema-residual GP model.

Target  : residual = h05_exp - h05_miedema  (Miedema is the prior baseline)
Features: 15 symmetric descriptors
            8  Miedema-derived  (phi, nws, V)
            6  tabular          (VEC, r_atomic, T_melt, EN_Pauling)
            1  Miedema h05      (full model prediction as a feature)
Noise   : per-pair heteroskedastic proxy from n_pts in Deffrennes
            σ_i = σ_base / sqrt(n_pts_i / n_ref)  clipped at σ_base
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent

# Noise model constants (kJ/mol)
_SIGMA_BASE = 1.5   # CALPHAD uncertainty for a fully-assessed pair (n_pts=99)
_N_REF      = 99    # standard Deffrennes grid size
_SIGMA_MIN  = _SIGMA_BASE  # floor: never below base uncertainty

# Elements outside Miedema's metallic design domain.
# Pairs containing any of these are excluded when scope='metals'.
SEMIMETALS: frozenset[str] = frozenset({
    "Si", "Ge", "As", "Sb", "Se", "Te", "Bi", "P"
})

FEATURE_NAMES: list[str] = [
    # ── Miedema-derived (8) ────────────────────────────────────────────
    "abs_delta_phi",    # |φ*_A - φ*_B|
    "abs_delta_nws13",  # |n_ws^(1/3)_A - n_ws^(1/3)_B|
    "mean_V23",         # (V_A^(2/3) + V_B^(2/3)) / 2
    "delta_phi_sq",     # (φ*_A - φ*_B)²
    "delta_nws_sq",     # (n_ws^(1/3)_A - n_ws^(1/3)_B)²
    "f_AB",             # 9.4·Δnws² − Δφ²  (Miedema interaction function)
    "V_fAB",            # mean_V23 · f_AB
    "prod_nws",         # n_ws^(1/3)_A × n_ws^(1/3)_B
    # ── Tabular (6) ────────────────────────────────────────────────────
    "mean_VEC",         # (VEC_A + VEC_B) / 2
    "abs_delta_VEC",    # |VEC_A - VEC_B|
    "abs_delta_r",      # |r_A - r_B|  (Å)  — Hume-Rothery size criterion
    "mean_Tm",          # (T_m,A + T_m,B) / 2  (K) — enters Ω parameter
    "abs_delta_Tm",     # |T_m,A - T_m,B|
    "geom_mean_EN",     # sqrt(EN_A · EN_B)
    # ── Miedema full prediction (1) ────────────────────────────────────
    "h05_miedema",      # ΔH_mix(0.5) from Miedema — encodes full model nonlinearity
]


def _compute_miedema_features(pair: str, mp: pd.DataFrame) -> dict[str, float]:
    """Symmetric Miedema-derived features for pair 'A-B'."""
    a, b   = pair.split("-")
    pa, pb = mp.loc[a], mp.loc[b]

    phi_a, phi_b = float(pa["phi"]),  float(pb["phi"])
    nws_a, nws_b = float(pa["nws"]),  float(pb["nws"])
    v_a,   v_b   = float(pa["v"]),    float(pb["v"])

    nws13_a = nws_a ** (1.0 / 3.0)
    nws13_b = nws_b ** (1.0 / 3.0)
    v23_a   = v_a   ** (2.0 / 3.0)
    v23_b   = v_b   ** (2.0 / 3.0)

    d_phi    = phi_a - phi_b
    d_nws13  = nws13_a - nws13_b
    mean_v23 = (v23_a + v23_b) / 2.0
    d_phi_sq = d_phi   ** 2
    d_nws_sq = d_nws13 ** 2
    f_AB     = 9.4 * d_nws_sq - d_phi_sq

    return {
        "abs_delta_phi":   abs(d_phi),
        "abs_delta_nws13": abs(d_nws13),
        "mean_V23":        mean_v23,
        "delta_phi_sq":    d_phi_sq,
        "delta_nws_sq":    d_nws_sq,
        "f_AB":            f_AB,
        "V_fAB":           mean_v23 * f_AB,
        "prod_nws":        nws13_a * nws13_b,
    }


def _compute_tabular_features(pair: str, ep: pd.DataFrame) -> dict[str, float]:
    """Symmetric tabular features from element_properties.csv."""
    a, b   = pair.split("-")
    ea, eb = ep.loc[a], ep.loc[b]

    vec_a, vec_b = float(ea["VEC"]),       float(eb["VEC"])
    r_a,   r_b   = float(ea["r_atomic"]),  float(eb["r_atomic"])
    tm_a,  tm_b  = float(ea["T_melt"]),    float(eb["T_melt"])
    en_a,  en_b  = float(ea["EN_Pauling"]),float(eb["EN_Pauling"])

    return {
        "mean_VEC":      (vec_a + vec_b) / 2.0,
        "abs_delta_VEC": abs(vec_a - vec_b),
        "abs_delta_r":   abs(r_a - r_b),
        "mean_Tm":       (tm_a + tm_b) / 2.0,
        "abs_delta_Tm":  abs(tm_a - tm_b),
        "geom_mean_EN":  np.sqrt(en_a * en_b),
    }


def noise_from_npts(n_pts: np.ndarray) -> np.ndarray:
    """
    Per-pair aleatoric noise in kJ/mol from Deffrennes data density.

    σ_i = max(σ_base, σ_base / sqrt(n_pts_i / n_ref))
    """
    n_pts  = np.maximum(n_pts, 1)
    sigma  = _SIGMA_BASE / np.sqrt(n_pts / _N_REF)
    return np.maximum(sigma, _SIGMA_MIN)


def load_gp_data(
    global_path: str | Path | None = None,
    miedema_params_path: str | Path | None = None,
    element_props_path: str | Path | None = None,
    scope: str = "metals",
    exclude_generated: bool = True,
) -> pd.DataFrame:
    """
    Load Deffrennes global dataset and compute all GP inputs.

    Parameters
    ----------
    scope : "metals"  → exclude pairs containing semimetals (default, recommended)
            "all"     → include all 314 pairs
    exclude_generated : if True (default), remove 12 pairs whose Deffrennes RK
        coefficients were fitted to Miedema-generated points (suffix 'g' in
        Data quality column).  These pairs have residual ≈ 0 by construction,
        which would circularly teach the GP that Miedema is exact for them.
        Selected pairs (manifest): data/processed/training_pairs.csv (222 pairs)

    Returns DataFrame with columns:
        pair, h05, h05_miedema, residual, noise_kJmol, n_pts, <FEATURE_NAMES>

    residual    = h05 - h05_miedema   (GP target)
    noise_kJmol = per-pair σ from n_pts (aleatoric proxy)
    """
    from src.data_preparation.miedema import MiedemaModel

    global_path        = Path(global_path or ROOT / "data/processed/binary_h05_dataset.csv")
    miedema_params_path = Path(miedema_params_path or ROOT / "data/miedema_params.csv")
    element_props_path  = Path(element_props_path or ROOT / "data/periodic_table/element_properties.csv")

    df     = pd.read_csv(global_path)
    mp_raw = pd.read_csv(miedema_params_path)
    ep_raw = pd.read_csv(element_props_path)
    mp     = mp_raw.set_index("elem")
    ep     = ep_raw.set_index("elem")
    model  = MiedemaModel(mp_raw)

    if scope == "metals":
        is_metal = df["pair"].apply(
            lambda p: not any(e in SEMIMETALS for e in p.split("-"))
        )
        df = df[is_metal].reset_index(drop=True)

    if exclude_generated:
        clean_path = ROOT / "data/processed/training_pairs.csv"
        clean_pairs = set(pd.read_csv(clean_path)["pair"])
        df = df[df["pair"].isin(clean_pairs)].reset_index(drop=True)

    df["h05_miedema"] = [model.h_mix_fn(*p.split("-"), 0.5) for p in df["pair"]]
    df["residual"]    = df["h05"] - df["h05_miedema"]
    df["noise_kJmol"] = noise_from_npts(df["n_pts"].values)

    # Build feature columns
    for pair in df["pair"]:
        pass  # validate element coverage (raises KeyError if missing)

    mied_feats = pd.DataFrame(
        [_compute_miedema_features(p, mp) for p in df["pair"]], index=df.index
    )
    tab_feats = pd.DataFrame(
        [_compute_tabular_features(p, ep) for p in df["pair"]], index=df.index
    )

    for col in mied_feats.columns:
        df[col] = mied_feats[col]
    for col in tab_feats.columns:
        df[col] = tab_feats[col]
    # h05_miedema is already a column — no alias needed

    # Return: base cols first, then feature cols (h05_miedema deduplicated)
    base_cols = ["pair", "h05", "h05_miedema", "residual", "noise_kJmol", "n_pts"]
    feat_cols = [f for f in FEATURE_NAMES if f not in base_cols]
    return df[base_cols + feat_cols]


class GPDataset:
    """
    Normalised train/test tensors for GP, including per-point noise.

    Normalisation
    -------------
    X         : StandardScaler fit on training set
    y         : (y - y_mean_train) / y_std_train
    noise_var : (noise_kJmol / y_std_train)²   [in normalised residual units]
    """

    def __init__(self, df: pd.DataFrame, train_idx, test_idx=None):
        X_raw     = df[FEATURE_NAMES].values.astype(np.float64)
        y_raw     = df["residual"].values.astype(np.float64)
        noise_raw = df["noise_kJmol"].values.astype(np.float64)

        self.x_scaler = StandardScaler()
        X_tr = self.x_scaler.fit_transform(X_raw[train_idx])

        self.y_mean = float(y_raw[train_idx].mean())
        self.y_std  = float(y_raw[train_idx].std()) or 1.0

        y_tr          = (y_raw[train_idx] - self.y_mean) / self.y_std
        noise_var_tr  = (noise_raw[train_idx] / self.y_std) ** 2

        self.train_x         = torch.tensor(X_tr,          dtype=torch.float64)
        self.train_y         = torch.tensor(y_tr,          dtype=torch.float64)
        self.train_noise_var = torch.tensor(noise_var_tr,  dtype=torch.float64)
        self.pairs           = df["pair"].values[train_idx]

        if test_idx is not None:
            X_te         = self.x_scaler.transform(X_raw[test_idx])
            y_te         = (y_raw[test_idx] - self.y_mean) / self.y_std
            noise_var_te = (noise_raw[test_idx] / self.y_std) ** 2

            self.test_x         = torch.tensor(X_te,         dtype=torch.float64)
            self.test_y         = torch.tensor(y_te,         dtype=torch.float64)
            self.test_noise_var = torch.tensor(noise_var_te, dtype=torch.float64)
            self.test_pairs     = df["pair"].values[test_idx]

    def unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * self.y_std + self.y_mean

    def transform_x(self, X_raw: np.ndarray) -> torch.Tensor:
        return torch.tensor(self.x_scaler.transform(X_raw), dtype=torch.float64)
