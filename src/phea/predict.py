"""
Single entry point: composition → P(HEA).

Three-layer pipeline
--------------------
  1. GP posterior  (models/gp_full_model.pt)
       → (μ_ΔH, σ_ΔH) for the multicomponent alloy
  2. Thermophysical features
       → δ, ΔS/R, T_m  from element_properties.csv
  3. LogisticRegression + temperature scaling  (out/models/calibration_model.pkl)
       → P(HEA) ∈ [0, 1]

Usage
-----
    from src.phea.predict import HEAPredictor

    pred = HEAPredictor.load()

    # Single composition
    p = pred.predict({"Al": 0.2, "Co": 0.2, "Cr": 0.2, "Fe": 0.2, "Ni": 0.2})

    # Batch (list of dicts)
    df = pred.predict_batch([comp1, comp2, ...])

    # Atlas-scale (same element set, many compositions)
    import numpy as np
    arr = pred.predict_array(["Al","Co","Cr","Fe","Ni"], fracs_matrix)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from src.phea.features import FEATURE_NAMES

ROOT = Path(__file__).parent.parent.parent


class HEAPredictor:
    """
    Calibrated P(HEA) predictor.

    Parameters
    ----------
    gp_predictor    : GPPredictor  (src.gp.predict)
    pipeline        : sklearn Pipeline  (StandardScaler + LogisticRegression)
    temperature     : float  (temperature scaling parameter T > 1 → less extreme)
    feat_engine     : MulticomponentFeatures  (src.phea.features)
    """

    def __init__(self, gp_predictor, pipeline, temperature: float, feat_engine):
        self.gp          = gp_predictor
        self.pipeline    = pipeline
        self.temperature = float(temperature)
        self.feat        = feat_engine

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_pt:  str | Path | None = None,
        calib_pkl: str | Path | None = None,
        ep_path:   str | Path | None = None,
    ) -> "HEAPredictor":
        """
        Load GP model and calibration model from disk.

        Parameters
        ----------
        model_pt  : path to gp_full_model.pt       (default: models/gp_full_model.pt)
        calib_pkl : path to calibration_model.pkl  (default: out/models/calibration_model.pkl)
        ep_path   : path to element_properties.csv (default: data/periodic_table/...)
        """
        from src.gp.predict import GPPredictor
        from src.phea.features import MulticomponentFeatures

        model_pt  = Path(model_pt)  if model_pt  else ROOT / "models" / "gp_full_model.pt"
        calib_pkl = Path(calib_pkl) if calib_pkl else ROOT / "out" / "models" / "calibration_model.pkl"

        gp = GPPredictor.load(str(model_pt))

        with open(calib_pkl, "rb") as fh:
            ckpt = pickle.load(fh)
        pipeline    = ckpt["pipeline"]
        temperature = float(ckpt["temperature"])

        feat = MulticomponentFeatures(gp, ep_path=ep_path)

        return cls(gp, pipeline, temperature, feat)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _features_to_phea(self, X: np.ndarray) -> np.ndarray:
        """
        Apply pipeline + temperature scaling to feature matrix (N, 8).
        Returns P(HEA) array of shape (N,).
        """
        raw_proba = self.pipeline.predict_proba(X)[:, 1]
        raw_proba = np.clip(raw_proba, 1e-7, 1 - 1e-7)
        logits    = np.log(raw_proba / (1 - raw_proba))
        return expit(logits / self.temperature)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, composition: dict[str, float]) -> float:
        """
        P(HEA) for a single composition.

        Parameters
        ----------
        composition : {element: mole_fraction}, fractions must sum to ~1,
                      all x_i >= 0.05 recommended (atlas range).

        Returns
        -------
        float in [0, 1]
        """
        feat_dict = self.feat.compute(composition)
        X = np.array([[feat_dict[k] for k in FEATURE_NAMES]])
        return float(self._features_to_phea(X)[0])

    def predict_full(self, composition: dict[str, float]) -> dict[str, float]:
        """
        P(HEA) plus all intermediate features for a single composition.

        Returns
        -------
        dict with keys: P_HEA, + all FEATURE_NAMES
        """
        feat_dict = self.feat.compute(composition)
        X = np.array([[feat_dict[k] for k in FEATURE_NAMES]])
        p_hea = float(self._features_to_phea(X)[0])
        return {"P_HEA": p_hea, **feat_dict}

    def predict_batch(
        self,
        compositions: Sequence[dict[str, float]],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        P(HEA) for a list of composition dicts.

        Returns
        -------
        pd.DataFrame with columns: P_HEA, then the 8 FEATURE_NAMES
        (mu_dH, sigma_dH, pen_lo, pen_hi, pen_omega, delta, dS_R, T_m)
        """
        feat_df = self.feat.compute_batch(compositions, verbose=verbose)
        X       = feat_df[FEATURE_NAMES].values
        p_hea   = self._features_to_phea(X)
        feat_df.insert(0, "P_HEA", p_hea)
        return feat_df

    def predict_array(
        self,
        elements: Sequence[str],
        fracs_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised P(HEA) for N compositions sharing the same element set.

        Parameters
        ----------
        elements     : sequence of k element symbols
        fracs_matrix : ndarray (N, k), rows sum to 1

        Returns
        -------
        ndarray (N,) — P(HEA) for each composition
        """
        feat_arr = self.feat.compute_array(elements, fracs_matrix)
        return self._features_to_phea(feat_arr)

    def predict_array_filtered(
        self,
        elements: Sequence[str],
        fracs_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised P(HEA) with deployment-range filter.

        Compositions whose mu_dH falls outside [DH_DEPLOY_LO, DH_DEPLOY_HI]
        receive P(HEA) = 0.0 as a conservative safeguard.  Above +8 kJ/mol the
        calibration layer has no non-HEA training examples to constrain pen_hi,
        so it would extrapolate the penalty without identification; we abstain
        (P=0) rather than extrapolate.  This is conservative *against* recall: a
        few confirmed HEAs do lie above +8 (mu up to 11.4) and are zeroed by this
        rule.  Below -50 kJ/mol the GP is outside its training coverage (training
        min -46.5).  The out-of-range flag can be recovered by the caller via
        (mu_dH < DH_DEPLOY_LO) | (mu_dH > DH_DEPLOY_HI).

        Parameters
        ----------
        elements     : sequence of k element symbols
        fracs_matrix : ndarray (N, k), rows sum to 1

        Returns
        -------
        ndarray (N,) — P(HEA) in [0, 1]; 0.0 for out-of-range compositions
        """
        from src.phea.features import DH_DEPLOY_LO, DH_DEPLOY_HI

        feat_arr = self.feat.compute_array(elements, fracs_matrix)
        mu_dH    = feat_arr[:, 0]  # first column

        p_hea    = self._features_to_phea(feat_arr)

        out_of_range = (mu_dH < DH_DEPLOY_LO) | (mu_dH > DH_DEPLOY_HI)
        p_hea[out_of_range] = 0.0
        return p_hea

    def precompute_pairs(self, elements: Sequence[str]) -> None:
        """Pre-populate GP pair cache for an element set."""
        self.feat.precompute_pairs(elements)

    def precompute_pool(self) -> None:
        """Pre-populate GP pair cache for all 44 pool elements."""
        self.feat.precompute_pool()
