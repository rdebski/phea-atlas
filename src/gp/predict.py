"""
Full-data GP training and inference for arbitrary binary pairs.

Unlike LOSO evaluation (evaluate.py), this module trains once on ALL
metallic Deffrennes pairs and then generalises to any element combination
via feature-based prediction.

Out-of-distribution behaviour: GP reverts toward prior mean (≈0, i.e.
h05_miedema) with increasing uncertainty as the query moves away from
training data. This is the correct signal for data-scarce systems.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch

from .data_prep import (
    load_gp_data,
    GPDataset,
    FEATURE_NAMES,
    _compute_miedema_features,
    _compute_tabular_features,
    noise_from_npts,
    _SIGMA_BASE,
    _N_REF,
)
from .model import build_model
from .train import train_gp

ROOT = Path(__file__).parent.parent.parent


class GPPredictor:
    """
    Trained GP that predicts h05 = h05_miedema + residual for any binary pair.

    Usage
    -----
    predictor = GPPredictor.train(n_iter=300)
    mean, std, mied = predictor.predict("Al", "Ni")   # kJ/mol

    Notes
    -----
    - Elements must be in Miedema params and element_properties tables.
    - Pairs not in the training set yield GP prior mean (≈0 residual)
      with uncertainty ≈ GP output scale — honest extrapolation.
    - Use model-only std (epistemic) for screening; total std for calibration.
    """

    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        dataset: GPDataset,
        mp: pd.DataFrame,
        ep: pd.DataFrame,
        miedema_model,
    ):
        self.model = model
        self.likelihood = likelihood
        self.dataset = dataset
        self.mp = mp
        self.ep = ep
        self.miedema_model = miedema_model

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        n_iter: int = 300,
        lr: float = 0.05,
        verbose: bool = True,
        scope: str = "metals",
    ) -> "GPPredictor":
        """Train GP on the full Deffrennes dataset (scope='metals': 224 pairs)."""
        from src.data_preparation.miedema import MiedemaModel

        df  = load_gp_data(scope=scope)
        idx = np.arange(len(df))
        ds  = GPDataset(df, idx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, likelihood, _ = train_gp(
                ds.train_x, ds.train_y, ds.train_noise_var,
                n_iter=n_iter, lr=lr, verbose=verbose,
            )

        mp_raw = pd.read_csv(ROOT / "data/miedema_params.csv")
        ep_raw = pd.read_csv(ROOT / "data/periodic_table/element_properties.csv")
        mp = mp_raw.set_index("elem")
        ep = ep_raw.set_index("elem")
        miedema = MiedemaModel(mp_raw)

        return cls(model, likelihood, ds, mp, ep, miedema)

    def save(self, path: str | Path) -> None:
        """Save model state and normalisation parameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "likelihood_state": self.likelihood.state_dict(),
                "x_scaler_mean": self.dataset.x_scaler.mean_,
                "x_scaler_scale": self.dataset.x_scaler.scale_,
                "y_mean": self.dataset.y_mean,
                "y_std": self.dataset.y_std,
                "n_features": len(FEATURE_NAMES),
                "n_train": len(self.dataset.train_x),
            },
            path,
        )
        print(f"Saved GP predictor to {path}")

    @classmethod
    def load(cls, path: str | Path, scope: str = "metals") -> "GPPredictor":
        """Load a saved predictor; rebuilds feature pipeline from CSV files."""
        from src.data_preparation.miedema import MiedemaModel
        from sklearn.preprocessing import StandardScaler

        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Rebuild GPDataset with the actual training data.
        # The saved normalisation parameters must match what was used during training.
        # We re-apply them so ds.train_x / train_y are in the correct normalised space.
        df  = load_gp_data(scope=scope)
        idx = np.arange(len(df))
        ds  = GPDataset(df, idx)
        # Override scaler/target stats with the saved values to guarantee consistency
        # even if the loaded CSV has minor floating-point differences.
        ds.x_scaler.mean_  = ckpt["x_scaler_mean"]
        ds.x_scaler.scale_ = ckpt["x_scaler_scale"]
        ds.y_mean = float(ckpt["y_mean"])
        ds.y_std  = float(ckpt["y_std"])
        # Re-normalise training tensors with the loaded (canonical) parameters
        from sklearn.preprocessing import StandardScaler
        X_raw = df[FEATURE_NAMES].values.astype(np.float64)
        y_raw = df["residual"].values.astype(np.float64)
        noise_raw = df["noise_kJmol"].values.astype(np.float64)
        X_scaled = ds.x_scaler.transform(X_raw)
        y_scaled  = (y_raw  - ds.y_mean) / ds.y_std
        nv_scaled = (noise_raw / ds.y_std) ** 2
        ds.train_x         = torch.tensor(X_scaled,  dtype=torch.float64)
        ds.train_y         = torch.tensor(y_scaled,   dtype=torch.float64)
        ds.train_noise_var = torch.tensor(nv_scaled,  dtype=torch.float64)

        # Build model with the actual training tensors, then load hyperparameters.
        # Support both key formats ('model_state' and legacy 'model').
        model, likelihood = build_model(ds.train_x, ds.train_y, ds.train_noise_var)
        model.load_state_dict(ckpt.get("model_state", ckpt.get("model")))
        likelihood.load_state_dict(ckpt.get("likelihood_state", ckpt.get("likelihood")))
        model.eval()
        likelihood.eval()

        mp_raw = pd.read_csv(ROOT / "data/miedema_params.csv")
        ep_raw = pd.read_csv(ROOT / "data/periodic_table/element_properties.csv")
        mp = mp_raw.set_index("elem")
        ep = ep_raw.set_index("elem")
        miedema = MiedemaModel(mp_raw)

        return cls(model, likelihood, ds, mp, ep, miedema)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _features_for_pair(self, a: str, b: str) -> np.ndarray:
        """Compute 15-feature vector (alphabetical canonical order)."""
        if a > b:
            a, b = b, a
        pair = f"{a}-{b}"
        mied = _compute_miedema_features(pair, self.mp)
        tab  = _compute_tabular_features(pair, self.ep)
        h05m = self.miedema_model.h_mix_fn(a, b, 0.5)
        feat = {**mied, **tab, "h05_miedema": h05m}
        return np.array([feat[f] for f in FEATURE_NAMES])

    def predict(
        self,
        a: str,
        b: str,
        use_model_std: bool = False,
    ) -> tuple[float, float, float]:
        """
        Predict h05 for binary pair (A, B).

        Parameters
        ----------
        a, b           : element symbols (order does not matter)
        use_model_std  : if False (default), return total std (model + noise floor,
                         consistent with LOSO calibration ECE=0.028);
                         if True, return epistemic (model-only) std.

        Returns
        -------
        h05_mean    : GP corrected prediction  [kJ/mol]
        h05_std     : uncertainty              [kJ/mol]
        h05_miedema : Miedema baseline         [kJ/mol]
        """
        if a > b:
            a, b = b, a

        X_raw = self._features_for_pair(a, b).reshape(1, -1)
        X_t   = self.dataset.transform_x(X_raw)

        noise_var = torch.tensor(
            [(_SIGMA_BASE / self.dataset.y_std) ** 2],
            dtype=torch.float64,
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred   = self.model(X_t)
            obs_pred = self.likelihood(f_pred, noise=noise_var)

        res_mean = self.dataset.unscale_y(obs_pred.mean).item()
        std = (f_pred.stddev.item() if use_model_std else obs_pred.stddev.item()) * self.dataset.y_std

        h05m = self.miedema_model.h_mix_fn(a, b, 0.5)
        return h05m + res_mean, std, h05m

    def predict_joint(
        self,
        elements: list[str],
        use_model_std: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Joint GP prediction for all C(N,2) pairs with full posterior covariance.

        Unlike N independent calls to predict(), this returns the full P×P
        posterior covariance matrix, capturing correlations between pairs that
        share an element.  Used to assess the independence assumption in
        multicomponent σ_ΔH propagation.

        Parameters
        ----------
        elements      : list of element symbols (any order)
        use_model_std : if False (default), include observation noise on diagonal
                        (consistent with pipeline calibration)

        Returns
        -------
        mu_kJ    : shape (P,)   GP mean h05 per pair [kJ/mol]
        K_kJ     : shape (P,P)  posterior covariance  [kJ²/mol²]
        h05_mied : shape (P,)   Miedema h05 per pair  [kJ/mol]

        Pairs are ordered i < j over sorted(elements).
        """
        from itertools import combinations

        elems_sorted = sorted(elements)
        pair_canon = [(a, b) for a, b in combinations(elems_sorted, 2)]
        P = len(pair_canon)

        X_list, h05_mied_list = [], []
        for a, b in pair_canon:
            X_list.append(self._features_for_pair(a, b))
            h05_mied_list.append(self.miedema_model.h_mix_fn(a, b, 0.5))

        X_raw = np.stack(X_list)                    # (P, 15)
        X_t   = self.dataset.transform_x(X_raw)     # (P, 15) scaled

        noise_var_s = (_SIGMA_BASE / self.dataset.y_std) ** 2
        noise_batch = torch.full((P,), noise_var_s, dtype=torch.float64)

        # Disable fast_pred_var: we need the full covariance matrix, not just variance
        with torch.no_grad():
            f_pred   = self.model(X_t)
            obs_pred = self.likelihood(f_pred, noise=noise_batch)
            K_scaled = (f_pred.covariance_matrix if use_model_std
                        else obs_pred.covariance_matrix)

        K_kJ     = K_scaled.numpy() * self.dataset.y_std ** 2
        mu_scaled = f_pred.mean if use_model_std else obs_pred.mean
        res_mean  = self.dataset.unscale_y(mu_scaled).numpy()  # [kJ/mol]

        h05_mied = np.array(h05_mied_list)
        mu_kJ    = h05_mied + res_mean
        return mu_kJ, K_kJ, h05_mied

    def predict_all_pairs(
        self,
        elements: list[str],
        use_model_std: bool = False,
    ) -> dict[tuple[str, str], tuple[float, float, float]]:
        """
        Predict h05 for all C(N,2) binary sub-pairs of a multicomponent system.

        Returns
        -------
        dict mapping (a, b) canonical pairs → (h05_mean, h05_std, h05_miedema)
        """
        results = {}
        for i, a in enumerate(elements):
            for b in elements[i + 1 :]:
                a_c, b_c = (a, b) if a < b else (b, a)
                results[(a_c, b_c)] = self.predict(a_c, b_c, use_model_std=use_model_std)
        return results
