"""
Exact GP for Miedema residual: δ(A,B) = h05_exp - h05_miedema.

Architecture
------------
mean    : ZeroMean  (residuals centred before passing in)
kernel  : ScaleKernel( Matérn ν=2.5 with ARD )
noise   : FixedNoiseGaussianLikelihood  — per-pair heteroskedastic σ from n_pts
          with learn_additional_noise=True  (global noise floor, learned)

Why Matérn ν=2.5 over RBF
--------------------------
Matérn ν=2.5 assumes the target is twice mean-square differentiable, which is
more appropriate than the infinitely-smooth RBF for physical residuals that may
have sharp transitions (e.g. near-semimetal or noble-metal systems).

Permutation invariance k(A,B)=k(B,A) is guaranteed because all 15 input
features are symmetric by construction (see data_prep.FEATURE_NAMES).

Note on Student-t likelihood
-----------------------------
A VariationalGP + StudentTLikelihood was evaluated but rejected: the ELBO
optimisation systematically drives ν → 2 (infinite variance) regardless of
priors, because 5-10 genuine outlier pairs (Au-Zr, Ni-Sn, …) dominate the
heavy-tail objective. CI coverage drops to ~80% — worse than the calibrated
Gaussian. Calibration of the Gaussian GP is instead improved post-hoc via
temperature scaling (see calibrate_temperature in evaluate.py).
"""

from __future__ import annotations

import gpytorch
import torch


class ResidualGP(gpytorch.models.ExactGP):
    """Heteroskedastic Matérn GP for the Miedema correction residual."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        n_features = train_x.shape[1]

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_features)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


def build_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_noise_var: torch.Tensor,
) -> tuple[ResidualGP, gpytorch.likelihoods.FixedNoiseGaussianLikelihood]:
    """
    Construct heteroskedastic GP model + likelihood.

    Parameters
    ----------
    train_x         : (N, D) normalised feature tensor
    train_y         : (N,)   normalised residual tensor
    train_noise_var : (N,)   per-point noise variance in normalised units
                             derived from n_pts via data_prep.noise_from_npts
    """
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_noise_var,
        learn_additional_noise=True,
    )
    likelihood.second_noise = 0.1

    model = ResidualGP(train_x, train_y, likelihood)
    model.covar_module.outputscale = 1.0
    model.covar_module.base_kernel.lengthscale = torch.ones(1, train_x.shape[1])

    return model, likelihood
