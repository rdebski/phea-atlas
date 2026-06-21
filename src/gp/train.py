"""
GP training via marginal log-likelihood maximisation (Adam).

Early stop when MLL improvement < tol for `patience` consecutive steps.
"""

from __future__ import annotations

import gpytorch
import torch

from .model import ResidualGP, build_model


def train_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_noise_var: torch.Tensor,
    n_iter: int = 300,
    lr: float = 0.05,
    patience: int = 40,
    tol: float = 1e-4,
    verbose: bool = True,
) -> tuple[ResidualGP, gpytorch.likelihoods.FixedNoiseGaussianLikelihood, list[float]]:
    """
    Train heteroskedastic GP by maximising the exact MLL.

    Parameters
    ----------
    train_x, train_y, train_noise_var : normalised float64 tensors
    n_iter    : maximum Adam iterations
    lr        : Adam learning rate
    patience  : early-stop patience (steps without tol improvement)
    tol       : minimum absolute MLL improvement to reset counter
    verbose   : print progress every 50 steps

    Returns
    -------
    model, likelihood, loss_history
    """
    model, likelihood = build_model(train_x, train_y, train_noise_var)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses: list[float] = []
    best_loss  = float("inf")
    no_improve = 0

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss   = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        l = loss.item()
        losses.append(l)

        if best_loss - l > tol:
            best_loss  = l
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (i + 1) % 50 == 0:
            add_n = likelihood.second_noise.item()
            os    = model.covar_module.outputscale.item()
            ls    = model.covar_module.base_kernel.lengthscale.detach().squeeze()
            print(
                f"  iter {i+1:4d} | loss {l:+8.4f} | "
                f"add_noise {add_n:.4f} | σ² {os:.4f} | "
                f"ℓ [{float(ls.min()):.3f}, {float(ls.max()):.3f}]"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at iter {i+1} ({patience} steps without improvement)")
            break

    model.eval()
    likelihood.eval()
    return model, likelihood, losses
