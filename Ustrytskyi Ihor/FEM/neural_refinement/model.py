"""
MLP model for predicting stress corrections:
    Δσ = σ_refined − σ_coarse

Input dim: 17 features (coordinates, coarse stresses, error indicator, material, geometry)
Output dim: 4 corrections (Δσ_rr, Δσ_zz, Δσ_rz, Δσ_θθ)

Usage:
    from FEM.neural_refinement.model import StressRefinementMLP, InputNormalizer
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InputNormalizer(nn.Module):
    """Stores per-feature mean and std for online normalisation."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std",  std.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std.clamp(min=1e-8)

    @classmethod
    def fit(cls, X: torch.Tensor) -> "InputNormalizer":
        """Fit normaliser from training features X [N, F]."""
        return cls(X.mean(dim=0), X.std(dim=0))


class StressRefinementMLP(nn.Module):
    """
    Residual-style MLP that predicts stress *corrections* (Δσ).

    Architecture:
        [17] → BN → 256 → LN+SiLU+Drop → 256 → LN+SiLU+Drop → 128 → SiLU → 4
    """

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dims: list[int] | None = None,
        output_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers: list[nn.Module] = []
        in_d = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
