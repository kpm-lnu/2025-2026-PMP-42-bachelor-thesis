"""
Training script for StressRefinementMLP.

Usage:
    python -m FEM.neural_refinement.train --dataset dataset.h5 --output model.pt --epochs 200

The script:
1. Loads features and targets from HDF5.
2. Fits an InputNormalizer on the training split.
3. Trains the MLP to predict stress *corrections* (Δσ = σ_ref − σ_coarse).
4. Saves model checkpoint to model.pt.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .model import StressRefinementMLP, InputNormalizer

FEATURE_NAMES = [
    'r', 'z',
    'sigma_rr', 'sigma_zz', 'sigma_rz', 'sigma_tt',
    'eta_e', 'h_e',
    'cycle',
    'nu', 'mu', 'p',
    'r_min', 'r_max', 'z_min', 'z_max',
]
TARGET_NAMES = ['sigma_rr_ref', 'sigma_zz_ref', 'sigma_rz_ref', 'sigma_tt_ref']
COARSE_STRESS_IDX = [2, 3, 4, 5]  # indices of σ_rr,σ_zz,σ_rz,σ_tt in feature vector


def load_dataset(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load features and compute corrections (Δσ = σ_ref − σ_coarse)."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Run: pip install h5py")

    with h5py.File(path, 'r') as f:
        X = torch.tensor(f['features'][:], dtype=torch.float32)
        Y_abs = torch.tensor(f['targets'][:],  dtype=torch.float32)

    # Corrections: refined − coarse (coarse stress columns in X)
    Y_coarse = X[:, COARSE_STRESS_IDX]
    Y = Y_abs - Y_coarse  # shape [N, 4]

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target corrections — mean: {Y.mean(dim=0).tolist()}, std: {Y.std(dim=0).tolist()}")
    return X, Y


def train(
    dataset_path: str,
    output_path: str,
    val_ratio: float = 0.1,
    epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-3,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.1,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X, Y = load_dataset(dataset_path)
    n_total = len(X)
    n_val   = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    ds_train, ds_val = random_split(TensorDataset(X, Y), [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    X_train = X[ds_train.indices]
    normalizer = InputNormalizer.fit(X_train)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    if hidden_dims is None:
        hidden_dims = [256, 256, 128]

    model = StressRefinementMLP(input_dim=X.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    model = model.to(device)
    normalizer = normalizer.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    best_val_loss = math.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(normalizer(xb))
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(normalizer(xb))
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= n_val
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # Save best checkpoint
    torch.save({
        'model': best_state,
        'normalizer_mean': normalizer.mean.cpu(),
        'normalizer_std':  normalizer.std.cpu(),
        'input_dim': X.shape[1],
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'feature_names': FEATURE_NAMES,
        'target_names': TARGET_NAMES,
        'coarse_stress_idx': COARSE_STRESS_IDX,
        'best_val_loss': best_val_loss,
    }, output_path)
    print(f"\nSaved best model (val_loss={best_val_loss:.6f}) → {output_path}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train StressRefinementMLP")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output",  default="model.pt")
    p.add_argument("--epochs",  type=int,   default=200)
    p.add_argument("--batch",   type=int,   default=2048)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--val",     type=float, default=0.1)
    p.add_argument("--device",  default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_path=args.dataset,
        output_path=args.output,
        val_ratio=args.val,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
    )
