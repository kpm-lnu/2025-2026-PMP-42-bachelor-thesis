"""
Inference wrapper for the trained StressRefinementMLP.

Given a coarse FEM mesh (after cycle 0 + IBEM error estimation),
predicts refined stress corrections at every mesh node.

Usage:
    predictor = NeuralRefinementPredictor("model.pt")
    corrections = predictor.predict_node_stress_corrections(mesh, eta_all, cycle=0, config=config)
    # corrections: {node_id: (Δσ_rr, Δσ_zz, Δσ_rz, Δσ_θθ)}
    for nid, delta in corrections.items():
        mesh.nodes[nid].neural_stress_correction = delta
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import StressRefinementMLP, InputNormalizer

COARSE_STRESS_IDX = [2, 3, 4, 5]  # σ_rr, σ_zz, σ_rz, σ_tt columns in feature vector


class NeuralRefinementPredictor:
    """Load a trained checkpoint and predict stress corrections for a mesh."""

    def __init__(self, model_path: str, device: str | None = None):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(model_path, map_location=self.device)
        input_dim   = ckpt['input_dim']
        hidden_dims = ckpt.get('hidden_dims', [256, 256, 128])
        dropout     = ckpt.get('dropout', 0.0)

        self.model = StressRefinementMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device).eval()

        self.normalizer = InputNormalizer(
            mean=ckpt['normalizer_mean'].to(self.device),
            std=ckpt['normalizer_std'].to(self.device),
        )
        self.feature_names    = ckpt.get('feature_names', [])
        self.coarse_stress_idx = ckpt.get('coarse_stress_idx', COARSE_STRESS_IDX)

    # ------------------------------------------------------------------

    def predict_node_stress_corrections(
        self,
        mesh,
        eta_all: dict[int, float],
        cycle: int,
        config: Any,
    ) -> dict[int, tuple[float, float, float, float]]:
        """
        Build feature vectors for every node, run the MLP, and return
        a dict {node_id: (Δσ_rr, Δσ_zz, Δσ_rz, Δσ_tt)}.
        """
        from ..postprocessors.fem_postprocessor import _recover_raw

        nodal_stress = _recover_raw(mesh)

        # Pre-compute per-element eta and area for neighbour lookups
        node_eta: dict[int, list[float]] = {nid: [] for nid in mesh.nodes}
        node_h:   dict[int, list[float]] = {nid: [] for nid in mesh.nodes}
        node_sigma: dict[int, tuple] = nodal_stress

        for eid, elem in mesh.elements.items():
            eta_e = eta_all.get(eid, 0.0)
            # Approximate element size from bounding box
            coords = np.array([[mesh.nodes[nid].r, mesh.nodes[nid].z]
                                for nid in elem.node_ids], dtype=float)
            dr = coords[:, 0].max() - coords[:, 0].min()
            dz = coords[:, 1].max() - coords[:, 1].min()
            h_e = math.sqrt(max(dr * dz, 1e-30))
            for nid in elem.node_ids:
                node_eta[nid].append(eta_e)
                node_h[nid].append(h_e)

        r_min = min(n.r for n in mesh.nodes.values())
        r_max = max(n.r for n in mesh.nodes.values())
        z_min = min(n.z for n in mesh.nodes.values())
        z_max = max(n.z for n in mesh.nodes.values())
        p_val  = float(getattr(config, 'p',  0.0))
        nu_val = float(getattr(config, 'nu', 0.3))
        mu_val = float(getattr(config, 'mu', 1.0))

        node_ids_list = list(mesh.nodes.keys())
        features = []
        for nid in node_ids_list:
            node = mesh.nodes[nid]
            s = node_sigma[nid]
            eta_e = float(np.mean(node_eta[nid])) if node_eta[nid] else 0.0
            h_e   = float(np.mean(node_h[nid]))   if node_h[nid]   else 0.0
            features.append([
                node.r, node.z,
                s[0], s[1], s[2], s[3],   # sigma_rr, sigma_zz, sigma_rz, sigma_tt
                eta_e, h_e,
                float(cycle),
                nu_val, mu_val, p_val,
                r_min, r_max, z_min, z_max,
            ])

        X = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            delta = self.model(self.normalizer(X)).cpu().numpy()  # [N, 4]

        result: dict[int, tuple[float, float, float, float]] = {}
        for i, nid in enumerate(node_ids_list):
            result[nid] = (float(delta[i, 0]), float(delta[i, 1]),
                           float(delta[i, 2]), float(delta[i, 3]))
        return result

    # ------------------------------------------------------------------

    def refined_nodal_stresses(
        self,
        mesh,
        eta_all: dict[int, float],
        cycle: int,
        config: Any,
    ) -> dict[int, tuple[float, float, float, float]]:
        """Return absolute refined stresses = coarse + correction."""
        from ..postprocessors.fem_postprocessor import _recover_raw
        coarse = _recover_raw(mesh)
        corrections = self.predict_node_stress_corrections(mesh, eta_all, cycle, config)
        result = {}
        for nid, delta in corrections.items():
            c = coarse[nid]
            result[nid] = (c[0] + delta[0], c[1] + delta[1],
                           c[2] + delta[2], c[3] + delta[3])
        return result
