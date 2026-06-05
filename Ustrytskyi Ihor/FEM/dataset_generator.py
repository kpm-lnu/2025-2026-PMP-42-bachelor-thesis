"""
Automated dataset generator for neural refinement training.

Generates diverse axisymmetric elasticity problems, runs the full
adaptive FEM+IBEM cycle, and saves (coarse_features, refined_targets)
pairs to an HDF5 file.

Usage:
    python -m FEM.dataset_generator --n-problems 50 --n-cycles 3 --output dataset.h5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES = [
    'r', 'z',
    'sigma_rr', 'sigma_zz', 'sigma_rz', 'sigma_tt',
    'eta_e', 'h_e',
    'cycle',
    'nu', 'mu', 'p',
    'r_min', 'r_max', 'z_min', 'z_max',
]

TARGET_NAMES = ['sigma_rr_ref', 'sigma_zz_ref', 'sigma_rz_ref', 'sigma_tt_ref']

INPUT_DIM = len(FEATURE_NAMES)  # 17


def save_rows_to_hdf5(rows: list[dict], path: str, append: bool = False) -> None:
    """Write (or append) feature/target rows to an HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 dataset storage. Run: pip install h5py")

    if not rows:
        return

    feats = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
    tgts  = np.array([[r[k] for k in TARGET_NAMES]  for r in rows], dtype=np.float32)

    mode = 'a' if append and Path(path).exists() else 'w'
    with h5py.File(path, mode) as f:
        if 'features' not in f:
            maxshape_f = (None, INPUT_DIM)
            maxshape_t = (None, len(TARGET_NAMES))
            f.create_dataset('features', data=feats, maxshape=maxshape_f, chunks=(1024, INPUT_DIM))
            f.create_dataset('targets',  data=tgts,  maxshape=maxshape_t, chunks=(1024, len(TARGET_NAMES)))
            f.create_dataset('meta', data=json.dumps({
                'feature_names': FEATURE_NAMES,
                'target_names': TARGET_NAMES,
                'input_dim': INPUT_DIM,
            }))
        else:
            # Append to existing datasets
            n_old = f['features'].shape[0]
            n_new = feats.shape[0]
            f['features'].resize(n_old + n_new, axis=0)
            f['features'][n_old:] = feats
            f['targets'].resize(n_old + n_new, axis=0)
            f['targets'][n_old:] = tgts


class DatasetGenerator:
    """Generate training data by running randomised FEM+IBEM adaptive problems."""

    def __init__(
        self,
        n_problems: int = 50,
        n_cycles: int = 4,
        output_path: str = "dataset.h5",
        seed: int = 42,
        mesh_resolution: tuple[int, int] = (3, 3),
        refinement_threshold: float = 0.3,
        refinement_mode: str = 'eta',
    ):
        self.n_problems = n_problems
        self.n_cycles = n_cycles
        self.output_path = output_path
        self.seed = seed
        self.mesh_resolution = mesh_resolution
        self.refinement_threshold = refinement_threshold
        self.refinement_mode = refinement_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> None:
        """Run all problems and save dataset to HDF5."""
        rng = np.random.default_rng(self.seed)

        # Remove existing file so we start fresh
        if Path(self.output_path).exists():
            Path(self.output_path).unlink()

        n_ok = 0
        for prob_idx in range(self.n_problems):
            params = self._sample_problem_params(rng)
            print(f"\n[DatasetGenerator] Problem {prob_idx + 1}/{self.n_problems}: {params}")
            try:
                self._run_one(params)
                n_ok += 1
            except Exception as exc:
                print(f"  Problem {prob_idx + 1} failed: {exc}")
                continue

        # Report final count from HDF5
        total_rows = self._count_hdf5_rows()
        print(f"\n[DatasetGenerator] Done. {n_ok}/{self.n_problems} problems succeeded, "
              f"{total_rows} total rows saved to '{self.output_path}'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_hdf5_rows(self) -> int:
        try:
            import h5py
            if not Path(self.output_path).exists():
                return 0
            with h5py.File(self.output_path, 'r') as f:
                return int(f['features'].shape[0]) if 'features' in f else 0
        except Exception:
            return -1

    def _sample_problem_params(self, rng: np.random.Generator) -> dict:
        r_min = float(rng.uniform(0.5, 1.5))
        delta_r = float(rng.uniform(0.5, 2.0))
        r_max = r_min + delta_r
        z_min = 0.0
        z_max = float(rng.uniform(0.5, 2.0))
        p     = float(rng.uniform(10.0, 500.0))
        nu    = float(rng.uniform(0.1, 0.45))
        E     = float(rng.uniform(0.5e5, 2.0e5))
        mu    = E / (2.0 * (1.0 + nu))
        return dict(r_min=r_min, r_max=r_max, z_min=z_min, z_max=z_max,
                    p=p, nu=nu, mu=mu, E=E)

    def _run_one(self, params: dict) -> list[dict]:
        """Run adaptive_refine_and_solve with dataset collection enabled."""
        from .main import ExperimentConfig, adaptive_refine_and_solve
        from .material import Material
        from .factory import FEMFactory, ElementType
        from .services.io_service import NullIOService

        r_min = params['r_min']
        r_max = params['r_max']
        z_min = params['z_min']
        z_max = params['z_max']
        p     = params['p']
        nu    = params['nu']
        mu    = params['mu']
        E     = params['E']

        config = ExperimentConfig(
            r_min=r_min, r_max=r_max, z_min=z_min, z_max=z_max,
            r_func=lambda z: 0,
            mesh_resolutions=[self.mesh_resolution],
            p=p, nu=nu, mu=mu,
            fixed_z=(z_min + z_max) / 2,
            fixed_r=r_min,
            compare_eps=1e-6,
            element_types=[ElementType.LINEAR],
            n_points_range=[2],
            adaptive_enabled=True,
            adaptive_max_cycles=self.n_cycles,
            adaptive_min_cycles=self.n_cycles,
            adaptive_refine_fraction_threshold=self.refinement_threshold,
            refinement_mode=self.refinement_mode,
            do_plots=False,
            do_console_prints=False,
            # Dataset flags
            collect_dataset=True,
            dataset_output_path=self.output_path,
        )

        mat = Material("gen", E=E, nu=nu)
        factory = FEMFactory(
            r_min, r_max, z_min, z_max, 0, 0,
            material=mat, node_dof=2,
        ).init(ElementType.LINEAR)

        rN, zN = self.mesh_resolution
        _, _, _ = adaptive_refine_and_solve(
            factory, config,
            rN, zN,
            ElementType.LINEAR,
            custom_n_points=2,
            do_plots=False,
            io=NullIOService(),
        )
        # Rows are already saved inside adaptive_refine_and_solve via save_rows_to_hdf5.
        # Return empty list to avoid double-writing.
        return []


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate FEM neural-refinement dataset")
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--n-cycles",   type=int, default=4)
    parser.add_argument("--output",     type=str, default="dataset.h5")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--mesh-rN",    type=int, default=3)
    parser.add_argument("--mesh-zN",    type=int, default=3)
    parser.add_argument("--threshold",  type=float, default=0.3)
    parser.add_argument("--mode",       type=str, default="eta")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    gen = DatasetGenerator(
        n_problems=args.n_problems,
        n_cycles=args.n_cycles,
        output_path=args.output,
        seed=args.seed,
        mesh_resolution=(args.mesh_rN, args.mesh_zN),
        refinement_threshold=args.threshold,
        refinement_mode=args.mode,
    )
    gen.generate()
