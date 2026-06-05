"""
IOService — centralised output layer (plotting + formatted console output).

Responsibilities
~~~~~~~~~~~~~~~~
* Every matplotlib figure in the project is created here.
* Every formatted ``print`` (tables, progress lines) goes through this service.
* No domain module (solver, mesh, postprocessor) should call ``plt`` or
  ``print`` directly — they delegate to an ``IOService`` instance instead.

Design notes
~~~~~~~~~~~~
* The class is deliberately a *concrete* service, not a Protocol.
  Swap it for a no-op ``NullIOService`` in tests / headless runs.
* ``block`` kwarg mirrors ``plt.show(block=…)`` semantics.
* Helper geometry functions (edge interpolation for slicing) are kept as
  private module-level functions — they contain no I/O and are pure math.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

from ..analysis_utils import analytical_solution
from ..postprocessors.stress_recovery import (
    recover_all_nodal_stresses,
    recover_nodal_stresses,
)
from ..factory import ElementType


# ---------------------------------------------------------------------------
# Internal geometry helpers (pure math, no I/O)
# ---------------------------------------------------------------------------

def _quad_edge_pairs_Q4(node_ids):
    """4-node quad (bilinear): edges are 2-node segments."""
    return [
        (node_ids[0], node_ids[1]),
        (node_ids[1], node_ids[2]),
        (node_ids[2], node_ids[3]),
        (node_ids[3], node_ids[0]),
    ]


def _quad_edge_triplets_Q8(node_ids):
    """8-node quad (biquadratic): edges are 3-node segments."""
    return [
        (node_ids[0], node_ids[4], node_ids[1]),
        (node_ids[1], node_ids[5], node_ids[2]),
        (node_ids[2], node_ids[6], node_ids[3]),
        (node_ids[3], node_ids[7], node_ids[0]),
    ]


def _interp_Q1_edge_z(r1, z1, ur1, r2, z2, ur2, zfix, eps=1e-14):
    dz = z2 - z1
    if abs(dz) < eps:
        if abs(z1 - zfix) < eps:
            return [(r1, ur1), (r2, ur2)]
        return []
    t = (zfix - z1) / dz
    if -eps <= t <= 1 + eps:
        t = min(1.0, max(0.0, t))
        r = r1 + t * (r2 - r1)
        ur = ur1 + t * (ur2 - ur1)
        return [(r, ur)]
    return []


def _interp_Q2_edge_z(r1, z1, ur1, rm, zm, urm, r2, z2, ur2, zfix, eps=1e-14):
    a = 0.5 * (z1 + z2) - zm
    b = 0.5 * (z2 - z1)
    c = zm - zfix
    roots: list[float] = []
    if abs(a) < eps:
        if abs(b) < eps:
            if abs(c) < eps:
                return [(r1, ur1), (rm, urm), (r2, ur2)]
            return []
        roots = [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc < -1e-12:
            return []
        disc = max(0.0, disc)
        s = math.sqrt(disc)
        roots = [(-b - s) / (2 * a), (-b + s) / (2 * a)]
    out: list[tuple[float, float]] = []
    for t in roots:
        if t < -1 - eps or t > 1 + eps:
            continue
        t = min(1.0, max(-1.0, t))
        N1 = 0.5 * t * (t - 1.0)
        Nm = 1.0 - t * t
        N2 = 0.5 * t * (t + 1.0)
        r = N1 * r1 + Nm * rm + N2 * r2
        ur = N1 * ur1 + Nm * urm + N2 * ur2
        out.append((r, ur))
    return out


# ---------------------------------------------------------------------------
# Zoom-pan helper (attached to individual axes)
# ---------------------------------------------------------------------------

class _ZoomPan:
    """Mouse-wheel zoom for a single matplotlib Axes."""

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self._cid = ax.figure.canvas.mpl_connect("scroll_event", self._zoom)

    def _zoom(self, event):
        if event.xdata is None or event.ydata is None:
            return
        base = 1.1
        sf = 1 / base if event.button == "up" else (base if event.button == "down" else 1)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        rx = (xlim[1] - event.xdata) / (xlim[1] - xlim[0])
        ry = (ylim[1] - event.ydata) / (ylim[1] - ylim[0])
        nw = (xlim[1] - xlim[0]) * sf
        nh = (ylim[1] - ylim[0]) * sf
        self.ax.set_xlim([event.xdata - nw * (1 - rx), event.xdata + nw * rx])
        self.ax.set_ylim([event.ydata - nh * (1 - ry), event.ydata + nh * ry])
        self.ax.figure.canvas.draw()


# ---------------------------------------------------------------------------
# IOService
# ---------------------------------------------------------------------------

class IOService:
    """Centralised output layer — all plots and formatted prints go here."""

    COLORS = ["b", "g", "r", "c", "m", "y", "k"]

    def __init__(self, *, enabled: bool = True, block: bool = False):
        self._enabled = enabled
        self._block = block

    # -- guards -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _skip(self) -> bool:
        return not self._enabled

    # ===================================================================
    # Console output
    # ===================================================================

    def print_adaptive_cycle(self, cycle: int, n_refine: int, n_total: int) -> None:
        if self._skip():
            return
        print(f"Adaptive cycle {cycle}: refining {n_refine} elements out of {n_total}")

    def print_new_mesh(self, rN: int, zN: int) -> None:
        if self._skip():
            return
        print(f"--------New mesh ({rN}x{zN})-----------")

    def print_experiment_header(self, rN: int, zN: int, elem_type, n_points: int) -> None:
        if self._skip():
            return
        print("-" * 120)
        print(f"rN: {rN}, zN: {zN}, elem_type: {elem_type}, n_points: {n_points}")

    def print_benchmark_header(self) -> None:
        if self._skip():
            return
        print("Benchmarking mortar modes (wall time, seconds):")

    def print_benchmark_result(self, label: str, mean: float, std: float, n: int) -> None:
        if self._skip():
            return
        print(f"\033[32m  {label}: {mean:.4f} ± {std:.4f} (n={n})\033[0m")

    # -- node tables --------------------------------------------------------

    def print_node_table_ur(self, mesh, *, a: float, b: float, p: float, mu: float, nu: float) -> None:
        if self._skip():
            return
        nodes = sorted(mesh.nodes.values(), key=lambda n: n.node_id)
        header = f"{'node_id':>8}  {'(r, z)':>28}  {'u_r':>14}  {'u_r_analytical':>18}"
        print("\nTable: u_r at mesh nodes (with analytical)")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for n in nodes:
            coord = f"({n.r: .6f}, {n.z: .6f})"
            ur_anal, *_ = analytical_solution(n.r, a, b, p, mu, nu)
            print(f"{n.node_id:8d}  {coord:>28}  {n.displacements[0]:14.6e}  {ur_anal:18.6e}")
        print("-" * len(header))

    def print_node_table_sigma(
        self,
        mesh,
        material,
        component: str = "sigma_rr",
        *,
        a: float,
        b: float,
        p: float,
        mu: float,
        nu: float,
        tol: float = 1e-6,
    ) -> None:
        if self._skip():
            return
        nodal = recover_all_nodal_stresses(mesh, material, mesh.shape_func, tol=tol)
        comp_label = {
            "sigma_rr": r"σ_rr",
            "sigma_zz": r"σ_zz",
            "sigma_rz": r"σ_rz",
            "sigma_tt": r"σ_φφ",
        }.get(component, component)
        nodes = sorted(mesh.nodes.values(), key=lambda n: n.node_id)
        header = f"{'node_id':>8}  {'(r, z)':>28}  {component:>14}  {'analytical':>14}"
        print(f"\nTable: {comp_label} at mesh nodes (with analytical)")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for n in nodes:
            s = nodal.get(n.node_id, {})
            val = s.get(component, np.nan)
            coord = f"({n.r: .6f}, {n.z: .6f})"
            ur_a, uz_a, s_rr_a, s_zz_a, s_phi_a, s_rz_a = analytical_solution(n.r, a, b, p, mu, nu)
            ana_map = {
                "sigma_rr": s_rr_a,
                "sigma_zz": s_zz_a,
                "sigma_rz": s_rz_a,
                "sigma_tt": s_phi_a,
            }
            ana = ana_map.get(component, np.nan)
            print(f"{n.node_id:8d}  {coord:>28}  {val:14.6e}  {ana:14.6e}")
        print("-" * len(header))

    def print_error_tables(
        self,
        errors: Dict[str, Dict[str, np.ndarray]],
        analytical: Dict[str, np.ndarray],
        var_names: Sequence[str],
        tol: float = 1e-12,
    ) -> None:
        if self._skip():
            return
        print("\nAbsolute Errors:")
        header = "{:15} {:15} {:15} {:15}".format("Variable", "Mean", "Max", "Min")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for var in var_names:
            abs_err = errors["absolute_errors"][var]
            print("{:15} {:15.5e} {:15.5e} {:15.5e}".format(var, np.mean(abs_err), np.max(abs_err), np.min(abs_err)))
        print("-" * len(header))

        print("\nRelative Errors (in percentage, only for variables with nonzero analytical solution):")
        header_perc = "{:15} {:15} {:15} {:15}".format("Variable", "Mean (%)", "Max (%)", "Min (%)")
        print("-" * len(header_perc))
        print(header_perc)
        print("-" * len(header_perc))
        for var in var_names:
            if np.any(np.abs(analytical[var]) > tol):
                rel_err = errors["relative_errors"][var]
                print(
                    "{:15} {:15.5e} {:15.5e} {:15.5e}".format(
                        var, np.mean(rel_err) * 100, np.max(rel_err) * 100, np.min(rel_err) * 100
                    )
                )
        print("-" * len(header_perc))

    # ===================================================================
    # Plotting — adaptive overlay
    # ===================================================================

    def plot_adaptive_overlay(
        self,
        cycle_results: List[Dict],
        fixed_z: float,
        compare_eps: float,
        r_min: float,
        r_max: float,
        p: float,
        mu: float,
        nu: float,
    ) -> None:
        if self._skip():
            return
        if not cycle_results:
            return
        r_analytical = np.linspace(r_min, r_max, 200)
        ur_analytical = (1 / (2 * mu * (r_max**2 - r_min**2))) * (
            (1 - 2 * nu) * r_min**2 * p * r_analytical + p * r_min**2 * r_max**2 / r_analytical
        )
        plt.figure(figsize=(8, 6))
        for entry in cycle_results:
            snap = entry["snapshot"]
            r_vals, ur_vals = [], []
            for _nid, r, z, ur, uz in snap["nodes"]:
                if abs(z - fixed_z) < compare_eps:
                    r_vals.append(r)
                    ur_vals.append(ur)
            if len(r_vals) < 2:
                continue
            r_vals = np.array(r_vals)
            ur_vals = np.array(ur_vals)
            idx = np.argsort(r_vals)
            r_vals, ur_vals = r_vals[idx], ur_vals[idx]
            plt.plot(r_vals, ur_vals, marker="o", linestyle="-",
                     label=f"cycle {entry['cycle']} (nel={entry['num_elements']})")
        plt.plot(r_analytical, ur_analytical, "k--", label="analytical")
        plt.xlabel("r")
        plt.ylabel("u_r")
        plt.title(f"1. Radial displacement overlay at z={fixed_z}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=self._block)

    def plot_adaptive_overlay_slice(
        self,
        cycle_results: List[Dict],
        fixed_z: float,
        r_min: float,
        r_max: float,
        p: float,
        mu: float,
        nu: float,
        num_points: int = 200,
    ) -> None:
        """Edge-intersection based cross-section plot."""
        if self._skip():
            return
        if not cycle_results:
            return
        r_anal = np.linspace(r_min, r_max, num_points)
        ur_anal = (1 / (2 * mu * (r_max**2 - r_min**2))) * (
            (1 - 2 * nu) * r_min**2 * p * r_anal + p * r_min**2 * r_max**2 / r_anal
        )
        plt.figure(figsize=(8, 6))
        for entry in cycle_results:
            snap = entry["snapshot"]
            node_data = {}
            for nid, r, z, ur, uz in snap["nodes"]:
                node_data[nid] = (float(r), float(z), float(ur))
            points: list[tuple[float, float]] = []
            for e in snap.get("elements", []):
                nids = e["node_ids"]
                if len(nids) == 4:
                    for i0, i1 in _quad_edge_pairs_Q4(nids):
                        r1, z1, ur1 = node_data[i0]
                        r2, z2, ur2 = node_data[i1]
                        points.extend(_interp_Q1_edge_z(r1, z1, ur1, r2, z2, ur2, fixed_z))
                elif len(nids) == 8:
                    for i0, im, i1 in _quad_edge_triplets_Q8(nids):
                        r1, z1, ur1 = node_data[i0]
                        rm, zm, urm = node_data[im]
                        r2, z2, ur2 = node_data[i1]
                        points.extend(_interp_Q2_edge_z(r1, z1, ur1, rm, zm, urm, r2, z2, ur2, fixed_z))
            if not points:
                continue
            uniq: dict[float, float] = {}
            for r, ur in points:
                key = round(r, 12)
                if key not in uniq or abs(ur) > abs(uniq[key]):
                    uniq[key] = ur
            rs = np.array(sorted(uniq.keys()))
            urs = np.array([uniq[k] for k in rs])
            mask = (rs >= r_min) & (rs <= r_max)
            if np.count_nonzero(mask) >= 2:
                plt.plot(rs[mask], urs[mask], "-o",
                         label=f"cycle {entry['cycle']} (nel={entry['num_elements']})")
        plt.plot(r_anal, ur_anal, "k--", label="analytical")
        plt.xlabel("r")
        plt.ylabel("u_r")
        plt.title(f"2. Radial displacement overlay at z={fixed_z}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=self._block)

    # ===================================================================
    # Plotting — field heatmaps
    # ===================================================================

    def plot_ur_heatmap(self, mesh, *, cmap="viridis", levels=64, show_edges=False, interpolate=True) -> None:
        if self._skip():
            return
        node_id_to_idx = {nid: i for i, nid in enumerate(sorted(mesh.nodes.keys()))}
        nodes = [mesh.nodes[nid] for nid in sorted(mesh.nodes.keys())]
        r = np.array([n.r for n in nodes])
        z = np.array([n.z for n in nodes])
        u_r = np.array([n.displacements[0] for n in nodes])
        triangles = []
        for elem in mesh.elements.values():
            nids = elem.node_ids
            if len(nids) == 4:
                idx = [node_id_to_idx[nid] for nid in nids]
                triangles.append([idx[0], idx[1], idx[2]])
                triangles.append([idx[0], idx[2], idx[3]])
            elif len(nids) == 8:
                idx = [node_id_to_idx[nids[i]] for i in [0, 1, 2, 3]]
                triangles.append([idx[0], idx[1], idx[2]])
                triangles.append([idx[0], idx[2], idx[3]])
        tri = mtri.Triangulation(r, z, triangles=triangles)
        fig, ax = plt.subplots(figsize=(8, 6))
        if interpolate:
            tcf = ax.tricontourf(tri, u_r, levels=levels, cmap=cmap)
        else:
            tcf = ax.tripcolor(tri, u_r, shading="flat", cmap=cmap)
        if show_edges:
            ax.triplot(tri, color="k", linewidth=0.3, alpha=0.5)
        cbar = fig.colorbar(tcf, ax=ax)
        cbar.set_label(r"u_r")
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title("Heatmap of radial displacement $u_r$")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show(block=self._block)

    def plot_sigma_heatmap_dense_per_element(
        self,
        mesh,
        component: str = "sigma_rr",
        *,
        cmap: str = "coolwarm",
        levels: int = 64,
        show_edges: bool = False,
        material=None,
        tol: float = 1e-6,
        nsub: int = 12,
        method: str = "gouraud",
        antialiased: bool = False,
    ) -> None:
        """Dense FE-style sampling with independent triangulation per element."""
        if self._skip():
            return
        if material is None:
            material = getattr(mesh, "material", None)
        if material is None:
            raise ValueError("mesh.material is required to recover stresses.")

        nodal_stresses = recover_all_nodal_stresses(mesh, material, mesh.shape_func, tol=tol)

        # For sigma_eff we must interpolate each component separately and
        # compute the nonlinear von-Mises norm at each sampling point.
        _is_eff = (component == "sigma_eff")
        _eff_keys = ("sigma_rr", "sigma_zz", "sigma_tt", "sigma_rz")

        def node_sigma(nid: int) -> float:
            s = nodal_stresses.get(nid, {})
            if _is_eff:
                srr = s.get("sigma_rr", 0.0)
                szz = s.get("sigma_zz", 0.0)
                stt = s.get("sigma_tt", 0.0)
                srz = s.get("sigma_rz", 0.0)
                return float(np.sqrt(0.5 * (
                    (srr - szz) ** 2
                    + (srr - stt) ** 2
                    + (szz - stt) ** 2
                    + 6.0 * srz ** 2
                )))
            return float(s.get(component, 0.0))

        def _node_components(nid: int):
            """Return (σ_rr, σ_zz, σ_tt, σ_rz) for a node."""
            s = nodal_stresses.get(nid, {})
            return tuple(s.get(k, 0.0) for k in _eff_keys)

        def _von_mises(srr, szz, stt, srz):
            return np.sqrt(0.5 * (
                (srr - szz) ** 2
                + (srr - stt) ** 2
                + (szz - stt) ** 2
                + 6.0 * srz ** 2
            ))

        def eval_N(xi, eta, nen_expected: int):
            out = mesh.shape_func.evaluate(xi, eta)
            N = out[0] if isinstance(out, (tuple, list)) else out
            N = np.asarray(N, dtype=float).reshape(-1)
            if N.size != nen_expected:
                raise ValueError(f"shape_func.evaluate returned {N.size} values but element expects {nen_expected}.")
            return N

        xis = np.linspace(-1.0, 1.0, nsub + 1)
        etas = np.linspace(-1.0, 1.0, nsub + 1)
        m = nsub + 1
        local_tris = []
        for j in range(nsub):
            for i in range(nsub):
                p00 = i + m * j
                p10 = i + 1 + m * j
                p11 = i + 1 + m * (j + 1)
                p01 = i + m * (j + 1)
                local_tris.append([p00, p10, p11])
                local_tris.append([p00, p11, p01])
        local_tris = np.asarray(local_tris, dtype=int)

        vmin, vmax = np.inf, -np.inf
        for elem in mesh.elements.values():
            nids = list(elem.node_ids)
            nen = len(nids)
            if nen < 3:
                continue
            if _is_eff:
                # Interpolate each component, then compute von Mises
                ec = np.array([_node_components(nid) for nid in nids], dtype=float)  # (nen, 4)
                for eta in etas:
                    for xi in xis:
                        N = eval_N(xi, eta, nen)
                        comps = N @ ec  # (4,) = interpolated (srr, szz, stt, srz)
                        v = float(_von_mises(*comps))
                        vmin, vmax = min(vmin, v), max(vmax, v)
            else:
                es = np.array([node_sigma(nid) for nid in nids], dtype=float)
                for eta in etas:
                    for xi in xis:
                        v = float(np.dot(eval_N(xi, eta, nen), es))
                        vmin, vmax = min(vmin, v), max(vmax, v)

        if not np.isfinite(vmin):
            print(f"No elements to plot for {component}.")
            return

        comp_label = {
            "sigma_rr": r"$\sigma_{rr}$",
            "sigma_zz": r"$\sigma_{zz}$",
            "sigma_rz": r"$\sigma_{rz}$",
            "sigma_tt": r"$\sigma_{\phi\phi}$",
            "sigma_eff": r"$\sigma_{\mathrm{eff}}$",
        }.get(component, component)

        fig, ax = plt.subplots(figsize=(8, 6))
        last_artist = None
        for elem in mesh.elements.values():
            nids = list(elem.node_ids)
            nen = len(nids)
            if nen < 3:
                continue
            er = np.array([mesh.nodes[nid].r for nid in nids], dtype=float)
            ez = np.array([mesh.nodes[nid].z for nid in nids], dtype=float)
            if _is_eff:
                ec = np.array([_node_components(nid) for nid in nids], dtype=float)  # (nen, 4)
            else:
                es = np.array([node_sigma(nid) for nid in nids], dtype=float)
            Rloc = np.empty(m * m, dtype=float)
            Zloc = np.empty(m * m, dtype=float)
            Vloc = np.empty(m * m, dtype=float)
            k = 0
            for eta in etas:
                for xi in xis:
                    N = eval_N(xi, eta, nen)
                    Rloc[k] = float(np.dot(N, er))
                    Zloc[k] = float(np.dot(N, ez))
                    if _is_eff:
                        comps = N @ ec  # interpolated (srr, szz, stt, srz)
                        Vloc[k] = float(_von_mises(*comps))
                    else:
                        Vloc[k] = float(np.dot(N, es))
                    k += 1
            tri = mtri.Triangulation(Rloc, Zloc, triangles=local_tris)
            if method == "contour":
                last_artist = ax.tricontourf(tri, Vloc, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                shading = "gouraud" if method == "gouraud" else "flat"
                last_artist = ax.tripcolor(
                    tri, Vloc, shading=shading, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolors="none", antialiased=antialiased,
                )
            if show_edges and nen >= 4:
                corner_ids = [nids[i] for i in [0, 1, 2, 3]]
                bx = [mesh.nodes[n].r for n in corner_ids] + [mesh.nodes[corner_ids[0]].r]
                by = [mesh.nodes[n].z for n in corner_ids] + [mesh.nodes[corner_ids[0]].z]
                ax.plot(bx, by, color="k", linewidth=0.3, alpha=0.4)

        cbar = fig.colorbar(last_artist, ax=ax)
        cbar.set_label(comp_label)
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(f"Dense-sampled heatmap of {comp_label}")
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show(block=self._block)

    # ===================================================================
    # Plotting — heatmap recovery-mode comparison (2×2 grid)
    # ===================================================================

    def plot_sigma_heatmap_recovery_comparison(
        self,
        mesh,
        component: str = "sigma_rr",
        *,
        cmap: str = "coolwarm",
        nsub: int = 12,
        method: str = "gouraud",
        antialiased: bool = False,
    ) -> None:
        """2×2 heatmap grid comparing raw / L² / SPR / mortar recovery."""
        if self._skip():
            return
        from ..postprocessors.fem_postprocessor import (
            _recover_raw, _recover_l2, _recover_spr, _recover_mortar,
        )

        _is_eff = component == "sigma_eff"
        comp_idx = {
            "sigma_rr": 0, "sigma_zz": 1, "sigma_rz": 2, "sigma_tt": 3,
        }.get(component)
        if comp_idx is None and not _is_eff:
            raise ValueError(f"Unknown stress component: {component}")

        comp_label = {
            "sigma_rr": r"$\sigma_{rr}$",
            "sigma_zz": r"$\sigma_{zz}$",
            "sigma_rz": r"$\sigma_{rz}$",
            "sigma_tt": r"$\sigma_{\varphi\varphi}$",
            "sigma_eff": r"$\sigma_{\mathrm{eff}}$",
        }.get(component, component)

        modes = [
            ('Raw',   _recover_raw),
            ('L²',    _recover_l2),
            ('SPR',   _recover_spr),
            ('Mortar', _recover_mortar),
        ]

        # -- compute nodal stresses for every mode --------------------------
        all_nodal: list[tuple[str, dict | None]] = []
        for label, fn in modes:
            try:
                all_nodal.append((label, fn(mesh)))
            except Exception as exc:
                print(f"Recovery mode {label} failed: {exc}")
                all_nodal.append((label, None))

        # -- sub-triangulation template -------------------------------------
        xis_arr = np.linspace(-1.0, 1.0, nsub + 1)
        etas_arr = np.linspace(-1.0, 1.0, nsub + 1)
        m = nsub + 1
        local_tris: list[list[int]] = []
        for j in range(nsub):
            for i in range(nsub):
                p00, p10 = i + m * j, i + 1 + m * j
                p11, p01 = i + 1 + m * (j + 1), i + m * (j + 1)
                local_tris.append([p00, p10, p11])
                local_tris.append([p00, p11, p01])
        local_tris_arr = np.asarray(local_tris, dtype=int)

        def _vm(srr, szz, srz, stt):
            return float(np.sqrt(0.5 * (
                (srr - szz) ** 2 + (srr - stt) ** 2
                + (szz - stt) ** 2 + 6.0 * srz ** 2
            )))

        def _node_val(nodal_dict, nid):
            s = nodal_dict.get(nid, (0.0, 0.0, 0.0, 0.0))
            return _vm(s[0], s[1], s[2], s[3]) if _is_eff else float(s[comp_idx])

        # -- global colour range across all modes ---------------------------
        vmin, vmax = np.inf, -np.inf
        for _, nodal in all_nodal:
            if nodal is None:
                continue
            for nid in mesh.nodes:
                v = _node_val(nodal, nid)
                vmin, vmax = min(vmin, v), max(vmax, v)
        if not np.isfinite(vmin):
            return

        # -- 2×2 figure -----------------------------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Recovery comparison — {comp_label}', fontsize=13)

        for ax, (label, nodal) in zip(axs.flat, all_nodal):
            if nodal is None:
                ax.set_title(f'{label} (failed)')
                continue

            last_artist = None
            for elem in mesh.elements.values():
                nids = list(elem.node_ids)
                nen = len(nids)
                if nen < 3:
                    continue

                er = np.array([mesh.nodes[nid].r for nid in nids], dtype=float)
                ez = np.array([mesh.nodes[nid].z for nid in nids], dtype=float)
                if _is_eff:
                    ec = np.array(
                        [list(nodal.get(nid, (0, 0, 0, 0))) for nid in nids],
                        dtype=float,
                    )
                else:
                    es = np.array(
                        [nodal.get(nid, (0, 0, 0, 0))[comp_idx] for nid in nids],
                        dtype=float,
                    )

                Rloc = np.empty(m * m)
                Zloc = np.empty(m * m)
                Vloc = np.empty(m * m)
                k = 0
                for eta_v in etas_arr:
                    for xi_v in xis_arr:
                        N, _, _ = elem.shape_func.evaluate(xi_v, eta_v)
                        N = np.asarray(N, dtype=float)
                        Rloc[k] = float(N @ er)
                        Zloc[k] = float(N @ ez)
                        if _is_eff:
                            c = N @ ec
                            Vloc[k] = _vm(c[0], c[1], c[2], c[3])
                        else:
                            Vloc[k] = float(N @ es)
                        k += 1

                tri = mtri.Triangulation(Rloc, Zloc, triangles=local_tris_arr)
                shading = "gouraud" if method == "gouraud" else "flat"
                last_artist = ax.tripcolor(
                    tri, Vloc, shading=shading, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolors="none",
                    antialiased=antialiased,
                )

            if last_artist is not None:
                fig.colorbar(last_artist, ax=ax, shrink=0.8)
            ax.set_title(label)
            ax.set_xlabel('r')
            ax.set_ylabel('z')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(False)

        plt.tight_layout()
        plt.show(block=self._block)

    # ===================================================================
    # Plotting — nodal scatter / dot plots
    # ===================================================================

    def plot_grid_node_values(
        self,
        mesh,
        field: str = "u_r",
        *,
        annotate: bool = True,
        max_annotate: int = 250,
        fmt: str = "{:.3e}",
        cmap: str = "viridis",
        s: int = 60,
    ) -> None:
        if self._skip():
            return
        nodes = list(mesh.nodes.values())
        if not nodes:
            return
        r = np.array([n.r for n in nodes], dtype=float)
        z = np.array([n.z for n in nodes], dtype=float)

        if field == "u_r":
            vals = np.array([n.displacements[0] for n in nodes], dtype=float)
            label, title, cmap_use = r"$u_r$", r"Nodal $u_r$ values", cmap
        elif field == "u_z":
            vals = np.array([n.displacements[1] for n in nodes], dtype=float)
            label, title, cmap_use = r"$u_z$", r"Nodal $u_z$ values", cmap
        elif field in {"sigma_rr", "sigma_zz", "sigma_rz", "sigma_tt", "sigma_eff"}:
            material = getattr(mesh, "material", None)
            if material is None:
                raise ValueError("mesh.material is required to plot nodal stresses")
            nodal_stresses = recover_all_nodal_stresses(mesh, material, mesh.shape_func)
            if field == "sigma_eff":
                def _eff(nd):
                    s = nodal_stresses.get(nd.node_id, {})
                    srr = s.get("sigma_rr", 0.0)
                    szz = s.get("sigma_zz", 0.0)
                    stt = s.get("sigma_tt", 0.0)
                    srz = s.get("sigma_rz", 0.0)
                    return float(np.sqrt(0.5 * (
                        (srr - szz) ** 2
                        + (srr - stt) ** 2
                        + (szz - stt) ** 2
                        + 6.0 * srz ** 2
                    )))
                vals = np.array([_eff(n) for n in nodes], dtype=float)
            else:
                vals = np.array(
                    [float(nodal_stresses.get(n.node_id, {}).get(field, 0.0)) for n in nodes], dtype=float
                )
            label_map = {
                "sigma_rr": r"$\sigma_{rr}$",
                "sigma_zz": r"$\sigma_{zz}$",
                "sigma_rz": r"$\sigma_{rz}$",
                "sigma_tt": r"$\sigma_{\phi\phi}$",
                "sigma_eff": r"$\sigma_{\mathrm{eff}}$",
            }
            label = label_map.get(field, field)
            title = f"Nodal {label} values"
            cmap_use = "coolwarm"
        else:
            raise ValueError(f"Unknown field {field!r}")

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(r, z, c=vals, cmap=cmap_use, s=s, edgecolor="k")
        if annotate:
            n_nodes = len(nodes)
            stride = max(1, int(np.ceil(n_nodes / max(1, int(max_annotate)))))
            z_span = float(np.max(z) - np.min(z))
            dz = 0.01 * z_span if z_span > 0 else 0.01
            for i in range(0, n_nodes, stride):
                ax.text(r[i], z[i] + dz, fmt.format(vals[i]),
                        fontsize=8, ha="center", va="bottom", color="black", alpha=0.85)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label)
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(title)
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show(block=self._block)

    def plot_grid_node_indices_and_ur(self, mesh) -> None:
        self.plot_grid_node_values(mesh, field="u_r", annotate=True, fmt="{:.3f}")

    def plot_grid_node_indices_and_sigma(
        self, mesh, component: str = "sigma_rr", *, material=None, tol: float = 1e-6,
    ) -> None:
        if self._skip():
            return
        if material is None:
            material = getattr(mesh, "material", None)
        if material is None:
            raise ValueError("Material is required to recover stresses.")
        nodal_stresses = recover_all_nodal_stresses(mesh, material, mesh.shape_func, tol=tol)
        nodes = list(mesh.nodes.values())
        r, z, vals, node_ids = [], [], [], []
        for n in nodes:
            s = nodal_stresses.get(n.node_id)
            if s is None:
                continue
            if component == "sigma_eff":
                srr = s.get("sigma_rr", 0.0)
                szz = s.get("sigma_zz", 0.0)
                stt = s.get("sigma_tt", 0.0)
                srz = s.get("sigma_rz", 0.0)
                v = float(np.sqrt(0.5 * (
                    (srr - szz) ** 2
                    + (srr - stt) ** 2
                    + (szz - stt) ** 2
                    + 6.0 * srz ** 2
                )))
            elif component not in s:
                continue
            else:
                v = s[component]
            r.append(n.r)
            z.append(n.z)
            vals.append(v)
            node_ids.append(n.node_id)
        if not vals:
            print(f"No stresses recovered for component {component}.")
            return
        r_arr, z_arr, v_arr = np.array(r), np.array(z), np.array(vals)
        comp_label = {
            "sigma_rr": r"$\sigma_{rr}$",
            "sigma_zz": r"$\sigma_{zz}$",
            "sigma_rz": r"$\sigma_{rz}$",
            "sigma_tt": r"$\sigma_{\phi\phi}$",
            "sigma_eff": r"$\sigma_{\mathrm{eff}}$",
        }.get(component, component)

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(r_arr, z_arr, c=v_arr, cmap="coolwarm", s=60, edgecolor="k")
        for i in range(len(node_ids)):
            ax.text(r_arr[i], z_arr[i] + 0.01, f"{v_arr[i]:.3e}",
                    fontsize=8, ha="center", va="bottom", color="black", alpha=0.8)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(comp_label)
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(f"Grid node indices and {comp_label} after FEM solution")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show(block=self._block)

    # ===================================================================
    # Plotting — displacement / stress overlay with error panels
    # ===================================================================

    def plot_u_sigma(
        self,
        mesh_list,
        r_min: float,
        r_max: float,
        p: float,
        mu: float,
        nu: float,
        fixed_z: float,
        fixed_r: float,
        material,
        compare_eps: float,
        experiments: List[Dict],
        block_plot: bool = False,
    ) -> None:
        """6-panel displacement + stress overlay, plus 6-panel absolute-error plot."""
        if self._skip():
            return

        r_analytical = np.linspace(r_min, r_max, 100)
        ur_analytical_full = (1 / (2 * mu * (r_max**2 - r_min**2))) * (
            (1 - 2 * nu) * r_min**2 * p * r_analytical + p * r_min**2 * r_max**2 / r_analytical
        )
        sigma_rr_af, sigma_zz_af, sigma_pp_af, sigma_rz_af = [], [], [], []
        for r in r_analytical:
            _, _, s_rr, s_zz, s_pp, s_rz = analytical_solution(r, r_min, r_max, p, mu, nu)
            sigma_rr_af.append(s_rr)
            sigma_zz_af.append(s_zz)
            sigma_pp_af.append(s_pp)
            sigma_rz_af.append(s_rz)

        interp_kind = "linear"
        overlay_data: list[dict] = []

        for idx, mesh in enumerate(mesh_list):
            # --- fixed-z displacements ---
            r_d, ur_d, uz_d = [], [], []
            for _nid, node in mesh.nodes.items():
                if abs(node.z - fixed_z) < compare_eps:
                    r_d.append(node.r)
                    ur_d.append(node.displacements[0])
                    uz_d.append(node.displacements[1])
            r_d, ur_d, uz_d = (np.array(x) for x in (r_d, ur_d, uz_d))
            si = np.argsort(r_d)
            r_d, ur_d, uz_d = r_d[si], ur_d[si], uz_d[si]

            # --- recovered nodal stresses ---
            nodal_stresses, _ = recover_nodal_stresses(
                mesh, material, experiments[idx]["shape_func"], fixed_z, fixed_r, tol=compare_eps
            )
            rs, srr, szz, srz, spp = [], [], [], [], []
            for nid, node in mesh.nodes.items():
                if abs(node.z - fixed_z) < compare_eps and nid in nodal_stresses:
                    rs.append(node.r)
                    srr.append(nodal_stresses[nid]["sigma_rr"])
                    szz.append(nodal_stresses[nid]["sigma_zz"])
                    srz.append(nodal_stresses[nid]["sigma_rz"])
                    spp.append(nodal_stresses[nid]["sigma_tt"])
            rs = np.array(rs)
            si2 = np.argsort(rs)
            rs = rs[si2]
            srr = np.array(srr)[si2]
            szz = np.array(szz)[si2]
            srz = np.array(srz)[si2]
            spp = np.array(spp)[si2]

            et = 1 if experiments[idx]["elem_type"] == ElementType.LINEAR else 2
            overlay_data.append(
                {
                    "mesh_label": f"({experiments[idx]['rN']}x{experiments[idx]['zN']}), ET{et}, Q{experiments[idx]['n_points']}",
                    "r": r_d,
                    "ur": ur_d,
                    "uz": uz_d,
                    "r_nodal": rs,
                    "sigma_rr": srr,
                    "sigma_zz": szz,
                    "sigma_rz": srz,
                    "sigma_phi_phi": spp,
                }
            )

        # --- error figure ---
        fig2, axs2 = plt.subplots(2, 3, figsize=(16, 8))
        for i, data in enumerate(overlay_data):
            c = self.COLORS[i % len(self.COLORS)]
            r_d = data["r"]
            ur_anal = np.array([analytical_solution(rv, r_min, r_max, p, mu, nu)[0] for rv in r_d])
            uz_anal = np.array([analytical_solution(rv, r_min, r_max, p, mu, nu)[1] for rv in r_d])
            axs2[0, 0].plot(r_d, np.abs(data["ur"] - ur_anal), marker="o", color=c, label=data["mesh_label"])
            axs2[1, 0].plot(r_d, np.abs(data["uz"] - uz_anal), marker="o", color=c, label=data["mesh_label"])

            rs = data["r_nodal"]
            sv = [analytical_solution(rv, r_min, r_max, p, mu, nu) for rv in rs]
            axs2[0, 1].plot(rs, np.abs(data["sigma_rr"] - np.array([v[2] for v in sv])), marker="o", color=c, label=data["mesh_label"])
            axs2[0, 2].plot(rs, np.abs(data["sigma_zz"] - np.array([v[3] for v in sv])), marker="o", color=c, label=data["mesh_label"])
            axs2[1, 1].plot(rs, np.abs(data["sigma_rz"] - np.array([v[5] for v in sv])), marker="o", color=c, label=data["mesh_label"])
            axs2[1, 2].plot(rs, np.abs(data["sigma_phi_phi"] - np.array([v[4] for v in sv])), marker="o", color=c, label=data["mesh_label"])

        titles = [
            "Abs Error in $u_r$", "Abs Error in $u_z$",
            r"Abs Error in $\sigma_{rr}$", r"Abs Error in $\sigma_{zz}$",
            r"Abs Error in $\sigma_{rz}$", r"Abs Error in $\sigma_{\phi\phi}$",
        ]
        for k, ax in enumerate(axs2.flat):
            ax.set_xlabel("Radius (r)")
            ax.set_title(f"{titles[k]}, z={fixed_z}")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show(block=block_plot)

    # ===================================================================
    # Plotting — mesh connectivity
    # ===================================================================

    def plot_mesh_connectivity(self, mesh, use_quadratic_edge=False, shape_func=None) -> None:
        if self._skip():
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        for elem in mesh.elements.values():
            corner_ids = elem.node_ids[:4]
            coords = np.array([[mesh.nodes[n].r, mesh.nodes[n].z] for n in corner_ids])
            if use_quadratic_edge and shape_func is not None:
                num_samples = 20
                edge_defs = [
                    (0, 1, lambda s: (-s, -1)),
                    (1, 2, lambda s: (1, s)),
                    (2, 3, lambda s: (s, 1)),
                    (3, 0, lambda s: (-1, s)),
                ]
                for c1, c2, _ in edge_defs:
                    if c1 == 0 and c2 == 1:
                        enids = [elem.node_ids[0], elem.node_ids[4], elem.node_ids[1]]
                    elif c1 == 1 and c2 == 2:
                        enids = [elem.node_ids[1], elem.node_ids[5], elem.node_ids[2]]
                    elif c1 == 2 and c2 == 3:
                        enids = [elem.node_ids[2], elem.node_ids[6], elem.node_ids[3]]
                    else:
                        enids = [elem.node_ids[3], elem.node_ids[7], elem.node_ids[0]]
                    s_vals = np.linspace(-1, 1, num_samples)
                    pts = []
                    for s in s_vals:
                        N1 = 0.5 * s * (s - 1)
                        N2 = 1 - s**2
                        N3 = 0.5 * s * (s + 1)
                        rr = N1 * mesh.nodes[enids[0]].r + N2 * mesh.nodes[enids[1]].r + N3 * mesh.nodes[enids[2]].r
                        zz = N1 * mesh.nodes[enids[0]].z + N2 * mesh.nodes[enids[1]].z + N3 * mesh.nodes[enids[2]].z
                        pts.append([rr, zz])
                    pts_arr = np.array(pts)
                    ax.plot(pts_arr[:, 0], pts_arr[:, 1], "b-", linewidth=1)
            else:
                coords = np.vstack([coords, coords[0]])
                ax.plot(coords[:, 0], coords[:, 1], "k-", linewidth=1)
            for nid in elem.node_ids:
                pt = [mesh.nodes[nid].r, mesh.nodes[nid].z]
                color = "bo" if nid in elem.node_ids[:4] else "ro"
                ax.plot(pt[0], pt[1], color, markersize=4)
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(f"Mesh Connectivity for {mesh.shape_func.nodes_count}-Node Elements")
        ax.grid(True)
        ax.axis("equal")
        plt.show(block=True)

    # ===================================================================
    # Plotting — fields along 1D cuts & 3D surfaces
    # ===================================================================

    def plot_fields_along(
        self,
        mesh,
        u_func,
        r_min: float,
        r_max: float,
        z_min: float,
        z_max: float,
        *,
        fixed_z: Optional[float] = None,
        fixed_r: Optional[float] = None,
        r_func=None,
        num_points: int = 100,
    ) -> None:
        if self._skip():
            return

        # 1-D cut
        if fixed_z is not None:
            t = (fixed_z - z_min) / (z_max - z_min)
            r_shift = r_func(t) if r_func else 0.0
            r_vals = np.linspace(r_min + r_shift, r_max + r_shift, num_points)
            z_vals = np.full_like(r_vals, fixed_z)
            x_vals, xlabel = r_vals, "Radius $r$"
            title = f"Fields at $z={fixed_z}$"
            one_d = True
        elif fixed_r is not None:
            t = np.linspace(0.0, 1.0, num_points)
            delta_r = (r_max - r_min) * np.vectorize(r_func)(t) if r_func else np.zeros_like(t)
            r_vals = r_min + fixed_r * delta_r
            z_vals = z_min + t * (z_max - z_min)
            x_vals, xlabel = z_vals, "Axial position $z$"
            title = f"Fields radius={fixed_r} at bottom"
            one_d = True
        else:
            one_d = False

        if one_d:
            u_r_a, u_z_a, s_rr_a, s_zz_a, s_rz_a, s_tt_a = [], [], [], [], [], []
            for rq, zq in zip(r_vals, z_vals):
                ur, uz, srr, szz, srz, stt = u_func(mesh, rq, zq)
                u_r_a.append(ur)
                u_z_a.append(uz)
                s_rr_a.append(srr)
                s_zz_a.append(szz)
                s_rz_a.append(srz)
                s_tt_a.append(stt)
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(title, y=0.98)
            for ax_i, vals, lbl in [
                (axs[0, 0], u_r_a, r"$u_r$"),
                (axs[1, 0], u_z_a, r"$u_z$"),
                (axs[0, 1], s_rr_a, r"$\sigma_{rr}$"),
                (axs[0, 2], s_zz_a, r"$\sigma_{zz}$"),
                (axs[1, 1], s_rz_a, r"$\sigma_{rz}$"),
                (axs[1, 2], s_tt_a, r"$\sigma_{\phi\phi}$"),
            ]:
                ax_i.plot(x_vals, vals, "-o")
                ax_i.set(xlabel=xlabel, ylabel=lbl)
                ax_i.grid(True)
            plt.tight_layout()
            plt.show(block=self._block)

        # Full-domain 3-D surface
        t_lin = np.linspace(0.0, 1.0, num_points)
        s_lin = np.linspace(0.0, 1.0, num_points)
        T, S = np.meshgrid(t_lin, s_lin, indexing="ij")
        Z = z_min + T * (z_max - z_min)
        delta_r = (r_max - r_min) * np.vectorize(r_func)(T) if r_func else np.zeros_like(T)
        R_inner = r_min + delta_r
        R_outer = r_max + delta_r
        R = R_inner + S * (R_outer - R_inner)
        UR = np.zeros_like(R)
        UZ = np.zeros_like(R)
        SRR = np.zeros_like(R)
        SZZ = np.zeros_like(R)
        SRZ = np.zeros_like(R)
        STT = np.zeros_like(R)
        for i in range(num_points):
            for j in range(num_points):
                ur, uz, srr, szz, srz, stt = u_func(mesh, R[i, j], Z[i, j])
                UR[i, j], UZ[i, j] = ur, uz
                SRR[i, j], SZZ[i, j], SRZ[i, j], STT[i, j] = srr, szz, srz, stt

        fig = plt.figure(figsize=(18, 12))
        labels = [r"$u_r$", r"$u_z$", r"$\sigma_{rr}$", r"$\sigma_{zz}$", r"$\sigma_{rz}$", r"$\sigma_{\phi\phi}$"]
        data = [UR, UZ, SRR, SZZ, SRZ, STT]
        for k, (dat, ttl) in enumerate(zip(data, labels), start=1):
            ax = fig.add_subplot(2, 3, k, projection="3d")
            surf = ax.plot_surface(R, Z, dat, cmap="viridis", linewidth=0, antialiased=False)
            ax.set_xlabel("r")
            ax.set_ylabel("z")
            ax.set_title(ttl)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.show(block=self._block)

    # ===================================================================
    # Plotting — deformed mesh
    # ===================================================================

    def plot_deformation(self, mesh, scale_factor: float = 1.0) -> None:
        if self._skip():
            return
        original = np.array([[n.r, n.z] for n in mesh.nodes.values()])
        deformed = np.array(
            [[n.r + scale_factor * n.displacements[0], n.z + scale_factor * n.displacements[1]] for n in mesh.nodes.values()]
        )
        plt.figure(figsize=(8, 6))
        plt.scatter(original[:, 0], original[:, 1], color="blue", label="Original Mesh", alpha=0.5)
        plt.scatter(deformed[:, 0], deformed[:, 1], color="red", label=f"Deformed Mesh (x{scale_factor})", alpha=0.8)
        for i in range(len(original)):
            plt.plot([original[i, 0], deformed[i, 0]], [original[i, 1], deformed[i, 1]], "k-", alpha=0.3)
        plt.xlabel("Radial Coordinate (r)")
        plt.ylabel("Axial Coordinate (z)")
        plt.title("Mesh Deformation")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show(block=self._block)

    # ===================================================================
    # Plotting — IBEM heatmaps & z-slice FEM vs IBEM
    # ===================================================================

    def plot_ibem_heatmaps(self, fields, mesh=None) -> None:
        """Tricontour heatmaps for IBEM displacement and stress fields."""
        if self._skip():
            return
        if not fields:
            print("No IBEM fields to plot.")
            return

        r = np.array([f["r"] for f in fields])
        z = np.array([f["z"] for f in fields])
        tri = mtri.Triangulation(r, z)

        # Mask triangles outside the mesh domain (for L-form / non-convex shapes)
        if mesh is not None and hasattr(mesh, 'elements') and mesh.elements:
            valid_regions = []
            for elem in mesh.elements.values():
                node_ids = elem.node_ids[:4] if len(elem.node_ids) >= 4 else elem.node_ids
                coords = np.array([[mesh.nodes[nid].r, mesh.nodes[nid].z] for nid in node_ids])
                r_min, r_max = coords[:, 0].min(), coords[:, 0].max()
                z_min, z_max = coords[:, 1].min(), coords[:, 1].max()
                valid_regions.append((r_min, r_max, z_min, z_max))

            mask = np.zeros(len(tri.triangles), dtype=bool)
            for i, triangle_indices in enumerate(tri.triangles):
                tri_r = r[triangle_indices].mean()
                tri_z = z[triangle_indices].mean()
                is_valid = False
                for r_min, r_max, z_min, z_max in valid_regions:
                    margin = 0.1 * max(r_max - r_min, z_max - z_min, 0.01)
                    if (r_min - margin <= tri_r <= r_max + margin and
                            z_min - margin <= tri_z <= z_max + margin):
                        is_valid = True
                        break
                mask[i] = not is_valid
            if mask.any():
                tri.set_mask(mask)

        def _plot_scalar(values, title, cmap="viridis"):
            fig, ax = plt.subplots(figsize=(6, 4))
            tcf = ax.tricontourf(tri, values, levels=40, cmap=cmap)
            ax.triplot(tri, color="k", linewidth=0.2, alpha=0.4)
            fig.colorbar(tcf, ax=ax, shrink=0.8, label=title)
            ax.set_xlabel("r")
            ax.set_ylabel("z")
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            plt.show(block=self._block)

        _plot_scalar(np.array([f["ur"] for f in fields]), "IBEM $u_r$")
        _plot_scalar(np.array([f["uz"] for f in fields]), "IBEM $u_z$")
        _plot_scalar(np.array([f["srr"] for f in fields]), r"IBEM $\sigma_{rr}$", cmap="magma")
        _plot_scalar(np.array([f["szz"] for f in fields]), r"IBEM $\sigma_{zz}$", cmap="magma")
        _plot_scalar(np.array([f["srz"] for f in fields]), r"IBEM $\sigma_{rz}$", cmap="magma")
        _plot_scalar(np.array([f["stt"] for f in fields]), r"IBEM $\sigma_{\theta\theta}$", cmap="magma")

    def plot_z_slice_fem_vs_ibem(self, mesh, ibem_slice_fields, z_slices, r_sorted) -> None:
        """Plot FEM and IBEM stresses along z-slices.

        For each z-slice, produces a 2×2 figure (one subplot per stress
        component: σ_rr, σ_zz, σ_rz, σ_φφ) comparing four FEM nodal-stress
        recovery strategies against the IBEM reference.

        Parameters
        ----------
        mesh : Mesh
        ibem_slice_fields : dict  z_target -> list of IBEM field dicts (same order as r_sorted)
        z_slices : list of float
        r_sorted : np.ndarray of r coordinates
        """
        if self._skip():
            return
        from ..postprocessors.fem_postprocessor import (
            FEMPostProcessor,
            RECOVERY_RAW,
            RECOVERY_L2,
            RECOVERY_SPR,
            RECOVERY_MORTAR,
        )

        modes = [
            (RECOVERY_RAW,    'Raw',    'o-',  1.0),
            (RECOVERY_L2,     'L²',     's-',  1.0),
            (RECOVERY_SPR,    'SPR',    '^-',  1.0),
            (RECOVERY_MORTAR, 'Mortar', 'D-',  1.0),
        ]
        n_r_pts = len(r_sorted)

        # Pre-compute FEM stresses for all modes (cached per instance)
        mode_postprocessors = []
        for mode_key, label, _, _ in modes:
            try:
                pp = FEMPostProcessor(mesh, recovery_mode=mode_key)
                mode_postprocessors.append((mode_key, label, pp))
            except Exception as exc:
                print(f"Skipping recovery mode {mode_key}: {exc}")

        comp_info = [
            (0, r'$\sigma_{rr}$'),
            (1, r'$\sigma_{zz}$'),
            (2, r'$\sigma_{rz}$'),
            (3, r'$\sigma_{\varphi\varphi}$'),
        ]
        fem_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
        fem_markers = ['o', 's', '^', 'D']

        for z_target in z_slices:
            points = np.column_stack([r_sorted, np.full(n_r_pts, z_target)])

            # IBEM reference
            ibem_rows = ibem_slice_fields.get(z_target, [])
            ibem_at_slice = np.array(
                [[f['srr'], f['szz'], f['srz'], f['stt']] for f in ibem_rows],
                dtype=float,
            ) if ibem_rows else np.full((n_r_pts, 4), np.nan)

            # Compute σ_φφ from IBEM's own fields: σ_φφ = E·(u_r/r) + ν·(σ_rr + σ_zz)
            E_val = mesh.material.E
            nu_val = mesh.material.nu
            if ibem_rows:
                ibem_stt_computed = np.array([
                    E_val * (f['ur'] / f['r']) + nu_val * (f['srr'] + f['szz'])
                    if abs(f['r']) > 1e-30 else np.nan
                    for f in ibem_rows
                ], dtype=float)
            else:
                ibem_stt_computed = np.full(n_r_pts, np.nan)

            # FEM stresses per mode
            fem_results = {}
            for mode_key, label, pp in mode_postprocessors:
                try:
                    fem_results[mode_key] = (label, pp.stresses_at(points))
                except Exception as exc:
                    print(f"Recovery mode {mode_key} failed at z={z_target}: {exc}")

            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'FEM recovery modes vs IBEM at z = {z_target:.4f}', fontsize=13)

            for ax, (col_idx, comp_label) in zip(axs.flat, comp_info):
                # Plot each FEM recovery mode
                for i, (mode_key, label, _pp) in enumerate(mode_postprocessors):
                    if mode_key not in fem_results:
                        continue
                    lbl, stresses = fem_results[mode_key]
                    ax.plot(
                        r_sorted, stresses[:, col_idx],
                        fem_markers[i] + '-',
                        color=fem_colors[i],
                        label=f'FEM {lbl}',
                        markersize=3,
                        linewidth=1.0,
                    )
                # Plot IBEM reference
                ax.plot(
                    r_sorted, ibem_at_slice[:, col_idx],
                    'x--', color='red', label='IBEM',
                    markersize=4, linewidth=1.5, alpha=0.8,
                )
                # On σ_φφ subplot, add the computed IBEM σ_φφ = E·u_r/r + ν·(σ_rr+σ_zz)
                if col_idx == 3:
                    ax.plot(
                        r_sorted, ibem_stt_computed,
                        '*--', color='darkred',
                        label=r'IBEM $E\,u_r/r + \nu(\sigma_{rr}+\sigma_{zz})$',
                        markersize=5, linewidth=1.2, alpha=0.9,
                    )
                ax.set_xlabel('r')
                ax.set_ylabel(comp_label)
                ax.set_title(comp_label)
                ax.legend(fontsize=8)
                ax.grid(True)

            plt.tight_layout()
            plt.show(block=self._block)

        # Domain heatmaps comparing recovery modes
        for comp in ('sigma_rr', 'sigma_zz', 'sigma_rz', 'sigma_tt'):
            self.plot_sigma_heatmap_recovery_comparison(mesh, comp)


# ---------------------------------------------------------------------------
# NullIOService — drop-in replacement that silences all output
# ---------------------------------------------------------------------------

class NullIOService(IOService):
    """No-op variant for headless / testing runs."""

    def __init__(self):
        super().__init__(enabled=False)
