from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from ..interfaces import PostProcessor
from .fem_sampling import find_isoparametric_coords, u_r_z_sigma_rr_zz_rz_tt

# Recovery mode constants
RECOVERY_RAW = 'raw'
RECOVERY_L2 = 'l2'
RECOVERY_SPR = 'spr'
RECOVERY_MORTAR = 'mortar'
RECOVERY_MODES = (RECOVERY_RAW, RECOVERY_L2, RECOVERY_SPR, RECOVERY_MORTAR)


# ---------------------------------------------------------------------------
# Nodal stress recovery strategies
# ---------------------------------------------------------------------------

def _recover_raw(mesh) -> Dict[int, Tuple[float, float, float, float]]:
    """Current behaviour: shape-function-weighted Gauss-point averaging per element,
    then average over elements sharing a node."""
    D = mesh.material.get_elastic_matrix()
    stress_sum = {nid: np.zeros(4) for nid in mesh.nodes}
    stress_count = {nid: 0 for nid in mesh.nodes}

    for elem in mesh.elements.values():
        node_ids = elem.node_ids
        n = len(node_ids)
        u_e = np.zeros(2 * n)
        coords = np.zeros((n, 2))
        for a, nid in enumerate(node_ids):
            coords[a] = [mesh.nodes[nid].r, mesh.nodes[nid].z]
            u_e[a] = mesh.nodes[nid].displacements[0]
            u_e[n + a] = mesh.nodes[nid].displacements[1]
        for gp in elem.quadrature.gauss_points_2D():
            xi, eta = float(gp['xi']), float(gp['eta'])
            N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
            B, _, _ = elem.getB(N, dN_dxi, dN_deta, n, coords, n * mesh.node_dof)
            sig = D @ (B @ u_e)
            for a, nid in enumerate(node_ids):
                stress_sum[nid] += N[a] * sig
        for nid in node_ids:
            stress_count[nid] += 1

    result: Dict[int, Tuple[float, float, float, float]] = {}
    for nid in mesh.nodes:
        if stress_count[nid] > 0:
            s = stress_sum[nid] / stress_count[nid]
            result[nid] = (float(s[0]), float(s[1]), float(s[2]), float(s[3]))
        else:
            result[nid] = (0.0, 0.0, 0.0, 0.0)
    return result


def _recover_l2(mesh) -> Dict[int, Tuple[float, float, float, float]]:
    """L²-projection with lumped mass: M_i σ*_i = f_i.

    Properly integrates N_i · σ^h with quadrature weights and Jacobian
    (including axisymmetric 2πr factor), then divides by lumped mass.
    """
    D = mesh.material.get_elastic_matrix()
    f = {nid: np.zeros(4) for nid in mesh.nodes}   # integrated rhs
    m = {nid: 0.0 for nid in mesh.nodes}             # lumped mass

    for elem in mesh.elements.values():
        node_ids = elem.node_ids
        n = len(node_ids)
        u_e = np.zeros(2 * n)
        coords = np.zeros((n, 2))
        for a, nid in enumerate(node_ids):
            coords[a] = [mesh.nodes[nid].r, mesh.nodes[nid].z]
            u_e[a] = mesh.nodes[nid].displacements[0]
            u_e[n + a] = mesh.nodes[nid].displacements[1]

        for gp in elem.quadrature.gauss_points_2D():
            xi, eta, w = float(gp['xi']), float(gp['eta']), float(gp['weight'])
            N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
            B, r_cur, detJ = elem.getB(N, dN_dxi, dN_deta, n, coords, n * mesh.node_dof)
            dV = r_cur * detJ * w       # axisymmetric volume measure
            sig = D @ (B @ u_e)         # raw discontinuous stress at GP
            for a, nid in enumerate(node_ids):
                Na = float(N[a])
                f[nid] += Na * sig * dV
                m[nid] += Na * dV

    result: Dict[int, Tuple[float, float, float, float]] = {}
    for nid in mesh.nodes:
        if m[nid] > 0.0:
            s = f[nid] / m[nid]
            result[nid] = (float(s[0]), float(s[1]), float(s[2]), float(s[3]))
        else:
            result[nid] = (0.0, 0.0, 0.0, 0.0)
    _fix_hanging_nodes(mesh, result)
    return result


def _recover_spr(mesh) -> Dict[int, Tuple[float, float, float, float]]:
    """Superconvergent Patch Recovery (Zienkiewicz-Zhu SPR).

    For each node, collect Gauss-point stress samples from all elements
    sharing that node, fit a polynomial via least-squares, and evaluate
    the fit at the node location.  Polynomial basis: [1, r, z, rz].
    Falls back to simple average when least-squares is rank-deficient.
    """
    D = mesh.material.get_elastic_matrix()

    # Pre-compute Gauss-point data for each element once.
    # Each entry: (elem, list of (r_gp, z_gp, sigma_gp[4]))
    elem_gp_data: Dict[int, List[Tuple[float, float, np.ndarray]]] = {}
    elem_to_nodes: Dict[int, list] = {}
    for eid, elem in mesh.elements.items():
        node_ids = elem.node_ids
        n = len(node_ids)
        u_e = np.zeros(2 * n)
        coords = np.zeros((n, 2))
        for a, nid in enumerate(node_ids):
            coords[a] = [mesh.nodes[nid].r, mesh.nodes[nid].z]
            u_e[a] = mesh.nodes[nid].displacements[0]
            u_e[n + a] = mesh.nodes[nid].displacements[1]
        gp_list = []
        for gp in elem.quadrature.gauss_points_2D():
            xi, eta = float(gp['xi']), float(gp['eta'])
            N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
            B, _, _ = elem.getB(N, dN_dxi, dN_deta, n, coords, n * mesh.node_dof)
            sig = D @ (B @ u_e)
            r_gp = float(N @ coords[:, 0])
            z_gp = float(N @ coords[:, 1])
            gp_list.append((r_gp, z_gp, sig))
        elem_gp_data[eid] = gp_list
        elem_to_nodes[eid] = list(node_ids)

    # Build node-to-element adjacency
    node_to_elems: Dict[int, List[int]] = {nid: [] for nid in mesh.nodes}
    for eid, nids in elem_to_nodes.items():
        for nid in nids:
            node_to_elems[nid].append(eid)

    result: Dict[int, Tuple[float, float, float, float]] = {}
    for nid, node in mesh.nodes.items():
        # Collect GP samples from patch
        samples_r, samples_z, samples_sig = [], [], []
        for eid in node_to_elems[nid]:
            for r_gp, z_gp, sig in elem_gp_data[eid]:
                samples_r.append(r_gp)
                samples_z.append(z_gp)
                samples_sig.append(sig)
        if not samples_r:
            result[nid] = (0.0, 0.0, 0.0, 0.0)
            continue

        rr = np.array(samples_r)
        zz = np.array(samples_z)
        S = np.array(samples_sig)  # (n_samples, 4)

        # Polynomial basis [1, r, z, r*z]
        P = np.column_stack([np.ones_like(rr), rr, zz, rr * zz])
        try:
            # Least-squares fit: P @ a = S
            a_coeffs, _, rank, _ = np.linalg.lstsq(P, S, rcond=None)
            p_node = np.array([1.0, node.r, node.z, node.r * node.z])
            s = p_node @ a_coeffs
        except np.linalg.LinAlgError:
            s = S.mean(axis=0)

        result[nid] = (float(s[0]), float(s[1]), float(s[2]), float(s[3]))
    _fix_hanging_nodes(mesh, result)
    return result


def _fix_hanging_nodes(
    mesh,
    nodal: Dict[int, Tuple[float, float, float, float]],
) -> None:
    """Overwrite stress at hanging (interior slave) nodes by linearly
    interpolating from the master-side endpoint stresses.  Operates in-place.

    This enforces C0 continuity across non-conforming (mortar) interfaces:
    the coarse element interpolates through the master endpoints, and
    the corrected hanging-node values are consistent with that interpolation.
    """
    if not hasattr(mesh, '_detect_nonconforming_spans'):
        return
    try:
        spans = mesh._detect_nonconforming_spans()
    except Exception:
        return

    for master_nodes, slave_nodes in spans:
        if len(slave_nodes) <= 2:
            continue
        interior = slave_nodes[1:-1]
        a_id, b_id = master_nodes[0], master_nodes[-1]
        Ar, Az = mesh.nodes[a_id].r, mesh.nodes[a_id].z
        Br, Bz = mesh.nodes[b_id].r, mesh.nodes[b_id].z
        vr, vz = Br - Ar, Bz - Az
        L2 = vr * vr + vz * vz
        if L2 < 1e-30:
            continue
        sig_a = np.array(nodal.get(a_id, (0.0, 0.0, 0.0, 0.0)))
        sig_b = np.array(nodal.get(b_id, (0.0, 0.0, 0.0, 0.0)))
        for s_id in interior:
            Sr, Sz = mesh.nodes[s_id].r, mesh.nodes[s_id].z
            t = ((Sr - Ar) * vr + (Sz - Az) * vz) / L2
            t = max(0.0, min(1.0, t))
            s = (1.0 - t) * sig_a + t * sig_b
            nodal[s_id] = (float(s[0]), float(s[1]), float(s[2]), float(s[3]))


def _recover_mortar(mesh) -> Dict[int, Tuple[float, float, float, float]]:
    """Mortar-aware recovery: use raw recovery for regular nodes, but for
    hanging nodes (interior slave nodes of mortar interfaces) interpolate
    stress from the master (coarser) side using their mortar projection
    parameter along the master segment.
    """
    nodal = _recover_raw(mesh)
    _fix_hanging_nodes(mesh, nodal)
    return nodal


# ---------------------------------------------------------------------------
# FEMPostProcessor
# ---------------------------------------------------------------------------

class FEMPostProcessor(PostProcessor):
    """FEM field evaluator with selectable nodal stress recovery strategy.

    Parameters
    ----------
    mesh : Mesh object with material, elements, nodes
    recovery_mode : one of 'raw', 'l2', 'spr', 'mortar'
    """

    _RECOVERY_FN = {
        RECOVERY_RAW: _recover_raw,
        RECOVERY_L2: _recover_l2,
        RECOVERY_SPR: _recover_spr,
        RECOVERY_MORTAR: _recover_mortar,
    }

    def __init__(self, mesh: Any, recovery_mode: str = RECOVERY_RAW):
        self.mesh = mesh
        if recovery_mode not in RECOVERY_MODES:
            raise ValueError(
                f"Unknown recovery_mode={recovery_mode!r}. Choose from {RECOVERY_MODES}."
            )
        self.recovery_mode = recovery_mode
        self._nodal_cache: Optional[Dict[int, Tuple[float, float, float, float]]] = None

    def _ensure_nodal_stresses(self):
        if self._nodal_cache is not None:
            return
        self._nodal_cache = self._RECOVERY_FN[self.recovery_mode](self.mesh)

    def stresses_at(self, points: np.ndarray) -> np.ndarray:
        self._ensure_nodal_stresses()
        out = []
        for r, z in points:
            out.append(self._interpolate_stress(float(r), float(z)))
        return np.asarray(out)

    def displacements_at(self, points: np.ndarray) -> np.ndarray:
        out = []
        for r, z in points:
            ur, uz, _, _, _, _ = u_r_z_sigma_rr_zz_rz_tt(self.mesh, float(r), float(z))
            out.append([ur, uz])
        return np.asarray(out)

    def _interpolate_stress(self, r_q: float, z_q: float) -> List[float]:
        """Find the containing element and interpolate cached nodal stresses."""
        for elem in self.mesh.elements.values():
            try:
                xi, eta = find_isoparametric_coords(elem, self.mesh, r_q, z_q)
            except ValueError:
                continue
            N, _, _ = elem.shape_func.evaluate(xi, eta)
            srr = szz = srz = stt = 0.0
            for a, nid in enumerate(elem.node_ids):
                w = float(N[a])
                s = self._nodal_cache.get(nid, (0.0, 0.0, 0.0, 0.0))
                srr += w * s[0]
                szz += w * s[1]
                srz += w * s[2]
                stt += w * s[3]
            return [srr, szz, srz, stt]
        raise ValueError(f"Point ({r_q},{z_q}) outside mesh")
