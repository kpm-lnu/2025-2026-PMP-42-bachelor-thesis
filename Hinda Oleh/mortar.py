import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from numpy.polynomial.legendre import leggauss

import scipy.sparse as sp


def _is_sparse(A) -> bool:
    return sp is not None and sp.issparse(A)


def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """Return (t, proj, dist2) where proj = a + t (b - a), t in [0, 1]."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-30:
        proj = a.copy()
        d2 = float(np.dot(p - proj, p - proj))
        return 0.0, proj, d2
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    d2 = float(np.dot(p - proj, p - proj))
    return t, proj, d2


def _project_to_polyline(p: np.ndarray, poly: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """
    Project point p onto a polyline given by vertices poly[0..N-1].
    Returns (seg_idx, t, proj) where the active segment is
    [poly[seg_idx], poly[seg_idx + 1]] and t in [0, 1].
    """
    best_d2 = float("inf")
    best = (0, 0.0, poly[0].copy())
    for i in range(len(poly) - 1):
        t, proj, d2 = _closest_point_on_segment(p, poly[i], poly[i + 1])
        if d2 < best_d2:
            best_d2 = d2
            best = (i, t, proj)
    return best


def _polyline_coords(mesh, node_ids: List[int]) -> np.ndarray:
    return np.array([[mesh.nodes[n].r, mesh.nodes[n].z] for n in node_ids], dtype=float)


def _polyline_cumulative_lengths(coords: np.ndarray) -> np.ndarray:
    s = np.zeros(len(coords), dtype=float)
    for i in range(1, len(coords)):
        ds = float(np.linalg.norm(coords[i] - coords[i - 1]))
        s[i] = s[i - 1] + ds
    return s


def _project_nodes_to_reference_polyline(
    moving_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_s: np.ndarray,
) -> np.ndarray:
    """
    Project each point in moving_coords onto the reference polyline ref_coords and return
    its arc-length coordinate on the reference polyline.
    """
    s_vals = np.zeros(len(moving_coords), dtype=float)
    for i, p in enumerate(moving_coords):
        seg, t, _ = _project_to_polyline(p, ref_coords)
        seg = max(0, min(len(ref_s) - 2, int(seg)))
        t = max(0.0, min(1.0, float(t)))
        s_vals[i] = (1.0 - t) * ref_s[seg] + t * ref_s[seg + 1]
    return s_vals


def _find_active_segment(s_nodes: np.ndarray, s: float, tol: float = 1e-12) -> int:
    """
    Return i such that s lies in [s_nodes[i], s_nodes[i+1]].
    At the right endpoint, use the last segment.
    """
    n_seg = len(s_nodes) - 1
    if n_seg <= 0:
        raise ValueError("At least two segment boundary nodes are required.")
    if s <= s_nodes[0] + tol:
        return 0
    if s >= s_nodes[-1] - tol:
        return n_seg - 1
    i = int(np.searchsorted(s_nodes, s, side="right") - 1)
    return max(0, min(n_seg - 1, i))


def _merge_breakpoints(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    vals = np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
    vals.sort()
    merged = []
    for v in vals:
        if not merged or abs(v - merged[-1]) > tol:
            merged.append(float(v))
    return np.array(merged, dtype=float)


def _lagrange_shape(node_positions: np.ndarray, s: float) -> np.ndarray:
    """
    Evaluate the 1D Lagrange basis associated with the nodal abscissae node_positions at s.
    Works for linear and quadratic traces.
    """
    x = np.asarray(node_positions, dtype=float)
    n = len(x)
    values = np.ones(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = x[i] - x[j]
            if abs(denom) <= 1e-30:
                raise ValueError("Degenerate trace segment: repeated local node positions.")
            values[i] *= (s - x[j]) / denom
    return values


def _segment_boundary_positions(node_s: np.ndarray, order: int) -> np.ndarray:
    if order == 1:
        return np.asarray(node_s, dtype=float)
    if order == 2:
        if (len(node_s) - 1) % 2 != 0:
            raise ValueError(
                "For order=2, interface nodes must have 2k+1 entries: corner-mid-corner groups."
            )
        return np.asarray(node_s[::2], dtype=float)
    raise NotImplementedError("Only order=1 and order=2 are supported.")


def _segment_node_ids(node_ids: List[int], seg_idx: int, order: int) -> List[int]:
    if order == 1:
        return [node_ids[seg_idx], node_ids[seg_idx + 1]]
    if order == 2:
        base = 2 * seg_idx
        return [node_ids[base], node_ids[base + 1], node_ids[base + 2]]
    raise NotImplementedError("Only order=1 and order=2 are supported.")


def _segment_node_positions(node_s: np.ndarray, seg_idx: int, order: int) -> np.ndarray:
    if order == 1:
        return np.asarray([node_s[seg_idx], node_s[seg_idx + 1]], dtype=float)
    if order == 2:
        base = 2 * seg_idx
        return np.asarray([node_s[base], node_s[base + 1], node_s[base + 2]], dtype=float)
    raise NotImplementedError("Only order=1 and order=2 are supported.")


@dataclass
class MortarInterface:
    """
    Classical ordinary mortar interface for Q1/Q2 trace spaces on a 2D line interface.

    Standards implemented here
    --------------------------
    - primal traces on both sides are the standard one-dimensional trace spaces of order
      ``order`` (P1 for bilinear/Q1, P2 for biquadratic/Q2),
    - the multiplier space is the classical one-sided trace space on the SLAVE side,
    - for an open interface, multiplier DOFs are attached to all slave trace nodes except the
      two interface endpoints (zero boundary condition at the interface ends),
    - mortar integrals are evaluated on the common refinement of the master/slave coarse
      interface partitions using overlap sub-intervals.

    Input conventions
    -----------------
    order = 1
        ``master_nodes`` and ``slave_nodes`` list all interface nodes in order.

    order = 2
        ``master_nodes`` and ``slave_nodes`` must list the quadratic trace nodes in order
        ``[corner, mid, corner, mid, ..., corner]``.

    Notes
    -----
    - The interface geometry is reduced to a one-dimensional reference coordinate given by the
      master polyline through ``master_nodes``. This is exact for straight coincident interfaces.
      For genuinely curved Q2 geometry, a higher-order geometric map would be needed for full
      isoparametric exactness.
    - ``normal_dof`` selects which displacement component(s) are constrained; this is a component
      tie, not a geometric normal/tangential formulation.
    """

    master_nodes: List[int]
    slave_nodes: List[int]
    normal_dof: Optional[int] = None
    n_gp: int = 3
    order: int = 1
    projection_tol: float = 1e-10

    def assemble_augmented(self, K_global, f_global, mesh):
        """
        Build the mortar coupling matrix B for the saddle-point system
            [K  B^T][u] = [f]
            [B   0 ][λ]   [0]

        Returns
        -------
        B : (n_lambda, n_u) sparse/dense array
            Classical ordinary mortar coupling matrix for Q1/Q2 traces.
        """
        if self.order not in (1, 2):
            raise NotImplementedError("Only order=1 (Q1) and order=2 (Q2) are supported.")

        if len(self.master_nodes) < 2 or len(self.slave_nodes) < 2:
            n_u = int(K_global.shape[0])
            if _is_sparse(K_global):
                return sp.lil_matrix((0, n_u), dtype=float)
            return np.zeros((0, n_u), dtype=float)

        if self.order == 2:
            if (len(self.master_nodes) - 1) % 2 != 0:
                raise ValueError(
                    "For order=2, master_nodes must have 2k+1 entries in corner-mid-corner order."
                )
            if (len(self.slave_nodes) - 1) % 2 != 0:
                raise ValueError(
                    "For order=2, slave_nodes must have 2k+1 entries in corner-mid-corner order."
                )

        comps = [0, 1] if self.normal_dof is None else [int(self.normal_dof)]
        n_comp = len(comps)
        n_u = int(K_global.shape[0])

        master_coords = _polyline_coords(mesh, self.master_nodes)
        slave_coords = _polyline_coords(mesh, self.slave_nodes)

        master_s = _polyline_cumulative_lengths(master_coords)
        total_len = float(master_s[-1])
        if total_len <= 1e-30:
            raise ValueError("Degenerate master interface polyline.")

        # Use the master polyline as the geometric reference for integration and project the
        # slave trace nodes to this reference.
        slave_s = _project_nodes_to_reference_polyline(slave_coords, master_coords, master_s)

        # Consistency checks: ordering and coincident endpoints.
        if np.any(np.diff(slave_s) <= self.projection_tol):
            raise ValueError(
                "slave_nodes must follow the interface in strict order and define non-degenerate trace nodes."
            )
        if abs(slave_s[0] - master_s[0]) > self.projection_tol or abs(slave_s[-1] - master_s[-1]) > self.projection_tol:
            raise ValueError(
                "Master/slave interface endpoints do not coincide after projection; classical mortar "
                "requires the same geometric interface."
            )

        # Standard ordinary mortar space on an open interface: slave trace space with zero
        # boundary condition at the interface endpoints. For Q2, this means all slave trace nodes
        # except the first and last endpoints remain multiplier DOFs, including mid-edge nodes.
        interior_slave_nodes = self.slave_nodes[1:-1]
        n_lambda = len(interior_slave_nodes) * n_comp

        if _is_sparse(K_global):
            B = sp.lil_matrix((n_lambda, n_u), dtype=float)
        else:
            B = np.zeros((n_lambda, n_u), dtype=float)

        if n_lambda == 0:
            return B

        lambda_row_of_node = {nid: i for i, nid in enumerate(interior_slave_nodes)}

        master_seg_s = _segment_boundary_positions(master_s, self.order)
        slave_seg_s = _segment_boundary_positions(slave_s, self.order)

        if np.any(np.diff(master_seg_s) <= self.projection_tol):
            raise ValueError("Master interface contains degenerate coarse trace segments.")
        if np.any(np.diff(slave_seg_s) <= self.projection_tol):
            raise ValueError("Slave interface contains degenerate coarse trace segments.")

        # Common refinement / overlap partition of the coarse interface segments.
        breakpoints = _merge_breakpoints(master_seg_s, slave_seg_s, tol=self.projection_tol)
        min_gp = 2 if self.order == 1 else 3
        xi_gp, w_gp = leggauss(max(min_gp, int(self.n_gp)))

        for k in range(len(breakpoints) - 1):
            a = float(breakpoints[k])
            b = float(breakpoints[k + 1])
            h = b - a
            if h <= self.projection_tol:
                continue

            s_mid = 0.5 * (a + b)
            i_m = _find_active_segment(master_seg_s, s_mid, tol=self.projection_tol)
            i_s = _find_active_segment(slave_seg_s, s_mid, tol=self.projection_tol)

            m_nids = _segment_node_ids(self.master_nodes, i_m, self.order)
            s_nids = _segment_node_ids(self.slave_nodes, i_s, self.order)
            m_loc_s = _segment_node_positions(master_s, i_m, self.order)
            s_loc_s = _segment_node_positions(slave_s, i_s, self.order)

            for xi, w in zip(xi_gp, w_gp):
                s_gp = 0.5 * ((1.0 - xi) * a + (1.0 + xi) * b)
                ds = 0.5 * h * float(w)

                Nm = _lagrange_shape(m_loc_s, s_gp)
                Ns = _lagrange_shape(s_loc_s, s_gp)

                # Classical multiplier basis on the slave trace space: same local trace basis on
                # the active slave coarse segment, but with endpoint functions removed globally.
                mu_vals = Ns
                mu_nids = s_nids

                for lc, c in enumerate(comps):
                    for local_mu, mu_nid in enumerate(mu_nids):
                        row_base = lambda_row_of_node.get(mu_nid, None)
                        if row_base is None:
                            # Endpoint multiplier basis functions are not present in M_h^0.
                            continue

                        mu = float(mu_vals[local_mu])
                        if abs(mu) <= 1e-15:
                            continue

                        row = row_base * n_comp + lc

                        # Master contribution: + ∫ mu * u_m ds
                        for a_loc, mnid in enumerate(m_nids):
                            dof_m = mesh.nodes[mnid].dof_indices[c]
                            B[row, dof_m] += mu * Nm[a_loc] * ds

                        # Slave contribution: - ∫ mu * u_s ds
                        for a_loc, snid in enumerate(s_nids):
                            dof_s = mesh.nodes[snid].dof_indices[c]
                            B[row, dof_s] -= mu * Ns[a_loc] * ds

        return B
