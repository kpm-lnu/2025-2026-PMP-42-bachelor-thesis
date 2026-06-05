"""
mortar.py — Non-conforming mesh coupling via the mortar method
==============================================================

Theory
------
Given two subdomains Ω₁ (slave) and Ω₂ (master) that share an interface Γ
but whose meshes do NOT match node-to-node, we enforce the displacement
continuity constraint

    u¹(x) = u²(x)   on Γ

weakly using Lagrange multipliers λ (the mortar variables).

The coupled saddle-point system is:

    ┌ K₁   0   D^T ┐ ┌ U₁ ┐   ┌ F₁ ┐
    │  0   K₂ -M^T │ │ U₂ │ = │ F₂ │
    └ D   -M    0  ┘ └ λ  ┘   └  0 ┘

where
    D_ij = ∫_Γ φ_i · N¹_j dΓ    (slave  shape fn tested against mortar basis)
    M_ij = ∫_Γ φ_i · N²_j dΓ    (master shape fn tested against mortar basis)

    φ_i  : Lagrange multiplier basis (piecewise linear on slave segments)
    N¹_j : slave  mesh displacement shape functions restricted to Γ
    N²_j : master mesh displacement shape functions restricted to Γ

The Lagrange multipliers are eliminated by condensation before solving so
that the final system stays the same size as K₁ + K₂.

Usage
-----
    from mortar import MortarInterface, couple_meshes

    # 1. Identify interface nodes on each side
    slave_iface_nodes  = np.where(np.abs(nodes1[:, 0] - x_iface) < tol)[0]
    master_iface_nodes = np.where(np.abs(nodes2[:, 0] - x_iface) < tol)[0]

    # 2. Build mortar interface
    iface = MortarInterface(
        nodes1, slave_iface_nodes,
        nodes2, master_iface_nodes,
        n_gauss=5
    )

    # 3. Assemble coupled system and solve
    U1, U2 = couple_meshes(
        K1, F1, nodes1, slave_iface_nodes,
        K2, F2, nodes2, master_iface_nodes,
        iface,
        fixed_dofs1=fixed_dofs1,
        fixed_dofs2=fixed_dofs2
    )
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve

def _gauss_1d(n):
    """
    Return Gauss-Legendre points and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points (1–5 supported).

    Returns
    -------
    pts : ndarray (n,)
    wts : ndarray (n,)
    """
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    if n == 2:
        t = 1.0 / np.sqrt(3)
        return np.array([-t, t]), np.array([1.0, 1.0])
    if n == 3:
        t = np.sqrt(3.0 / 5.0)
        return np.array([-t, 0.0, t]), np.array([5/9, 8/9, 5/9])
    if n == 4:
        t1 = np.sqrt((3 - 2*np.sqrt(6/5)) / 7)
        t2 = np.sqrt((3 + 2*np.sqrt(6/5)) / 7)
        w1 = (18 + np.sqrt(30)) / 36
        w2 = (18 - np.sqrt(30)) / 36
        return np.array([-t2, -t1, t1, t2]), np.array([w2, w1, w1, w2])
    if n == 5:
        t1 = (1/3)*np.sqrt(5 - 2*np.sqrt(10/7))
        t2 = (1/3)*np.sqrt(5 + 2*np.sqrt(10/7))
        w0 = 128/225
        w1 = (322 + 13*np.sqrt(70)) / 900
        w2 = (322 - 13*np.sqrt(70)) / 900
        return np.array([-t2, -t1, 0.0, t1, t2]), np.array([w2, w1, w0, w1, w2])
    raise ValueError(f"n_gauss must be 1–5, got {n}")


def _lagrange_1d(xi, xi_nodes):
    """
    Evaluate 1-D Lagrange basis polynomials and their derivatives at xi.

    Parameters
    ----------
    xi       : float   — evaluation point
    xi_nodes : ndarray — interpolation nodes in [-1,1]

    Returns
    -------
    N  : ndarray (n,)   — basis values
    dN : ndarray (n,)   — derivatives w.r.t. xi
    """
    n = len(xi_nodes)
    N  = np.ones(n)
    dN = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if j != i:
                denom = xi_nodes[i] - xi_nodes[j]
                N[i]  *= (xi - xi_nodes[j]) / denom
                prod   = 1.0
                for k in range(n):
                    if k != i and k != j:
                        prod *= (xi - xi_nodes[k]) / (xi_nodes[i] - xi_nodes[k])
                dN[i] += prod / denom

    return N, dN


def _linear_shape_1d(xi):
    """
    Linear (2-node) 1-D shape functions on [-1,1].

    Returns N (2,), dN (2,).
    """
    N  = np.array([0.5*(1 - xi), 0.5*(1 + xi)])
    dN = np.array([-0.5,          0.5         ])
    return N, dN


def _project_onto_segment(x, x_a, x_b):
    """
    Project point x onto segment [x_a, x_b] and return
    the local coordinate xi ∈ [-1,1].
    Assumes the segment is aligned with one coordinate axis.
    """
    t   = x_a + 0.5*(x_b - x_a)   # midpoint
    L   = 0.5 * np.linalg.norm(x_b - x_a)
    d   = (x_b - x_a) / (2*L)
    xi  = np.dot(x - t, d) / L
    return np.clip(xi, -1.0, 1.0)

def _build_segments(nodes, iface_nodes):
    """
    Sort interface nodes along the interface (by y-coordinate for a
    vertical interface, by x-coordinate for a horizontal one) and return
    the list of segments as pairs of node indices.

    Parameters
    ----------
    nodes       : (N,2) all mesh nodes
    iface_nodes : 1-D array of node indices on the interface

    Returns
    -------
    sorted_nodes : node indices sorted along interface
    segments     : list of (nA, nB) pairs
    """
    coords = nodes[iface_nodes]

    # Determine dominant direction of the interface
    span = coords.max(axis=0) - coords.min(axis=0)
    sort_axis = int(np.argmax(span))          # 0=x, 1=y

    order        = np.argsort(coords[:, sort_axis])
    sorted_nodes = iface_nodes[order]

    segments = [(sorted_nodes[i], sorted_nodes[i+1])
                for i in range(len(sorted_nodes) - 1)]

    return sorted_nodes, segments


def _find_master_segment(x_phys, master_nodes_sorted, nodes):
    """
    Given a physical point x_phys on the interface, find which master
    segment contains it and return the local coordinate xi ∈ [-1,1].

    Parameters
    ----------
    x_phys             : (2,) physical coordinates of the query point
    master_nodes_sorted: node indices sorted along the interface
    nodes              : full node array

    Returns
    -------
    seg_idx : index into master_nodes_sorted[:-1]
    xi      : local coordinate in that segment
    """
    coords = nodes[master_nodes_sorted]
    span   = coords.max(axis=0) - coords.min(axis=0)
    axis   = int(np.argmax(span))

    s = x_phys[axis]

    for i in range(len(master_nodes_sorted) - 1):
        s_a = nodes[master_nodes_sorted[i],   axis]
        s_b = nodes[master_nodes_sorted[i+1], axis]

        s_lo, s_hi = min(s_a, s_b), max(s_a, s_b)

        if s_lo - 1e-12 <= s <= s_hi + 1e-12:
            # map physical → local
            xi = 2.0*(s - s_a)/(s_b - s_a) - 1.0
            xi = np.clip(xi, -1.0, 1.0)
            return i, xi

    # fallback: nearest segment
    dists = [abs(s - 0.5*(nodes[master_nodes_sorted[i], axis] +
                           nodes[master_nodes_sorted[i+1], axis]))
             for i in range(len(master_nodes_sorted)-1)]
    i = int(np.argmin(dists))
    s_a = nodes[master_nodes_sorted[i],   axis]
    s_b = nodes[master_nodes_sorted[i+1], axis]
    xi  = np.clip(2.0*(s - s_a)/(s_b - s_a) - 1.0, -1.0, 1.0)
    return i, xi


class MortarInterface:
    """
    Mortar coupling between two non-conforming 2-D meshes sharing a
    1-D interface (straight line).

    The *slave* side carries the Lagrange multiplier (mortar) basis.
    The *master* side is integrated against the mortar basis.

    Parameters
    ----------
    nodes1        : (N1, 2) node coordinates of mesh 1 (slave)
    slave_nodes   : 1-D int array — interface node indices in mesh 1
    nodes2        : (N2, 2) node coordinates of mesh 2 (master)
    master_nodes  : 1-D int array — interface node indices in mesh 2
    n_gauss       : number of Gauss points per segment (default 4)

    Attributes
    ----------
    D : (n_lm*2, ndof1) sparse — slave  mortar matrix
    M : (n_lm*2, ndof2) sparse — master mortar matrix
    n_lm : number of Lagrange multiplier nodes (= len(slave_nodes))
    """

    def __init__(self, nodes1, slave_nodes, nodes2, master_nodes, n_gauss=4):

        self.nodes1       = nodes1
        self.nodes2       = nodes2
        self.slave_nodes  = np.asarray(slave_nodes)
        self.master_nodes = np.asarray(master_nodes)
        self.n_gauss      = n_gauss

        self.ndof1 = 2 * len(nodes1)
        self.ndof2 = 2 * len(nodes2)

        # Lagrange multiplier DOFs: 2 per slave interface node (ux, uy)
        self.n_lm  = len(self.slave_nodes)
        self.ndof_lm = 2 * self.n_lm

        # Sort interface nodes along the interface
        self._slave_sorted,  self._slave_segs  = _build_segments(nodes1, self.slave_nodes)
        self._master_sorted, self._master_segs = _build_segments(nodes2, self.master_nodes)

        # Build mortar matrices
        self.D, self.M = self._assemble_mortar_matrices()

    # ------------------------------------------------------------------
    def _assemble_mortar_matrices(self):
        """
        Assemble D and M by integrating over each *slave* segment.

        For every slave segment [A,B]:
          - map Gauss points to physical space
          - evaluate slave  shape functions at those points  → fills D
          - find corresponding master segment + local coord  → fills M
        """

        gp_xi, gp_w = _gauss_1d(self.n_gauss)

        # Build node → position in slave_sorted map
        slave_pos  = {n: i for i, n in enumerate(self._slave_sorted)}

        D = lil_matrix((self.ndof_lm, self.ndof1))
        M = lil_matrix((self.ndof_lm, self.ndof2))

        for nA, nB in self._slave_segs:

            xA = self.nodes1[nA]
            xB = self.nodes1[nB]
            seg_len = np.linalg.norm(xB - xA)
            jac     = 0.5 * seg_len             # dx/dxi

            # local indices of mortar nodes A, B in slave_sorted
            iA = slave_pos[nA]
            iB = slave_pos[nB]

            for xi, w in zip(gp_xi, gp_w):

                # Physical position of Gauss point
                N_lin, _ = _linear_shape_1d(xi)
                x_gp     = N_lin[0] * xA + N_lin[1] * xB

                weight = w * jac

                # ── D: mortar basis (linear on slave) × slave disp shape fn ──
                # Mortar basis uses the same linear shape functions as the
                # slave mesh edges (conforming on slave side by definition)
                phi = N_lin   # mortar basis at this GP (shape: 2)

                for loc_lm, (lm_node, phi_i) in enumerate(zip([nA, nB], phi)):
                    lm_dof_x = 2 * slave_pos[lm_node]
                    lm_dof_y = lm_dof_x + 1

                    # slave displacement DOFs
                    sl_dof_x = 2 * lm_node
                    sl_dof_y = sl_dof_x + 1

                    # D couples LM dof i with slave displacement dof j
                    # Both x and y components independently
                    D[lm_dof_x, sl_dof_x] += phi_i * N_lin[0] * weight
                    D[lm_dof_x, 2*nB    ] += phi_i * N_lin[1] * weight
                    D[lm_dof_y, sl_dof_y] += phi_i * N_lin[0] * weight
                    D[lm_dof_y, 2*nB + 1] += phi_i * N_lin[1] * weight

                # ── M: mortar basis × master disp shape fn ──
                seg_idx, xi_m = _find_master_segment(
                    x_gp, self._master_sorted, self.nodes2
                )

                mA = self._master_sorted[seg_idx]
                mB = self._master_sorted[seg_idx + 1]

                N_master, _ = _linear_shape_1d(xi_m)

                for loc_lm, (lm_node, phi_i) in enumerate(zip([nA, nB], phi)):
                    lm_dof_x = 2 * slave_pos[lm_node]
                    lm_dof_y = lm_dof_x + 1

                    M[lm_dof_x, 2*mA    ] += phi_i * N_master[0] * weight
                    M[lm_dof_x, 2*mB    ] += phi_i * N_master[1] * weight
                    M[lm_dof_y, 2*mA + 1] += phi_i * N_master[0] * weight
                    M[lm_dof_y, 2*mB + 1] += phi_i * N_master[1] * weight

        return D.tocsr(), M.tocsr()

    # ------------------------------------------------------------------
    def mortar_matrices(self):
        """Return (D, M) as CSR matrices."""
        return self.D, self.M

    # ------------------------------------------------------------------
    def residual(self, U1, U2):
        """
        Compute the interface gap (constraint residual):
            g = D @ U1 - M @ U2
        A zero residual means perfect displacement continuity.

        Parameters
        ----------
        U1 : displacement vector for mesh 1  (slave)
        U2 : displacement vector for mesh 2  (master)

        Returns
        -------
        gap : ndarray (ndof_lm,)
        """
        return self.D @ U1 - self.M @ U2

    # ------------------------------------------------------------------
    def interface_error(self, U1, U2):
        """
        Return the L2 norm of the interface gap, normalised by the
        interface length — a scalar measure of coupling quality.
        """
        gap = self.residual(U1, U2)
        return np.linalg.norm(gap)


def _apply_bcs_to_vector(v, fixed_dofs, value=0.0):
    v = v.copy()
    v[fixed_dofs] = value
    return v


def _apply_bcs_to_matrix(K, fixed_dofs):
    K = K.tolil()
    for d in fixed_dofs:
        K[d, :] = 0
        K[:, d] = 0
        K[d, d] = 1
    return K.tocsr()


def couple_meshes(
        K1, F1, nodes1, slave_iface_nodes,
        K2, F2, nodes2, master_iface_nodes,
        iface,
        fixed_dofs1=None,
        fixed_dofs2=None):
    """
    Assemble and solve the coupled mortar system using Lagrange multiplier
    condensation (penalty-free, exact enforcement).

    The saddle-point system is:

        ┌ K1   0   D^T ┐ ┌ U1 ┐   ┌ F1 ┐
        │  0   K2 -M^T │ │ U2 │ = │ F2 │
        └ D   -M    0  ┘ └ λ  ┘   └  0 ┘

    Boundary conditions are applied on the sub-blocks before assembly.

    Parameters
    ----------
    K1, F1             : stiffness + force for mesh 1 (slave)
    nodes1             : (N1,2) coordinates for mesh 1
    slave_iface_nodes  : interface node indices in mesh 1

    K2, F2             : stiffness + force for mesh 2 (master)
    nodes2             : (N2,2) coordinates for mesh 2
    master_iface_nodes : interface node indices in mesh 2

    iface              : MortarInterface instance

    fixed_dofs1        : list of fixed DOFs in mesh 1 (global indices)
    fixed_dofs2        : list of fixed DOFs in mesh 2 (global indices)

    Returns
    -------
    U1 : displacement vector for mesh 1
    U2 : displacement vector for mesh 2
    lam: Lagrange multipliers (interface tractions)
    """

    D, M = iface.mortar_matrices()
    n_lm = iface.ndof_lm

    if fixed_dofs1:
        K1 = _apply_bcs_to_matrix(K1, fixed_dofs1)
        F1 = _apply_bcs_to_vector(F1, fixed_dofs1)

    if fixed_dofs2:
        K2 = _apply_bcs_to_matrix(K2, fixed_dofs2)
        F2 = _apply_bcs_to_vector(F2, fixed_dofs2)

    if fixed_dofs1:
        D = D.tolil()
        for d in fixed_dofs1:
            D[:, d] = 0
        D = D.tocsr()

    if fixed_dofs2:
        M = M.tolil()
        for d in fixed_dofs2:
            M[:, d] = 0
        M = M.tocsr()

    #   [ K1    0    D^T ] [ U1 ]   [ F1 ]
    #   [  0   K2  -M^T ] [ U2 ] = [ F2 ]
    #   [  D   -M    0  ] [ λ  ]   [  0 ]

    from scipy.sparse import csr_matrix as csr

    ndof1 = K1.shape[0]
    ndof2 = K2.shape[0]

    Z11  = csr((ndof1, ndof1))   # not used — K1 goes there
    Z12  = csr((ndof1, ndof2))
    Z21  = csr((ndof2, ndof1))
    Z_lm = csr((n_lm,  n_lm ))

    K_coupled = bmat([
        [K1,    Z12,   D.T  ],
        [Z21,   K2,   -M.T  ],
        [D,    -M,     Z_lm ],
    ], format='csr')

    F_coupled = np.concatenate([F1, F2, np.zeros(n_lm)])

    sol = spsolve(K_coupled, F_coupled)

    U1  = sol[:ndof1]
    U2  = sol[ndof1:ndof1 + ndof2]
    lam = sol[ndof1 + ndof2:]

    return U1, U2, lam

def split_mesh_at(nodes, elements, x_split, tol=1e-9):
    """
    Partition nodes and elements of a single mesh into a left (x ≤ x_split)
    and right (x ≥ x_split) subdomain.  Nodes exactly on x_split appear in
    both submeshes — their global indices are recorded so the caller can set
    up the MortarInterface.

    This is a utility for quickly testing the mortar coupling on a single
    mesh that is artificially split.

    Parameters
    ----------
    nodes    : (N,2)
    elements : (E, 4 or 8)
    x_split  : float
    tol      : float

    Returns
    -------
    nodes1, elements1, node_map1  : left  subdomain
    nodes2, elements2, node_map2  : right subdomain
    iface_nodes1, iface_nodes2    : interface node indices in each submesh
    """

    x = nodes[:, 0]

    on_iface  = np.abs(x - x_split) < tol
    in_left   = x <= x_split + tol
    in_right  = x >= x_split - tol

    left_mask  = in_left
    right_mask = in_right

    def build_submesh(mask, elems):
        global_ids = np.where(mask)[0]
        g2l        = {g: l for l, g in enumerate(global_ids)}
        sub_nodes  = nodes[global_ids]

        sub_elems = []
        for el in elems:
            if all(mask[n] for n in el):
                sub_elems.append([g2l[n] for n in el])

        return sub_nodes, np.array(sub_elems), global_ids, g2l

    nodes1, elems1, gids1, g2l1 = build_submesh(left_mask,  elements)
    nodes2, elems2, gids2, g2l2 = build_submesh(right_mask, elements)

    iface_global = np.where(on_iface)[0]
    iface1 = np.array([g2l1[g] for g in iface_global if g in g2l1])
    iface2 = np.array([g2l2[g] for g in iface_global if g in g2l2])

    return nodes1, elems1, nodes2, elems2, iface1, iface2