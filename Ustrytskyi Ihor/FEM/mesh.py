from __future__ import annotations
from abc import ABC, abstractmethod

from .node import Node
from .element import AxisymmetricElement
from .material import Material
from .shapeFunction import ShapeFunction2D, ShapeFunction1D
from .quadrature import Quadrature
import math
import torch
import numpy as np
from .mortar import MortarInterface

class Mesh(ABC):
    def __init__(self, material: Material, shape_func: ShapeFunction2D, shape_func_boundary: ShapeFunction1D, quadrature: Quadrature, node_dof: int):
        self.nodes = {}
        self.elements = {}
        self.material = material
        self.shape_func = shape_func
        self.shape_func_boundary = shape_func_boundary
        self.quadrature = quadrature
        self.node_dof = node_dof
        self.leftBoundaryNodes = []
        self.rightBoundaryNodes = []
        self.topBoundaryNodes = []
        self.bottomBoundaryNodes = []
        self.innerBoundaryNodesVertical = []
        self.innerBoundaryNodesHorizontal = []
        self.boundary_nodes = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
            "inner_vertical": [],
            "inner_horizontal": [],
            "all": [],
        }
        self.boundary_segments = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
            "inner_vertical": [],
            "inner_horizontal": [],
            "all": [],
        }
        self.edge_mid_node: dict[tuple[int,int], int] = {}
    
    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def add_element(self, element: AxisymmetricElement):
        self.elements[element.elem_id] = element

    def _prune_unused_nodes_and_boundaries(self):
        """Drop nodes not referenced by any element and clean boundary lists."""
        used_nodes = {nid for elem in self.elements.values() for nid in elem.node_ids}
        unused = [nid for nid in self.nodes if nid not in used_nodes]
        for nid in unused:
            self.nodes.pop(nid, None)

        def _filtered(seq):
            return [nid for nid in seq if nid in used_nodes]

        self.leftBoundaryNodes = _filtered(self.leftBoundaryNodes)
        self.rightBoundaryNodes = _filtered(self.rightBoundaryNodes)
        self.bottomBoundaryNodes = _filtered(self.bottomBoundaryNodes)
        self.topBoundaryNodes = _filtered(self.topBoundaryNodes)
        self.innerBoundaryNodesVertical = _filtered(self.innerBoundaryNodesVertical)
        self.innerBoundaryNodesHorizontal = _filtered(self.innerBoundaryNodesHorizontal)
        self._finalize_boundaries()

    def _build_boundary_segments(self):
        self.boundary_nodes = {
            "left": list(self.leftBoundaryNodes),
            "right": list(self.rightBoundaryNodes),
            "top": list(self.topBoundaryNodes),
            "bottom": list(self.bottomBoundaryNodes),
            "inner_vertical": list(self.innerBoundaryNodesVertical),
            "inner_horizontal": list(self.innerBoundaryNodesHorizontal),
            "all": [],
        }
        segments = {"left": [], "right": [], "top": [], "bottom": [], "inner_vertical": [], "inner_horizontal": [], "all": []}

        # Side-specific segments from pre-classified nodes (rectangle-friendly)
        for side, seq in self.boundary_nodes.items():
            if side == "all":
                continue
            if len(seq) < 2:
                continue
            segments[side] = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

        # Shape-agnostic external edges (covers L-form and semiring cuts)
        edge_counts: dict[tuple[int, ...], int] = {}
        edge_sequences: dict[tuple[int, ...], tuple[int, ...]] = {}

        for elem in self.elements.values():
            ids = elem.node_ids
            if len(ids) >= 8:
                edge_list = [
                    (ids[0], ids[4], ids[1]),
                    (ids[1], ids[5], ids[2]),
                    (ids[2], ids[6], ids[3]),
                    (ids[3], ids[7], ids[0]),
                ]
            elif len(ids) == 4:
                edge_list = [
                    (ids[0], ids[1]),
                    (ids[1], ids[2]),
                    (ids[2], ids[3]),
                    (ids[3], ids[0]),
                ]
            else:
                continue

            for seq in edge_list:
                key = tuple(sorted(seq))
                edge_counts[key] = edge_counts.get(key, 0) + 1
                edge_sequences.setdefault(key, seq)

        all_segments = []
        all_nodes: list[int] = []
        for key, count in edge_counts.items():
            if count != 1:
                continue
            seq = edge_sequences[key]
            all_nodes.extend(seq)
            all_segments.extend((seq[i], seq[i + 1]) for i in range(len(seq) - 1))

        self.boundary_nodes["all"] = sorted(set(all_nodes), key=lambda nid: (self.nodes[nid].r, self.nodes[nid].z))
        segments["all"] = all_segments
        self.boundary_segments = segments

    def _finalize_boundaries(self):
        self.leftBoundaryNodes    = sorted(set(self.leftBoundaryNodes),    key=lambda nid: self.nodes[nid].z)
        self.rightBoundaryNodes   = sorted(set(self.rightBoundaryNodes),   key=lambda nid: self.nodes[nid].z)
        self.bottomBoundaryNodes  = sorted(set(self.bottomBoundaryNodes),  key=lambda nid: self.nodes[nid].r)
        self.topBoundaryNodes     = sorted(set(self.topBoundaryNodes),     key=lambda nid: self.nodes[nid].r)
        self.innerBoundaryNodesVertical   = sorted(set(self.innerBoundaryNodesVertical),   key=lambda nid: (self.nodes[nid].z, self.nodes[nid].r))
        self.innerBoundaryNodesHorizontal = sorted(set(self.innerBoundaryNodesHorizontal), key=lambda nid: (self.nodes[nid].r, self.nodes[nid].z))
        self._build_boundary_segments()

    # -------------------- Visualization --------------------
    def plot_connectivity(self, show=True, title_suffix=""):
        """Quick plotting of current mesh connectivity.

        For 4-node elements: straight edges.
        For 8-node elements: mid-side curvature approximated by quadratic interpolation (10 samples per edge).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot.")
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        plotted_mid = (self.shape_func.nodes_count == 8)
        for elem in self.elements.values():
            ids = elem.node_ids
            if len(ids) < 4:
                continue
            # Corner nodes (assume first 4 in order bl, br, tr, tl)
            corners = ids[:4]
            corner_coords = np.array([[self.nodes[i].r, self.nodes[i].z] for i in corners])
            if plotted_mid and len(ids) >= 8:
                # edges with mid nodes: (0-1-4), (1-2-5), (2-3-6), (3-0-7)
                edge_triplets = [
                    (ids[0], ids[4], ids[1]),
                    (ids[1], ids[5], ids[2]),
                    (ids[2], ids[6], ids[3]),
                    (ids[3], ids[7], ids[0])
                ]
                for a, m, b in edge_triplets:
                    A = self.nodes[a]; M = self.nodes[m]; B = self.nodes[b]
                    s_vals = np.linspace(-1, 1, 20)
                    edge_pts = []
                    for s in s_vals:
                        N1 = 0.5 * s * (s - 1)
                        N2 = 1 - s**2
                        N3 = 0.5 * s * (s + 1)
                        r = N1 * A.r + N2 * M.r + N3 * B.r
                        z = N1 * A.z + N2 * M.z + N3 * B.z
                        edge_pts.append((r, z))
                    edge_pts = np.array(edge_pts)
                    ax.plot(edge_pts[:,0], edge_pts[:,1], 'k-', linewidth=0.8)
            else:
                loop = np.vstack([corner_coords, corner_coords[0]])
                ax.plot(loop[:,0], loop[:,1], 'k-', linewidth=0.8)
            # plot nodes
            for nid in ids:
                n = self.nodes[nid]
                ax.plot(n.r, n.z, 'bo' if nid in corners else 'ro', markersize=3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('r'); ax.set_ylabel('z')
        ax.set_title(f'Mesh connectivity {title_suffix}')
        ax.grid(True)
        if show:
            plt.show(block=False)

    @abstractmethod
    def generate(self, r_min, r_max, z_min, z_max, rN, zN, r_func=lambda t:0, z_forced=None):
        pass

class Mesh4Nodes(Mesh):
    # ----------------------------------------------------------------------------------
    # H-REFINEMENT (uniform subdivision of selected 4-node elements into 4 children)
    # ----------------------------------------------------------------------------------
    def refine_elements(self, to_refine, auto_plot: bool = True):
        """Refine selected bilinear (Q4) elements.

        Accepts:
          - list[int]: element IDs (each refined 2x2 by default)
          - list[tuple]: (eid, ns, nt) or (eid, ns) where nt=ns
          - dict[int, tuple[int,int]|int]: mapping eid -> (ns, nt) or eid -> ns (nt=ns)

        Produces ns x nt child Q4 elements via bilinear interpolation of the parent corners.
        Boundary node lists are updated when refined edges lie on a global boundary.
        """
        # Normalize input to list of (eid, ns, nt)
        if to_refine is None:
            return
        tasks: list[tuple[int, int, int]] = []
        if isinstance(to_refine, dict):
            for eid, sz in to_refine.items():
                if isinstance(sz, (tuple, list)) and len(sz) == 2:
                    ns, nt = int(sz[0]), int(sz[1])
                elif isinstance(sz, int):
                    ns = nt = int(sz)
                else:
                    ns = nt = 2
                tasks.append((int(eid), max(1, ns), max(1, nt)))
        elif isinstance(to_refine, (list, tuple)):
            for item in to_refine:
                if isinstance(item, int):
                    tasks.append((item, 2, 2))
                elif isinstance(item, (list, tuple)):
                    if len(item) == 3:
                        eid, ns, nt = item
                        tasks.append((int(eid), max(1, int(ns)), max(1, int(nt))))
                    elif len(item) == 2:
                        eid, ns = item
                        tasks.append((int(eid), max(1, int(ns)), max(1, int(ns))))
        else:
            # fallback: assume single id
            if isinstance(to_refine, int):
                tasks.append((to_refine, 2, 2))
            else:
                return

        if not tasks:
            return

        # Coordinate-based node reuse cache (shared with Q8 approach)
        if not hasattr(self, "_coord_to_node"):
            self._coord_to_node = {}
        _COORD_KEY_PREC = 12
        for nid, nd in self.nodes.items():
            key = (round(float(nd.r), _COORD_KEY_PREC), round(float(nd.z), _COORD_KEY_PREC))
            self._coord_to_node[key] = nid

        next_node_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        new_elements = {}
        next_elem_id = max(self.elements.keys()) + 1 if self.elements else 0

        def add_node_xy(r, z):
            nonlocal next_node_id
            r = float(r); z = float(z)
            key = (round(r, _COORD_KEY_PREC), round(z, _COORD_KEY_PREC))
            nid_existing = self._coord_to_node.get(key)
            if nid_existing is not None:
                return nid_existing
            nid = next_node_id; next_node_id += 1
            self.add_node(Node(nid, r, z))
            self._coord_to_node[key] = nid
            return nid

        def insert_boundary_if_on(edge_nodes_pair, nid):
            a, b = edge_nodes_pair
            if a in self.leftBoundaryNodes and b in self.leftBoundaryNodes:
                self.leftBoundaryNodes.append(nid)
            if a in self.rightBoundaryNodes and b in self.rightBoundaryNodes:
                self.rightBoundaryNodes.append(nid)
            if a in self.bottomBoundaryNodes and b in self.bottomBoundaryNodes:
                self.bottomBoundaryNodes.append(nid)
            if a in self.topBoundaryNodes and b in self.topBoundaryNodes:
                self.topBoundaryNodes.append(nid)

        for eid, ns, nt in tasks:
            if eid not in self.elements:
                continue
            elem = self.elements[eid]
            if len(elem.node_ids) != 4:
                continue
            n1, n2, n3, n4 = elem.node_ids  # bl, br, tr, tl
            r1, z1 = self.nodes[n1].r, self.nodes[n1].z
            r2, z2 = self.nodes[n2].r, self.nodes[n2].z
            r3, z3 = self.nodes[n3].r, self.nodes[n3].z
            r4, z4 = self.nodes[n4].r, self.nodes[n4].z

            # Build grid of (ns+1) x (nt+1) points via bilinear mapping
            grid = [[None]*(nt+1) for _ in range(ns+1)]
            for i in range(ns+1):
                s = i/float(ns)
                for j in range(nt+1):
                    t = j/float(nt)
                    if i == 0 and j == 0:
                        nid = n1
                    elif i == ns and j == 0:
                        nid = n2
                    elif i == ns and j == nt:
                        nid = n3
                    elif i == 0 and j == nt:
                        nid = n4
                    else:
                        rb = (1-s)*(1-t)*r1 + s*(1-t)*r2 + s*t*r3 + (1-s)*t*r4
                        zb = (1-s)*(1-t)*z1 + s*(1-t)*z2 + s*t*z3 + (1-s)*t*z4
                        nid = add_node_xy(rb, zb)
                        # if on parent edges, insert into appropriate boundary list
                        if j == 0:
                            insert_boundary_if_on((n1, n2), nid)
                        if j == nt:
                            insert_boundary_if_on((n4, n3), nid)
                        if i == 0:
                            insert_boundary_if_on((n1, n4), nid)
                        if i == ns:
                            insert_boundary_if_on((n2, n3), nid)
                    grid[i][j] = nid

            # Create ns x nt children
            for i in range(ns):
                for j in range(nt):
                    c1 = grid[i][j]
                    c2 = grid[i+1][j]
                    c3 = grid[i+1][j+1]
                    c4 = grid[i][j+1]
                    cnodes = [c1, c2, c3, c4]
                    new_elements[next_elem_id] = AxisymmetricElement(
                        next_elem_id, cnodes, self.material, elem.shape_func, elem.quadrature
                    )
                    next_elem_id += 1

            # remove parent
            del self.elements[eid]

        # Append new elements
        self.elements.update(new_elements)

        # Resort boundary lists (deduplicate) after possible insertions
        self._finalize_boundaries()

    # ----------------------------------------------------------------------------------
    # NON-CONFORMING INTERFACE (MORTAR) AUTO-DETECTION (Linear elements only)
    # ----------------------------------------------------------------------------------
    def _edge_iter(self):
        """Yield (elem_id, (nA,nB)) with consistent orientation (lexicographic)."""
        for eid, elem in self.elements.items():
            if len(elem.node_ids) != 4:
                continue
            n1, n2, n3, n4 = elem.node_ids
            edges = [(n1, n2), (n2, n3), (n3, n4), (n4, n1)]
            for (a, b) in edges:
                if (self.nodes[a].r, self.nodes[a].z) <= (self.nodes[b].r, self.nodes[b].z):
                    yield eid, (a, b)
                else:
                    yield eid, (b, a)

    def _line_key(self, a, b, ang_tol=1e-8, c_tol=1e-8):
        """Return canonical key for line passing through nodes a,b (angle,c)."""
        A = self.nodes[a]; B = self.nodes[b]
        dr = B.r - A.r; dz = B.z - A.z
        L = math.hypot(dr, dz)
        if L < 1e-14:
            return None
        dr /= L; dz /= L
        angle = math.atan2(dz, dr)
        if angle < 0:
            angle += math.pi  # symmetry: reverse direction gives same line
        # normal
        nr, nz = -dz, dr
        c = nr * A.r + nz * A.z
        return (round(angle / ang_tol) * ang_tol, round(c / c_tol) * c_tol)

    def _detect_nonconforming_spans(self, tol: float = 1e-9):
        """Internal: detect nonconforming edge spans.

        Returns a list of (master_nodes, slave_nodes) for each coarse edge span [a,b]
        that has interior nodes on the same straight line (created by refinement).
        """
        # Group edges by line key
        line_groups = {}
        for eid, (a, b) in self._edge_iter():
            key = self._line_key(a, b)
            if key is None:
                continue
            line_groups.setdefault(key, []).append((eid, a, b))

        spans: list[tuple[list[int], list[int]]] = []
        for key, edges in line_groups.items():
            if len(edges) < 2:
                continue  # need at least two edges to have mismatch
            # Collect all distinct node ids on this line
            nodes_on_line = set()
            for (_, a, b) in edges:
                nodes_on_line.add(a); nodes_on_line.add(b)
            if len(nodes_on_line) <= 2:
                continue  # still conforming
            # Build a 1D parameter along the line using first edge as direction
            _first_eid, a0, b0 = edges[0]
            A0 = self.nodes[a0]; B0 = self.nodes[b0]
            dr = B0.r - A0.r; dz = B0.z - A0.z
            L0 = math.hypot(dr, dz)
            if L0 < 1e-14:
                continue
            dr /= L0; dz /= L0
            def proj(nid: int) -> float:
                node = self.nodes[nid]
                return dr * node.r + dz * node.z
            # Sort nodes along direction and map to param values
            sorted_nodes = sorted(nodes_on_line, key=lambda nid: proj(nid))
            s_map = {nid: proj(nid) for nid in sorted_nodes}
            # For each edge span on this line, if there are interior nodes strictly
            # between its endpoints, treat that edge as the coarse segment and
            # create a mortar interface for that segment alone.
            seen_spans = set()  # avoid duplicates per (start,end) pair
            for (_eid, a, b) in edges:
                sa, sb = s_map[a], s_map[b]
                if sa > sb:
                    a, b = b, a
                    sa, sb = sb, sa
                span_key = (a, b)
                if span_key in seen_spans:
                    continue
                # Collect interior nodes in open interval (sa, sb)
                interior = [nid for nid in sorted_nodes if sa < s_map[nid] < sb]
                if len(interior) == 0:
                    continue  # this edge is already fine resolution on this span
                slave_nodes = [a] + interior + [b]
                if len(slave_nodes) <= 2:
                    continue
                master_nodes = [a, b]
                spans.append((master_nodes, slave_nodes))
                seen_spans.add(span_key)
        return spans

    def detect_nonconforming_interfaces_mortar(self, normal_dof: int | None = None, n_gp: int = 2, tol: float = 1e-9):
        """Detect nonconforming spans and return original MortarInterface (Lagrange-multiplier) objects.

        Returns list[MortarInterface]. These interfaces build an augmented saddle-point system.
        """
        pairs = self._detect_nonconforming_spans(tol=tol)
        interfaces = [
            MortarInterface(master_nodes=mn, slave_nodes=sn, normal_dof=normal_dof, n_gp=int(n_gp))
            for (mn, sn) in pairs
        ]
        if interfaces:
            print('Detected nonconforming interfaces (mortar):', interfaces)
        else:
            print('Detected nonconforming interfaces (mortar): []')
        return interfaces

    def build_and_attach_mortar_interfaces(
        self,
        solver,
        n_gp: int = 2,
        normal_dof: int | None = None,
        tol: float = 1e-9,
    ):
        """Detect and attach original mortar interfaces."""
        if not hasattr(solver, 'mortar_interfaces'):
            solver.mortar_interfaces = []
        interfaces = self.detect_nonconforming_interfaces_mortar(normal_dof=normal_dof, n_gp=n_gp, tol=tol)
        solver.mortar_interfaces.extend(interfaces)
        return interfaces
    
    def generate(self, r_min, r_max, z_min, z_max, rN, zN, r_func=lambda t:0, z_forced=None):
        dr = (r_max - r_min) / rN

        # Build z-coordinate array, merging forced z-values if provided
        z_values = [z_min + j * (z_max - z_min) / zN for j in range(zN + 1)]
        if z_forced:
            dz_min = (z_max - z_min) / zN
            snap_tol = dz_min * 0.1  # snap grid values within 10% of element height
            for zf in z_forced:
                if zf < z_min - 1e-12 or zf > z_max + 1e-12:
                    continue
                # Check if any existing value is close enough to snap
                snapped = False
                for k, zv in enumerate(z_values):
                    if abs(zv - zf) < snap_tol:
                        z_values[k] = zf  # snap to forced value
                        snapped = True
                        break
                if not snapped:
                    z_values.append(zf)
            z_values = sorted(set(z_values))
        actual_zN = len(z_values) - 1

        used_positions = {(i, j) for j in range(actual_zN + 1) for i in range(rN + 1)}

        # --- Create nodes with contiguous IDs (row-major order) ---
        grid_to_id = {}
        node_id = 0
        for j in range(actual_zN + 1):
            for i in range(rN + 1):
                if (i, j) not in used_positions:
                    continue
                t = (z_values[j] - z_min) / (z_max - z_min) if z_max != z_min else 0
                r = r_min + (r_max - r_min) * r_func(t) + i * dr
                z = z_values[j]
                nid = node_id
                grid_to_id[(i, j)] = nid
                if i == 0:
                    self.leftBoundaryNodes.append(nid)
                if i == rN:
                    self.rightBoundaryNodes.append(nid)
                if j == 0:
                    self.bottomBoundaryNodes.append(nid)
                if j == actual_zN:
                    self.topBoundaryNodes.append(nid)
                self.add_node(Node(nid, r, z))
                node_id += 1

        self.innerBoundaryNodesVertical = []
        self.innerBoundaryNodesHorizontal = []

        # --- Create elements ---
        elem_id = 0
        for j in range(actual_zN):
            for i in range(rN):
                n1 = grid_to_id[(i, j)]         # left bottom
                n2 = grid_to_id[(i + 1, j)]     # right bottom
                n3 = grid_to_id[(i, j + 1)]     # left top
                n4 = grid_to_id[(i + 1, j + 1)] # right top

                coords = np.array([[self.nodes[nid].r, self.nodes[nid].z] for nid in [n1, n2, n4, n3]])
                
                quadrature = self.quadrature

                self.add_element(AxisymmetricElement(elem_id, [n1, n2, n4, n3], self.material, self.shape_func, quadrature))
                elem_id += 1

        self._finalize_boundaries()
    
class Mesh8Nodes(Mesh):
    # ----------------------------------------------------------------------------------
    # H-REFINEMENT (uniform subdivision of selected 8-node elements into 4 children)
    # ----------------------------------------------------------------------------------
    def refine_elements(self, to_refine, auto_plot: bool = True):
        """
        Refine selected 8-node (Q8) elements into ns×nt child Q8 elements.

        Parameters
        ----------
        to_refine :
            Either:
            - list[int]: element ids; each is refined 2×2
            - list[tuple]: (eid, ns) for ns×ns or (eid, ns, nt) for ns×nt
        """
        if not to_refine:
            return

        import numpy as np

        # ---------- parse to_refine into {eid: (ns, nt)} ----------
        refine_spec = {}
        for item in to_refine:
            if isinstance(item, int):
                refine_spec[item] = (2, 2)
            elif isinstance(item, (tuple, list)):
                if len(item) == 2:
                    eid, ns = item
                    refine_spec[int(eid)] = (int(ns), int(ns))
                elif len(item) == 3:
                    eid, ns, nt = item
                    refine_spec[int(eid)] = (int(ns), int(nt))
            # silently ignore malformed entries

        if not refine_spec:
            return

        # ---------- persistent caches across calls ----------
        if not hasattr(self, "_edge_point_cache"):   # key: (minA, maxB, t_rounded) -> nid
            self._edge_point_cache = {}
        if not hasattr(self, "_segment_mid_cache"):  # key: tuple(sorted((nid1, nid2))) -> nid
            self._segment_mid_cache = {}

        # Ensure boundary lists exist (so we can safely append)  # FIX: create lists if missing
        if not hasattr(self, 'leftBoundaryNodes'):   self.leftBoundaryNodes   = []
        if not hasattr(self, 'rightBoundaryNodes'):  self.rightBoundaryNodes  = []
        if not hasattr(self, 'bottomBoundaryNodes'): self.bottomBoundaryNodes = []
        if not hasattr(self, 'topBoundaryNodes'):    self.topBoundaryNodes    = []

        # boundary membership sets (for quick hints)
        left_set   = set(self.leftBoundaryNodes)
        right_set  = set(self.rightBoundaryNodes)
        bottom_set = set(self.bottomBoundaryNodes)
        top_set    = set(self.topBoundaryNodes)

        # ---- coordinate-based node reuse map (global within the mesh) ----
        if not hasattr(self, "_coord_to_node"):
            self._coord_to_node = {}
        _COORD_KEY_PREC = 12
        for nid, nd in self.nodes.items():
            key = (round(float(nd.r), _COORD_KEY_PREC), round(float(nd.z), _COORD_KEY_PREC))
            self._coord_to_node[key] = nid

        def _register_boundary_membership_from_parents(new_nid, a, b):
            """If both parents lie on the same boundary list, add the new node to that list too."""
            if a in left_set and b in left_set:
                self.leftBoundaryNodes.append(new_nid); left_set.add(new_nid)
            if a in right_set and b in right_set:
                self.rightBoundaryNodes.append(new_nid); right_set.add(new_nid)
            if a in bottom_set and b in bottom_set:
                self.bottomBoundaryNodes.append(new_nid); bottom_set.add(new_nid)
            if a in top_set and b in top_set:
                self.topBoundaryNodes.append(new_nid); top_set.add(new_nid)

        next_node_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        next_elem_id = max(self.elements.keys()) + 1 if self.elements else 0
        new_elements = {}

        def add_node_xy(r, z):
            nonlocal next_node_id
            r = float(r); z = float(z)
            key = (round(r, _COORD_KEY_PREC), round(z, _COORD_KEY_PREC))
            nid_existing = self._coord_to_node.get(key)
            if nid_existing is not None:
                return nid_existing
            nid = next_node_id
            next_node_id += 1
            self.add_node(Node(nid, r, z))
            self._coord_to_node[key] = nid
            return nid

        # ---- Q8 edge point at param t in [-1,1] (parent edge [nA - nM - nB]) ----
        def get_edge_point_Q8(nA, nM, nB, t: float):
            """
            Evaluate quadratic Lagrange on the parent edge at param t ∈ [-1,1].
            Cache key is independent of mid id; canonicalize (A,B,t).
            """
            if nA <= nB:
                Akey, Bkey = nA, nB
                t_eval = t
                A_eval, M_eval, B_eval = nA, nM, nB
            else:
                Akey, Bkey = nB, nA
                t_eval = -t
                A_eval, M_eval, B_eval = nB, nM, nA

            t_key = float(round(t_eval, 6))
            key = (Akey, Bkey, t_key)

            if key in self._edge_point_cache:
                return self._edge_point_cache[key]

            NA = 0.5 * t_eval * (t_eval - 1.0)
            NM = 1.0 - t_eval * t_eval
            NB = 0.5 * t_eval * (t_eval + 1.0)
            A = self.nodes[A_eval]; M = self.nodes[M_eval]; B = self.nodes[B_eval]
            r = NA * A.r + NM * M.r + NB * B.r
            z = NA * A.z + NM * M.z + NB * B.z

            nid = add_node_xy(r, z)
            _register_boundary_membership_from_parents(nid, Akey, Bkey)
            self._edge_point_cache[key] = nid
            return nid

        def get_mid_segment(nP, nQ):
            """Cached geometric midpoint between any two existing nodes (used for interior child edges)."""
            key = tuple(sorted((nP, nQ)))
            if key in self._segment_mid_cache:
                return self._segment_mid_cache[key]
            P = self.nodes[nP]; Q = self.nodes[nQ]
            nid = add_node_xy(0.5 * (P.r + Q.r), 0.5 * (P.z + Q.z))
            _register_boundary_membership_from_parents(nid, nP, nQ)
            self._segment_mid_cache[key] = nid
            return nid

        # ------------------------ MAIN LOOP ------------------------
        for eid, (ns, nt) in refine_spec.items():
            if eid not in self.elements or len(self.elements[eid].node_ids) != 8:
                continue

            n1, n2, n4, n3, m12, m23, m34, m41 = self.elements[eid].node_ids

            # Parametric division points
            s_vals = np.linspace(-1.0, 1.0, ns + 1)  # left→right
            v_vals = np.linspace(-1.0, 1.0, nt + 1)  # bottom→top

            # Sample parent edges with consistent left→right, bottom→top orientation
            bottom_nodes = [get_edge_point_Q8(n1, m12, n2, t=s)  for s in s_vals]     # left→right
            top_nodes    = [get_edge_point_Q8(n4, m34, n3, t=-s) for s in s_vals]     # FIX: use -s (left→right)

            # Build grid of (ns+1)×(nt+1) node ids
            grid = [[None] * (nt + 1) for _ in range(ns + 1)]
            for i in range(ns + 1):
                grid[i][0]  = bottom_nodes[i]
                grid[i][nt] = top_nodes[i]
                Pb = self.nodes[bottom_nodes[i]]
                Pt = self.nodes[top_nodes[i]]
                for j in range(1, nt):
                    w = j / nt
                    r = (1.0 - w) * Pb.r + w * Pt.r
                    z = (1.0 - w) * Pb.z + w * Pt.z
                    grid[i][j] = add_node_xy(r, z)

            # Create children, left-to-right, bottom-to-top
            for i in range(ns):
                s_mid = 0.5 * (s_vals[i] + s_vals[i + 1])
                for j in range(nt):
                    v_mid = 0.5 * (v_vals[j] + v_vals[j + 1])

                    bl = grid[i][j]
                    br = grid[i + 1][j]
                    tr = grid[i + 1][j + 1]
                    tl = grid[i][j + 1]

                    # Mid-edges: use Q8 mapping only on outer edges; interior uses midpoints
                    if j == 0:
                        m_blbr = get_edge_point_Q8(n1, m12, n2, t=s_mid)
                    else:
                        m_blbr = get_mid_segment(bl, br)

                    if i == ns - 1:
                        m_brtr = get_edge_point_Q8(n2, m23, n4, t=v_mid)   # bottom→top
                    else:
                        m_brtr = get_mid_segment(br, tr)

                    if j == nt - 1:
                        m_trtl = get_edge_point_Q8(n4, m34, n3, t=-s_mid)  # FIX: use -s_mid for left→right
                    else:
                        m_trtl = get_mid_segment(tr, tl)

                    if i == 0:
                        m_tlbl = get_edge_point_Q8(n3, m41, n1, t=-v_mid)  # top→bottom edge, so -v
                    else:
                        m_tlbl = get_mid_segment(tl, bl)

                    cnodes = [bl, br, tr, tl, m_blbr, m_brtr, m_trtl, m_tlbl]
                    new_elements[next_elem_id] = AxisymmetricElement(
                        next_elem_id, cnodes, self.elements[eid].material,
                        self.elements[eid].shape_func, self.elements[eid].quadrature
                    )
                    next_elem_id += 1

            # remove parent
            del self.elements[eid]

        # install children
        self.elements.update(new_elements)

        # deduplicate and sort boundary lists
        self.leftBoundaryNodes    = sorted(set(self.leftBoundaryNodes),    key=lambda nid: self.nodes[nid].z)
        self.rightBoundaryNodes   = sorted(set(self.rightBoundaryNodes),   key=lambda nid: self.nodes[nid].z)
        self.bottomBoundaryNodes  = sorted(set(self.bottomBoundaryNodes),  key=lambda nid: self.nodes[nid].r)
        self.topBoundaryNodes     = sorted(set(self.topBoundaryNodes),     key=lambda nid: self.nodes[nid].r)

        # ------------------------ SUMMARY PRINTS ------------------------
        print("\n=== Mesh after Q8 refinement ===")
        for eid in sorted(self.elements.keys()):
            e = self.elements[eid]
            print(f"Element {eid}: node_ids = {list(e.node_ids)}")

        def _print_boundary(name, seq, key=None):
            arr = sorted(set(seq), key=(key if key else (lambda nid: nid)))
            print(f"{name} (count={len(arr)}): {arr}")

        _print_boundary('leftBoundaryNodes',   self.leftBoundaryNodes,   key=lambda nid: self.nodes[nid].z)
        _print_boundary('rightBoundaryNodes',  self.rightBoundaryNodes,  key=lambda nid: self.nodes[nid].z)
        _print_boundary('bottomBoundaryNodes', self.bottomBoundaryNodes, key=lambda nid: self.nodes[nid].r)
        _print_boundary('topBoundaryNodes',    self.topBoundaryNodes,    key=lambda nid: self.nodes[nid].r)

    # ----------------------------------------------------------------------------------
    # NON-CONFORMING INTERFACE (MORTAR) AUTO-DETECTION (Quadratic elements)
    # ----------------------------------------------------------------------------------
    def _edge_iter(self):
        """Yield (elem_id, (nA,nB)) for each edge in quadratic elements."""
        for eid, elem in self.elements.items():
            ids = elem.node_ids
            # For 8-node quad: edges are (0,1,4), (1,2,5), (2,3,6), (3,0,7)
            edges = [
                (ids[0], ids[1]),
                (ids[1], ids[2]),
                (ids[2], ids[3]),
                (ids[3], ids[0])
            ]
            for ab in edges:
                yield eid, ab

    def _line_key(self, a, b, ang_tol=1e-8, c_tol=1e-8):
        A = self.nodes[a]; B = self.nodes[b]
        dr = B.r - A.r; dz = B.z - A.z
        L = math.hypot(dr, dz)
        if L < 1e-14:
            return None
        dr /= L; dz /= L
        angle = math.atan2(dz, dr)
        if angle < 0:
            angle += 2 * math.pi
        nr, nz = -dz, dr
        c = nr * A.r + nz * A.z
        return (round(angle / ang_tol) * ang_tol, round(c / c_tol) * c_tol)

    def _detect_nonconforming_spans(self, tol: float = 1e-9):
        """
        Detect nonconforming interfaces by:
        1) grouping element edges by straight line (orientation-invariant),
        2) collecting all *corner* nodes on that line and forming consecutive
            corner-to-corner segments,
        3) per segment, aggregating nodes from both sides and comparing lists,
        4) merging adjacent nonconforming segments into a single span, and
        5) recomputing the slave list on the merged hull.

        Returns: list of (master_nodes=[a,b], slave_nodes=[a, s1, ..., b]).
        """
        import math
        from collections import defaultdict

        # ---------- helpers ----------
        def elem_corner_edges(elem):
            ids = elem.node_ids
            if len(ids) < 4:
                return []
            # Q8 order you use: [n1, n2, n4, n3, ...]
            n1, n2, n4, n3 = ids[0], ids[1], ids[2], ids[3]
            return [(n1, n2), (n2, n4), (n4, n3), (n3, n1)]

        def line_key_and_frame(a, b):
            """Orientation-invariant line key + a consistent tangent/normal frame."""
            A = self.nodes[a]; B = self.nodes[b]
            dr, dz = (B.r - A.r), (B.z - A.z)
            L = math.hypot(dr, dz)
            if L < 1e-14:
                return None, None
            tr, tz = dr / L, dz / L            # tangent
            nr, nz = -tz, tr                    # one normal
            # canonicalize normal so key is unique
            if nr < 0 or (abs(nr) <= 1e-15 and nz < 0):
                nr, nz = -nr, -nz
            nang = math.atan2(nz, nr)
            c = nr * A.r + nz * A.z            # Hesse offset
            key = (round(nang, 12), round(c, 12))
            frame = (tr, tz, nr, nz, c, A.r, A.z)
            return key, frame

        def s_of(tr, tz, r0, z0, nid):
            P = self.nodes[nid]
            return tr * (P.r - r0) + tz * (P.z - z0)

        def centroid_side(eid, nr, nz, c):
            ids = self.elements[eid].node_ids[:4]
            A, B, C, D = (self.nodes[i] for i in ids)
            rc = 0.25 * (A.r + B.r + C.r + D.r)
            zc = 0.25 * (A.z + B.z + C.z + D.z)
            return 1 if (nr * rc + nz * zc - c) >= 0 else -1

        def ordered_unique_with_endpoints(a, b, hits):
            """hits: list of (s, nid). Ensures [a, ..., b] with unique ids ordered by s."""
            if not hits:
                return []
            hits.sort(key=lambda t: t[0])
            out, seen = [a], {a}
            for _s, nid in hits:
                if nid not in seen:
                    out.append(nid); seen.add(nid)
            if out[-1] != b:
                out.append(b)
            return out

        def coords_tuple(nlist, prec=11):
            return tuple((round(self.nodes[n].r, prec), round(self.nodes[n].z, prec)) for n in nlist)

        # ---------- 1) group edges by straight line ----------
        groups = defaultdict(list)   # key -> list of edges dicts {eid,a,b}
        frames = {}                  # key -> (tr,tz,nr,nz,c,r0,z0)
        for eid, elem in self.elements.items():
            for (a, b) in elem_corner_edges(elem):
                key, frame = line_key_and_frame(a, b)
                if key is None:
                    continue
                if key not in frames:
                    frames[key] = frame
                groups[key].append({'eid': eid, 'a': a, 'b': b})

        spans_out = []

        # ---------- process each line separately ----------
        for key, edges in groups.items():
            tr, tz, nr, nz, c, r0, z0 = frames[key]

            # 2) Collect ALL *corner* nodes lying on the line and sort by s
            corner_set = set()
            for e in edges:
                corner_set.add(e['a']); corner_set.add(e['b'])
            corners = sorted(corner_set, key=lambda nid: s_of(tr, tz, r0, z0, nid))

            if len(corners) < 2:
                continue

            # Helper: aggregate nodes per side on a finite corner-to-corner segment [A,B]
            def side_lists_on_segment(A, B):
                sA, sB = s_of(tr, tz, r0, z0, A), s_of(tr, tz, r0, z0, B)
                smin, smax = (sA, sB) if sA <= sB else (sB, sA)
                side_hits = {+1: [], -1: []}  # side -> [(s, nid)]

                # iterate all elements; include those with at least one edge colinear & overlapping the segment
                for eid, elem in self.elements.items():
                    overlaps = False
                    for (eA, eB) in elem_corner_edges(elem):
                        distA = abs(nr * self.nodes[eA].r + nz * self.nodes[eA].z - c)
                        distB = abs(nr * self.nodes[eB].r + nz * self.nodes[eB].z - c)
                        if distA > tol or distB > tol:
                            continue
                        seA = s_of(tr, tz, r0, z0, eA); seB = s_of(tr, tz, r0, z0, eB)
                        emin, emax = (seA, seB) if seA <= seB else (seB, seA)
                        if emax >= smin - tol and emin <= smax + tol:
                            overlaps = True
                            break
                    if not overlaps:
                        continue

                    # collect this element's nodes that lie on the finite segment
                    hits = []
                    for nid in elem.node_ids:
                        s = s_of(tr, tz, r0, z0, nid)
                        dist = abs(nr * self.nodes[nid].r + nz * self.nodes[nid].z - c)
                        if (smin - tol) <= s <= (smax + tol) and dist <= tol:
                            hits.append((s, nid))
                    if not hits:
                        continue
                    side = centroid_side(eid, nr, nz, c)
                    side_hits[side].extend(hits)

                pos = ordered_unique_with_endpoints(A, B, side_hits[+1])
                neg = ordered_unique_with_endpoints(A, B, side_hits[-1])
                return pos, neg

            # 3) Flag which consecutive corner segments are nonconforming
            bad_segments = []  # list of (A,B)
            for i in range(len(corners) - 1):
                A, B = corners[i], corners[i+1]
                pos, neg = side_lists_on_segment(A, B)
                # need both sides to compare
                if not pos or not neg:
                    continue
                # conforming?
                if tuple(pos) == tuple(neg) or coords_tuple(pos) == coords_tuple(neg):
                    continue
                bad_segments.append((A, B))

            if not bad_segments:
                continue

            # 4) Merge adjacent bad segments into maximal spans along the corner chain
            merged_spans = []
            cur_a, cur_b = bad_segments[0]
            for (A, B) in bad_segments[1:]:
                # If current end equals next start, merge; otherwise, start a new span
                if cur_b == A:
                    cur_b = B
                else:
                    merged_spans.append((cur_a, cur_b))
                    cur_a, cur_b = A, B
            merged_spans.append((cur_a, cur_b))

            # 5) Recompute slave list on each merged hull and emit
            for (A, B) in merged_spans:
                pos, neg = side_lists_on_segment(A, B)
                if not pos or not neg:
                    continue
                if tuple(pos) == tuple(neg) or coords_tuple(pos) == coords_tuple(neg):
                    continue
                if len(pos) >= len(neg):
                    slave_full, master_full = pos, neg
                else:
                    slave_full, master_full = neg, pos

                if len(slave_full) > len(master_full):
                    spans_out.append((master_full, slave_full))


        # Deduplicate results
        uniq, seen = [], set()
        for master, slave in spans_out:
            key = (tuple(master), tuple(slave))
            if key not in seen:
                seen.add(key)
                uniq.append((master, slave))
        return uniq

    def detect_nonconforming_interfaces_mortar(self, normal_dof: int | None = None, n_gp: int = 3, order: int = 2, tol: float = 1e-9):
        """Detect nonconforming spans and return MortarInterface objects for Q8 elements."""
        pairs = self._detect_nonconforming_spans(tol=tol)
        interfaces = [
            MortarInterface(master_nodes=mn, slave_nodes=sn, normal_dof=normal_dof, n_gp=int(n_gp), order=order)
            for (mn, sn) in pairs
        ]
        if interfaces:
            print('Detected nonconforming interfaces (mortar):', interfaces)
        else:
            print('Detected nonconforming interfaces (mortar): []')
        return interfaces

    def build_and_attach_mortar_interfaces(
        self,
        solver,
        n_gp: int = 3,
        normal_dof: int | None = None,
        order: int = 2,
        tol: float = 1e-9,
    ):
        """Detect and attach original mortar interfaces."""
        if not hasattr(solver, 'mortar_interfaces'):
            solver.mortar_interfaces = []
        interfaces = self.detect_nonconforming_interfaces_mortar(normal_dof=normal_dof, n_gp=n_gp, order=order, tol=tol)
        solver.mortar_interfaces.extend(interfaces)
        return interfaces

    def generate(self, r_min, r_max, z_min, z_max, rN, zN, r_func=lambda t:0, z_forced=None):
        dr = (r_max - r_min) / rN

        # Build z-coordinate array, merging forced z-values if provided
        z_values = [z_min + j * (z_max - z_min) / zN for j in range(zN + 1)]
        if z_forced:
            dz_min = (z_max - z_min) / zN
            snap_tol = dz_min * 0.1  # snap grid values within 10% of element height
            for zf in z_forced:
                if zf < z_min - 1e-12 or zf > z_max + 1e-12:
                    continue
                # Check if any existing value is close enough to snap
                snapped = False
                for k, zv in enumerate(z_values):
                    if abs(zv - zf) < snap_tol:
                        z_values[k] = zf  # snap to forced value
                        snapped = True
                        break
                if not snapped:
                    z_values.append(zf)
            z_values = sorted(set(z_values))
        actual_zN = len(z_values) - 1

        used_positions = {(i, j) for j in range(actual_zN + 1) for i in range(rN + 1)}

        # --- Create corner nodes with contiguous IDs (row-major order) ---
        grid_to_id = {}
        node_id = 0
        for j in range(actual_zN + 1):
            for i in range(rN + 1):
                if (i, j) not in used_positions:
                    continue
                t = (z_values[j] - z_min) / (z_max - z_min) if z_max != z_min else 0
                r = r_min + (r_max - r_min) * r_func(t) + i * dr
                z = z_values[j]
                nid = node_id
                grid_to_id[(i, j)] = nid
                if i == 0:
                    self.leftBoundaryNodes.append(nid)
                if i == rN:
                    self.rightBoundaryNodes.append(nid)
                if j == 0:
                    self.bottomBoundaryNodes.append(nid)
                if j == actual_zN:
                    self.topBoundaryNodes.append(nid)
                self.add_node(Node(nid, r, z))
                node_id += 1

        self.innerBoundaryNodesVertical = []
        self.innerBoundaryNodesHorizontal = []

        mid_node_dict = {}
    
        def get_mid_node(i1, i2):
            """Return (and create if necessary) the index of the mid-edge node.

            – Horizontal edge (constant z):        Cartesian midpoint.  
            – Vertical edge (varying z):           point on the curved profile
            r(t) = r_min + (r_max-r_min)·r_func(t) + i·dr evaluated at t_mid.
            """
            nonlocal node_id
            key = tuple(sorted((i1, i2)))
            if key in mid_node_dict:
                return mid_node_dict[key]

            n1, n2 = self.nodes[i1], self.nodes[i2]

            # ------------------------------------------------------------------
            # Detect edge orientation
            # ------------------------------------------------------------------
            if abs(n1.z - n2.z) < 1e-12:            # horizontal edge → straight
                r_mid = 0.5 * (n1.r + n2.r)
                z_mid = n1.z

            else:                                   # vertical edge → curved
                z_mid = 0.5 * (n1.z + n2.z)

                # radial index term  i·dr  (constant along the vertical edge)
                t1          = (n1.z - z_min) / (z_max - z_min)
                r_offset_1  = r_min + (r_max - r_min) * r_func(t1)
                i_term      = n1.r - r_offset_1      # = i · dr

                # evaluate r_offset at mid-height
                t_mid       = (z_mid - z_min) / (z_max - z_min)
                r_offset_mid = r_min + (r_max - r_min) * r_func(t_mid)

                r_mid = r_offset_mid + i_term

            # ------------------------------------------------------------------
            # Register and return the new node
            # ------------------------------------------------------------------
            idx = node_id
            node_id += 1
            self.add_node(Node(idx, r_mid, z_mid))
            mid_node_dict[key] = idx
            return idx

        # --- Create elements ---
        elem_id = 0
        for j in range(actual_zN):
            for i in range(rN):
                n1 = grid_to_id[(i, j)]         # left bottom
                n2 = grid_to_id[(i + 1, j)]     # right bottom
                n3 = grid_to_id[(i, j + 1)]     # left top
                n4 = grid_to_id[(i + 1, j + 1)] # right top

                m12 = get_mid_node(n1, n2) # bottom middle
                m23 = get_mid_node(n2, n4) # right middle
                m34 = get_mid_node(n3, n4) # top middle
                m41 = get_mid_node(n3, n1) # left middle

                if i == 0:
                    self.leftBoundaryNodes.append(m41)
                if i == rN-1:
                    self.rightBoundaryNodes.append(m23)
                if j == 0:
                    self.bottomBoundaryNodes.append(m12)
                if j == actual_zN-1:
                    self.topBoundaryNodes.append(m34)

                quadrature = self.quadrature
                self.add_element(AxisymmetricElement(
                        elem_id,
                        [n1, n2, n4, n3, m12, m23, m34, m41],
                        self.material,
                        self.shape_func,
                        quadrature
                    ))
                elem_id += 1

        self._finalize_boundaries()


        print("\n=== Mesh initial ===")
        # Elements and their node lists
        for eid in sorted(self.elements.keys()):
            e = self.elements[eid]
            print(f"Element {eid}: node_ids = {list(e.node_ids)}")

        # Boundary node IDs (deduped & sorted for readability)
        def _print_boundary(name, seq, key=None):
            if not hasattr(self, name):
                return
            arr = sorted(set(seq), key=(key if key else (lambda nid: nid)))
            print(f"{name} (count={len(arr)}): {seq}")

        _print_boundary('leftBoundaryNodes',   self.leftBoundaryNodes,   key=lambda nid: self.nodes[nid].z)
        _print_boundary('rightBoundaryNodes',  self.rightBoundaryNodes,  key=lambda nid: self.nodes[nid].z)
        _print_boundary('bottomBoundaryNodes', self.bottomBoundaryNodes, key=lambda nid: self.nodes[nid].r)
        _print_boundary('topBoundaryNodes',    self.topBoundaryNodes,    key=lambda nid: self.nodes[nid].r)