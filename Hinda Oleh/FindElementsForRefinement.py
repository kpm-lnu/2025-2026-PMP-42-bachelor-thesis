import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class FindElementsForRefinement:
    def find(self, mesh, threshold, bcs=None, mode="eta"):
        raise NotImplementedError

class FindElementsForRefinementIBEM(FindElementsForRefinement):
    def __init__(self, static_boundary=False, static_segments_per_side=None, io=None):
        self.static_boundary = static_boundary
        self.static_segments_per_side = static_segments_per_side or {
            'left': 8, 'bottom': 8, 'right': 4, 'top': 4,
            'inner_vertical': 4, 'inner_horizontal': 4,
        }
        self._cached_static_segments = None  # built once, reused across cycles
        self._io = io  # IOService instance (None = use local fallback plots)

    def _build_static_segments(self, mesh, bcs):
        """Build IBEM boundary segments at a fixed resolution (uniform subdivision per side).

        The segments are independent of the FEM mesh refinement state — only the
        *geometry* of each boundary side (its two endpoints) is needed, plus the
        boundary-condition type/value that applies to that side.
        """
        from collections import defaultdict
        from .boundaryConditions import DirichletBC, NeumannBC

        # ---------- Collect BC maps PER SIDE (avoid corner contamination) ----------
        side_node_lists = {
            'left': list(mesh.leftBoundaryNodes),
            'bottom': list(mesh.bottomBoundaryNodes),
            'right': list(mesh.rightBoundaryNodes),
            'top': list(mesh.topBoundaryNodes),
        }

        boundary_sets = {s: set(nids) for s, nids in side_node_lists.items()}

        _mk_vals  = lambda: defaultdict(lambda: [0.0, 0.0])
        _mk_flags = lambda: defaultdict(lambda: [False, False])
        side_neumann_vals  = {s: _mk_vals()  for s in boundary_sets}
        side_neumann_flags = {s: _mk_flags() for s in boundary_sets}
        side_dirichlet_vals  = {s: _mk_vals()  for s in boundary_sets}
        side_dirichlet_flags = {s: _mk_flags() for s in boundary_sets}

        for bc in bcs:
            if isinstance(bc, DirichletBC):
                for side_name, side_set in boundary_sets.items():
                    if bc.node_id in side_set:
                        side_dirichlet_vals[side_name][bc.node_id][bc.dof] = bc.value
                        side_dirichlet_flags[side_name][bc.node_id][bc.dof] = True
            elif isinstance(bc, NeumannBC):
                bc_nodes = set(bc.edge_nodes)
                for side_name, side_set in boundary_sets.items():
                    if bc_nodes.issubset(side_set):
                        for nid in bc_nodes:
                            side_neumann_vals[side_name][nid][bc.dof] = bc.traction_value
                            side_neumann_flags[side_name][nid][bc.dof] = True
                        break

        # Sort each side's nodes along its natural coordinate
        sort_axis = {
            'left': lambda nid: mesh.nodes[nid].z,
            'right': lambda nid: mesh.nodes[nid].z,
            'bottom': lambda nid: mesh.nodes[nid].r,
            'top': lambda nid: mesh.nodes[nid].r,
            'inner_vertical': lambda nid: mesh.nodes[nid].z,
            'inner_horizontal': lambda nid: mesh.nodes[nid].r,
        }

        # ---------- Helpers ----------
        def _component_mode_side(vec, node_ids, side_name):
            tol = 1e-10
            needed = []
            if abs(vec[0]) > tol:
                needed.append(0)
            if abs(vec[1]) > tol:
                needed.append(1)
            if not needed:
                return 'neumann'
            d_flags = side_dirichlet_flags[side_name]
            n_flags = side_neumann_flags[side_name]
            for dof in needed:
                all_dir = all(d_flags[nid][dof] for nid in node_ids)
                all_neu = all(n_flags[nid][dof] for nid in node_ids)
                if all_dir:
                    return 'dirichlet'
                if all_neu:
                    return 'neumann'
            return 'zero'

        def _component_values_side(vec, node_ids, mode, side_name):
            d_vals = side_dirichlet_vals[side_name]
            n_vals = side_neumann_vals[side_name]
            n_flags = side_neumann_flags[side_name]
            values = []
            for nid in node_ids:
                if mode == 'dirichlet':
                    ur = d_vals[nid][0]
                    uz = d_vals[nid][1]
                elif mode == 'neumann':
                    ur = n_vals[nid][0] if n_flags[nid][0] else 0.0
                    uz = n_vals[nid][1] if n_flags[nid][1] else 0.0
                else:
                    ur = 0.0
                    uz = 0.0
                values.append(float(ur * vec[0] + uz * vec[1]))
            return values

        segments = []
        for side_name, node_ids in side_node_lists.items():
            if len(node_ids) < 2:
                continue
            n_seg = self.static_segments_per_side.get(side_name, 4)
            sorted_ids = sorted(node_ids, key=sort_axis[side_name])
            # Endpoints of the whole side
            p0 = np.array([mesh.nodes[sorted_ids[0]].r, mesh.nodes[sorted_ids[0]].z])
            p1 = np.array([mesh.nodes[sorted_ids[-1]].r, mesh.nodes[sorted_ids[-1]].z])
            side_vec = p1 - p0
            side_len = np.linalg.norm(side_vec)
            if side_len < 1e-12:
                continue

            # Tangent/normal for BC projection
            tangent = side_vec / side_len
            normal = np.array([tangent[1], -tangent[0]])
            tang_mode = _component_mode_side(tangent, sorted_ids, side_name)
            norm_mode = _component_mode_side(normal, sorted_ids, side_name)

            # Side-level BC values (use the first mesh node's value as representative)
            # Since all nodes on a side share the same BC type, pick a representative.
            tang_val_repr = _component_values_side(tangent, [sorted_ids[0]], tang_mode, side_name)[0]
            norm_val_repr = _component_values_side(normal, [sorted_ids[0]], norm_mode, side_name)[0]

            for i in range(n_seg):
                t0 = i / n_seg
                t1 = (i + 1) / n_seg
                seg_start = p0 + t0 * side_vec
                seg_end = p0 + t1 * side_vec
                kt = [float(seg_start[0]), float(seg_start[1]),
                      float(seg_end[0]), float(seg_end[1])]
                zgu = [tang_val_repr, norm_val_repr,
                       tang_val_repr, norm_val_repr]
                segments.append({
                    'kod': 't',
                    'kge': 1,
                    'kt': kt,
                    'vgu': ('f' if tang_mode == 'dirichlet' else 't') + ('f' if norm_mode == 'dirichlet' else 't'),
                    'zgu': zgu,
                })

        if not segments:
            raise ValueError("No static boundary segments could be built.")
        return segments

    def _write_ibem_input(self, mesh, bcs, file_path="ibem_input.txt", field_params=None, static_segments=None, extra_query_points=None):
        """Serialize boundary mesh + BCs into legacy IBEM .dan-like format.

        Returns the number of Gauss-point query rows written (before extra_query_points).
        """
        from collections import defaultdict
        from .boundaryConditions import DirichletBC, NeumannBC
        if not mesh.nodes:
            raise ValueError("Mesh has no nodes to export.")

        material = mesh.material
        mu = material.E / (2 * (1 + material.nu))
        nu = material.nu
        pa1 = mesh.shape_func_boundary.nodes_count - 1
        nodes_needed = pa1 + 1
        step = max(1, mesh.shape_func_boundary.nodes_count - 1)

        # --- Build boundary side sets FIRST (needed for per-side BC maps) ---
        boundary_sets = {
            "bottom": set(mesh.bottomBoundaryNodes),
            "right": set(mesh.rightBoundaryNodes),
            "top": set(mesh.topBoundaryNodes),
            "left": set(mesh.leftBoundaryNodes),
        }

        # --- Build PER-SIDE BC maps to prevent corner value contamination ---
        # At corner nodes shared by two sides (e.g. bottom & right), the global
        # neumann_vals dict would contain whichever side's BC was written last.
        # Per-side maps ensure each segment uses only BCs defined for its own side.
        _mk_vals  = lambda: defaultdict(lambda: [0.0, 0.0])
        _mk_flags = lambda: defaultdict(lambda: [False, False])

        side_neumann_vals  = {s: _mk_vals()  for s in boundary_sets}
        side_neumann_flags = {s: _mk_flags() for s in boundary_sets}
        side_dirichlet_vals  = {s: _mk_vals()  for s in boundary_sets}
        side_dirichlet_flags = {s: _mk_flags() for s in boundary_sets}

        for bc in bcs:
            if isinstance(bc, DirichletBC):
                # A point Dirichlet BC (e.g. corner pin) may belong to several sides
                for side_name, side_set in boundary_sets.items():
                    if bc.node_id in side_set:
                        side_dirichlet_vals[side_name][bc.node_id][bc.dof] = bc.value
                        side_dirichlet_flags[side_name][bc.node_id][bc.dof] = True
            elif isinstance(bc, NeumannBC):
                bc_nodes = set(bc.edge_nodes)
                # Assign to the side whose node set contains ALL of the BC's edge_nodes
                for side_name, side_set in boundary_sets.items():
                    if bc_nodes.issubset(side_set):
                        for nid in bc_nodes:
                            side_neumann_vals[side_name][nid][bc.dof] = bc.traction_value
                            side_neumann_flags[side_name][nid][bc.dof] = True
                        break

        def component_mode(vec, node_ids, side_name):
            """Decide if the component is Dirichlet, Neumann, or zero on this side."""
            tol = 1e-10
            needed = []
            if abs(vec[0]) > tol:
                needed.append(0)
            if abs(vec[1]) > tol:
                needed.append(1)
            if not needed:
                return 'neumann'

            d_flags = side_dirichlet_flags[side_name]
            n_flags = side_neumann_flags[side_name]
            for dof in needed:
                all_dir = all(d_flags[nid][dof] for nid in node_ids)
                all_neu = all(n_flags[nid][dof] for nid in node_ids)
                if all_dir:
                    return 'dirichlet'
                if all_neu:
                    return 'neumann'
            # Mixed BCs on this side: treat as zero load to avoid leaking forces from other sides.
            return 'zero'

        def component_values(vec, node_ids, mode, side_name):
            d_vals = side_dirichlet_vals[side_name]
            n_vals = side_neumann_vals[side_name]
            n_flags = side_neumann_flags[side_name]
            values = []
            for nid in node_ids:
                if mode == 'dirichlet':
                    ur = d_vals[nid][0]
                    uz = d_vals[nid][1]
                elif mode == 'neumann':
                    ur = n_vals[nid][0] if n_flags[nid][0] else 0.0
                    uz = n_vals[nid][1] if n_flags[nid][1] else 0.0
                else:  # zero for mixed boundaries
                    ur = 0.0
                    uz = 0.0
                values.append(float(ur * vec[0] + uz * vec[1]))
            return values

        def element_edges(elem):
            ids = elem.node_ids
            if len(ids) == 4:
                return [
                    (ids[0], ids[1]),
                    (ids[1], ids[2]),
                    (ids[2], ids[3]),
                    (ids[3], ids[0]),
                ]
            if len(ids) >= 8:
                return [
                    (ids[0], ids[4], ids[1]),
                    (ids[1], ids[5], ids[2]),
                    (ids[2], ids[6], ids[3]),
                    (ids[3], ids[7], ids[0]),
                ]
            return []

        def edge_boundary_side(edge_nodes):
            """Return the boundary side name if edge lies on one, else None."""
            node_set = set(edge_nodes)
            for side_name, bset in boundary_sets.items():
                if node_set.issubset(bset):
                    return side_name
            return None

        segments = []
        if static_segments is not None:
            # Use pre-built static boundary segments
            segments = static_segments
        else:
            for elem_id in sorted(mesh.elements.keys()):
                elem = mesh.elements[elem_id]
                for edge_nodes in element_edges(elem):
                    side_name = edge_boundary_side(edge_nodes)
                    if side_name is None:
                        continue
                    coords = [(mesh.nodes[nid].r, mesh.nodes[nid].z) for nid in edge_nodes]
                    start = np.array(coords[0])
                    end = np.array(coords[-1])
                    delta = end - start
                    length = np.linalg.norm(delta)
                    if length < 1e-12:
                        continue
                    tangent = delta / length
                    normal = np.array([tangent[1], -tangent[0]])
                    tang_mode = component_mode(tangent, edge_nodes, side_name)
                    norm_mode = component_mode(normal, edge_nodes, side_name)
                    tang_vals = component_values(tangent, edge_nodes, tang_mode, side_name)
                    norm_vals = component_values(normal, edge_nodes, norm_mode, side_name)
                    zgu = []
                    for tv, nv in zip(tang_vals, norm_vals):
                        zgu.extend([tv, nv])
                    kt = []
                    for r, z in coords:
                        kt.extend([float(r), float(z)])
                    kge = max(1, len(edge_nodes) - 1)
                    segments.append({
                        'kod': 't',# if not segments else 'f',
                        'kge': kge,
                        'kt': kt,
                        'vgu': ('f' if tang_mode == 'dirichlet' else 't') + ('f' if norm_mode == 'dirichlet' else 't'),
                        'zgu': zgu
                    })

        if not segments:
            raise ValueError("No boundary segments detected for IBEM export.")

        r_vals = [node.r for node in mesh.nodes.values()]
        z_vals = [node.z for node in mesh.nodes.values()]
        if field_params is None:
            field_params = (1, 1, min(r_vals), min(z_vals), max(r_vals), max(z_vals))
        ksegf, ktf, rl, zn, rp, zv = field_params

        def fmt(val):
            if isinstance(val, float):
                if abs(val) < 1e-12:
                    val = 0.0
                return f"{val:.10g}"
            return str(val)

        with open(file_path, 'w', encoding='ascii') as fh:
            fh.write(f"{fmt(mu)}\n")
            fh.write(f"{fmt(nu)}\n")
            fh.write(f"{pa1}\n")
            fh.write(f"{len(segments)}\n\n")
            for seg in segments:
                fh.write(f"{seg['kod']}\n")
                fh.write(f"{seg['kge']}\n")
                fh.write(" ".join(fmt(val) for val in seg['kt']) + "\n")
                fh.write(f"{seg['vgu']}\n")
                fh.write(" ".join(fmt(val) for val in seg['zgu']) + "\n\n")
            fh.write(f"{ksegf}\n")
            fh.write(f"{ktf}\n")
            fh.write(" ".join(fmt(val) for val in (rl, zn, rp, zv)) + "\n")

            # Append Gauss-point coordinates (query points for IBEM fields)
            fh.write("\n")
            n_gauss_pts = 0
            for eid in sorted(mesh.elements.keys()):
                elem = mesh.elements[eid]
                node_ids = elem.node_ids
                coords = np.array([[mesh.nodes[nid].r, mesh.nodes[nid].z] for nid in node_ids], dtype=float)
                for gp in elem.quadrature.gauss_points_2D():
                    xi = gp["xi"]
                    eta = gp["eta"]
                    N, _, _ = elem.shape_func.evaluate(xi, eta)
                    N = np.asarray(N, dtype=float).reshape(-1)
                    if N.shape[0] != coords.shape[0]:
                        raise ValueError(
                            f"Shape function size mismatch for element {eid}: "
                            f"N has {N.shape[0]} entries, coords has {coords.shape[0]} nodes"
                        )
                    r_gp = float(np.dot(N, coords[:, 0]))
                    z_gp = float(np.dot(N, coords[:, 1]))
                    fh.write(f"{fmt(r_gp)} {fmt(z_gp)}\n")
                    n_gauss_pts += 1

            # Append extra query points (e.g. z-slice sampling points)
            if extra_query_points is not None:
                for r_q, z_q in extra_query_points:
                    fh.write(f"{fmt(float(r_q))} {fmt(float(z_q))}\n")

        print('IBEM BE: ', len(segments))
        return n_gauss_pts

    def _run_ibem_console(self, ibem_input_path: str = "ibem_input.txt", exe_name: str = "FEM\\IBEMConsole.exe"):
        """Run IBEMConsole.exe synchronously so we wait for completion."""
        input_path = Path(ibem_input_path).resolve()
        search_paths = [
            Path(__file__).resolve().parent.parent / exe_name,
            input_path.parent / exe_name,
        ]
        exe_path = next((p for p in search_paths if p.exists()), None)
        if exe_path is None:
            print(f"IBEMConsole.exe not found. Searched: {[str(p) for p in search_paths]}")
            return

        try:
            print(f"Running {exe_path} ...")
            completed = subprocess.run(
                [str(exe_path)],
                cwd=input_path.parent,
                capture_output=True,
                text=True,
                check=True,
            )
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr)
            print("IBEMConsole.exe finished successfully.")
        except subprocess.CalledProcessError as exc:
            print(f"IBEMConsole.exe failed with code {exc.returncode}")
            if exc.stdout:
                print(exc.stdout)
            if exc.stderr:
                print(exc.stderr)
        except Exception as exc:
            print(f"IBEMConsole.exe execution error: {exc}")

    def _parse_ibem_output(self, file_path: str = "ibem_output.txt"):
        """Parse ibem_output.txt into structured data (potentials (optional) + field rows)."""
        path = Path(file_path)
        if not path.exists():
            print(f"IBEM output not found: {path}")
            return {"potentials": [], "fields": []}

        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

        def to_float(val: str) -> float:
            return float(val.replace(",", "."))

        potentials = []
        fields = []
        state = "search"
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Tochky i potencialy"):
                state = "potentials_header"
                continue
            if state == "potentials_header":
                if line.lower().startswith("r"):
                    state = "potentials"
                continue
            if state == "potentials":
                if line.startswith("***"):
                    state = "search"
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) >= 4:
                    try:
                        r, z, fr, fz = map(to_float, parts[:4])
                        potentials.append({"r": r, "z": z, "fr": fr, "fz": fz})
                    except ValueError:
                        pass
                continue

            # detect fields header (contains ur uz ... srz)
            if " ur" in line and "srz" in line and line.lower().startswith("r"):
                state = "fields"
                continue

            if state == "fields":
                if line.startswith("***"):
                    state = "search"
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) >= 14:
                    try:
                        r, z, ur, uz, tr, tz, err, ett, ezz, erz, srr, stt, szz, srz = map(to_float, parts[:14])
                        fields.append({
                            "r": r, "z": z, "ur": ur, "uz": uz,
                            "tr": tr, "tz": tz, "err": err, "ett": ett,
                            "ezz": ezz, "erz": erz, "srr": srr,
                            "stt": stt, "szz": szz, "srz": srz,
                        })
                    except ValueError:
                        print('ValueError in line:', line)
                        pass
                continue

        print(f"Parsed IBEM output: {len(potentials)} potentials, {len(fields)} field rows")
        return {"potentials": potentials, "fields": fields}

    def _plot_ibem_heatmaps(self, fields, mesh=None):
        if not fields:
            print("No IBEM fields to plot.")
            return

        r = np.array([f["r"] for f in fields])
        z = np.array([f["z"] for f in fields])
        tri = mtri.Triangulation(r, z)
        
        # Mask triangles outside the mesh domain (for L-form and non-convex shapes)
        if mesh is not None and hasattr(mesh, 'elements') and mesh.elements:
            # Build a set of valid (r,z) regions from mesh elements
            valid_regions = []
            for elem in mesh.elements.values():
                node_ids = elem.node_ids[:4] if len(elem.node_ids) >= 4 else elem.node_ids
                coords = np.array([[mesh.nodes[nid].r, mesh.nodes[nid].z] for nid in node_ids])
                # Store bounding box and center for quick filtering
                r_min, r_max = coords[:, 0].min(), coords[:, 0].max()
                z_min, z_max = coords[:, 1].min(), coords[:, 1].max()
                center = coords.mean(axis=0)
                valid_regions.append((r_min, r_max, z_min, z_max, center))
            
            # Mask triangles whose centroids are far from any mesh element
            mask = np.zeros(len(tri.triangles), dtype=bool)
            for i, triangle_indices in enumerate(tri.triangles):
                # Compute triangle centroid
                tri_r = r[triangle_indices].mean()
                tri_z = z[triangle_indices].mean()
                
                # Check if centroid is near any valid mesh element
                is_valid = False
                for r_min, r_max, z_min, z_max, center in valid_regions:
                    # Expand bounding box slightly for tolerance
                    margin = 0.1 * max(r_max - r_min, z_max - z_min, 0.01)
                    if (r_min - margin <= tri_r <= r_max + margin and 
                        z_min - margin <= tri_z <= z_max + margin):
                        is_valid = True
                        break
                
                mask[i] = not is_valid
            
            if mask.any():
                tri.set_mask(mask)

        def plot_scalar(values, title, cmap="viridis"):
            fig, ax = plt.subplots(figsize=(6, 4))
            tcf = ax.tricontourf(tri, values, levels=40, cmap=cmap)
            ax.triplot(tri, color="k", linewidth=0.2, alpha=0.4)
            fig.colorbar(tcf, ax=ax, shrink=0.8, label=title)
            ax.set_xlabel("r")
            ax.set_ylabel("z")
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()

        plot_scalar(np.array([f["ur"] for f in fields]), "IBEM $u_r$")
        plot_scalar(np.array([f["uz"] for f in fields]), "IBEM $u_z$")

        plot_scalar(np.array([f["srr"] for f in fields]), "IBEM $\\sigma_{rr}$", cmap="magma")
        plot_scalar(np.array([f["szz"] for f in fields]), "IBEM $\\sigma_{zz}$", cmap="magma")
        plot_scalar(np.array([f["srz"] for f in fields]), "IBEM $\\sigma_{rz}$", cmap="magma")
        plot_scalar(np.array([f["stt"] for f in fields]), "IBEM $\\sigma_{\\theta\\theta}$", cmap="magma")

    def _plot_top_boundary_comparison(
        self,
        mesh,
        fem_nodal,
        ibem_disp_per_node,
        ibem_stress_per_node,
        analytical_solution,
        pressure_val,
        mu_val,
        nu_val,
        r_min_mesh,
        r_max_mesh,
        target_z=0.5,
        tol=1e-6,
    ):
        if not mesh.nodes:
            return

        top_node_ids = [nid for nid, node in mesh.nodes.items() if abs(node.z - target_z) <= tol]
        if not top_node_ids:
            top_z = 0.5#max(node.z for node in mesh.nodes.values())
            top_node_ids = [nid for nid, node in mesh.nodes.items() if abs(node.z - top_z) <= tol]
        if len(top_node_ids) < 2:
            print("Top boundary plot skipped: not enough nodes on z=1.")
            return

        top_node_ids.sort(key=lambda nid: mesh.nodes[nid].r)
        r_vals = []
        ur_fem = []
        ur_ibem = []
        ur_ana = []
        uz_fem = []
        uz_ibem = []
        uz_ana = []
        srr_fem = []
        srr_ibem = []
        srr_ana = []
        szz_fem = []
        szz_ibem = []
        szz_ana = []
        srz_fem = []
        srz_ibem = []
        srz_ana = []
        stt_fem = []
        stt_ibem = []
        stt_ana = []

        for nid in top_node_ids:
            node = mesh.nodes[nid]
            r_vals.append(node.r)
            ur_fem.append(node.displacements[0])
            uz_fem.append(node.displacements[1])
            ur_ibem.append(ibem_disp_per_node.get(nid, (np.nan, np.nan))[0])
            uz_ibem.append(ibem_disp_per_node.get(nid, (np.nan, np.nan))[1])
            ur_a, uz_a, srr_a, szz_a, sphi_a, srz_a = analytical_solution(
                node.r, r_min_mesh, r_max_mesh, pressure_val, mu_val, nu_val
            )
            ur_ana.append(ur_a)
            uz_ana.append(uz_a)
            fem_s = fem_nodal.get(nid, {})
            srr_fem.append(fem_s.get("sigma_rr", np.nan))
            szz_fem.append(fem_s.get("sigma_zz", np.nan))
            srz_fem.append(fem_s.get("sigma_rz", np.nan))
            stt_fem.append(fem_s.get("sigma_tt", np.nan))
            ib_s = ibem_stress_per_node.get(nid, np.array([np.nan] * 4))
            srr_ibem.append(ib_s[0])
            szz_ibem.append(ib_s[1])
            srz_ibem.append(ib_s[2])
            stt_ibem.append(ib_s[3])
            srr_ana.append(srr_a)
            szz_ana.append(szz_a)
            srz_ana.append(srz_a)
            stt_ana.append(sphi_a)

        r_vals = np.array(r_vals)

        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        axes = axes.ravel()
        series = [
            (axes[0], "u_r", ur_fem, ur_ibem, ur_ana),
            (axes[1], "u_z", uz_fem, uz_ibem, uz_ana),
            (axes[2], "sigma_rr", srr_fem, srr_ibem, srr_ana),
            (axes[3], "sigma_zz", szz_fem, szz_ibem, szz_ana),
            (axes[4], "sigma_rz", srz_fem, srz_ibem, srz_ana),
            (axes[5], "sigma_tt", stt_fem, stt_ibem, stt_ana),
        ]

        for idx, (ax, title, fem_vals, ibem_vals, ana_vals) in enumerate(series):
            fem_arr = np.array(fem_vals, dtype=float)
            ibem_arr = np.array(ibem_vals, dtype=float)
            ana_arr = np.array(ana_vals, dtype=float)
            ax.plot(r_vals, fem_arr, "o-", label="FEM", markersize=4)
            ax.plot(r_vals, ibem_arr, "s-", label="IBEM", markersize=4)
            ax.plot(r_vals, ana_arr, "x-", label="Analytical", markersize=4)
            ax.set_title(f"{title} at z={target_z}")
            ax.grid(True, linestyle="--", alpha=0.4)
            if idx >= 3:
                ax.set_xlabel("r")
        axes[0].set_ylabel("Displacement")
        axes[3].set_ylabel("Stress")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", ncol=3)
        fig.suptitle(f"Comparison at (z={target_z})")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        plt.show(block=False)

    def _print_table_draw_FEM_vs_IBEM_vs_analytical(self, mesh, fem_nodal, ibem_disp_per_node, ibem_stress_per_node):
        from .analysis_utils import analytical_solution  # moved to analysis_utils
        try:
            r_min_mesh = min(n.r for n in mesh.nodes.values())
            r_max_mesh = max(n.r for n in mesh.nodes.values())
            mu_val = getattr(mesh.material, 'mu', mesh.material.E / (2 * (1 + mesh.material.nu)))
            pressure_val = getattr(mesh, 'refinement_pressure', 0)
            nodes_sorted = sorted(mesh.nodes.items(), key=lambda kv: kv[0])

            def print_table(title, rows):
                df = pd.DataFrame(rows)
                print(f"\n{title}")
                print(df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

            disp_tables = {"ur": [], "uz": []}
            for nid, node in nodes_sorted:
                ur_fem, uz_fem = node.displacements[:2]
                ur_ibem, uz_ibem = ibem_disp_per_node.get(nid, (np.nan, np.nan))
                ur_ana, uz_ana, *_ = analytical_solution(node.r, r_min_mesh, r_max_mesh, pressure_val, mu_val, mesh.material.nu)
                disp_tables["ur"].append({
                    'node': nid, 'r': node.r, 'z': node.z,
                    'ur_fem': ur_fem, 'ur_ibem': ur_ibem, 'ur_ana': ur_ana,
                    '|ur_fem - ur_ana|': abs(ur_fem - ur_ana),
                    '|ur_ibem - ur_ana|': abs(ur_ibem - ur_ana),
                })
                disp_tables["uz"].append({
                    'node': nid, 'r': node.r, 'z': node.z,
                    'uz_fem': uz_fem, 'uz_ibem': uz_ibem, 'uz_ana': uz_ana,
                    '|uz_fem - uz_ana|': abs(uz_fem - uz_ana),
                    '|uz_ibem - uz_ana|': abs(uz_ibem - uz_ana),
                })
            print_table("Displacements (u_r) FEM vs IBEM vs analytical", disp_tables["ur"])
            print_table("Displacements (u_z) FEM vs IBEM vs analytical", disp_tables["uz"])

            stress_tables = {"srr": [], "szz": [], "srz": [], "stt": []}
            for nid, node in nodes_sorted:
                fem_s = fem_nodal.get(nid, {})
                ibem_s = ibem_stress_per_node.get(nid, np.array([np.nan] * 4))
                srr_ib, szz_ib, srz_ib, stt_ib = ibem_s.tolist()
                _, _, srr_ana, szz_ana, sphi_ana, srz_ana = analytical_solution(node.r, r_min_mesh, r_max_mesh, pressure_val, mu_val, mesh.material.nu)
                stress_tables["srr"].append({
                    'node': nid, 'r': node.r, 'z': node.z,
                    'srr_fem': fem_s.get('sigma_rr', np.nan), 'srr_ibem': srr_ib, 'srr_ana': srr_ana,
                    '|srr_fem - srr_ana|': abs(fem_s.get('sigma_rr', np.nan) - srr_ana),
                    '|srr_ibem - srr_ana|': abs(srr_ib - srr_ana),
                })
                stress_tables["szz"].append({
                    'node': nid, 'r': node.r, 'z': node.z,
                    'szz_fem': fem_s.get('sigma_zz', np.nan), 'szz_ibem': szz_ib, 'szz_ana': szz_ana,
                    '|szz_fem - szz_ana|': abs(fem_s.get('sigma_zz', np.nan) - szz_ana),
                    '|szz_ibem - szz_ana|': abs(szz_ib - szz_ana),
                })
                stress_tables["srz"].append({
                    'node': nid, 'r': node.r, 'z': node.z,
                    'srz_fem': fem_s.get('sigma_rz', np.nan), 'srz_ibem': srz_ib, 'srz_ana': srz_ana,
                    '|srz_fem - srz_ana|': abs(fem_s.get('sigma_rz', np.nan) - srz_ana),
                    '|srz_ibem - srz_ana|': abs(srz_ib - srz_ana),
                })
                stress_tables["stt"].append({
                    'node': nid, 'r': node.r, 'z': node.z,
                    'stt_fem': fem_s.get('sigma_tt', np.nan), 'stt_ibem': stt_ib, 'stt_ana': sphi_ana,
                    '|stt_fem - stt_ana|': abs(fem_s.get('sigma_tt', np.nan) - sphi_ana),
                    '|stt_ibem - stt_ana|': abs(stt_ib - sphi_ana),
                })

            print_table("Stresses (sigma_rr) FEM vs IBEM vs analytical", stress_tables["srr"])
            print_table("Stresses (sigma_zz) FEM vs IBEM vs analytical", stress_tables["szz"])
            print_table("Stresses (sigma_rz) FEM vs IBEM vs analytical", stress_tables["srz"])
            print_table("Stresses (sigma_tt) FEM vs IBEM vs analytical", stress_tables["stt"])

            self._plot_top_boundary_comparison(
                mesh,
                fem_nodal=fem_nodal,
                ibem_disp_per_node=ibem_disp_per_node,
                ibem_stress_per_node=ibem_stress_per_node,
                analytical_solution=analytical_solution,
                pressure_val=pressure_val,
                mu_val=mu_val,
                nu_val=mesh.material.nu,
                r_min_mesh=r_min_mesh,
                r_max_mesh=r_max_mesh,
                target_z=1.0
            )
            self._plot_top_boundary_comparison(
                mesh,
                fem_nodal=fem_nodal,
                ibem_disp_per_node=ibem_disp_per_node,
                ibem_stress_per_node=ibem_stress_per_node,
                analytical_solution=analytical_solution,
                pressure_val=pressure_val,
                mu_val=mu_val,
                nu_val=mesh.material.nu,
                r_min_mesh=r_min_mesh,
                r_max_mesh=r_max_mesh,
                target_z=0.5
            )
            self._plot_top_boundary_comparison(
                mesh,
                fem_nodal=fem_nodal,
                ibem_disp_per_node=ibem_disp_per_node,
                ibem_stress_per_node=ibem_stress_per_node,
                analytical_solution=analytical_solution,
                pressure_val=pressure_val,
                mu_val=mu_val,
                nu_val=mesh.material.nu,
                r_min_mesh=r_min_mesh,
                r_max_mesh=r_max_mesh,
                target_z=0.25
            )
            self._plot_top_boundary_comparison(
                mesh,
                fem_nodal=fem_nodal,
                ibem_disp_per_node=ibem_disp_per_node,
                ibem_stress_per_node=ibem_stress_per_node,
                analytical_solution=analytical_solution,
                pressure_val=pressure_val,
                mu_val=mu_val,
                nu_val=mesh.material.nu,
                r_min_mesh=r_min_mesh,
                r_max_mesh=r_max_mesh,
                target_z=0.0
            )
        except Exception as exc:
            print(f"FEM/IBEM/analytical table print skipped: {exc}")
    
    def _plot_z_slice_fem_vs_ibem(self, mesh, ibem_slice_fields, z_slices, r_sorted):
        """Plot all FEM stress components (SPR + mortar fix) on one figure per z-slice."""
        from .postprocessors.fem_postprocessor import (
            FEMPostProcessor, RECOVERY_SPR,
        )

        fem_pp = FEMPostProcessor(mesh, recovery_mode=RECOVERY_SPR)
        n_r_pts = len(r_sorted)

        comp_info = [
            (0, r'$\sigma_{rr}$', 'blue', 'o'),
            (1, r'$\sigma_{zz}$', 'green', 's'),
            (2, r'$\sigma_{rz}$', 'orange', '^'),
            (3, r'$\sigma_{\varphi\varphi}$', 'purple', 'D'),
        ]

        # Collect stresses for all z-slices, then plot side by side
        slice_data = []
        for z_target in z_slices:
            points = np.column_stack([r_sorted, np.full(n_r_pts, z_target)])
            try:
                fem_stresses = fem_pp.stresses_at(points)
                slice_data.append((z_target, fem_stresses))
            except Exception as exc:
                print(f"SPR stresses_at failed at z={z_target}: {exc}")

        if slice_data:
            n_slices = len(slice_data)
            fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 5), squeeze=False)
            for idx, (z_target, fem_stresses) in enumerate(slice_data):
                ax = axes[0, idx]
                for col_idx, label, color, marker in comp_info:
                    ax.plot(
                        r_sorted, fem_stresses[:, col_idx],
                        marker=marker, color=color, label=label,
                        markersize=3, linewidth=1.2,
                    )
                ax.set_xlabel('r')
                ax.set_ylabel('Stress')
                ax.set_title(f'FEM stresses (SPR + mortar fix) at z = {z_target:.4f}')
                ax.legend()
                ax.grid(True)
            plt.tight_layout()
            plt.show()

    def find(self, mesh, threshold, bcs=None, mode="eta"):
        try:
            print("IBEM boundary input saved to ibem_input.txt")
            # Static boundary mode: build segments once, reuse across cycles
            static_segs = None
            if self.static_boundary:
                if self._cached_static_segments is None:
                    print(f"Building static IBEM boundary: {self.static_segments_per_side}")
                    self._cached_static_segments = self._build_static_segments(mesh, bcs)
                    print(f"Static IBEM boundary: {len(self._cached_static_segments)} segments")
                else:
                    print(f"Reusing cached static IBEM boundary: {len(self._cached_static_segments)} segments")
                static_segs = self._cached_static_segments
            # Build z-slice query points to pass into the IBEM input
            r_min_m = min(n.r for n in mesh.nodes.values())
            r_max_m = max(n.r for n in mesh.nodes.values())
            z_min_m = min(n.z for n in mesh.nodes.values())
            z_max_m = max(n.z for n in mesh.nodes.values())
            z_mid_slice = (z_min_m + z_max_m) / 2
            z_slices = [z_mid_slice, z_mid_slice + 4.05]
            n_r_pts = 50
            r_slice_sorted = np.linspace(r_min_m, r_max_m, n_r_pts)
            extra_pts = []
            for z_t in z_slices:
                for r_v in r_slice_sorted:
                    extra_pts.append((r_v, z_t))

            n_gauss = self._write_ibem_input(mesh, bcs, file_path="./ibem_input.txt", static_segments=static_segs, extra_query_points=extra_pts)
            self._run_ibem_console("ibem_input.txt")
            ibem_data = self._parse_ibem_output("ibem_output.txt")
            all_fields = ibem_data.get('fields', []) if ibem_data else []
            print(f"IBEM parsed: {len(ibem_data['potentials'])} potentials, {len(all_fields)} field rows (gauss={n_gauss}, extra={len(extra_pts)})")

            # Split: first n_gauss rows are Gauss-point fields, the rest are z-slice query points
            ibem_fields = all_fields[:n_gauss]
            ibem_extra = all_fields[n_gauss:]

            # Build dict z_target -> list of IBEM field rows (n_r_pts per slice)
            ibem_slice_fields = {}
            offset = 0
            for z_t in z_slices:
                ibem_slice_fields[z_t] = ibem_extra[offset:offset + n_r_pts]
                offset += n_r_pts

            if self._io is not None:
                self._io.plot_ibem_heatmaps(ibem_fields, mesh=mesh)
                try:
                    self._io.plot_z_slice_fem_vs_ibem(mesh, ibem_slice_fields, z_slices, r_slice_sorted)
                except Exception as exc:
                    print(f"FEM vs IBEM z-slice plot skipped: {exc}")
            else:
                self._plot_ibem_heatmaps(ibem_fields, mesh=mesh)
                try:
                    self._plot_z_slice_fem_vs_ibem(mesh, ibem_slice_fields, z_slices, r_slice_sorted)
                except Exception as exc:
                    print(f"FEM vs IBEM SPR z-slice plot skipped: {exc}")
        except Exception as exc:
            print(f"IBEM parse failed: {exc}")
            return []
        if not ibem_fields:
            print("IBEM refinement skipped: no IBEM field data available.")
            return []

        # NOTE: For the IBEM estimator we compare stresses at Gauss points.
        # FEM stresses are therefore evaluated directly at Gauss points via B @ u_e,
        # instead of recovering nodal stresses and interpolating back.
        try:
            D = mesh.material.get_elastic_matrix()
        except Exception as exc:
            print(f"FEM elastic matrix unavailable: {exc}")
            return []

        # IBEM fields are sampled at Gauss points (written at end of ibem_input.txt).
        # Build a lookup so we can fetch IBEM stresses directly at integration points.
        def _coord_key(r, z, ndigits=10):
            return (round(float(r), ndigits), round(float(z), ndigits))

        ibem_by_key = {}
        for row in ibem_fields:
            ibem_by_key[_coord_key(row['r'], row['z'])] = row

        field_coords = np.array([[f['r'], f['z']] for f in ibem_fields], dtype=float)
        match_eps = 1e-8

        def _get_ibem_row(r, z):
            key = _coord_key(r, z)
            row = ibem_by_key.get(key)
            if row is not None:
                return row
            if field_coords.size == 0:
                return None
            mask = (np.abs(field_coords[:, 0] - r) <= match_eps) & (np.abs(field_coords[:, 1] - z) <= match_eps)
            if not np.any(mask):
                return None
            idx = int(np.argmax(mask))
            row = ibem_fields[idx]
            ibem_by_key[key] = row
            return row

        field_stress = np.array([[f['srr'], f['szz'], f['srz'], f['stt']] for f in ibem_fields], dtype=float)
        ibem_norm = np.linalg.norm(field_stress) / max(len(field_stress), 1)
        if ibem_norm == 0:
            return []

        print("FEM/IBEM comparison uses Gauss-point stresses (direct FEM evaluation).")

        # Try to compute analytical stresses for θ (effectivity index) calculation
        has_analytical = True
        try:
            from .analysis_utils import analytical_solution
            r_min_mesh = min(n.r for n in mesh.nodes.values())
            r_max_mesh = max(n.r for n in mesh.nodes.values())
            mu_val = getattr(mesh.material, 'mu', mesh.material.E / (2 * (1 + mesh.material.nu)))
            pressure_val = getattr(mesh, 'refinement_pressure', 0)
            print('pressure_val', pressure_val)
            nu_val = mesh.material.nu
            if has_analytical:
                print("Analytical stresses available for theta calculation (evaluated at Gauss points)")
        except Exception as exc:
            print(f"Analytical solution unavailable for theta: {exc}")
            has_analytical = False

        def sigma_eff_sq(b11, b22, b12, b33):
            """
            Same Frobenius norm-squared under σ13=σ23=0:
            ||σ||_F^2 = b11^2 + b22^2 + b33^2 + 2 b12^2
            """
            return (b11 * b11) + (b22 * b22) + (b33 * b33) + 2.0 * (b12 * b12)


        # Normalize/alias mode keys
        mode_in = (mode or "eta").strip()
        mode_key = {
            "eta": "eta",
            "eta_e": "eta",
            "θ~b": "theta_tilde_B",
            "tthetab": "theta_tilde_B",
            "t_theta_b": "theta_tilde_B",
            "theta_tilde_b": "theta_tilde_B",
            "thetab_tilde": "theta_tilde_B",
            "θ~t": "theta_tilde_T",
            "tthetat": "theta_tilde_T",
            "t_theta_t": "theta_tilde_T",
            "theta_tilde_t": "theta_tilde_T",
            "thetat_tilde": "theta_tilde_T",
            "eta_t": "eta_T",
            "eta_e_t": "eta_T",
            "eta_exact": "eta_T",
        }.get(mode_in.lower(), mode_in)


        # Keep both filtered (above threshold) and full metric dictionaries for plotting
        # eta: current implementation (area-normalized per element and globally)
        eta_values = {}
        eta_all = {}
        delta_FB_all = {}  # Δσ_FB,Ωe := sqrt(∫||σ_F-σ_B||^2 dΩ) / sqrt(∫1 dΩ)

        # tilde-theta^B: same structure as eta, but WITHOUT dividing by areas
        theta_tilde_B_values = {}
        theta_tilde_B_all = {}
        delta_FB_tilde_all = {}  # sqrt(∫||σ_F-σ_B||^2 dΩ)

        # tilde-theta^T: like tilde-theta^B but using analytical reference (σ_T = σ_analytical)
        theta_tilde_T_values = {}
        theta_tilde_T_all = {}
        delta_FA_tilde_all = {}  # sqrt(∫||σ_F-σ_A||^2 dΩ)

        # eta^T: area-normalized analytical eta (η_e^T = Δσ_FA,Ωe / σ̄_A,Ω)
        eta_T_values = {}
        eta_T_all = {}

        # Global integrals for σ̄_B,Ω (over the full domain Ω)
        int_sigmaB_sq_global = 0.0
        int_one_global = 0.0

        # Global integrals for σ̄_A,Ω (analytical, for θ calculation)
        int_sigmaA_sq_global = 0.0
        delta_FA_all = {}  # element-level Δσ_FA,Ωe (area-normalized)

        for eid, elem in mesh.elements.items():
            node_ids = elem.node_ids
            coords = np.array([[mesh.nodes[nid].r, mesh.nodes[nid].z] for nid in node_ids], dtype=float)

            # ---- FEM element displacement vector u_e (u_r first, then u_z) ----
            n_el_nodes = len(node_ids)
            u_e = np.zeros(2 * n_el_nodes, dtype=float)
            for a, nid in enumerate(node_ids):
                u_e[a] = mesh.nodes[nid].displacements[0]
                u_e[n_el_nodes + a] = mesh.nodes[nid].displacements[1]

            dof = mesh.node_dof * elem.shape_func.nodes_count
            ok = True
            ok_analytical = has_analytical

            int_delta_sq = 0.0
            int_sigmaB_sq = 0.0 # local, used for global accumulation
            int_one = 0.0       # local, used for global accumulation
            int_delta_sq_ana = 0.0  # analytical-based error integral
            int_sigmaA_sq = 0.0     # analytical stress norm integral

            for gp in elem.quadrature.gauss_points_2D():
                xi = gp["xi"]
                eta = gp["eta"]
                w = gp["weight"]

                N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)

                N = np.asarray(N, dtype=float).reshape(-1)
                dN_dxi = np.asarray(dN_dxi, dtype=float).reshape(-1)
                dN_deta = np.asarray(dN_deta, dtype=float).reshape(-1)

                # ---- Geometry Jacobian: dΩ = |detJ| dξ dη ----
                # Use the same Jacobian computation as element.getB
                try:
                    B, _, detJ = elem.getB(N, dN_dxi, dN_deta, n_el_nodes, coords, dof)
                except Exception:
                    ok = False
                    break
                detJ_abs = abs(float(detJ))
                if detJ_abs == 0.0:
                    continue

                dOmega = float(w) * detJ_abs

                # Physical coordinates of this Gauss point (must match points written to IBEM input)
                r_gp = float(np.dot(N, coords[:, 0]))
                z_gp = float(np.dot(N, coords[:, 1]))

                # IBEM stresses at this Gauss point (no interpolation)
                ibem_row = _get_ibem_row(r_gp, z_gp)
                if ibem_row is None:
                    ok = False
                    break

                # ---- FEM stresses evaluated at this Gauss point ----
                epsilon_gp = B @ u_e
                fem_gp = D @ epsilon_gp
                ibem_gp = np.array([ibem_row['srr'], ibem_row['szz'], ibem_row['srz'], ibem_row['stt']], dtype=float)

                ds11, ds22, ds12, ds33 = (fem_gp - ibem_gp)
                b11, b22, b12, b33 = ibem_gp

                # ============================================================
                # Δσ_FB(x)^2 at the Gauss point
                # ============================================================
                delta_sq_gp = sigma_eff_sq(ds11, ds22, ds12, ds33)

                # ============================================================
                # ||σ_B(x)||_F^2 at the Gauss point (for σ̄_B,Ωe)
                # ============================================================
                sigmaB_sq_gp = sigma_eff_sq(b11, b22, b12, b33)

                # Accumulate integrals
                int_delta_sq += delta_sq_gp * dOmega
                int_sigmaB_sq += sigmaB_sq_gp * dOmega
                int_one += dOmega

                # Analytical-based error (for θ)
                if ok_analytical:
                    try:
                        _, _, srr_a, szz_a, sphi_a, srz_a = analytical_solution(
                            r_gp, r_min_mesh, r_max_mesh, pressure_val, mu_val, nu_val
                        )
                        ana_gp = np.array([srr_a, szz_a, srz_a, sphi_a], dtype=float)
                        ds_ana = fem_gp - ana_gp
                        delta_sq_ana_gp = sigma_eff_sq(ds_ana[0], ds_ana[1], ds_ana[2], ds_ana[3])
                        sigmaA_sq_gp = sigma_eff_sq(ana_gp[0], ana_gp[1], ana_gp[2], ana_gp[3])
                        int_delta_sq_ana += delta_sq_ana_gp * dOmega
                        int_sigmaA_sq += sigmaA_sq_gp * dOmega
                    except Exception as exc:
                        ok_analytical = False


            if int_one <= 0.0:
                continue

            if not ok:
                continue

            # Accumulate global integrals over Ω
            int_sigmaB_sq_global += int_sigmaB_sq
            int_one_global += int_one

            delta_FB_all[eid] = np.sqrt(int_delta_sq) / np.sqrt(int_one)  # Δσ_FB,Ωe
            delta_FB_tilde_all[eid] = np.sqrt(int_delta_sq)

            # Analytical-based per-element error (for θ)
            if ok_analytical and int_one > 0.0:
                delta_FA_all[eid] = np.sqrt(int_delta_sq_ana) / np.sqrt(int_one)
                delta_FA_tilde_all[eid] = np.sqrt(int_delta_sq_ana)
                int_sigmaA_sq_global += int_sigmaA_sq


        # Compute global σ̄_B,Ω and finalize η_e per element
        if int_one_global > 0.0:
            sigmaB_global = np.sqrt(int_sigmaB_sq_global) / np.sqrt(int_one_global)
        else:
            sigmaB_global = 0.0

        # Global (raw integral) reference for tilde-theta^B
        sigmaB_global_tilde = np.sqrt(int_sigmaB_sq_global) if int_sigmaB_sq_global > 0.0 else 0.0

        # Global (raw integral) reference for tilde-theta^T
        sigmaA_global_tilde = np.sqrt(int_sigmaA_sq_global) if int_sigmaA_sq_global > 0.0 else 0.0

        if sigmaB_global > 0.0:
            for eid, delta_FB in delta_FB_all.items():
                eta = delta_FB / sigmaB_global
                eta_all[eid] = eta
                if eta > threshold:
                    eta_values[eid] = eta

        if sigmaB_global_tilde > 0.0:
            for eid, delta_FB_tilde in delta_FB_tilde_all.items():
                val = delta_FB_tilde / sigmaB_global_tilde
                theta_tilde_B_all[eid] = val
                if val > threshold:
                    theta_tilde_B_values[eid] = val

        if sigmaA_global_tilde > 0.0:
            for eid, delta_FA_tilde in delta_FA_tilde_all.items():
                val = delta_FA_tilde / sigmaA_global_tilde
                theta_tilde_T_all[eid] = val
                if val > threshold:
                    theta_tilde_T_values[eid] = val

        # Compute η_e^T (area-normalized analytical eta) and θ (effectivity index)
        theta_all = {}
        theta_computed = False
        if has_analytical and int_one_global > 0.0:
            sigmaA_global = np.sqrt(int_sigmaA_sq_global) / np.sqrt(int_one_global)
            if sigmaA_global > 0.0:
                # η_e^T = Δσ_FA,Ωe / σ̄_A,Ω
                for eid, delta_FA in delta_FA_all.items():
                    val = delta_FA / sigmaA_global
                    eta_T_all[eid] = val
                    if val > threshold:
                        eta_T_values[eid] = val

                # θ_e = √(∫_Ωe Δσ_FB²(x)dΩe) / √(∫_Ωe Δσ_FT²(x)dΩe)
                for eid in delta_FB_tilde_all:
                    if eid in delta_FA_tilde_all:
                        theta_all[eid] = delta_FB_tilde_all[eid] / delta_FA_tilde_all[eid]
                    else:
                        theta_all[eid] = delta_FB_tilde_all[eid]
                theta_computed = True
                print(f"Theta effectivity index computed for {len(theta_all)} elements "
                      f"(mean={np.mean(list(theta_all.values())):.4f}, "
                      f"std={np.std(list(theta_all.values())):.4f})")
        if not theta_computed:
            theta_all = dict(eta_all)

        # Collect candidates after processing all elements
        metric_map_filtered = {
            "eta": eta_values,
            "theta_tilde_B": theta_tilde_B_values,
            "theta_tilde_T": theta_tilde_T_values,
            "eta_T": eta_T_values,
        }
        metric_map_all = {
            "eta": eta_all,
            "theta_tilde_B": theta_tilde_B_all,
            "theta_tilde_T": theta_tilde_T_all,
            "eta_T": eta_T_all,
        }

        chosen_all = metric_map_all.get(mode_key, eta_all)
        if not chosen_all:
            return []

        chosen_filtered = metric_map_filtered.get(mode_key, eta_values)
        candidates = sorted(chosen_filtered.keys(), key=lambda eid: chosen_filtered[eid], reverse=True) if chosen_filtered else []

        # Plot of element outlines: selected metric + (optional) θ effectivity (only for eta)
        show_theta = False # bool(theta_computed and mode_key == "eta")
        n_plots = 2 if show_theta else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        title_map = {
            "eta": r'$\eta_e$ per element',
            "theta_tilde_B": r'$\tilde{\eta}_e$ per element',
            "theta_tilde_T": r'$\tilde{\theta}^T_e$ per element',
            "eta_T": r'$\eta_e^T$ per element',
        }

        # Selected metric plot
        ax = axes[0]
        for eid, elem in mesh.elements.items():
            node_ids = elem.node_ids
            corner_ids = node_ids[:4] if len(node_ids) >= 4 else node_ids
            coords = np.array([[mesh.nodes[nid].r, mesh.nodes[nid].z] for nid in corner_ids])
            loop = np.vstack([coords, coords[0]])
            ax.plot(loop[:, 0], loop[:, 1], 'k-', linewidth=0.8)
            if eid in chosen_all:
                ctr = coords.mean(axis=0)
                val = chosen_all[eid]
                color = 'red' if val > threshold else 'green'
                ax.text(ctr[0], ctr[1], f"{val:.2g}", color=color, fontsize=7, ha='center', va='center')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title_map.get(mode_key, title_map["eta"]))
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.grid(True, linestyle='--', alpha=0.3)

        # θ plot (effectivity index)
        if show_theta and n_plots > 1:
            ax2 = axes[1]
            for eid, elem in mesh.elements.items():
                node_ids = elem.node_ids
                corner_ids = node_ids[:4] if len(node_ids) >= 4 else node_ids
                coords = np.array([[mesh.nodes[nid].r, mesh.nodes[nid].z] for nid in corner_ids])
                loop = np.vstack([coords, coords[0]])
                ax2.plot(loop[:, 0], loop[:, 1], 'k-', linewidth=0.8)
                if eid in theta_all:
                    ctr = coords.mean(axis=0)
                    val = theta_all[eid]
                    # Blue if θ ≈ 1 (good), orange otherwise
                    color = 'blue'# if 0.8 <= val <= 1.2 else 'orange'
                    ax2.text(ctr[0], ctr[1], f"{val:.2g}", color=color, fontsize=7, ha='center', va='center')
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_title(r'$\theta_e$ per element')
            ax2.set_xlabel('r')
            ax2.set_ylabel('z')
            ax2.grid(True, linestyle='--', alpha=0.3)

        # fig.savefig(f"rect_1_0.3_eta_{mode_key}.png", dpi=600)

        import matplotlib
        matplotlib.rcParams['savefig.dpi'] = 600

        fig.tight_layout()
        plt.show(block=True)

        return candidates
