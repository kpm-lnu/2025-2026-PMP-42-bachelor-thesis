from __future__ import annotations
import numpy as np


def _ref_coords_for_shape(shape_func):
    """
    Return a list of reference (xi, eta) for each local node of the element's
    shape function, if known. Falls back to common Q4/Q8 serendipity ordering.
    """
    # If the shape function exposes reference coords, prefer those
    for attr in ("reference_coords", "node_reference_coords", "local_coords", "node_coords"):
        coords = getattr(shape_func, attr, None)
        if coords is not None:
            try:
                # ensure list[tuple]
                return [(float(c[0]), float(c[1])) for c in coords]
            except Exception:
                pass
    n = getattr(shape_func, "nodes_count", None)
    if n == 4:
        # Q4
        return [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    if n == 8:
        # Q8 serendipity: corners then midsides
        return [
            (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
            (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
        ]
    return []


def recover_nodal_stresses(mesh, material, shape_func, fixed_z, fixed_r, tol=1e-6, eval_mode: str = "node"):
    elem_gp_stresses = {}
    for elem in mesh.elements.values():
        elem_gp_stresses[elem.elem_id] = []

        node_ids = elem.node_ids
        n_el_nodes = len(node_ids)

        nodes_coords = np.zeros((n_el_nodes, 2))
        dof = mesh.node_dof * elem.shape_func.nodes_count
        
        u_e = np.zeros(2 * n_el_nodes)
        for a, nid in enumerate(node_ids):
            nodes_coords[a, 0] = mesh.nodes[nid].r
            nodes_coords[a, 1] = mesh.nodes[nid].z
            u_e[a] = mesh.nodes[nid].displacements[0]
            u_e[n_el_nodes + a] = mesh.nodes[nid].displacements[1]
        for gp in elem.quadrature.gauss_points_2D():
            xi = gp["xi"]
            eta = gp["eta"]
            N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
            B, _, _ = elem.getB(N, dN_dxi, dN_deta, len(nodes_coords), nodes_coords, dof)
            epsilon_gp = B @ u_e
            sigma_gp = material.get_elastic_matrix() @ epsilon_gp

            elem_gp_stresses[elem.elem_id].append({
                'sigma_rr': sigma_gp[0],
                'sigma_zz': sigma_gp[1],
                'sigma_rz': sigma_gp[2],
                'sigma_tt': sigma_gp[3],
                'r_gp': np.dot(N, nodes_coords[:, 0]),
                'z_gp': np.dot(N, nodes_coords[:, 1])
            })
    
    nodal_stresses_fixed_z = {}
    for node_id, node in mesh.nodes.items():
        if abs(node.z - fixed_z) < tol:
            collected = {'sigma_rr': [], 'sigma_zz': [], 'sigma_rz': [], 'sigma_tt': []}
            if eval_mode == "gauss":
                # Average all adjacent elements' Gauss-point stresses
                for elem in mesh.elements.values():
                    if node_id in elem.node_ids:
                        for gp_stress in elem_gp_stresses[elem.elem_id]:
                            collected['sigma_rr'].append(gp_stress['sigma_rr'])
                            collected['sigma_zz'].append(gp_stress['sigma_zz'])
                            collected['sigma_rz'].append(gp_stress['sigma_rz'])
                            collected['sigma_tt'].append(gp_stress['sigma_tt'])
            else:
                # Evaluate stress at the node param coords in each adjacent element, then average
                for elem in mesh.elements.values():
                    if node_id not in elem.node_ids:
                        continue
                    node_ids = elem.node_ids
                    n_el_nodes = len(node_ids)
                    nodes_coords = np.zeros((n_el_nodes, 2))
                    dof = mesh.node_dof * elem.shape_func.nodes_count
                    u_e = np.zeros(2 * n_el_nodes)
                    for a, nid in enumerate(node_ids):
                        nodes_coords[a, 0] = mesh.nodes[nid].r
                        nodes_coords[a, 1] = mesh.nodes[nid].z
                        u_e[a] = mesh.nodes[nid].displacements[0]
                        u_e[n_el_nodes + a] = mesh.nodes[nid].displacements[1]
                    # local index of this node in the element
                    a_local = node_ids.index(node_id)
                    ref_coords = _ref_coords_for_shape(elem.shape_func)
                    if ref_coords and a_local < len(ref_coords):
                        xi, eta = ref_coords[a_local]
                        N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
                        B, _, _ = elem.getB(N, dN_dxi, dN_deta, len(nodes_coords), nodes_coords, dof)
                        epsilon_node = B @ u_e
                        sigma_node = material.get_elastic_matrix() @ epsilon_node
                        collected['sigma_rr'].append(sigma_node[0])
                        collected['sigma_zz'].append(sigma_node[1])
                        collected['sigma_rz'].append(sigma_node[2])
                        collected['sigma_tt'].append(sigma_node[3])
                    else:
                        # Fallback: use element mean of Gauss-point stresses
                        for gp_stress in elem_gp_stresses[elem.elem_id]:
                            collected['sigma_rr'].append(gp_stress['sigma_rr'])
                            collected['sigma_zz'].append(gp_stress['sigma_zz'])
                            collected['sigma_rz'].append(gp_stress['sigma_rz'])
                            collected['sigma_tt'].append(gp_stress['sigma_tt'])

            if collected['sigma_rr']:
                nodal_stresses_fixed_z[node_id] = {
                    'sigma_rr': np.mean(collected['sigma_rr']),
                    'sigma_zz': np.mean(collected['sigma_zz']),
                    'sigma_rz': np.mean(collected['sigma_rz']),
                    'sigma_tt': np.mean(collected['sigma_tt'])
                }

    nodal_stresses_fixed_r = {}
    for node_id, node in mesh.nodes.items():
        if abs(node.r - fixed_r) < tol:
            collected = {'sigma_rr': [], 'sigma_zz': [], 'sigma_rz': [], 'sigma_tt': []}
            if eval_mode == "gauss":
                for elem in mesh.elements.values():
                    if node_id in elem.node_ids:
                        for gp_stress in elem_gp_stresses[elem.elem_id]:
                            collected['sigma_rr'].append(gp_stress['sigma_rr'])
                            collected['sigma_zz'].append(gp_stress['sigma_zz'])
                            collected['sigma_rz'].append(gp_stress['sigma_rz'])
                            collected['sigma_tt'].append(gp_stress['sigma_tt'])
            else:
                for elem in mesh.elements.values():
                    if node_id not in elem.node_ids:
                        continue
                    node_ids = elem.node_ids
                    n_el_nodes = len(node_ids)
                    nodes_coords = np.zeros((n_el_nodes, 2))
                    dof = mesh.node_dof * elem.shape_func.nodes_count
                    u_e = np.zeros(2 * n_el_nodes)
                    for a, nid in enumerate(node_ids):
                        nodes_coords[a, 0] = mesh.nodes[nid].r
                        nodes_coords[a, 1] = mesh.nodes[nid].z
                        u_e[a] = mesh.nodes[nid].displacements[0]
                        u_e[n_el_nodes + a] = mesh.nodes[nid].displacements[1]
                    a_local = node_ids.index(node_id)
                    ref_coords = _ref_coords_for_shape(elem.shape_func)
                    if ref_coords and a_local < len(ref_coords):
                        xi, eta = ref_coords[a_local]
                        N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
                        B, _, _ = elem.getB(N, dN_dxi, dN_deta, len(nodes_coords), nodes_coords, dof)
                        epsilon_node = B @ u_e
                        sigma_node = material.get_elastic_matrix() @ epsilon_node
                        collected['sigma_rr'].append(sigma_node[0])
                        collected['sigma_zz'].append(sigma_node[1])
                        collected['sigma_rz'].append(sigma_node[2])
                        collected['sigma_tt'].append(sigma_node[3])
                    else:
                        for gp_stress in elem_gp_stresses[elem.elem_id]:
                            collected['sigma_rr'].append(gp_stress['sigma_rr'])
                            collected['sigma_zz'].append(gp_stress['sigma_zz'])
                            collected['sigma_rz'].append(gp_stress['sigma_rz'])
                            collected['sigma_tt'].append(gp_stress['sigma_tt'])

            if collected['sigma_rr']:
                nodal_stresses_fixed_r[node_id] = {
                    'sigma_rr': np.mean(collected['sigma_rr']),
                    'sigma_zz': np.mean(collected['sigma_zz']),
                    'sigma_rz': np.mean(collected['sigma_rz']),
                    'sigma_tt': np.mean(collected['sigma_tt'])
                }
    return nodal_stresses_fixed_z, nodal_stresses_fixed_r


def recover_all_nodal_stresses(mesh, material, shape_func, tol=1e-6, eval_mode: str = "node"):
    """
    Recover stresses at all nodes.
    eval_mode:
      - "gauss": average all Gauss-point stresses from adjacent elements (current behavior)
      - "node": evaluate stress at the node param location in each adjacent element, then average
    """
    nodal_stresses = {}
    ref_cache = {}
    for node_id, node in mesh.nodes.items():
        collected = {'sigma_rr': [], 'sigma_zz': [], 'sigma_rz': [], 'sigma_tt': []}
        for elem in mesh.elements.values():
            if node_id not in elem.node_ids:
                continue
            n_el_nodes = len(elem.node_ids)
            nodes_coords = np.zeros((n_el_nodes, 2))
            dof = mesh.node_dof * elem.shape_func.nodes_count
            u_e = np.zeros(2 * n_el_nodes)
            for a, nid in enumerate(elem.node_ids):
                nodes_coords[a, 0] = mesh.nodes[nid].r
                nodes_coords[a, 1] = mesh.nodes[nid].z
                u_e[a] = mesh.nodes[nid].displacements[0]
                u_e[n_el_nodes + a] = mesh.nodes[nid].displacements[1]

            if eval_mode == "gauss":
                for gp in elem.quadrature.gauss_points_2D():
                    xi = gp["xi"]
                    eta = gp["eta"]
                    N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
                    B, _, _ = elem.getB(N, dN_dxi, dN_deta, len(nodes_coords), nodes_coords, dof)
                    epsilon_gp = B @ u_e
                    sigma_gp = material.get_elastic_matrix() @ epsilon_gp
                    collected['sigma_rr'].append(sigma_gp[0])
                    collected['sigma_zz'].append(sigma_gp[1])
                    collected['sigma_rz'].append(sigma_gp[2])
                    collected['sigma_tt'].append(sigma_gp[3])
            else:
                # nodal param evaluation
                key = id(elem.shape_func)
                if key not in ref_cache:
                    ref_cache[key] = _ref_coords_for_shape(elem.shape_func)
                ref_coords = ref_cache[key]
                a_local = elem.node_ids.index(node_id)
                if ref_coords and a_local < len(ref_coords):
                    xi, eta = ref_coords[a_local]
                    N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
                    B, _, _ = elem.getB(N, dN_dxi, dN_deta, len(nodes_coords), nodes_coords, dof)
                    epsilon_node = B @ u_e
                    sigma_node = material.get_elastic_matrix() @ epsilon_node
                    collected['sigma_rr'].append(sigma_node[0])
                    collected['sigma_zz'].append(sigma_node[1])
                    collected['sigma_rz'].append(sigma_node[2])
                    collected['sigma_tt'].append(sigma_node[3])
                else:
                    # fallback to gauss averaging for this element
                    for gp in elem.quadrature.gauss_points_2D():
                        xi = gp["xi"]
                        eta = gp["eta"]
                        N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
                        B, _, _ = elem.getB(N, dN_dxi, dN_deta, len(nodes_coords), nodes_coords, dof)
                        epsilon_gp = B @ u_e
                        sigma_gp = material.get_elastic_matrix() @ epsilon_gp
                        collected['sigma_rr'].append(sigma_gp[0])
                        collected['sigma_zz'].append(sigma_gp[1])
                        collected['sigma_rz'].append(sigma_gp[2])
                        collected['sigma_tt'].append(sigma_gp[3])

        if collected['sigma_rr']:
            nodal_stresses[node_id] = {
                'sigma_rr': np.mean(collected['sigma_rr']),
                'sigma_zz': np.mean(collected['sigma_zz']),
                'sigma_rz': np.mean(collected['sigma_rz']),
                'sigma_tt': np.mean(collected['sigma_tt'])
            }
    return nodal_stresses
