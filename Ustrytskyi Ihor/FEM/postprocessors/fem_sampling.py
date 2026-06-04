from __future__ import annotations
import numpy as np


def find_isoparametric_coords(elem, mesh, r_target: float, z_target: float, tol: float = 1e-8, max_iter: int = 10):
    """
    Solve for (ξ, η) in the parent element such that 
       (r_target, z_target) = Σ_i N_i(ξ,η) · (r_i, z_i)
    using Newton–Raphson.
    Returns (ξ, η) if it converges within the master element (|ξ|,|η| ≤ 1).
    Raises ValueError otherwise.
    """
    xi = 0.0
    eta = 0.0
    node_ids = elem.node_ids
    element_nodes = [mesh.nodes[nid] for nid in node_ids]
    coords = np.array([[node.r, node.z] for node in element_nodes])

    for _ in range(max_iter):
        N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
        mapped = N @ coords
        resid = np.array([r_target - mapped[0], z_target - mapped[1]])
        if np.linalg.norm(resid) < tol:
            break
        dr_dxi  = dN_dxi  @ coords[:, 0]
        dr_deta = dN_deta @ coords[:, 0]
        dz_dxi  = dN_dxi  @ coords[:, 1]
        dz_deta = dN_deta @ coords[:, 1]
        J = np.array([[dr_dxi, dr_deta],
                      [dz_dxi, dz_deta]])
        try:
            delta = np.linalg.solve(J, resid)
        except np.linalg.LinAlgError:
            raise ValueError("Singular Jacobian in isoparametric mapping")
        xi  += delta[0]
        eta += delta[1]
    else:
        raise ValueError("Newton did not converge for isoparametric coordinates")

    if abs(xi) > 1+1e-6 or abs(eta) > 1+1e-6:
        raise ValueError(f"Point ({r_target},{z_target}) lies outside this element")

    return xi, eta


def u_r_z_sigma_rr_zz_rz_tt(mesh, r_query: float, z_query: float):
    """
    Sample the FE solution (displacements and stresses) at global (r_query, z_query),
    using nodal-averaged values and caching for efficiency.

    Returns: (u_r, u_z, sigma_rr, sigma_zz, sigma_rz, sigma_tt)
    """
    nodal_disp = {}
    nodal_stress = {}
    disp_sum = {nid: np.zeros(2) for nid in mesh.nodes}
    disp_count = {nid: 0 for nid in mesh.nodes}
    stress_sum = {nid: np.zeros(4) for nid in mesh.nodes}
    stress_count = {nid: 0 for nid in mesh.nodes}

    D = mesh.material.get_elastic_matrix()
    for elem in mesh.elements.values():
        node_ids = elem.node_ids
        n = len(node_ids)
        u_e = np.zeros(2*n)
        coords = np.zeros((n,2))
        for a, nid in enumerate(node_ids):
            coords[a] = [mesh.nodes[nid].r, mesh.nodes[nid].z]
            u_e[a]     = mesh.nodes[nid].displacements[0]
            u_e[n + a] = mesh.nodes[nid].displacements[1]
        for gp in elem.quadrature.gauss_points_2D():
            xi = float(gp['xi'])
            eta = float(gp['eta'])
            N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
            B, _, _ = elem.getB(
                N, dN_dxi, dN_deta,
                elem.shape_func.nodes_count, 
                coords,
                elem.shape_func.nodes_count * mesh.node_dof
            )
            eps = B @ u_e
            sig = D @ eps
            for a, nid in enumerate(node_ids):
                disp_sum[nid] += N[a] * np.array([u_e[a], u_e[n + a]])
                stress_sum[nid] += N[a] * sig
                
        for a, nid in enumerate(node_ids):
            disp_count[nid] += 1
            stress_count[nid] += 1
    for nid in mesh.nodes:
        if disp_count[nid] > 0:
            nodal_disp[nid] = tuple(disp_sum[nid] / disp_count[nid])
        else:
            nodal_disp[nid] = (0.0, 0.0)
        if stress_count[nid] > 0:
            nodal_stress[nid] = tuple(stress_sum[nid] / stress_count[nid])
        else:
            nodal_stress[nid] = (0.0, 0.0, 0.0, 0.0)

    for elem in mesh.elements.values():
        try:
            xi, eta = find_isoparametric_coords(elem, mesh, r_query, z_query)
        except ValueError:
            continue
        N, _, _ = elem.shape_func.evaluate(xi, eta)
        node_ids = elem.node_ids
        u_r = u_z = sigma_rr = sigma_zz = sigma_rz = sigma_tt = 0.0
        for a, nid in enumerate(node_ids):
            w = N[a]
            dr, dz = nodal_disp[nid]
            u_r += w * dr
            u_z += w * dz
            srr, szz, srz, stt = nodal_stress[nid]
            sigma_rr += w * srr
            sigma_zz += w * szz
            sigma_rz += w * srz
            sigma_tt += w * stt
        return u_r, u_z, sigma_rr, sigma_zz, sigma_rz, sigma_tt

    raise ValueError(f"Point ({r_query},{z_query}) outside mesh")
