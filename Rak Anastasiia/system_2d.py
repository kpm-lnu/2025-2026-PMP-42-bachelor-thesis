import numpy as np
import sympy as sp
import mesh_2d as m2d
import base_functions_2d as b2f
from scipy.spatial import distance

def set_up_vector_point_sources(sources, strengths, nodes, p, m, ap):
    f_vec = np.zeros((ap * p + 1) * (ap * m + 1))

    used_nodes = set()

    for (x0, y0), strength in zip(sources, strengths):
        dists = [distance.euclidean((x0, y0), (x, y)) for (x, y) in nodes]
        min_idx = np.argmin(dists)

        if min_idx not in used_nodes:
            f_vec[min_idx] += strength
            used_nodes.add(min_idx)
        else:
            print(f"Увага: вузол {min_idx} вже має джерело, можливо перекриття.")

    return f_vec


def set_up_vector(f, nodes, elements, p, m, ap):
    f_vec = np.zeros((ap * p + 1) * (ap * m + 1))

    dN_dksi_list, dN_deta_list = compute_partial_derivatives(ap)

    if ap == 1:
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1.0, 1.0]

    elif ap == 2:
        gauss_points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        gauss_weights = [5 / 9, 8 / 9, 5 / 9]

    elif ap == 3:
        # gauss_points = [
        #     -np.sqrt((3 / 7) - (2 / 7) * np.sqrt(6 / 5)),
        #     -np.sqrt((3 / 7) + (2 / 7) * np.sqrt(6 / 5)),
        #     np.sqrt((3 / 7) - (2 / 7) * np.sqrt(6 / 5)),
        #     np.sqrt((3 / 7) + (2 / 7) * np.sqrt(6 / 5))
        # ]
        gauss_points = [
            -np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
            -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
            np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
            np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
        ]
        gauss_weights = [
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36,
        ]

        # gauss_weights = [
        #     (18 - np.sqrt(30)) / 36,
        #     (18 + np.sqrt(30)) / 36,
        #     (18 - np.sqrt(30)) / 36,
        #     (18 + np.sqrt(30)) / 36
        # ]

    for ioe, noe in enumerate(elements):
        x_coords = [nodes[i][0] for i in noe]
        y_coords = [nodes[i][1] for i in noe]

        for i in range(len(noe)):
            integral_value = 0.0

            for ksi_idx, ksi_point in enumerate(gauss_points):
                for eta_idx, eta_point in enumerate(gauss_points):
                    x_val, y_val = m2d.isoparametric_transform(ksi_point, eta_point, x_coords, y_coords, ap)
                    J = compute_jacobian(ksi_point, eta_point, x_coords, y_coords, dN_dksi_list, dN_deta_list)
                    detJ = np.abs(np.linalg.det(J))
                    f_val = f(x_val, y_val)
                    N_i = b2f.N(i, ksi_point, eta_point, ap)

                    integral_value += N_i * f_val * detJ * gauss_weights[ksi_idx] * gauss_weights[eta_idx]

            f_vec[noe[i]] += integral_value

    return f_vec


def compute_element_stiffness(elements, nodes, ap, k1, k2):
    ksi, eta = sp.symbols('ksi eta')
    element_matrices = []

    dN_dksi_list, dN_deta_list = compute_partial_derivatives(ap)

    if ap == 1:
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1.0, 1.0]
    elif ap == 2:
        gauss_points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        gauss_weights = [5 / 9, 8 / 9, 5 / 9]
    elif ap == 3:
        sqrt_3_7_plus = np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)
        sqrt_3_7_minus = np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7)

        # gauss_points = [-sqrt_3_7_minus, -sqrt_3_7_plus, sqrt_3_7_plus, sqrt_3_7_minus]

        # w_1 = (18 + np.sqrt(30)) / 36
        # w_2 = (18 - np.sqrt(30)) / 36
        #
        # gauss_weights = [w_2, w_1, w_1, w_2]
        gauss_points = [
            -np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
            -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
            np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
            np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
        ]
        gauss_weights = [
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36,
        ]

    for noe in elements:
        x_coords = [nodes[i][0] for i in noe]
        y_coords = [nodes[i][1] for i in noe]

        K_local = np.zeros((len(noe), len(noe)))

        for i in range(len(noe)):
            for j in range(len(noe)):
                integral_value = 0.0

                for ksi_idx, ksi_point in enumerate(gauss_points):
                    for eta_idx, eta_point in enumerate(gauss_points):
                        J = compute_jacobian(ksi_point, eta_point, x_coords, y_coords, dN_dksi_list, dN_deta_list)
                        detJ = np.linalg.det(J)
                        J_inv = np.linalg.inv(J)

                        dN_i_dksi = dN_dksi_list[i](ksi_point, eta_point)
                        dN_i_deta = dN_deta_list[i](ksi_point, eta_point)
                        dN_j_dksi = dN_dksi_list[j](ksi_point, eta_point)
                        dN_j_deta = dN_deta_list[j](ksi_point, eta_point)

                        x_gp = sum(x_coords[n] * b2f.N(n, ksi_point, eta_point, ap) for n in range(len(noe)))
                        y_gp = sum(y_coords[n] * b2f.N(n, ksi_point, eta_point, ap) for n in range(len(noe)))

                        k1_val = k1(x_gp, y_gp)
                        k2_val = k2(x_gp, y_gp)

                        dN_i = np.array([dN_i_dksi, dN_i_deta])  # тепер (2,)
                        grad_N_i = J_inv @ dN_i  # також (2,)
                        dN_j = np.array([dN_j_dksi, dN_j_deta])  # (2,)
                        grad_N_j = J_inv @ dN_j

                        integrand = float((k1_val * grad_N_i[0] * grad_N_j[0] +
                                           k2_val * grad_N_i[1] * grad_N_j[1]) * abs(detJ))
                        if isinstance(integrand, np.ndarray):
                            integrand = integrand.item()
                        integral_value += integrand * gauss_weights[ksi_idx] * gauss_weights[eta_idx]

                K_local[i, j] = float(integral_value)

        element_matrices.append(K_local)

    return element_matrices


def assemble_global_matrix(nodes, elements, p, m, element_matrices, ap):
    num_nodes = (ap * p + 1) * (ap * m + 1)
    K = np.zeros((num_nodes, num_nodes))

    for elem_idx, noe in enumerate(elements):
        em = element_matrices[elem_idx]

        for i in range(len(noe)):
            for j in range(len(noe)):
                im = noe[i]
                jm = noe[j]
                K[im, jm] += em[i, j]

    return K



def compute_element_mass_matrix(elements, nodes, ap):
    """
    Обчислює локальні матриці мас для всіх елементів.
    M_ij = інтеграл(N_i * N_j * det(J))
    """
    element_mass_matrices = []
    
    dN_dksi_list, dN_deta_list = compute_partial_derivatives(ap)

    if ap == 1:
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1.0, 1.0]
    elif ap == 2:
        gauss_points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        gauss_weights = [5 / 9, 8 / 9, 5 / 9]
    elif ap == 3:
        gauss_points = [
            -np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
            -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
             np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
             np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
        ]
        gauss_weights = [
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36,
        ]

    for noe in elements:
        x_coords = [nodes[i][0] for i in noe]
        y_coords = [nodes[i][1] for i in noe]

        M_local = np.zeros((len(noe), len(noe)))

        for i in range(len(noe)):
            for j in range(len(noe)):
                integral_value = 0.0

                for ksi_idx, ksi_point in enumerate(gauss_points):
                    for eta_idx, eta_point in enumerate(gauss_points):
                        J = compute_jacobian(ksi_point, eta_point, x_coords, y_coords, dN_dksi_list, dN_deta_list)
                        detJ = np.abs(np.linalg.det(J))

                        N_i = b2f.N(i, ksi_point, eta_point, ap)
                        N_j = b2f.N(j, ksi_point, eta_point, ap)

                        integrand = N_i * N_j * detJ
                        integral_value += integrand * gauss_weights[ksi_idx] * gauss_weights[eta_idx]

                M_local[i, j] = integral_value
                
        element_mass_matrices.append(M_local)

    return element_mass_matrices

def compute_element_reaction_matrix(elements, nodes, ap, beta):
    """
    Обчислює локальні матриці реакції для всіх елементів.
    R_ij = інтеграл(beta(x,y) * N_i * N_j * det(J))
    """
    element_reaction_matrices = []
    dN_dksi_list, dN_deta_list = compute_partial_derivatives(ap)

    if ap == 1:
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1.0, 1.0]
    elif ap == 2:
        gauss_points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        gauss_weights = [5 / 9, 8 / 9, 5 / 9]
    elif ap == 3:
        gauss_points = [
            -np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
            -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
             np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
             np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
        ]
        gauss_weights = [
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36,
        ]

    for noe in elements:
        x_coords = [nodes[i][0] for i in noe]
        y_coords = [nodes[i][1] for i in noe]

        R_local = np.zeros((len(noe), len(noe)))

        for i in range(len(noe)):
            for j in range(len(noe)):
                integral_value = 0.0

                for ksi_idx, ksi_point in enumerate(gauss_points):
                    for eta_idx, eta_point in enumerate(gauss_points):
                        J = compute_jacobian(ksi_point, eta_point, x_coords, y_coords, dN_dksi_list, dN_deta_list)
                        detJ = np.abs(np.linalg.det(J))

                        N_i = b2f.N(i, ksi_point, eta_point, ap)
                        N_j = b2f.N(j, ksi_point, eta_point, ap)

                        x_gp = sum(x_coords[n] * b2f.N(n, ksi_point, eta_point, ap) for n in range(len(noe)))
                        y_gp = sum(y_coords[n] * b2f.N(n, ksi_point, eta_point, ap) for n in range(len(noe)))
                        
                        beta_val = beta(x_gp, y_gp)

                        integrand = beta_val * N_i * N_j * detJ
                        integral_value += integrand * gauss_weights[ksi_idx] * gauss_weights[eta_idx]

                R_local[i, j] = integral_value
                
        element_reaction_matrices.append(R_local)

    return element_reaction_matrices

def crank_nicolson(K, M, f, u0, dt, num_steps):
    u = u0.copy()

    A = M + (dt / 2) * K
    B = M - (dt / 2) * K

    for n in range(num_steps):
        rhs = B @ u + dt * f
        u = np.linalg.solve(A, rhs)

    return u


def compute_jacobian(ksi, eta, x_coords, y_coords, dN_dksi_funcs, dN_deta_funcs):
    dx_dksi = sum(dN_dksi_funcs[j](ksi, eta) * x_coords[j] for j in range(len(x_coords)))
    dx_deta = sum(dN_deta_funcs[j](ksi, eta) * x_coords[j] for j in range(len(x_coords)))
    dy_dksi = sum(dN_dksi_funcs[j](ksi, eta) * y_coords[j] for j in range(len(y_coords)))
    dy_deta = sum(dN_deta_funcs[j](ksi, eta) * y_coords[j] for j in range(len(y_coords)))

    J = np.array([[dx_dksi, dx_deta],
                  [dy_dksi, dy_deta]])
    return J


def compute_partial_derivatives(ap):
    ksi, eta = sp.symbols('ksi eta')

    num_nodes = 4 if ap == 1 else (9 if ap == 2 else 16)
    dN_dksi_funcs = []
    dN_deta_funcs = []

    for i in range(num_nodes):
        Ni = b2f.N(i, ksi, eta, ap)
        dN_dksi = sp.diff(Ni, ksi)
        dN_deta = sp.diff(Ni, eta)

        dN_dksi_func = sp.lambdify((ksi, eta), dN_dksi, modules="numpy")
        dN_deta_func = sp.lambdify((ksi, eta), dN_deta, modules="numpy")

        dN_dksi_funcs.append(dN_dksi_func)
        dN_deta_funcs.append(dN_deta_func)

    return dN_dksi_funcs, dN_deta_funcs