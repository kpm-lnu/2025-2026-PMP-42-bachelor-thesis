import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from element_stiffness import element_stiffness_q4, element_stiffness_q8

def elasticity_matrix(E, nu, plane_stress=True):
    if plane_stress:

        D = E / (1 - nu**2) * np.array([
            [1,  nu, 0],
            [nu, 1,  0],
            [0,  0, (1 - nu) / 2]
        ])

    else:

        D = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2]
        ])

    return D

def assemble_q4(nodes, elements, E, nu, thickness=1.0):
    ndof = 2 * len(nodes)

    K = lil_matrix((ndof, ndof))
    F = np.zeros(ndof)

    D = elasticity_matrix(E, nu)

    for element in elements:

        # Element coordinates
        coords = np.array([nodes[node] for node in element])

        # Element stiffness matrix
        Ke = thickness * element_stiffness_q4(coords, D)

        # DOF mapping
        dofs = []

        for node in element:
            dofs.extend([2 * node, 2 * node + 1])

        # Assembly
        for i in range(8):
            for j in range(8):

                K[dofs[i], dofs[j]] += Ke[i, j]

    return K.tocsr(), F


def assemble_q8(nodes, elements, E, nu, thickness=1.0):
    """
    Global assembly for Q8 serendipity elements

    Node ordering:

        7 --- 6 --- 5
        |             |
        8             4
        |             |
        1 --- 2 --- 3
    """

    ndof = 2 * len(nodes)

    K = lil_matrix((ndof, ndof))
    F = np.zeros(ndof)

    D = elasticity_matrix(E, nu)

    for element in elements:
        coords = np.array([nodes[node] for node in element])

        if coords.shape != (8, 2):
            raise ValueError(
                f"Q8 element must contain 8 nodes. "
                f"Got shape {coords.shape}"
            )

        Ke = thickness * element_stiffness_q8(coords, D)

        dofs = []

        for node in element:
            dofs.extend([2 * node, 2 * node + 1])

        for i in range(16):
            for j in range(16):

                K[dofs[i], dofs[j]] += Ke[i, j]

    return K.tocsr(), F