import numpy as np
from shape_functions import shape_functions_q4, shape_functions_q8

def element_stiffness_q4(coords, D):
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    Ke = np.zeros((8, 8))

    for xi in gp:
        for eta in gp:
            N, dN_dxi = shape_functions_q4(xi, eta)

            # Jacobian
            J = dN_dxi.T @ coords

            detJ = np.linalg.det(J)

            if detJ <= 0:
                raise ValueError(
                    f"Negative Jacobian determinant: detJ = {detJ}"
                )

            invJ = np.linalg.inv(J)

            dN_dx = dN_dxi @ invJ

            B = np.zeros((3, 8))

            for i in range(4):

                B[0, 2 * i]     = dN_dx[i, 0]
                B[1, 2 * i + 1] = dN_dx[i, 1]

                B[2, 2 * i]     = dN_dx[i, 1]
                B[2, 2 * i + 1] = dN_dx[i, 0]

            Ke += B.T @ D @ B * detJ

    return Ke


import numpy as np

def element_stiffness_q8(coords, D):
    """
    Q8 serendipity element stiffness matrix (2D elasticity)

    coords : (8,2)
    D      : (3,3)
    """

    # 3-point Gauss rule
    gp = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    w  = np.array([5/9, 8/9, 5/9])

    Ke = np.zeros((16, 16))

    for i, xi in enumerate(gp):
        for j, eta in enumerate(gp):

            weight = w[i] * w[j]

            N, dN_dxi = shape_functions_q8(xi, eta)

            J = dN_dxi.T @ coords

            detJ = np.linalg.det(J)

            if detJ <= 0:
                raise ValueError(
                    f"Invalid Jacobian: detJ={detJ} at (xi={xi}, eta={eta})"
                )

            invJ = np.linalg.inv(J)

            dN_dx = dN_dxi @ invJ   # (8,2)

            B = np.zeros((3, 16))

            for a in range(8):
                dNdx = dN_dx[a, 0]
                dNdy = dN_dx[a, 1]

                idx = 2 * a

                B[0, idx]     = dNdx
                B[1, idx + 1] = dNdy
                B[2, idx]     = dNdy
                B[2, idx + 1] = dNdx

            Ke += (B.T @ D @ B) * detJ * weight

    return Ke
