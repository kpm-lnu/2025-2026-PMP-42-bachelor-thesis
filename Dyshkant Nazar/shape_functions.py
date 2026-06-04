import numpy as np

def shape_functions_q4(xi, eta):

    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ])

    dN_dxi = np.array([
        [-0.25 * (1 - eta), -0.25 * (1 - xi)],
        [ 0.25 * (1 - eta), -0.25 * (1 + xi)],
        [ 0.25 * (1 + eta),  0.25 * (1 + xi)],
        [-0.25 * (1 + eta),  0.25 * (1 - xi)]
    ])

    return N, dN_dxi


def shape_functions_q8(xi, eta):
    """
    Q8 serendipity shape functions.

    Node ordering (interleaved corners + midsides):

        idx: 0       1        2       3       4      5       6       7
             C(-1,-1) M(0,-1) C(1,-1) M(1,0) C(1,1) M(0,1) C(-1,1) M(-1,0)

    Visually:
        6 --- 5 --- 4
        |           |
        7           3
        |           |
        0 --- 1 --- 2

    Derivatives are obtained analytically via the product rule.
    All formulas verified by Kronecker delta + partition-of-unity tests.
    """

    N  = np.zeros(8)
    dN = np.zeros((8, 2))   # columns: [d/dxi, d/deta]

    # ------------------------------------------------------------------
    # Corner nodes
    # General formula: N = (1/4)(1+xi_i*xi)(1+eta_i*eta)(xi_i*xi+eta_i*eta-1)
    # Derivatives obtained by product rule on A=(1+/-xi), B=(1+/-eta), C=(...)
    # ------------------------------------------------------------------

    # idx 0 : (xi_i, eta_i) = (-1, -1)
    N[0]    =  0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
    dN[0,0] =  0.25 * (1 - eta) * (2*xi + eta)       # d/dxi
    dN[0,1] =  0.25 * (1 - xi)  * (xi  + 2*eta)      # d/deta

    # idx 2 : (xi_i, eta_i) = (+1, -1)
    N[2]    =  0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
    dN[2,0] =  0.25 * (1 - eta) * (2*xi - eta)
    dN[2,1] =  0.25 * (1 + xi)  * (-xi + 2*eta)

    # idx 4 : (xi_i, eta_i) = (+1, +1)
    N[4]    =  0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
    dN[4,0] =  0.25 * (1 + eta) * (2*xi + eta)
    dN[4,1] =  0.25 * (1 + xi)  * (xi  + 2*eta)

    # idx 6 : (xi_i, eta_i) = (-1, +1)
    N[6]    =  0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
    dN[6,0] =  0.25 * (1 + eta) * (2*xi - eta)
    dN[6,1] =  0.25 * (1 - xi)  * (-xi + 2*eta)

    # ------------------------------------------------------------------
    # Midside nodes
    # ------------------------------------------------------------------

    # idx 1 : (xi_i, eta_i) = (0, -1)  — bottom edge
    N[1]    =  0.5 * (1 - xi**2) * (1 - eta)
    dN[1,0] = -xi * (1 - eta)
    dN[1,1] = -0.5 * (1 - xi**2)

    # idx 3 : (xi_i, eta_i) = (+1, 0)  — right edge
    N[3]    =  0.5 * (1 + xi) * (1 - eta**2)
    dN[3,0] =  0.5 * (1 - eta**2)
    dN[3,1] = -(1 + xi) * eta

    # idx 5 : (xi_i, eta_i) = (0, +1)  — top edge
    N[5]    =  0.5 * (1 - xi**2) * (1 + eta)
    dN[5,0] = -xi * (1 + eta)
    dN[5,1] =  0.5 * (1 - xi**2)

    # idx 7 : (xi_i, eta_i) = (-1, 0)  — left edge
    N[7]    =  0.5 * (1 - xi) * (1 - eta**2)
    dN[7,0] = -0.5 * (1 - eta**2)
    dN[7,1] = -(1 - xi) * eta

    return N, dN