import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from assemble import assemble_q4, assemble_q8, elasticity_matrix
from boundary_conditions import apply_multiple_point_loads, apply_boundary_conditions


def solve_fem(
        nodes,
        elements,
        E,
        nu,
        loads=None,
        fixed_dofs=None,
        element_type="Q4",
        thickness=1.0):

    if element_type.upper() == "Q4":

        K, F = assemble_q4(
            nodes,
            elements,
            E,
            nu,
            thickness
        )

    elif element_type.upper() == "Q8":

        K, F = assemble_q8(
            nodes,
            elements,
            E,
            nu,
            thickness
        )

    else:
        raise ValueError("element_type must be 'Q4' or 'Q8'")

    if loads is not None:

        F = apply_multiple_point_loads(F, loads)


    if fixed_dofs is not None:

        K, F = apply_boundary_conditions(
            K,
            F,
            fixed_dofs
        )

    U = spsolve(K, F)

    return U, K, F