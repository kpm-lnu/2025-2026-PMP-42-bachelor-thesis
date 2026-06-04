def apply_point_load(F, node, direction, value):
    dof = 2 * node + direction
    F[dof] += value
    return F

def apply_multiple_point_loads(F, loads):
    for load in loads:
        if len(load) == 3:
            node, direction, value = load
            F = apply_point_load(F, node, direction, value)
        else:
            raise ValueError("Each load must have [node, direction, value]")
    return F

def apply_boundary_conditions(K, F, fixed_dofs):

    K = K.tolil()

    for dof in fixed_dofs:

        K[dof, :] = 0
        K[:, dof] = 0

        K[dof, dof] = 1
        F[dof] = 0

    return K.tocsr(), F