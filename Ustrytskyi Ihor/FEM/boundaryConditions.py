import numpy as np

class BoundaryCondition:
    def apply(self, global_stiffness, global_force, mesh):
        raise NotImplementedError("Must implement BoundaryCondition apply method.")


class DirichletBC(BoundaryCondition):
    def __init__(self, node_id, dof, value):
        self.node_id = node_id
        self.dof = dof
        self.value = value

    def apply(self, global_stiffness, global_force, mesh):
        node = mesh.nodes[self.node_id]
        global_dof = node.dof_indices[self.dof]
        
        size = global_stiffness.shape[0]
        for i in range(size):
            if i == global_dof:
                global_stiffness[global_dof, global_dof] = 1.0
            else:
                global_stiffness[global_dof, i] = 0.0
                global_stiffness[i, global_dof] = 0.0
        
        global_force[global_dof] = self.value

class NeumannBC(BoundaryCondition):
    def __init__(self, edge_nodes: list[int], dof: int, traction_value: float):
        self.edge_nodes = edge_nodes
        self.dof = dof
        self.traction_value = traction_value

    def apply(self, global_stiffness, global_force, mesh):
        xi_gp, w_gp = mesh.quadrature.gauss_points_boundary()

        step = mesh.shape_func_boundary.nodes_count - 1
        for i in range(0, len(self.edge_nodes) - step, step):
            segment_node_ids = self.edge_nodes[i : i + mesh.shape_func_boundary.nodes_count]
            segment_nodes = [mesh.nodes[nid] for nid in segment_node_ids]
            r = np.array([node.r for node in segment_nodes])
            z = np.array([node.z for node in segment_nodes])


            for xi, weight in zip(xi_gp, w_gp):
                N, dN_dxi = mesh.shape_func_boundary.evaluate(xi)

                r_gp = float(np.dot(N, r))
                dr_dxi = float(np.dot(dN_dxi, r))
                dz_dxi = float(np.dot(dN_dxi, z))

                J = (dr_dxi**2 + dz_dxi**2) ** 0.5
                dS = J * weight

                traction = self.traction_value * r_gp

                for j in range(len(segment_nodes)):
                    global_dof = segment_nodes[j].dof_indices[self.dof]
                    global_force[global_dof] += N[j] * traction * dS
