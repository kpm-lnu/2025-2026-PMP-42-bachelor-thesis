from .mesh import Mesh
from .boundaryConditions import BoundaryCondition
import numpy as np
import pandas as pd
from .timingUtils import stopwatch
from .mortar import MortarInterface

class AxisymmetricFEMSolver:
    def __init__(
        self,
        mesh: Mesh,
        bcs: list[BoundaryCondition],
        mortar_interfaces: list[MortarInterface] | None = None,
    ):
        self.mesh = mesh
        self.bcs = bcs
        self.mortar_interfaces = mortar_interfaces or []
        
        self.global_stiffness = None
        self.global_force = None
        self.displacements = None
        self.local_stiffnesses = [] # for debug only

    def assign_dof_indices(self):
        num_nodes = len(self.mesh.nodes)

        for node_id, node in self.mesh.nodes.items():
            node.dof_indices = [node_id, num_nodes + node_id]


    def assemble_system(self):
        n_dof = self.mesh.node_dof * len(self.mesh.nodes)
        print('n_dof', n_dof)
        self.global_stiffness = np.zeros((n_dof, n_dof), dtype=float)
        self.global_force = np.zeros(n_dof, dtype=float)
        
        for elem_id, elem in self.mesh.elements.items():
            Ke = elem.compute_element_stiffness(self.mesh)
            self.local_stiffnesses.append(Ke)
            
            num_nodes = len(self.mesh.nodes)
            dof_map = [None] * (self.mesh.node_dof * len(elem.node_ids))
            for a, node_id in enumerate(elem.node_ids):
                dof_map[a] = node_id
                dof_map[len(elem.node_ids) + a] = node_id + num_nodes
        
            for i_local, i_global in enumerate(dof_map):
                for j_local, j_global in enumerate(dof_map):
                    self.global_stiffness[i_global, j_global] += Ke[i_local, j_local]

    def _build_augmented_system_mortar(self):
        """Build augmented [K B^T; B 0], b=[f;0] for MortarInterface (LM mortar) interfaces."""
        if not self.mortar_interfaces:
            return None, None
        K = self.global_stiffness
        f = self.global_force
        n_u = int(K.shape[0])

        Bs = []
        for iface in self.mortar_interfaces:
            # Use the interface to build its constraint matrix against the displacement DOFs.
            # We assemble one combined augmented system for all interfaces.
            B = iface.assemble_augmented(K, f, self.mesh)
            if B is None:
                continue
            Bs.append(np.asarray(B, dtype=float))

        if not Bs:
            return None, None

        Btot = np.vstack(Bs)
        n_lam = int(Btot.shape[0])
        A = np.zeros((n_u + n_lam, n_u + n_lam), dtype=K.dtype)
        b = np.zeros((n_u + n_lam,), dtype=f.dtype)
        A[:n_u, :n_u] = K
        b[:n_u] = f
        A[:n_u, n_u:] = Btot.T
        A[n_u:, :n_u] = Btot
        return A, b

    def apply_boundary_conditions(self):
        for bc in self.bcs:
            bc.apply(self.global_stiffness, self.global_force, self.mesh)

    @stopwatch
    def solve(self, custom_n_points, element_type):
        self.displacements = np.linalg.solve(self.global_stiffness, self.global_force)
        
        labels = [*[f"r{i}" for i in range(int(len(self.displacements)/2))], *[f"z{i}" for i in range(int(len(self.displacements)/2))]]
        df = pd.DataFrame(self.displacements.reshape(1, -1), columns=labels)

        print("Displacements u:")
        print(df.to_string(float_format=lambda x: f"{x: .4f}"))

        
        num_nodes = len(self.mesh.nodes)

        for node_id, node in self.mesh.nodes.items():
            node.displacements[0] = self.displacements[node_id]
            node.displacements[1] = self.displacements[num_nodes + node_id]

    def run(self, custom_n_points, element_type):
        self.assign_dof_indices()
        self.assemble_system()

        if self.mortar_interfaces:
            A, b = self._build_augmented_system_mortar()
            if A is None:
                self.apply_boundary_conditions()
                self.solve(custom_n_points, element_type)
            else:
                for bc in self.bcs:
                    bc.apply(A, b, self.mesh)
                x = np.linalg.solve(A, b)
                n_u = self.global_stiffness.shape[0]
                self.displacements = x[:n_u]
                self.lambdas = x[n_u:]
                num_nodes = len(self.mesh.nodes)
                for node_id, node in self.mesh.nodes.items():
                    node.displacements[0] = self.displacements[node_id]
                    node.displacements[1] = self.displacements[num_nodes + node_id]
        else:
            self.apply_boundary_conditions()
            self.solve(custom_n_points, element_type)
        print("FEM solution complete.")
