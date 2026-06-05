from .material import Material
from .shapeFunction import ShapeFunction2D
from .quadrature import Quadrature
import numpy as np

class AxisymmetricElement:
    def __init__(self, elem_id: int, node_ids: list[int], material: Material, 
                 shape_func: ShapeFunction2D, quadrature: Quadrature):
        self.elem_id = elem_id
        self.node_ids = node_ids
        self.material = material
        self.shape_func = shape_func
        self.quadrature = quadrature

    def getB(self, N, dN_dxi, dN_deta, element_nodes_len, coords, dof):
        a = np.dot(dN_dxi, coords[:, 0])
        b = np.dot(dN_dxi, coords[:, 1])
        c = np.dot(dN_deta, coords[:, 0])
        d = np.dot(dN_deta, coords[:, 1])
        detJ = a*d - b*c
        inv_detJ = 1.0 / detJ

        dN_dr = inv_detJ * (d*dN_dxi -b*dN_deta)
        dN_dz = inv_detJ * (-c*dN_dxi + a*dN_deta)
        
        r_current = np.dot(N, coords[:, 0])

        strain_length = 4
        B = np.zeros((strain_length, dof))
        B[0, :element_nodes_len] = dN_dr
        B[1, element_nodes_len:] = dN_dz
        B[2, :element_nodes_len] = dN_dz
        B[2, element_nodes_len:] = dN_dr
        B[3, :element_nodes_len] = N / r_current

        return B, r_current, detJ

    def compute_element_stiffness(self, mesh, log=False):
        D = self.material.get_elastic_matrix()

        dof = mesh.node_dof * self.shape_func.nodes_count
        Ke = np.zeros((dof, dof), dtype=float)

        element_nodes = [mesh.nodes[nid] for nid in self.node_ids]

        coords = np.array([[node.r, node.z] for node in element_nodes])

        for gp in self.quadrature.gauss_points_2D():
            xi = gp["xi"]
            eta = gp["eta"]
            weight = gp["weight"]

            N, dN_dxi, dN_deta = self.shape_func.evaluate(xi, eta)
            B, r_current, detJ = self.getB(N, dN_dxi, dN_deta, len(element_nodes), coords, dof)

            dV = r_current * detJ * weight
            Ke += B.T @ D @ B * dV

        np.set_printoptions(precision=4, suppress=True)
        return Ke

class DatasetElement:
    @staticmethod
    def getB(N, dN_dxi, dN_deta, element_nodes_len, coords, dof):
        a = np.dot(dN_dxi, coords[:, 0])
        b = np.dot(dN_dxi, coords[:, 1])
        c = np.dot(dN_deta, coords[:, 0])
        d = np.dot(dN_deta, coords[:, 1])
        detJ = a*d - b*c
        inv_detJ = 1.0 / detJ

        dN_dr = inv_detJ * (d*dN_dxi -b*dN_deta)
        dN_dz = inv_detJ * (-c*dN_dxi + a*dN_deta)
        
        r_current = np.dot(N, coords[:, 0])

        strain_length = 4
        B = np.zeros((strain_length, dof))
        B[0, :element_nodes_len] = dN_dr
        B[1, element_nodes_len:] = dN_dz
        B[2, :element_nodes_len] = dN_dz
        B[2, element_nodes_len:] = dN_dr
        B[3, :element_nodes_len] = N / r_current

        return B, r_current, detJ
    
    @staticmethod
    def compute_element_stiffness(coords, shape_func, material, pointsWithWeights):
        D = material.get_elastic_matrix()

        dof = 2 * shape_func.nodes_count
        Ke = np.zeros((dof, dof), dtype=float)

        for i in range(len(pointsWithWeights) // 3):
            print('pointsWithWeights', pointsWithWeights[3*i],pointsWithWeights[3 * i + 1])
            N, dN_dxi, dN_deta = shape_func.evaluate(pointsWithWeights[3 * i], pointsWithWeights[3 * i + 1])

            
            B, r_current, detJ = DatasetElement.getB(N, dN_dxi, dN_deta, len(coords), coords, dof)
            

            dV = r_current * detJ * pointsWithWeights[3 * i + 2]
            Ke += B.T @ D @ B * dV

        return Ke