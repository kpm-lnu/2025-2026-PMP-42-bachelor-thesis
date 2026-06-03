# transforms form of bounds of needed
# applies boundary conditions
# Dirichlet, Neumann, Robin

from enum import Enum

import numpy as np
import scipy
import sympy as sp
from element import find_edge_idx
import local

x = sp.symbols('x')
y =  sp.symbols('y')
ksi = sp.symbols('ksi')
eta = sp.symbols('eta')

class Type(Enum):
    DIRICHLET = 3
    NEUMANN = 1
    ROBIN = 2

class Bound:
    def __init__(self, type: Type, points):
        self.type = type
        self.points = points
        self.nodes = []
        self.edges = []
        self.elems = []
        # self.degree = degree

    def set_all_nodes(self, nodes):
        self.nodes = nodes

    def clear(self):
        self.nodes = []
        self.edges = []
        self.elems = []

    def find_elems_and_edges(self, elements):
        i = 0
        while i < len(self.nodes)-1:
            edge = [self.nodes[i], self.nodes[i+1]]
            elem = [el for el in elements if set(edge) < set(el.nodes)]
            if len(elem)!=1:
                print(f'edge: {edge}')
                print('wrong elem array: len = ', len(elem))
                for el in elem:
                    print(f'element: {el.nodes}')
            assert (len(elem) == 1)
            # extend edge while next node belongs to the same element
            while i + len(edge) < len(self.nodes) and self.nodes[i + len(edge)] in elem[0].nodes:
                edge.append(self.nodes[i + len(edge)])
            i += len(edge) - 1
            self.edges.append(np.array(edge))
            self.elems.append(elem[0])
            
        # for edge in self.edges:
        #     elem = [el for el in elements if set(edge) < set(el.nodes)]
        #     if len(elem)!=1:
        #         print('for edge', edge)
        #         for el in elem:
        #             print(el.nodes)
        #     assert (len(elem) == 1)
        #     if not elem in self.elems:
        #         self.elems.append(elem[0])

    # def find_edges(self):
    #     for i in range(0, len(self.nodes)-1, self.degree):
    #         self.edges.append(self.nodes[i:i+self.degree+1])

def _edge_length_ratio(elem, idx_gamma):
    ref_lengths = [1.0, 1.4142135623730951, 1.0]
    v = elem.vertices
    pairs = [(0,1),(1,2),(2,0)]
    a, b = pairs[idx_gamma]
    phys_len = np.sqrt((v[b][0]-v[a][0])**2 + (v[b][1]-v[a][1])**2)
    return phys_len / ref_lengths[idx_gamma]

class BoundaryConds:
    def applyDirichlet(self, matrix, vector, bound, vertices, g):
        for node in bound.nodes:
            coords = vertices[node]
            matrix[2*node] = [0 for el in matrix[2*node]]
            matrix[2*node][2*node] = 1
            vector[2*node] = g(coords[0], coords[1])[0]

            matrix[2*node+1] = [0 for el in matrix[2*node+1]]
            matrix[2*node+1][2*node+1] = 1
            vector[2*node+1] = g(coords[0], coords[1])[1]
        return matrix, vector

    def applyNeumann(self, bound, base, p, vector):
        # NT (ksi, eta) ~> NT (x, y) because we integrate over bound
         # # NT*p = [ 6x2 ] * [ 2 ] = [ 6 ]
        # NTp = np.matvec(NT, p(x, y))
        # print('Neumann nodes: ', bound.nodes)
        # print('Neumann edges: ', bound.edges)
        for i in range(len(bound.elems)):
            elem = bound.elems[i]
            edge = bound.edges[i]
            # print('elem: ', elem.nodes, ', edge: ', edge)
            system = local.LTriangle.system(elem.vertices)
            idx_gamma = find_edge_idx(elem, edge)
            for node_j in edge:
                if node_j not in bound.nodes:
                    continue
                loc_j = np.where(elem.nodes==node_j)[0][0]
                integrals = local.LTriangle.integrate_gamma_xy(p, system, base[elem.approx-1][loc_j], idx_gamma)
                scale = _edge_length_ratio(elem, idx_gamma)
                vector[2 * node_j] += integrals[0] * scale
                vector[2 * node_j + 1] += integrals[1] * scale
        return vector

    def applyRobin(self, bound, mass_matrix, base_integrals, alpha, u0, matrix, vector):
        # print('Robin nodes: ', bound.nodes)
        # print('Robin edges: ', bound.edges)
        # print(u0)
        for i in range(len(bound.elems)):
            elem = bound.elems[i]
            edge = bound.edges[i]
            # print('elem: ', elem, ', edge: ', edge)
            idx_gamma = find_edge_idx(elem, edge)
            for node_j in edge:
                if node_j not in bound.nodes:
                    continue
                loc_j = np.where(elem.nodes==node_j)[0][0]
                scale = _edge_length_ratio(elem, idx_gamma)
                for node_i in elem.nodes:
                    loc_i = np.where(elem.nodes==node_i)[0][0]
                    matrix[2*node_j][2*node_i] += alpha * mass_matrix[elem.approx-1][2*loc_j][2*loc_i][idx_gamma] * scale
                    matrix[2*node_j+1][2*node_i+1] += alpha * mass_matrix[elem.approx-1][2*loc_j+1][2*loc_i+1][idx_gamma] * scale
                vector[2*node_j] += alpha * u0(x, y)[0] * base_integrals[elem.approx-1][loc_j][idx_gamma] * scale
                vector[2*node_j+1] += alpha * u0(x, y)[1] * base_integrals[elem.approx-1][loc_j][idx_gamma] * scale
        return matrix, vector