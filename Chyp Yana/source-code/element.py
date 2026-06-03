# triangles or squares
# stores nodes (numbers), probably jacobian so on
# processes element
# converts it to the form to assemble system of
from enum import Enum
import numpy as np
import local
import triangulation as tr

class TypeOfElement(Enum):
    TRIANGLE = 3
    SQUARE = 4

def find_edge_idx(elem, edge):
        idx_start = np.where(elem.nodes==edge[0])[0][0]
        idx_end = np.where(elem.nodes==edge[-1])[0][0]
        if [idx_start, idx_end] in ([0, 1], [1, 0]):
            return 0
        if [idx_start, idx_end] in ([1, 2], [2, 1]):
            return 1
        if [idx_start, idx_end] in ([2, 0], [0, 2]):
            return 2
        return -1

class Element:
    type: TypeOfElement
    nodes: np.array(int)
    vertices: np.array(object)
    ksieta_coords: np.array(object)
    jacobian: float
    approx: int

    def __init__(self, type, nodes, vertices, approx=1):
        if (type!=3 and type!=4):
            raise('wrong type of element')
        self.type = type
        self.nodes = nodes
        self.vertices = []
        for node in self.nodes:
            self.vertices.append(vertices[node])
        self.approx = approx
        if (type==3):
            self.calculate_jacobian(local.LTriangle)
        else:
            self.calculate_jacobian(local.LSquare)
        # self.add_points(vertices)


    def find_point_index(self, v, pts, eps=1e-9):
        for i in range(len(pts)):
            if abs(pts[i][0] - v[0]) < eps and abs(pts[i][1] - v[1]) < eps:
                return i
        return -1

    def add_points(self, vertices):
        if self.approx==2:
            for i in range(len(self.vertices)):
                v = [0.5*(self.vertices[i%3][0] + self.vertices[(i+1)%3][0]),
                     0.5*(self.vertices[i%3][1] + self.vertices[(i+1)%3][1])]
                # print(v)
                index = self.find_point_index(v, vertices, 1e-4)
                if index>=0:
                    # print('present')
                    self.nodes = np.append(self.nodes, [index])
                else:
                    # print('new')
                    # print(len(vertices))
                    self.vertices.append(v)
                    vertices = np.append(np.array(vertices), [v], axis=0)
                    # print(len(vertices))
                    self.nodes = np.append(self.nodes, [len(vertices)-1])
        elif self.approx==3:
            for i in range(3):
                for t in [1/3, 2/3]:
                    v = [float(t * self.vertices[(i+1)%3][0] + (1-t) * self.vertices[i][0]),
                         float(t * self.vertices[(i+1)%3][1] + (1-t) * self.vertices[i][1])]
                    index = self.find_point_index(v, vertices, 1e-4)
                    if index >= 0:
                        self.nodes = np.append(self.nodes, [index])
                    else:
                        self.vertices.append(v)
                        vertices = np.append(np.array(vertices), [v], axis=0)
                        self.nodes = np.append(self.nodes, [len(vertices)-1])
            # interior node at centroid
            v = [sum(self.vertices[i][0] for i in range(3)) / 3,
                 sum(self.vertices[i][1] for i in range(3)) / 3]
            index = self.find_point_index(v, vertices, 1e-4)
            if index >= 0:
                self.nodes = np.append(self.nodes, [index])
            else:
                self.vertices.append(v)
                vertices = np.append(np.array(vertices), [v], axis=0)
                self.nodes = np.append(self.nodes, [len(vertices)-1])
        return vertices

    def calculate_jacobian(self, localFigure):
        self.jacobian = localFigure.jacobi(self.vertices)

    def calculate_stress(self, D, strain):
        return D*strain

    def calculate_strain(self, B, u_e):
        return B*u_e
