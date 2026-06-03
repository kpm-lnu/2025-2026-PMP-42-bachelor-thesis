# use delaunay for triangulation
# smth else for squares
# stores array of elements of nodes
# maps coordinates to numbers of nodes
import math
from cmath import cos, sin

from bound_cond import Type


class Domain:
    def __init__(self, vertices):
        self.vertices = vertices
        self.bounds = []
        for i in range(len(self.vertices) - 1):
            self.bounds.append([self.vertices[i], self.vertices[i+1]])
        self.bounds.append([self.vertices[0], self.vertices[len(self.vertices) - 1]])

    def get_bounds(self):
        return self.bounds

    @staticmethod
    def check_on_edge(start, end, coords):
        tol = 0.00001
        return abs((coords[1] - start[1]) * (end[0] - start[0]) - (end[1] - start[1]) * (coords[0] - start[0])) < tol

    def find_nodes_at_bound(self, bound, vertices):
        nodes = []
        for i in range(len(vertices)):
            if self.check_on_edge(bound[0], bound[1], vertices[i]):
                nodes.append(i)
        nodes = sorted(nodes, key= lambda i: (vertices[i][0]-bound[0][0])**2 + (vertices[i][1]-bound[0][1])**2)
        return nodes
    def find_elems_at_bound(self, nodes_at_bound, elements):
        belems = []
        for i in range(len(nodes_at_bound)-1):
            elem = [el for el in elements if (nodes_at_bound[i] in el.nodes and nodes_at_bound[i+1] in el.nodes)]
            assert(len(elem)==1)
            belems.append(elem[0])
        return belems
    def apply_priority(self, bounds):
        n = len(bounds)
        for i in range(n):
            idxn = (i + 1) % n
            last = bounds[i].nodes[-1]
            next_first = bounds[idxn].nodes[0]
            if bounds[i].type == Type.ROBIN and bounds[idxn].type in (Type.ROBIN, Type.NEUMANN):
                bounds[idxn].nodes.pop(0)
            elif bounds[i].type == Type.NEUMANN:
                if bounds[idxn].type in (Type.ROBIN, Type.NEUMANN):
                    bounds[i].nodes.pop()
        return bounds

class CircularDomain:
    # vertices:
    def __init__(self, vertices, density=10):
        self.bounds = []
        for i in range(len(vertices) - 1):
            self.bounds.append([vertices[i], vertices[i+1]])
        self.bounds.append([vertices[0], vertices[len(vertices) - 1]])

        self.vertices = []
        self.vertices.append(vertices[0])
        self.vertices.append(vertices[1])
        a = math.pi/(2*density)
        self.r1 = vertices[1][1]-vertices[2][1]
        self.c = [vertices[1][0], vertices[1][1]-self.r1]
        for i in range(1, density):
            self.vertices.append([self.r1*math.sin(a*i)+self.c[0], self.r1*math.cos(a*i)+self.c[1]])
        self.vertices.append(vertices[2])
        self.vertices.append(vertices[3])
        self.r2 = vertices[0][1]-vertices[3][1]
        for i in range(1, density):
            self.vertices.append([self.r2 * math.cos(a * i) + self.c[0], self.r2 * math.sin(a * i) + self.c[1]])

    def get_bounds(self):
        return self.bounds

    def find_elems_at_bound(self, nodes_at_bound, elements):
        # print(nodes_at_bound)
        belems = []
        for i in range(len(nodes_at_bound)-1):
            elem = [el for el in elements if (nodes_at_bound[i] in el.nodes and nodes_at_bound[i+1] in el.nodes)]
            # print(nodes_at_bound[i], nodes_at_bound[i+1])
            assert(len(elem)==1)
            belems.append(elem[0])
        return belems

    @staticmethod
    def _on_segment(a, b, p, tol=0.0001):
        cross = abs((p[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(p[0]-a[0]))
        in_box = (min(a[0],b[0])-tol <= p[0] <= max(a[0],b[0])+tol and
                  min(a[1],b[1])-tol <= p[1] <= max(a[1],b[1])+tol)
        return cross < tol and in_box

    def find_nodes_at_bound(self, bound, vertices, refined=False):
        is_circular = not (
            (bound[0] == self.bounds[0][0] and bound[1] == self.bounds[0][1]) or
            (bound[0] == self.bounds[2][0] and bound[1] == self.bounds[2][1])
        )
        nodes = [i for i in range(len(vertices)) if self.check_on_edge(bound[0], bound[1], vertices[i], i, refined)]
        nodes = sorted(nodes, key=lambda i: (vertices[i][0]-bound[0][0])**2 + (vertices[i][1]-bound[0][1])**2)
        if not is_circular:
            return nodes

        r = self.r1 if (bound[0] == self.bounds[0][1] and bound[1] == self.bounds[2][0]) else self.r2
        other_r = self.r2 if r == self.r1 else self.r1
        tol_circ = 0.01
        on_other_circle = set(
            i for i in range(len(vertices))
            if abs((vertices[i][0]-self.c[0])**2 + (vertices[i][1]-self.c[1])**2 - other_r**2)/other_r**2 < tol_circ
        )
        result = []
        for i in range(len(nodes)-1):
            result.append(nodes[i])
            between = [
                j for j in range(len(vertices))
                if j not in set(nodes) and j not in on_other_circle and
                self._on_segment(vertices[nodes[i]], vertices[nodes[i+1]], vertices[j])
            ]
            between = sorted(between, key=lambda j: (vertices[j][0]-vertices[nodes[i]][0])**2 + (vertices[j][1]-vertices[nodes[i]][1])**2)
            result.extend(between)
        result.append(nodes[-1])
        return result

    def check_on_edge(self, start, end, coords, index, refined=False):
        tol_lin = 0.0001
        tol_circ = 0.02
        # if refined:
            # tol_circ /= 10000
        if (start==self.bounds[0][0] and end==self.bounds[0][1]) or (start==self.bounds[2][0] and end==self.bounds[2][1]):
            return abs((coords[1] - start[1]) * (end[0] - start[0]) - (end[1] - start[1]) * (coords[0] - start[0])) < tol_lin
        elif (start==self.bounds[0][1] and end==self.bounds[2][0]):
            dist = abs((coords[0]-self.c[0])**2 + (coords[1]-self.c[1])**2 - self.r1**2)/self.r1**2 
        else:
            dist = abs((coords[0] - self.c[0]) ** 2 + (coords[1] - self.c[1]) ** 2 - self.r2 ** 2)/self.r2**2
        if dist < tol_circ:
            # print(f'dist for {index}: {dist}')
            return True
        return False

    def apply_priority(self, bounds):
        n = len(bounds)
        for i in range(n):
            idxn = (i + 1) % n
            last = bounds[i].nodes[-1]
            next_first = bounds[idxn].nodes[0]
            if bounds[i].type == Type.ROBIN and bounds[idxn].type in (Type.ROBIN, Type.NEUMANN):
                bounds[idxn].nodes.pop(0)
            elif bounds[i].type == Type.NEUMANN:
                if bounds[idxn].type in (Type.ROBIN, Type.NEUMANN):
                    bounds[i].nodes.pop()
        return bounds

class LDomain:
    def __init__(self, vertices):
        self.vertices = vertices
        self.bounds = []
        for i in range(len(self.vertices) - 1):
            self.bounds.append([self.vertices[i], self.vertices[i + 1]])
        self.bounds.append([self.vertices[0], self.vertices[len(self.vertices) - 1]])

    def get_bounds(self):
        return self.bounds

    @staticmethod
    def check_on_edge(start, end, coords):
        tol = 0.001
        diff = abs((coords[1] - start[1]) * (end[0] - start[0]) - (end[1] - start[1]) * (coords[0] - start[0]))
        left = (start[0] <= coords[0] <= end[0] and start[1] <= coords[1] <= end[1])
        right = (end[0] <= coords[0] <= start[0] and end[1] <= coords[1] <= start[1])
        return (diff < tol) and (left or right)

    def find_nodes_at_bound(self, bound, vertices):
        nodes = []
        for i in range(len(vertices)):
            if self.check_on_edge(bound[0], bound[1], vertices[i]):
                nodes.append(i)
        nodes = sorted(nodes, key= lambda i: (vertices[i][0]-bound[0][0])**2 + (vertices[i][1]-bound[0][1])**2)
        return nodes
    def find_elems_at_bound(self, nodes_at_bound, elements):
        belems = []
        for i in range(len(nodes_at_bound)-1):
            elem = [el for el in elements if (nodes_at_bound[i] in el.nodes and nodes_at_bound[i+1] in el.nodes)]
            assert(len(elem)==1)
            belems.append(elem[0])
        return belems

    def apply_priority(self, bounds):
        n = len(bounds)
        for i in range(n):
            idxn = (i + 1) % n
            last = bounds[i].nodes[-1]
            next_first = bounds[idxn].nodes[0]
            if bounds[i].type == Type.ROBIN and bounds[idxn].type in (Type.ROBIN, Type.NEUMANN):
                bounds[idxn].nodes.pop(0)
            elif bounds[i].type == Type.NEUMANN:
                if bounds[idxn].type in (Type.ROBIN, Type.NEUMANN):
                    bounds[i].nodes.pop()
        return bounds