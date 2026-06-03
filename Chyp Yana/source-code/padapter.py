import numpy as np


import element as el
from norms import h1_norm_diff, h1_error_elements
import domain as dom
from solver import plot_mesh
import triangulation

class PAdapter:
    elements: np.array(object)
    vertices: np.array(object)
    # perimeter_neigh: np.array(object)
    def __init__(self, elements, vertices):
        self.elements = elements
        self.vertices = vertices
        self.new_elements = []
        self.new_vertices = []
        self.perimeter_neigh = []
        self.perimeter_bounds = []
        self.elems_to_rebuild_idx = []

    def calc_estimates(self, u_exact, u_approx, u_exact_grad, u_approx_grad):
        return h1_error_elements(self.elements, u_exact, u_approx, u_exact_grad, u_approx_grad)

    def mark_elements(self, error_estimates, theta=0.5):
        elems_to_refine = []
        self.elems_to_rebuild_idx = []
        # save elements or their numbers
        # maximum strategy
        eta = max(error_estimates)
        print('eta = ', eta)
        for i in range(len(self.elements)):
            if error_estimates[i] >= theta*eta:
                elems_to_refine.append(self.elements[i])
            else:
                self.elems_to_rebuild_idx.append(i)
        return elems_to_refine

    def find_by_edge(self, edge, target):
        for el in self.elements:
            if set({edge[0], edge[-1]}) < set(el.nodes):
                if set(el.nodes) != set(target.nodes):
                    # print(f'found {el.nodes} for edge {edge}')
                    return el
                
    def find_elem_idx(self, elem):
        for i in range(len(self.elements)):
            if set(self.elements[i].nodes) == set(elem.nodes):
                return i

    def split_neighbour(self, element, edge):
        idx = self.find_elem_idx(element)
        self.elements = np.delete(self.elements, idx)
        # print(f'split {element.nodes} by {edge}')
        # self.elements = np.pop()
        # edge = [start, extra, end]
        idx_edge = el.find_edge_idx(element, edge)
        idx_fixed = (idx_edge+2)%3
        nodes = [0, 0, 0]

        # nodes[idx_edge] = element.nodes[idx_edge]
        # nodes[(idx_edge+1)%3] = element.nodes[(idx_edge+1)%3]
        # nodes[idx_fixed] = element.nodes[idx_fixed]
        for i in range(len(edge)-1):
            # nodes = [edge[i], edge[i+1], element.nodes[idx_fixed]]
            # idx_start = np.where(element.nodes == edge[i])
            # idx_end = np.where(element.nodes == edge[(i+1)%3])
            nodes = [0, 0, 0]
            nodes[idx_edge] = edge[i]
            nodes[(idx_edge+1)%3] = edge[i+1]
            nodes[(idx_edge+2)%3] = element.nodes[idx_fixed]
            # print('nodes after split: ', nodes)
            # для approx=2 передбачити
            # чи правильний порядок вузлів
            elem = el.Element(3, nodes, self.vertices, element.approx)
            self.elements = np.append(np.array(self.elements), [elem], axis=0)
            # print(f'new element: {elem.nodes} added: {self.elements[-1].nodes}')
        return

    def refine_elem(self, element, marked_elements):
        refined = el.Element(3, element.nodes[:3], self.vertices, element.approx+1)
        self.vertices = refined.add_points(self.vertices)
        idx = self.find_elem_idx(element)
        # print(f'idx = {idx}, elements = {element.nodes}')
        # print(f'assignment at {idx}: {self.elements[idx].nodes} <- {refined.nodes}')
        self.elements[idx] = refined
        # print(f'refined element: ', self.elements[idx].nodes)
        self.new_elements.append(refined)
        neighbours = []
        for i in range(3):
            edge = [element.nodes[i], refined.nodes[i+3], element.nodes[(i+1)%3]]
            neigh = self.find_by_edge(edge, refined)
            if neigh:
                neighbours.append((edge, neigh))
        
        if len(neighbours)==0:
            return
        for neigh in neighbours:
            if neigh[1] not in marked_elements:
                # print(f'{neigh[1].nodes} not in marked_elements')
                if neigh[1].approx < refined.approx:
                    self.perimeter_neigh.append(neigh[0])
        #             self.split_neighbour(neigh[1], neigh[0])

    # after reapplication of boundary conditions
    def find_perimeter_bounds(self, bounds, domain, degree):
        i = 0
        for bound in bounds:
            bound.clear()
            if i==0 or i==2:
                type = 'lin'
            elif i==3:
                type = 'cir2'
            else:
                type = 'cir1'
            refined = (type == 'cir1')
            bound.set_all_nodes(domain.find_nodes_at_bound(bound.points, self.vertices, refined))
            print(f'bound {bound.points} nodes {bound.nodes}')
            bound.find_elems_and_edges(self.elements)
            for i in range(len(bound.elems)):
                elem = bound.elems[i]
                if elem.approx<degree:
                    self.perimeter_bounds.append(bound.edges[i])   
            i+=1         
    def cluster_edges(self):
        perimeter = list(set(tuple(e) for e in self.perimeter_neigh) | set(tuple(e) for e in self.perimeter_bounds))
        by_start = {}
        for e in perimeter:
            by_start.setdefault(e[0], []).append(e)
        end_nodes = {e[-1] for e in perimeter}
        visited = set()
        clusters = []
        for first in perimeter:
            if first in visited:
                continue
            chain = [first]
            visited.add(first)
            while len(visited) < len(perimeter):
                candidates = [e for e in by_start.get(chain[-1][-1], []) if e not in visited]
                if not candidates:
                    break
                chain.append(candidates[0])
                visited.add(candidates[0])
            clusters.append(chain)
            print('chain found: ', chain)
        return clusters

    def mesh_cluster(self, cluster, degree, min_angle=30, max_area=0.1):
        perim_nodes = []
        for edge in cluster:
            if int(edge[0]) not in perim_nodes:
                perim_nodes.append(int(edge[0]))
            if len(edge)==3 and int(edge[1]) not in perim_nodes:
                perim_nodes.append(int(edge[1]))
        coords = np.array([self.vertices[n] for n in perim_nodes], dtype=float)
        n = len(perim_nodes)
        print('perim nodes: ', perim_nodes)
        # for i in range(n):
            # print(f'node: {perim_nodes[i]} is {coords[i]}')
        print('elems to rebuild indices: ', self.elems_to_rebuild_idx)
        # for idx in self.elems_to_rebuild_idx:
            # print('elem to rebuild: ', self.elements[idx].nodes)
            # self.elements.pop(idx)
        
        domain = dom.Domain(coords)
        tr = triangulation.Triangulator(domain)
        mesh = tr.triangulate(len(perim_nodes), min_angle, max_area)
        new_verts = mesh['vertices']
        new_tris  = mesh['triangles']
        plot_mesh(new_verts, new_tris)  


        # first n local indices map exactly to perim_nodes
        # any interior Steiner points are new global vertices
        local_to_global = list(perim_nodes)
        for lv in new_verts[n:]:
            self.vertices = np.append(self.vertices, [lv], axis=0)
            local_to_global.append(len(self.vertices) - 1)

        # remove old low-degree elements whose corners are all inside this cluster
        cluster_node_set = set(local_to_global)
        self.elements = np.array([
            e for e in self.elements
            if not (e.approx < degree and set(e.nodes[:3]).issubset(cluster_node_set))
        ])

        new_elements = []
        for tri in new_tris:
            global_nodes = np.array([local_to_global[i] for i in tri])
            new_elem = el.Element(3, global_nodes, self.vertices, degree-1)
            self.vertices = new_elem.add_points(self.vertices)
            self.new_elements.append(new_elem)
        # self.elements = np.append(self.elements, new_elements)

    def refinement(self, marked_elements, bounds, domain, degree=2, min_angle=30):
        for element in marked_elements:
            if element.approx < degree:
                self.refine_elem(element, marked_elements)
        self.find_perimeter_bounds(bounds, domain, degree)
        clusters = self.cluster_edges()
        for cluster in clusters:
            self.mesh_cluster(cluster, degree, min_angle)
        self.perimeter_neigh = []
        self.perimeter_bounds = []

