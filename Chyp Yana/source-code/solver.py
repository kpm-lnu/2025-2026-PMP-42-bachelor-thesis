
from matplotlib import pyplot as plt
import numpy as np

import local
import element as el
import elasticdeform as elde
import bound_cond as bc
from norms import mean_exact_grad


def plot_mesh(points, triangles):
    plt.triplot(points[:, 0], points[:, 1], triangles)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.gca().set_aspect('equal')
    # for i, (x, y) in enumerate(points):
        # plt.text(x, y, str(i), fontsize=10, color='blue',
                # ha='center', va='center')
    plt.show()

def create_elements(vertices, triangles, degree):
    elements = []
    for triangle in triangles:
        elements.append(el.Element(3, triangle, vertices, degree))
    for element in elements:
        vertices = element.add_points(vertices)
    return elements, vertices

def initialize_base(young=2500, max_degree=3):
    base_1 = local.LTriangle.base(1)
    base_2 = local.LTriangle.base(2)
    base_3 = local.LTriangle.base(3)
    base = (base_1, base_2, base_3)
    ed = elde.ElasticDeform(base_1, base_2, base_3, young)
    # k_e1, k_e2, k_e3 = ed.stiffness_matrix()
    # k_e = (k_e1, k_e2, k_e3)
    m_e1, m_e2, m_e3 = ed.mass_matrix()
    m_e = (m_e1, m_e2, m_e3)
    base_integrals_1 = local.LTriangle.base_integrals(base_1)
    base_integrals_2 = local.LTriangle.base_integrals(base_2)
    base_integrals_3 = local.LTriangle.base_integrals(base_3)
    base_integrals = (base_integrals_1, base_integrals_2, base_integrals_3)
    return ed, base, (), m_e, base_integrals
    
def assemble_matvec(s, k_e, elements, nn, b):
    matrix = s.assemble_matrix(elements, k_e, nn)
    vector = s.assemble_vector(elements, nn, b)
    return matrix, vector

def assemble_matvec_physical(s, ed, elements, nn, b):
    k_e_phys = ed.stiffness_matrix_physical(elements)
    matrix = s.assemble_matrix_physical(elements, k_e_phys, nn)
    vector = s.assemble_vector(elements, nn, b)
    return matrix, vector

def apply_bounds_rectangle(domain, bound_types, elements, vertices, funcs, base, m_e, base_integrals, matrix, vector, alpha=None):
    bounds_coords = domain.get_bounds()
    bounds = []
    for i in range(len(bounds_coords)):
        bound = bc.Bound(type=bound_types[i], points=bounds_coords[i])
        bound.set_all_nodes(domain.find_nodes_at_bound(bounds_coords[i], vertices))
        # print('bound: ', bound.points)
        # print('nodes at bound: ', bound.nodes)
        bound.find_elems_and_edges(elements)
        bounds.append(bound)
    bounds = domain.apply_priority(bounds)
    bconds = bc.BoundaryConds()
    bconds.applyNeumann(bounds[1], base, funcs[1], vector)
    bconds.applyNeumann(bounds[3], base, funcs[3], vector)
    # bconds.applyRobin(bounds[0], m_e, base_integrals, alpha, funcs[0], matrix, vector)
    # bconds.applyRobin(bounds[2], m_e, base_integrals, alpha, funcs[2], matrix, vector)
    bconds.applyDirichlet(matrix, vector, bounds[0], vertices, lambda x, y: [0, 0])
    bconds.applyDirichlet(matrix, vector, bounds[2], vertices, lambda x, y: [0, 0])
    return matrix, vector, bounds

def apply_bounds_lame(domain, bound_types, elements, vertices, funcs, base, m_e, base_integrals, matrix, vector, alpha=None):
    bounds_coords = domain.get_bounds()
    bounds = []
    for i in range(len(bounds_coords)):
        bound = bc.Bound(type=bound_types[i], points=bounds_coords[i])
        bound.set_all_nodes(domain.find_nodes_at_bound(bounds_coords[i], vertices))
        # print('bound: ', bound.points)
        # print('nodes at bound: ', bound.nodes)
        bound.find_elems_and_edges(elements)
        bounds.append(bound)
    bounds = domain.apply_priority(bounds)
    bconds = bc.BoundaryConds()
    bconds.applyNeumann(bounds[1], base, funcs[1], vector)
    bconds.applyNeumann(bounds[3], base, funcs[3], vector)
    # bconds.applyRobin(bounds[0], m_e, base_integrals, alpha, funcs[0], matrix, vector)
    # bconds.applyRobin(bounds[2], m_e, base_integrals, alpha, funcs[2], matrix, vector)
    bconds.applyDirichlet(matrix, vector, bounds[0], vertices, lambda x, y: [0, 0])
    bconds.applyDirichlet(matrix, vector, bounds[2], vertices, lambda x, y: [0, 0])
    return matrix, vector, bounds

def apply_bounds_lshape(domain, bound_types, elements, vertices, funcs, base, m_e, base_integrals, matrix, vector, alpha=None):
    bounds_coords = domain.get_bounds()
    bounds = []
    for i in range(len(bounds_coords)):
        bound = bc.Bound(type=bound_types[i], points=bounds_coords[i])
        bound.set_all_nodes(domain.find_nodes_at_bound(bounds_coords[i], vertices))
        bound.find_elems_and_edges(elements)
        bounds.append(bound)
    bounds = domain.apply_priority(bounds)
    # for i in range(len(bounds)):
        # print('bound: ', bounds[i].points)
        # print('nodes at bound: ', bounds[i].nodes)
    bconds = bc.BoundaryConds()
    bconds.applyNeumann(bounds[1], base, funcs[1], vector)
    bconds.applyNeumann(bounds[2], base, funcs[2], vector)
    bconds.applyNeumann(bounds[3], base, funcs[3], vector)
    bconds.applyNeumann(bounds[4], base, funcs[4], vector)
    bconds.applyDirichlet(matrix, vector, bounds[0], vertices, funcs[0])
    bconds.applyDirichlet(matrix, vector, bounds[5], vertices, funcs[5])
    return matrix, vector, bounds

def solve(matrix, vector):
    u = np.linalg.solve(matrix, vector)
    # print('u = ')
    # print('[' + ', '.join([f"{u[i]:.4f}" for i in range(len(u)) if i%2==0]) + ']' )
    # print('[' + ', '.join([f"{u[i]:.4f}" for i in range(len(u)) if i%2==1]) + ']' )
    return u

def init_exact_val_grads_lame(ed, elements, vertices):
    u_exact = np.array([
        [ed.u_lame(vertices[node, 0], vertices[node, 1]), 
        ed.u_lame(vertices[node, 0], vertices[node, 1])]
        for node in range(len(vertices))])
    u_exact_grad = np.array([mean_exact_grad(ed.u_lame_grad, vertices[element.nodes]) for element in elements])
    return u_exact, u_exact_grad

def init_approx_val_grads(u, elements):
    u_approx = np.array([
        [u[2*i], u[2*i+1]] for i in range(len(u)//2)
    ])
    u_approx_grad = np.array([
        local.LTriangle.gradient(element.vertices, u_approx[element.nodes]) for element in elements])
    return u_approx, u_approx_grad

def init_exact_val_grads_lshape(ed, elements, vertices):
    u_exact = np.array([
        # [val := ed.u_lshape(vertices[node, 0], vertices[node, 1]), val]
        [0, 0]
        for node in range(len(vertices))])
    u_exact_grad = np.array([mean_exact_grad(ed.u_lshape_grad, vertices[element.nodes]) for element in elements])
    return u_exact, u_exact_grad

def init_exact_val_grads_rectangle(ed, elements, vertices):
    u_exact = np.array([[0, 0] for v in vertices])
    u_grad = np.array([[0, 0] for el in elements])
    return u_exact, u_grad

def plot_solution(u, elements, vertices, u_exact=None):
    scale = 1
    u_x, u_y = u[0::2], u[1::2]

    nodes_def = np.copy(vertices)
    nodes_def[:, 0] += scale * u_x
    nodes_def[:, 1] += scale * u_y

    plt.figure(figsize=(6, 6))

    # Plot undeformed mesh (gray)
    for elem in elements:
        elem_nodes = np.append(elem.nodes[0:3], elem.nodes[0])
        plt.plot(vertices[elem_nodes, 0], vertices[elem_nodes, 1], 'k--', linewidth=0.5)

    # Plot deformed mesh (colored)
    for elem in elements:
        elem_nodes = np.append(elem.nodes[0:3], elem.nodes[0])
        plt.plot(nodes_def[elem_nodes, 0], nodes_def[elem_nodes, 1], 'r-', linewidth=1.5)

    if u_exact is not None:
        nodes_exact = np.array([
            [vertices[node, 0] + scale*u_exact[node][0], 
            vertices[node, 1] + scale*u_exact[node][1]] for node in range(len(vertices))])
        # plot exact
        for elem in elements:
            elem_nodes = np.append(elem.nodes[0:3], elem.nodes[0])
            plt.plot(nodes_exact[elem_nodes, 0], nodes_exact[elem_nodes, 1], 'b--', linewidth=1)

    for i, (x, y) in enumerate(vertices):
        plt.text(x, y, str(i), fontsize=10, color='blue',
                ha='center', va='center')

    plt.title("Mesh Deformation Visualization")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.legend(["undeformed", "deformed"])
    plt.show()

def refine_mesh(padapter, elements, vertices, bounds, domain, error_estimates, theta=0.5):
    marked_elems = padapter.mark_elements(error_estimates, theta)
    print(f'len before vertices: {len(vertices)}, elements: {len(elements)}')
    padapter.refinement(marked_elems, bounds, domain)
    vertices, elements = padapter.vertices, padapter.new_elements
    print(f'len after vertices: {len(vertices)}, elements: {len(elements)}')
    return vertices, elements

def clean_vertices(vertices, elements):
    all_nodes = set(node for element in elements for node in element.nodes)
    # print('all_nodes: ', all_nodes)
    vertices_nodes = set(i for i in range(len(vertices)))
    nodes_to_exclude = vertices_nodes - all_nodes
    print('nodes_to_exclude: ', nodes_to_exclude)
    n = len(vertices)
    for node in nodes_to_exclude:
        vertices[node] = vertices[n-1]
        vertices = np.delete(vertices, n - 1, axis=0)
        for el in elements:
            for j in range(len(el.nodes)):
                if el.nodes[j] == n-1:
                    el.nodes[j] = node
        n-=1
    # all_nodes = set(node for element in elements for node in element.nodes)
    # print('all_nodes: ', all_nodes)
    return vertices, elements