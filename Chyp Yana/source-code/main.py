import numpy as np
from matplotlib import pyplot as plt

from norms import h1_error_elements
from solver import apply_bounds_rectangle, assemble_matvec_physical, create_elements, init_approx_val_grads, init_exact_val_grads_rectangle, initialize_base, plot_mesh, plot_solution, solve
import triangulation as tr
import element
import elasticdeform as elde
import local
import system as s
import domain
import bound_cond as bc

def b(x, y):
    return [0, 0]

domain = domain.Domain([[-2, 0], [-2, 1], [2, 1], [2, 0]])
# domain = domain.Domain([[0, 0], [0, 1], [1, 1], [1, 0]])
tr = tr.Triangulator(domain)
min_angle = 30
max_area = 0.05
mesh = tr.triangulate(len(domain.vertices), min_angle, max_area)
vertices_orig = mesh['vertices']
triangles_orig = mesh['triangles']
plot_mesh(vertices_orig, triangles_orig)

alpha=1000
young=5000
ed, base, k_e, m_e, base_integrals = initialize_base(young=young)

s = s.System(ed)
def b(x, y):
    return [0, 0]
def f1(x, y):
    return [0, 0]
def f2(x, y):
    return [0, 0]
def g1(x, y):
    return [0, 0]
def g2(x, y):
    return [0, 0]

bound_types = [bc.Type.DIRICHLET, bc.Type.NEUMANN, bc.Type.DIRICHLET, bc.Type.NEUMANN]

degree = 1
elements, vertices = create_elements(vertices_orig, triangles_orig, degree)
print(len(vertices))
matrix, vector = assemble_matvec_physical(s, ed, elements, len(vertices), b)
matrix, vector, bounds = apply_bounds_rectangle(domain, bound_types, elements, vertices, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_rectangle(ed, elements, vertices)
u_approx, u_approx_grad = init_approx_val_grads(u, elements)
errors1 = h1_error_elements(elements, u_exact, u_approx, u_exact_grad, u_approx_grad)
plot_solution(u, elements, vertices, u_exact)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))

mesh_h2 = tr.triangulate(len(domain.vertices), min_angle, max_area/2)
vertices_h2 = mesh_h2['vertices']
triangles_h2 = mesh_h2['triangles']
plot_mesh(vertices_h2, triangles_h2)
degree = 1
elements_h2, vertices_h2 = create_elements(vertices_h2, triangles_h2, degree)
print(len(vertices_h2))
matrix, vector = assemble_matvec_physical(s, ed, elements_h2, len(vertices_h2), b)
matrix, vector, bounds = apply_bounds_rectangle(domain, bound_types, elements_h2, vertices_h2, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_rectangle(ed, elements_h2, vertices_h2)
u_approx, u_approx_grad = init_approx_val_grads(u, elements_h2)
errors_h2 = h1_error_elements(elements_h2, u_exact, u_approx, u_exact_grad, u_approx_grad)
plot_solution(u, elements_h2, vertices_h2, u_exact)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
print('first degree h/2: ', sum(errors_h2))


mesh_h4 = tr.triangulate(len(domain.vertices), min_angle, max_area/4)
vertices_h4 = mesh_h4['vertices']
triangles_h4 = mesh_h4['triangles']
plot_mesh(vertices_h4, triangles_h4)
degree = 1
elements_h4, vertices_h4 = create_elements(vertices_h4, triangles_h4, degree)
print(len(vertices_h4))
matrix, vector = assemble_matvec_physical(s, ed, elements_h4, len(vertices_h4), b)
matrix, vector, bounds = apply_bounds_rectangle(domain, bound_types, elements_h4, vertices_h4, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_rectangle(ed, elements_h4, vertices_h4)
u_approx, u_approx_grad = init_approx_val_grads(u, elements_h4)
errors_h4 = h1_error_elements(elements_h4, u_exact, u_approx, u_exact_grad, u_approx_grad)
plot_solution(u, elements_h4, vertices_h4, u_exact)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
print('first degree h/2: ', sum(errors_h2))
print('first degree h/4: ', sum(errors_h4))


mesh_h8 = tr.triangulate(len(domain.vertices), min_angle, max_area/8)
vertices_h8 = mesh_h8['vertices']
triangles_h8 = mesh_h8['triangles']
plot_mesh(vertices_h8, triangles_h8)
degree = 1
elements_h8, vertices_h8 = create_elements(vertices_h8, triangles_h8, degree)
print(len(vertices_h8))
matrix, vector = assemble_matvec_physical(s, ed, elements_h8, len(vertices_h8), b)
matrix, vector, bounds = apply_bounds_rectangle(domain, bound_types, elements_h8, vertices_h8, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_rectangle(ed, elements_h8, vertices_h8)
u_approx, u_approx_grad = init_approx_val_grads(u, elements_h8)
errors_h8 = h1_error_elements(elements_h8, u_exact, u_approx, u_exact_grad, u_approx_grad)
plot_solution(u, elements_h8, vertices_h8, u_exact)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
print('first degree h/2: ', sum(errors_h2))
print('first degree h/4: ', sum(errors_h4))
print('first degree h/8: ', sum(errors_h8))

# degree = 1
# elements = []
# for triangle in triangles:
#     elements.append(element.Element(3, triangle, vertices, degree))
# for element in elements:
#     vertices = element.add_points(vertices)
    # print(element.nodes)


# base = local.LTriangle.base(degree)
# print("base = ", base)
# ed = elde.ElasticDeform(base, 100)
# NT = np.transpose(ed.N)
# k_e = ed.stiffness_matrix()
# print("k_e:")
# for row in k_e:
#     print('[' + ', '.join([f"{el:.4f}" for el in row]) + ']' )
# m_e = ed.mass_matrix()
# print("m_e:")
# for row in m_e:
    # print('[' + ', '.join([el for el in row]) + ']' )
    # print(el for el in row)


# s = s.System(ed)
# matrix = s.assemble_matrix(elements, k_e, len(vertices))
# for row in matrix:
#     print('[' + ', '.join([f"{el:.4f}" for el in row]) + ']' )

# r_e = ed.load_vector(b, p, elements[0])
# print('load vector r_e: ')
# print(r_e)

# vector = s.assemble_vector(elements, len(vertices), b)
# print('global load vector: ')
# for el in vector:
#     print(el)

# bounds = domain.get_bounds()
# bound_cond_types = [bc.Type.DIRICHLET, bc.Type.NEUMANN, bc.Type.NEUMANN, bc.Type.NEUMANN]
# bound_nodes = []
# bound_elems = []
# i = 0
# for bound in bounds:
#     nodes_at_bound = domain.find_nodes_at_bound(bound, vertices, type)
#     # print('nodes at bound: ', nodes_at_bound)
#     bound_nodes.append(nodes_at_bound)
#     bound_elems.append(domain.find_elems_at_bound(nodes_at_bound, elements))
#     i+=1
    # print('elems at bound')
    # for el in bound_elems[-1]:
    #     print(el.nodes)


# bconds = bc.BoundaryConds()

# bconds.applyNeumann(NT, vector, g1, bound_nodes[1], bound_elems[1])
# bconds.applyNeumann(NT, vector, g2, bound_nodes[3], bound_elems[3])
# # bconds.applyNeumann(NT, vector, f2, bound_nodes[3], bound_elems[3])
# bconds.applyDirichlet(matrix, vector, bound_nodes[0], vertices, f1)
# bconds.applyDirichlet(matrix, vector, bound_nodes[2], vertices, f2)

# for row in matrix:
#     print('[' + ', '.join([f"{el:.4f}" for el in row]) + ']' )
# for el in vector:
#     print(el)

# u = np.linalg.solve(matrix, vector)
# print('u = ')
# print('[' + ', '.join([f"{u[i]:.8f}" for i in range(len(u)) if i%2==0]) + ']' )
# print('[' + ', '.join([f"{u[i]:.4f}" for i in range(len(u)) if i%2==1]) + ']' )




# scale = 1.0  # deformation scale factor (tune as needed)
# # Extract displacements
# u_x = u[0::2]
# u_y = u[1::2]

# # Compute deformed coordinates
# nodes_def = np.copy(vertices)
# nodes_def[:, 0] += scale * u_x
# nodes_def[:, 1] += scale * u_y

# # --- Visualization ---
# plt.figure(figsize=(6, 6))

# # Plot undeformed mesh (gray)
# for elem in triangles:
#     elem_nodes = np.append(elem, elem[0])  # close the polygon
#     plt.plot(vertices[elem_nodes, 0], vertices[elem_nodes, 1], 'k--', linewidth=0.5)

# # Plot deformed mesh (colored)
# for elem in triangles:
#     elem_nodes = np.append(elem, elem[0])
#     plt.plot(nodes_def[elem_nodes, 0], nodes_def[elem_nodes, 1], 'r-', linewidth=1.5)

# for i, (x, y) in enumerate(vertices):
#     plt.text(x, y, str(i), fontsize=10, color='blue',
#              ha='center', va='center')

# plt.title("Mesh Deformation Visualization")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis('equal')
# plt.legend(["undeformed", "deformed"])
# plt.show()