import time

import numpy as np

import domain
from padapter import PAdapter
from solver import apply_bounds_lame, assemble_matvec, assemble_matvec_physical, clean_vertices, create_elements, init_approx_val_grads, init_exact_val_grads_lame, initialize_base, plot_mesh, plot_solution, refine_mesh, solve
import triangulation as tr
import system as s
import domain
import bound_cond as bc
from norms import h1_error_elements



domain = domain.CircularDomain([[0, 1], [0, 2], [2, 0], [1, 0]], 10)
# domain = domain.CircularDomain([[0, 0.5], [0, 1], [1, 0], [0.5, 0]])

tr = tr.Triangulator(domain)
min_angle = 30
max_area = 0.025
# mesh = tr.triangulate(len(domain.vertices), min_angle, max_area)

# vertices_orig = mesh['vertices']
# triangles_orig = mesh['triangles']
# plot_mesh(vertices_orig, triangles_orig)

# print('vertices: ', len(vertices_orig))
# print('triangles: ', len(triangles_orig))
# print(triangles)

alpha=1000
young=2500
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
    return [x, y]
bound_types = [bc.Type.DIRICHLET, bc.Type.NEUMANN, bc.Type.DIRICHLET, bc.Type.NEUMANN]

start = time.time()
mesh = tr.triangulate(len(domain.vertices), min_angle, max_area)
vertices_orig = mesh['vertices']
triangles_orig = mesh['triangles']
degree = 1
elements, vertices = create_elements(vertices_orig, triangles_orig, degree)
print(len(vertices))
# print(vertices)
# matrix, vector = assemble_matvec(s, k_e, elements, len(vertices), b)
matrix, vector = assemble_matvec_physical(s, ed, elements, len(vertices), b)
matrix, vector, bounds = apply_bounds_lame(domain, bound_types, elements, vertices, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lame(ed, elements, vertices)
u_approx, u_approx_grad = init_approx_val_grads(u, elements)
errors1 = h1_error_elements(elements, u_exact, u_approx, u_exact_grad, u_approx_grad)
print('deg 1 h time: ', time.time() - start)
# plot_solution(u, elements, vertices, u_exact)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))


start = time.time()
mesh_h2 = tr.triangulate(len(domain.vertices), min_angle, max_area/2)
vertices_h2 = mesh_h2['vertices']
triangles_h2 = mesh_h2['triangles']
# plot_mesh(vertices_h2, triangles_h2)
degree = 1
elements_h2, vertices_h2 = create_elements(vertices_h2, triangles_h2, degree)
# print(len(vertices_h2))
# print(vertices_h2)
# matrix, vector = assemble_matvec(s, k_e, elements_h2, len(vertices_h2), b)
matrix, vector = assemble_matvec_physical(s, ed, elements_h2, len(vertices_h2), b)
matrix, vector, bounds = apply_bounds_lame(domain, bound_types, elements_h2, vertices_h2, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lame(ed, elements_h2, vertices_h2)
u_approx, u_approx_grad = init_approx_val_grads(u, elements_h2)
errors_h2 = h1_error_elements(elements_h2, u_exact, u_approx, u_exact_grad, u_approx_grad)
# plot_solution(u, elements_h2, vertices_h2, u_exact)
print('deg 1 h/2 time: ', time.time() - start)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
print('first degree h/2: ', sum(errors_h2))

start =time.time()
mesh_h4 = tr.triangulate(len(domain.vertices), min_angle, max_area/4)
vertices_h4 = mesh_h4['vertices']
triangles_h4 = mesh_h4['triangles']
# plot_mesh(vertices_h4, triangles_h4)
degree = 1
elements_h4, vertices_h4 = create_elements(vertices_h4, triangles_h4, degree)
print(len(vertices_h4))
# print(vertices_h4)
# matrix, vector = assemble_matvec(s, k_e, elements_h4, len(vertices_h4), b)
matrix, vector = assemble_matvec_physical(s, ed, elements_h4, len(vertices_h4), b)
matrix, vector, bounds = apply_bounds_lame(domain, bound_types, elements_h4, vertices_h4, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lame(ed, elements_h4, vertices_h4)
u_approx, u_approx_grad = init_approx_val_grads(u, elements_h4)
errors_h4 = h1_error_elements(elements_h4, u_exact, u_approx, u_exact_grad, u_approx_grad)
# plot_solution(u, elements_h4, vertices_h4, u_exact)
print('deg 1 h/4 time: ', time.time() - start)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
print('first degree h/2: ', sum(errors_h2))
print('first degree h/4: ', sum(errors_h4))

start = time.time()
mesh_h8 = tr.triangulate(len(domain.vertices), min_angle, max_area/8)
vertices_h8 = mesh_h8['vertices']
triangles_h8 = mesh_h8['triangles']
# plot_mesh(vertices_h8, triangles_h8)
degree = 1
elements_h8, vertices_h8 = create_elements(vertices_h8, triangles_h8, degree)
print(len(vertices_h8))
# print(vertices_h8)
# matrix, vector = assemble_matvec(s, k_e, elements_h8, len(vertices_h8), b)
matrix, vector = assemble_matvec_physical(s, ed, elements_h8, len(vertices_h8), b)
matrix, vector, bounds = apply_bounds_lame(domain, bound_types, elements_h8, vertices_h8, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lame(ed, elements_h8, vertices_h8)
u_approx, u_approx_grad = init_approx_val_grads(u, elements_h8)
errors_h8 = h1_error_elements(elements_h8, u_exact, u_approx, u_exact_grad, u_approx_grad)
# plot_solution(u, elements_h8, vertices_h8, u_exact)
print('deg 1 h/8 time: ', time.time() - start)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
print('first degree h/2: ', sum(errors_h2))
print('first degree h/4: ', sum(errors_h4))
print('first degree h/8: ', sum(errors_h8))


start = time.time()
padapter12 = PAdapter(elements, vertices)
# error_estimates = padapter.calc_estimates(u_exact, u_approx, u_exact_grad, u_approx_grad)
vertices, elements = refine_mesh(padapter12, elements, vertices, bounds, domain, errors1, theta=0.5)
vertices, elements = clean_vertices(vertices, elements)
triangles = [list(elem.nodes[0:3]) for elem in elements]
# plot_mesh(vertices, triangles)
# matrix, vector = assemble_matvec(s, k_e, elements, len(vertices), b)
matrix, vector = assemble_matvec_physical(s, ed, elements, len(vertices), b)
zero_rows = np.where(~np.array(matrix).any(axis=1))[0]
print("Zero rows:", zero_rows)
matrix, vector, bounds = apply_bounds_lame(domain, bound_types, elements, vertices, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lame(ed, elements, vertices)
u_approx, u_approx_grad = init_approx_val_grads(u, elements)
# print(f'lens: u_approx = {len(u_approx)}, u_approx_grad = {len(u_approx_grad)}, u_exact = {len(u_exact)}, u_exact_grad = {len(u_exact_grad)}')
errors_ph1 = padapter12.calc_estimates(u_exact, u_approx, u_exact_grad, u_approx_grad)
# plot_solution(u, elements, vertices, u_exact)
print('ph 12 time: ', time.time() - start)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
# print('first degree h/2: ', sum(errors_h2))
print('p-adaptive 1-2: ', sum(errors_ph1))

start = time.time()
degree = 2
elements, vertices = create_elements(vertices_orig, triangles_orig, degree)
print(len(vertices))
triangles = [list(elem.nodes[0:3]) for elem in elements]
# plot_mesh(vertices, triangles)
# matrix, vector = assemble_matvec(s, k_e, elements, len(vertices), b)
matrix, vector = assemble_matvec_physical(s, ed, elements, len(vertices), b)
matrix, vector, bounds = apply_bounds_lame(domain, bound_types, elements, vertices, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lame(ed, elements, vertices)
u_approx, u_approx_grad = init_approx_val_grads(u, elements)
errors2 = h1_error_elements(elements, u_exact, u_approx, u_exact_grad, u_approx_grad)
# plot_solution(u, elements, vertices, u_exact)
print('deg 2 time: ', time.time() - start)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
# print('first degree h/2: ', sum(errors_h2))
# print('first degree h/4: ', sum(errors_h4))
# print('first degree h/8: ', sum(errors_h8))
print('second degree: ', sum(errors2))
print('p-adaptive 1-2: ', sum(errors_ph1))


# padapter23 = PAdapter(elements, vertices)
# vertices, elements = refine_mesh(padapter23, elements, vertices, errors1, theta=0.5)
# triangles = [list(elem.nodes[0:3]) for elem in elements]
# plot_mesh(vertices, triangles)
# # matrix, vector = assemble_matvec(s, k_e, elements, len(vertices), b)
# matrix, vector = assemble_matvec_physical(s, ed, elements, len(vertices), b)
# matrix, vector = apply_bounds_lame(domain, bound_types, elements, vertices, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
#                             base, m_e, base_integrals, matrix, vector, alpha)
# zero_rows = np.where(~np.array(matrix).any(axis=1))[0]
# # print("Zero rows:", zero_rows)
# u = solve(matrix, vector)
# u_exact, u_exact_grad = init_exact_val_grads_lame(ed, elements, vertices)
# # u_approx, u_approx_grad = init_approx_val_grads(u, elements)
# print(f'lens: u_approx = {len(u_approx)}, u_approx_grad = {len(u_approx_grad)}, u_exact = {len(u_exact)}, u_exact_grad = {len(u_exact_grad)}')
# errors_ph2 = padapter23.calc_estimates(u_exact, u_approx, u_exact_grad, u_approx_grad)
# plot_solution(u, elements, vertices, u_exact)


start =time.time()
degree = 3
elements, vertices = create_elements(vertices_orig, triangles_orig, degree)
# print(len(vertices))
triangles = [list(elem.nodes[0:3]) for elem in elements]
# plot_mesh(vertices, triangles)
# matrix, vector = assemble_matvec(s, k_e, elements, len(vertices), b)
matrix, vector = assemble_matvec_physical(s, ed, elements, len(vertices), b)
matrix, vector, bounds = apply_bounds_lame(domain, bound_types, elements, vertices, [lambda x, y: [0, 0], g1, lambda x, y: [0, 0], g2],
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lame(ed, elements, vertices)
u_approx, u_approx_grad = init_approx_val_grads(u, elements)
errors3 = h1_error_elements(elements, u_exact, u_approx, u_exact_grad, u_approx_grad)
# plot_solution(u, elements, vertices, u_exact)
print('deg 3 time: ', time.time() - start)
print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
# print('first degree h/2: ', sum(errors_h2))
# print('first degree h/4: ', sum(errors_h4))
# print('first degree h/8: ', sum(errors_h8))
print('second degree: ', sum(errors2))
# print('p-adaptive 1-2: ', sum(errors_ph1))
print('third degree: ', sum(errors3))
# print('ph-adaptive 2-3: ', sum(errors_ph2))




# notes:
# support 3rd degree
# check how gradient for local triangle is calculated, if it supports higher degrees
# define boundary conditions for L-shaped domain and check

# future:
# use p + mortar (or whatever) instead pf ph