import numpy as np
from matplotlib import pyplot as plt

from norms import h1_error_elements
from padapter import PAdapter
from solver import apply_bounds_lshape, assemble_matvec, create_elements, init_approx_val_grads, init_exact_val_grads_lshape, initialize_base, plot_mesh, plot_solution, refine_mesh, solve
import triangulation as tr
import elasticdeform as elde
import system as s
import domain
import bound_cond as bc


# domain = domain.LDomain([[0, 0], [0, 3], [1, 3], [1, 1], [2, 1], [2, 0]])
domain = domain.LDomain([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
tr = tr.Triangulator(domain)
min_angle = 30
max_area = 0.1
mesh = tr.triangulate(len(domain.vertices), min_angle, max_area)
# tr.plot(mesh)
# print(mesh)
vertices_orig = mesh['vertices']
triangles_orig = mesh['triangles']

plot_mesh(vertices_orig, triangles_orig)

print('vertices: ', len(vertices_orig))
print('triangles: ', len(triangles_orig))

alpha=1000
young=2500

ed, base, k_e, m_e, base_integrals = initialize_base(young)
s = s.System(ed)
def b(x, y):
    return [0, 0]
bound_types = [bc.Type.DIRICHLET, bc.Type.NEUMANN, bc.Type.DIRICHLET, bc.Type.DIRICHLET, bc.Type.NEUMANN, bc.Type.DIRICHLET]
bound_funcs = [lambda x,y: [0,0], lambda x,y:[0,0], lambda x,y:[1,1], lambda x,y:[1,1], lambda x,y:[0,0], lambda x,y:[0,0]]



degree = 1
elements, vertices = create_elements(vertices_orig, triangles_orig, degree)
print(len(vertices))
# print(vertices)
matrix, vector = assemble_matvec(s, k_e, elements, len(vertices), b)
matrix, vector = apply_bounds_lshape(domain, bound_types, elements, vertices, bound_funcs,
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lshape(ed, elements, vertices)
u_approx, u_approx_grad = init_approx_val_grads(u, elements)
errors1 = h1_error_elements(elements, u_exact, u_approx, u_exact_grad, u_approx_grad)
plot_solution(u, elements, vertices, u_exact)


padapter = PAdapter(elements, vertices)
# error_estimates = padapter.calc_estimates(u_exact, u_approx, u_exact_grad, u_approx_grad)
vertices, elements = refine_mesh(padapter, elements, vertices, errors1)
triangles = [list(elem.nodes[0:3]) for elem in elements]
plot_mesh(vertices, triangles)
matrix, vector = assemble_matvec(s, k_e, elements, len(vertices), b)
matrix, vector = apply_bounds_lshape(domain, bound_types, elements, vertices, bound_funcs,
                            base, m_e, base_integrals, matrix, vector, alpha)
zero_rows = np.where(~np.array(matrix).any(axis=1))[0]
# print("Zero rows:", zero_rows)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lshape(ed, elements, vertices)
u_approx, u_approx_grad = init_approx_val_grads(u, elements)
print(f'lens: u_approx = {len(u_approx)}, u_approx_grad = {len(u_approx_grad)}, u_exact = {len(u_exact)}, u_exact_grad = {len(u_exact_grad)}')
errors_ph = padapter.calc_estimates(u_exact, u_approx, u_exact_grad, u_approx_grad)
plot_solution(u, elements, vertices, u_exact)


degree = 2
elements, vertices = create_elements(vertices_orig, triangles_orig, degree)
print(len(vertices))
matrix, vector = assemble_matvec(s, k_e, elements, len(vertices), b)
matrix, vector = apply_bounds_lshape(domain, bound_types, elements, vertices, bound_funcs,
                            base, m_e, base_integrals, matrix, vector, alpha)
u = solve(matrix, vector)
u_exact, u_exact_grad = init_exact_val_grads_lshape(ed, elements, vertices)
u_approx, u_approx_grad = init_approx_val_grads(u, elements)
errors2 = h1_error_elements(elements, u_exact, u_approx, u_exact_grad, u_approx_grad)
plot_solution(u, elements, vertices, u_exact)

print('~~~~~~~h1 norm~~~~')
print('first degree: ', sum(errors1))
print('second degree: ', sum(errors2))
print('ph-adaptive: ', sum(errors_ph))