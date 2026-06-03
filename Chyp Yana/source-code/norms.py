
import math

import numpy as np


def l2_norm(phi):
    norm = 0
    for value in phi:
        norm += value**2
    return math.sqrrt(norm)

def l2_norm_diff(u_exact, u_approx):
    diff = [ math.sqrt((u_exact[i][0] - u_approx[i][0])**2 + (u_exact[i][1] - u_approx[i][1])**2) for i in range(len(u_approx))]
    return l2_norm(diff)

def h1_norm(phi, phi_grad):
    norm = 0
    for i in range(len(phi)):
        norm += phi[i][0]**2 + phi[i][1]**2
    for i in range(len(phi_grad)):
        norm += np.sum(np.array(phi_grad[i])**2)
    return math.sqrt(norm)

def h1_norm_diff(u_exact, u_approx, u_exact_grad, u_approx_grad):
    diff = [[u_exact[i][0] - u_approx[i][0], u_exact[i][1] - u_approx[i][1]] for i in range(len(u_exact))]
    diff_grad = [u_exact_grad[i] - u_approx_grad[i] for i in range(len(u_exact_grad))]
    return h1_norm(diff, diff_grad)

def mean_exact_grad(exact_grad, vertices):
    xc, yc = np.mean(vertices[:3,0]), np.mean(vertices[:3,1])
    grad = np.array(exact_grad(xc, yc))
    return grad

def h1_error_elements(elements, u_exact, u_approx, u_exact_grad, u_approx_grad):
    errors = []
    for i, element in enumerate(elements):
        corner_nodes = element.nodes[:3]
        u_e = u_exact[corner_nodes]
        u_a = u_approx[corner_nodes]
        norm = h1_norm_diff(u_e, u_a, u_exact_grad[i], u_approx_grad[i])
        errors.append(norm)
    return errors
