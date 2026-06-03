from abc import ABC, abstractmethod
import math

import numpy as np
import scipy
import sympy as sp
import scipy.integrate as scin
from sympy.core import numbers

local_triangle = [[0,0], [1,0], [0,1], [0.5, 0], [0.5, 0.5], [0, 0.5]]
local_det = 1

ksi = sp.symbols('ksi')
eta = sp.symbols('eta')
x = sp.symbols('x')
y = sp.symbols('y')

class Local(ABC):
    @abstractmethod
    def system(self, element):
        pass
    def jacobi_sys(self, system):
        pass
    def jacobi(self, figure):
        # for squares?
        pass
    def base(self, approx_degree=1):
        pass
    @abstractmethod
    def matrix(self):
        pass
    def integrate(self, f):
        pass


class LTriangle(Local):
    @staticmethod
    def system(triangle):
        x = triangle[0][0] + (triangle[1][0] - triangle[0][0]) * ksi + (triangle[2][0] - triangle[0][0]) * eta
        y = triangle[0][1] + (triangle[1][1] - triangle[0][1]) * ksi + (triangle[2][1] - triangle[0][1]) * eta
        return (x, y)

    @staticmethod
    def ksieta_from_xy(triangle, x, y):
        a = [[triangle[1][0] - triangle[0][0], triangle[2][0] - triangle[0][0]],
                  [triangle[1][1] - triangle[0][1], triangle[2][1] - triangle[0][1]]]
        b = [x - triangle[0][0],
            y - triangle[0][1]]
        return np.linalg.solve(a, b)

    def jacobi_sys(self, system):
        x = system[0]
        y = system[1]
        jacobi = [[sp.diff(x, 'ksi'), sp.diff(y, 'ksi')], [sp.diff(x, 'eta'), sp.diff(y, 'eta')]]
        return jacobi[0][0] * jacobi[1][1] - jacobi[0][1] * jacobi[1][0]

    @staticmethod
    def jacobi(triangle):
        det = (triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1])
        det -= (triangle[2][0] - triangle[0][0]) * (triangle[1][1] - triangle[0][1])
        return det*2

    @staticmethod
    def base(approx_degree=1):
        if approx_degree == 1:
            base = [1 - ksi - eta, ksi, eta]
        elif approx_degree == 2:
            base = [(1 - ksi - eta) * (1 - 2 * ksi - 2 * eta),
                    ksi * (2 * ksi - 1),
                    eta * (2 * eta - 1),
                    4 * ksi * (1 - ksi - eta),
                    4 * ksi * eta,
                    4 * eta * (1 - ksi - eta)]
        elif approx_degree == 3:
            L1, L2, L3 = 1 - ksi - eta, ksi, eta
            base = [
                1/2*L1*(3*L1-1)*(3*L1-2),
                1/2*L2*(3*L2-1)*(3*L2-2),
                1/2*L3*(3*L3-1)*(3*L3-2),
                9/2*L1*L2*(3*L1-1),
                9/2*L1*L2*(3*L2-1),
                9/2*L2*L3*(3*L2-1),
                9/2*L2*L3*(3*L3-1),
                9/2*L1*L3*(3*L3-1),
                9/2*L1*L3*(3*L1-1),
                27*L1*L2*L3,
            ]
        else:
            raise ('degree not impemented')
        return base
    @staticmethod
    def gradient(vertices, u_values):
        area = 0.5*abs(LTriangle.jacobi(vertices))
        x1,y1 = vertices[0]
        x2,y2 = vertices[1]
        x3,y3 = vertices[2]
        J = np.array([[x2-x1, x3-x1], [y2-y1, y3-y1]])
        invJ = np.linalg.inv(J)
        if len(u_values) == 3:  # degree 1
            grad_ref = np.array([[-1,-1], [1, 0], [0, 1]])
            grad_phys = grad_ref @ invJ.T
            return sum(u_values[i] * grad_phys[i] for i in range(3))
        else:
            degree = {6: 2, 9: 3, 10: 3}[len(u_values)]
            grad_ref = _grad_ref_at_centroid(degree)
            grad_phys = grad_ref @ invJ.T
            return sum(u_values[i] * grad_phys[i] for i in range(len(u_values)))
    
    @staticmethod
    def integrate(f):
        # if not callable(f):
        #     const_val = float(f) if isinstance(f, numbers.Number) else f
        #     f = lambda ksi, eta: const_val
        # f = f(ksi, eta)/
        # else:
        f = sp.lambdify((ksi, eta), f)
        return scin.dblquad(f, 0, 1, lambda ksi: 0, lambda ksi: 1-ksi)[0]
    
    @staticmethod
    def integrate_gamma(f):
        return [LTriangle.integrate_gamma_i(f, i) for i in range(3)]
    @staticmethod
    def integrate_gamma_i(f, idx):
        if callable(f):
            f_symp = sp.sympify((ksi, eta), f)
        else:
            f_symp = f
        t = sp.symbols('t')
        start, end = local_triangle[idx], local_triangle[(idx+1)%3]
        ksi_t = start[0] + t * (end[0] - start[0])
        eta_t = start[1] + t * (end[1] - start[1])
        jacobi = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        f_t = f_symp.subs('ksi', ksi_t).subs('eta', eta_t)
        return scipy.integrate.quad(sp.lambdify(t, f_t), 0, 1)[0] * jacobi

    @staticmethod
    def base_integrals(base):
        base_integrals = []
        for b in base:
            base_integrals.append(LTriangle.integrate_gamma(b))
        return base_integrals

    @staticmethod
    def integrate_gamma_xy(p, system, phi, gamma_idx):
        px = sp.sympify(p(x, y)[0])
        py = sp.sympify(p(x, y)[1])
        px_ = px.subs('x', system[0]).subs('y', system[1])
        py_ = py.subs('x', system[0]).subs('y', system[1])
        return [LTriangle.integrate_gamma_i(px_*phi, gamma_idx),
                LTriangle.integrate_gamma_i(py_*phi, gamma_idx)] 



_grad_cache = {}

def _grad_ref_at_centroid(degree):
    if degree not in _grad_cache:
        ksi_c, eta_c = sp.Rational(1,3), sp.Rational(1,3)
        base = LTriangle.base(degree)
        _grad_cache[degree] = np.array([
            [float(sp.diff(b, ksi).subs(ksi, ksi_c).subs(eta, eta_c)),
             float(sp.diff(b, eta).subs(ksi, ksi_c).subs(eta, eta_c))]
            for b in base
        ])
    return _grad_cache[degree]


# ltr = LTriangle()
# triangle = [[0, 0], [2, 1], [-1, 2]]
# system = ltr.system(triangle)
# print('system:')
# print(system)
#
# print('jacobi from system: ', local_jacobi_sys(system))
# print('jacobi from triangle: ', local_jacobi_tri(triangle))

# f_lamb = sp.lambdify([ksi, eta], f(ksi, eta))
# res = scin.dblquad(lambda ksi, eta: f_lamb(ksi, eta), 0, 1, lambda ksi: 0, lambda ksi: 1-ksi)[0]
# print(res)