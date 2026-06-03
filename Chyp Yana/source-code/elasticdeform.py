import math

import sympy as sp
import numpy as np

import local

x = sp.symbols('x')
y =  sp.symbols('y')
ksi = sp.symbols('ksi')
eta = sp.symbols('eta')

class ElasticDeform:
    def __init__(self, base_1, base_2, base_3, E=5, nu=0.25):
        # for degree = 1
        self.E = E
        self.nu = nu
        self.lame1 = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.lame2 = self.E / (2 * (1 + self.nu))
        # plane stress
        self.D = [[E/(1-nu**2) * element for element in row] for row in [[ 1, nu,        0],
                                                                    [nu,  1,        0],
                                                                    [ 0,  0, (1-nu)/2]]]
        # plane strain
        # self.D = [[ E/((1+nu)*(1-2*nu)) * element for element in row] for row in [[1-nu, nu, 0],
        #                                                                           [nu, 1-nu, 0],
        #                                                                           [0, 0, (1-2*nu)/2]]]
        L = [[self.d_dksi,         0],
             [        0, self.d_deta],
             [self.d_deta, self.d_dksi]]
        # self.N = [[base[0],       0, base[1],       0, base[2],       0],
        #      [      0, base[0],       0, base[1],       0, base[2]]]
        self.N_1 = [[], []]
        for b in base_1:
            self.N_1[0].append(b)
            self.N_1[0].append(0)
        for b in base_1:
            self.N_1[1].append(0)
            self.N_1[1].append(b)
        # print("N_1 = ", self.N_1)

        self.N_2 = [[], []]
        for b in base_2:
            self.N_2[0].append(b)
            self.N_2[0].append(0)
        for b in base_2:
            self.N_2[1].append(0)
            self.N_2[1].append(b)

        self.N_3 = [[], []]
        for b in base_3:
            self.N_3[0].append(b)
            self.N_3[0].append(0)
        for b in base_3:
            self.N_3[1].append(0)
            self.N_3[1].append(b)
        # print("N_2 = ", self.N_2)
        # B = L*N
        # self.B = [[self.d_dksi(base[0]),                  0, self.d_dksi(base[1]),                  0, self.d_dksi(base[2]),                  0],
        #      [                 0, self.d_deta(base[0]),                  0, self.d_deta(base[1]),                  0, self.d_deta(base[2])],
        #      [self.d_deta(base[0]), self.d_dksi(base[0]), self.d_deta(base[1]), self.d_dksi(base[1]), self.d_deta(base[2]), self.d_dksi(base[2])]]
        self.B_1 = [[], [], []]
        for b in base_1:
            self.B_1[0].append(self.d_dksi(b))
            self.B_1[0].append(0)
        for b in base_1:
            self.B_1[1].append(0)
            self.B_1[1].append(self.d_deta(b))
        for b in base_1:
            self.B_1[2].append(self.d_deta(b))
            self.B_1[2].append(self.d_dksi(b))
        # print("B = ", self.B_1)

        self.B_2 = [[], [], []]
        for b in base_2:
            self.B_2[0].append(self.d_dksi(b))
            self.B_2[0].append(0)
        for b in base_2:
            self.B_2[1].append(0)
            self.B_2[1].append(self.d_deta(b))
        for b in base_2:
            self.B_2[2].append(self.d_deta(b))
            self.B_2[2].append(self.d_dksi(b))
        # print("B = ", self.B_2)
        # BT = np.transpose(self.B)
        # BT * D = [ 6x3 ]
        # BT * D * B = [ 6x6 ]
        # K_e = integral(BT*D*B)_over(V_e) = [ 6x6 ]

        # sigma = D * epsilon
    def u_lame(self, x, y):
        r = math.sqrt(x ** 2 + y ** 2)
        return r / (6 * (self.lame1 + self.lame2)) + 2 / (3 * self.lame2 * r)

    def u_lame_grad(self, x, y):
        r = math.sqrt(x**2 + y**2)
        return [
            2*x / (6 * (self.lame1 + self.lame2)) - 4*x / (3 * self.lame2 * r**2),
            2*y / (6 * (self.lame1 + self.lame2)) - 4*y / (3 * self.lame2 * r**2)
        ]
    
    def u_lshape(self, x, y):
        r = math.sqrt(x**2 + y**2)
        epsilon = 0.001
        if r<=epsilon:
            return 0
        theta = math.asin(y/r)
        # if x<0: theta = math.pi - theta
        k_max = 10
        j = 1
        sum = 0
        mu = 2/3
        add = r**mu * math.sin(mu*theta)
        while abs(add) > epsilon and j<k_max:
            sum += add
            j+=1
            mu = 2/3*(2*j-1)
            add = r**mu * math.sin(mu*theta)
        print(f'for ({x}, {y}) theta {theta} iter {j}')
        return sum

    def u_lshape_grad(self, x, y):
        r = math.sqrt(x**2 + y**2)
        theta = math.asin(y/r)
        # if x<0: theta = math.pi - theta
        sin, cos = math.sin(2/3*theta), math.cos(2/3*theta)
        # return [
        #     2/3 * r**(-4/3) * (x*sin - y*cos),
        #     2/3 * r**(-4/3) * (y*sin + x*cos)
        # ]
        return [0, 0]

    def calculate_stress(self, D, strain):
        # sigma = D * strain = [ 3x3 ] * [ 3 ] = [ 3 ]
        return D*strain

    def calculate_strain(self, B, u_e):
        # epsilon = B * u_e = [ 3x6] * [ 6 ] = [ 3 ]
        return B*u_e

    def d_dksi(self, func):
        return sp.diff(func, ksi)
    def d_deta(self, func):
        return sp.diff(func, eta)


    def stiffness_matrix(self):
        B1T = np.transpose(self.B_1)
        # BT*D
        # n = len(BT)
        # m = len(self.D[0])
        # k = len(BT[0])
        prod = np.matmul(np.matmul(B1T, self.D), self.B_1)
        k_e1 = [[local.LTriangle.integrate(el) for el in row] for row in prod]

        B2T = np.transpose(self.B_2)
        prod = np.matmul(np.matmul(B2T, self.D), self.B_2)
        k_e2 = [[local.LTriangle.integrate(el) for el in row] for row in prod]
        return k_e1, k_e2

    def _k_e_physical(self, vertices, degree):
        x1,y1=vertices[0]; x2,y2=vertices[1]; x3,y3=vertices[2]
        J = np.array([[x2-x1, x3-x1],[y2-y1, y3-y1]], dtype=float)
        Jinv = np.linalg.inv(J)
        detJ = np.linalg.det(J)
        a, b = Jinv[0,0], Jinv[0,1]
        c, d = Jinv[1,0], Jinv[1,1]
        base = local.LTriangle.base(degree)
        n = len(base)
        B = np.zeros((3, 2*n), dtype=object)
        for i, phi in enumerate(base):
            dk = sp.diff(phi, ksi)
            de = sp.diff(phi, eta)
            B[0, 2*i]   = a*dk + b*de
            B[1, 2*i+1] = c*dk + d*de
            B[2, 2*i]   = c*dk + d*de
            B[2, 2*i+1] = a*dk + b*de
        prod = np.matmul(np.matmul(B.T, self.D), B)
        return [[detJ * local.LTriangle.integrate(el) for el in row] for row in prod]

    def stiffness_matrix_physical(self, elements):
        k_e1 = [self._k_e_physical(elem.vertices[:3], 1) for elem in elements if elem.approx == 1]
        k_e2 = [self._k_e_physical(elem.vertices[:3], 2) for elem in elements if elem.approx == 2]
        k_e3 = [self._k_e_physical(elem.vertices[:3], 3) for elem in elements if elem.approx == 3]
        return k_e1, k_e2, k_e3

    def mass_matrix(self):
        N1T = np.transpose(self.N_1)
        prod = np.matmul(N1T, self.N_1)
        m_e1 = [[local.LTriangle.integrate_gamma(el) for el in row] for row in prod]
        N2T = np.transpose(self.N_2)
        prod = np.matmul(N2T, self.N_2)
        m_e2 = [[local.LTriangle.integrate_gamma(el) for el in row] for row in prod]
        N3T = np.transpose(self.N_3)
        prod = np.matmul(N3T, self.N_3)
        m_e3 = [[local.LTriangle.integrate_gamma(el) for el in row] for row in prod]
        return m_e1, m_e2, m_e3

    def load_vector(self, b, element):
        # r_e = integral(NT*b)_over(V_e) + integral(NT*p)_over(S_e)
        # b = [b_x, b_y]; p = [p_x, p_y]
        system = local.LTriangle.system(element.vertices)
        bksi = sp.sympify(b(x, y)[0]).subs('x', system[0]).subs('y', system[1])
        beta = sp.sympify(b(x, y)[1]).subs('x', system[0]).subs('y', system[1])
        # NT*b = [ 6x2 ] * [ 2 ] = [ 6 ]
        N1T = np.transpose(self.N_1)
        NTb = np.matvec(N1T, [bksi, beta])
        r_e1 = [local.LTriangle.integrate(el) for el in NTb]
        N2T = np.transpose(self.N_2)
        NTb = np.matvec(N2T, [bksi, beta])
        r_e2 = [local.LTriangle.integrate(el) for el in NTb]
        N3T = np.transpose(self.N_3)
        NTb = np.matvec(N3T, [bksi, beta])
        r_e3 = [local.LTriangle.integrate(el) for el in NTb]
        return r_e1, r_e2, r_e3

# base = local.LTriangle.base()
# eldef = ElasticDeform(base)
# k_e = eldef.stiffness_matrix()
#
# for row in k_e:
#     print('[' + ', '.join([f"{el:.4f}" for el in row]) + ']' )
