import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# ПАРАМЕТРИ
# -------------------------------
b = 1.0
c = 1.0

# -------------------------------
# ТОЧНИЙ РОЗВ'ЯЗОК
# -------------------------------
def u_exact(x, eps):
    return x * (1 - x) * (1 - np.exp(-(1 - x) / eps))

def f(x, eps):
    h = 1e-5
    u_xx = (u_exact(x + h, eps) - 2 * u_exact(x, eps) + u_exact(x - h, eps)) / h**2
    u_x = (u_exact(x + h, eps) - u_exact(x - h, eps)) / (2 * h)
    return -eps * u_xx + b * u_x + c * u_exact(x, eps)

# -------------------------------
# СІТКА
# -------------------------------
def create_mesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    h = 1.0 / N
    return x, h

# -------------------------------
# ЛОКАЛЬНЕ ЧИСЛО ПЕКЛЕ І tau
# -------------------------------
def coth(z):
    return np.cosh(z) / np.sinh(z)

def tau_supg(h, eps):
    if abs(b) < 1e-14:
        return 0.0
    pe = abs(b) * h / (2.0 * eps)
    if pe < 1e-8:
        return h * h / (12.0 * eps)
    return h / (2.0 * abs(b)) * (coth(pe) - 1.0 / pe)

# -------------------------------
# СТАНДАРТНА ЛОКАЛЬНА МАТРИЦЯ
# -------------------------------
def local_matrix_standard(h, eps):
    a_diff = eps / h * np.array([[1.0, -1.0],
                                 [-1.0, 1.0]])

    a_adv = b / 2.0 * np.array([[-1.0, 1.0],
                                [-1.0, 1.0]])

    a_react = c * h / 6.0 * np.array([[2.0, 1.0],
                                      [1.0, 2.0]])
    return a_diff + a_adv + a_react

# -------------------------------
# SUPG-ДОДАТОК ДО МАТРИЦІ
# -------------------------------
def local_matrix_supg(h, eps):
    tau = tau_supg(h, eps)

    # dphi/dx для лінійних базисів на елементі
    dphi = np.array([-1.0 / h, 1.0 / h])

    # інтервалова масова матриця
    mass = h / 6.0 * np.array([[2.0, 1.0],
                               [1.0, 2.0]])

    # член tau * ∫ (b u_h') b v_h' dx
    m1 = tau * (b ** 2) / h * np.array([[1.0, -1.0],
                                        [-1.0, 1.0]])

    # член tau * ∫ (c u_h) b v_h' dx
    # інтеграл: tau * c * b * dphi_j * ∫ phi_i dx
    int_phi = np.array([h / 2.0, h / 2.0])
    m2 = np.zeros((2, 2))
    for j in range(2):
        for i in range(2):
            m2[j, i] = tau * c * b * dphi[j] * int_phi[i]

    return m1 + m2

# -------------------------------
# ЛОКАЛЬНИЙ ВЕКТОР ПРАВОЇ ЧАСТИНИ
# -------------------------------
def local_vector_standard(x1, x2, eps):
    h = x2 - x1
    xm = 0.5 * (x1 + x2)
    val = f(xm, eps)
    return val * h / 2.0 * np.array([1.0, 1.0])

def local_vector_supg(x1, x2, eps):
    h = x2 - x1
    xm = 0.5 * (x1 + x2)
    val = f(xm, eps)
    tau = tau_supg(h, eps)
    dphi = np.array([-1.0 / h, 1.0 / h])

    # tau * ∫ f b v_h' dx ≈ tau * f(xm) * b * dphi_j * h
    return tau * val * b * dphi * h

# -------------------------------
# ЗБІРКА СИСТЕМИ
# -------------------------------
def assemble_supg(N, eps):
    x, h = create_mesh(N)

    A = np.zeros((N + 1, N + 1))
    F = np.zeros(N + 1)

    for k in range(N):
        x1, x2 = x[k], x[k + 1]

        A_loc = local_matrix_standard(h, eps) + local_matrix_supg(h, eps)
        F_loc = local_vector_standard(x1, x2, eps) + local_vector_supg(x1, x2, eps)

        nodes = [k, k + 1]

        for m in range(2):
            for n in range(2):
                A[nodes[m], nodes[n]] += A_loc[m, n]
            F[nodes[m]] += F_loc[m]

    return A, F, x

# -------------------------------
# КРАЙОВІ УМОВИ
# -------------------------------
def apply_bc(A, F):
    A[0, :] = 0.0
    A[:, 0] = 0.0
    A[0, 0] = 1.0
    F[0] = 0.0

    A[-1, :] = 0.0
    A[:, -1] = 0.0
    A[-1, -1] = 1.0
    F[-1] = 0.0

    return A, F

# -------------------------------
# РОЗВ'ЯЗАННЯ
# -------------------------------
def solve_supg(N, eps):
    A, F, x = assemble_supg(N, eps)
    A, F = apply_bc(A, F)
    u = np.linalg.solve(A, F)
    return x, u

# -------------------------------
# ПОХИБКИ
# -------------------------------
def compute_errors(x, u, eps):
    u_ex = u_exact(x, eps)
    err = np.abs(u - u_ex)

    e_inf = np.max(err)
    h = x[1] - x[0]
    e_l2 = np.sqrt(np.sum(err**2) * h)

    return e_inf, e_l2

# -------------------------------
# ГРАФІК
# -------------------------------
def save_plot(x, u, eps, N, folder="plots_supg"):
    os.makedirs(folder, exist_ok=True)

    x_fine = np.linspace(0.0, 1.0, 1000)
    plt.figure(figsize=(8, 5))
    plt.plot(x_fine, u_exact(x_fine, eps), label="Exact", linewidth=2)
    plt.plot(x, u, 'o-', label="SUPG")
    plt.title(f"SUPG solution, eps={eps}, N={N}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(folder, f"supg_eps_{eps}_N_{N}.png")
    plt.savefig(filename, dpi=200)
    plt.close()

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    eps_values = [1e-1, 1e-2, 1e-3]
    N_values = [10, 20, 40, 80, 160]

    print(f"{'eps':>10} {'N':>10} {'E_inf':>15} {'E_L2':>15}")
    print("-" * 55)

    for eps in eps_values:
        for N in N_values:
            x, u = solve_supg(N, eps)
            e_inf, e_l2 = compute_errors(x, u, eps)
            print(f"{eps:10.1e} {N:10d} {e_inf:15.8e} {e_l2:15.8e}")

        save_plot(x, u, eps, N_values[-1])