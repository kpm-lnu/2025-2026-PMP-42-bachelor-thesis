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
    x = np.linspace(0, 1, N + 1)
    h = 1.0 / N
    return x, h

# -------------------------------
# ЛОКАЛЬНА МАТРИЦЯ
# -------------------------------
def local_matrix(h, eps):
    A_diff = eps / h * np.array([[1, -1],
                                 [-1, 1]])

    A_adv = b / 2 * np.array([[-1, 1],
                              [-1, 1]])

    A_react = c * h / 6 * np.array([[2, 1],
                                    [1, 2]])

    return A_diff + A_adv + A_react

# -------------------------------
# ЛОКАЛЬНИЙ ВЕКТОР
# -------------------------------
def local_vector(x1, x2, eps):
    xm = (x1 + x2) / 2
    val = f(xm, eps)
    return val * (x2 - x1) / 2 * np.array([1, 1])

# -------------------------------
# ЗБІРКА СИСТЕМИ
# -------------------------------
def assemble(N, eps):
    x, h = create_mesh(N)

    A = np.zeros((N + 1, N + 1))
    F = np.zeros(N + 1)

    for i in range(N):
        x1, x2 = x[i], x[i + 1]

        A_loc = local_matrix(h, eps)
        F_loc = local_vector(x1, x2, eps)

        nodes = [i, i + 1]

        for m in range(2):
            for n in range(2):
                A[nodes[m], nodes[n]] += A_loc[m, n]
            F[nodes[m]] += F_loc[m]

    return A, F, x

# -------------------------------
# КРАЙОВІ УМОВИ
# -------------------------------
def apply_bc(A, F):
    A[0, :] = 0
    A[:, 0] = 0
    A[0, 0] = 1
    F[0] = 0

    A[-1, :] = 0
    A[:, -1] = 0
    A[-1, -1] = 1
    F[-1] = 0

    return A, F

# -------------------------------
# РОЗВ'ЯЗАННЯ
# -------------------------------
def solve(N, eps):
    A, F, x = assemble(N, eps)
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
def save_plot(x, u, eps, N, folder="plots"):
    os.makedirs(folder, exist_ok=True)

    x_fine = np.linspace(0, 1, 1000)
    plt.figure(figsize=(8, 5))
    plt.plot(x_fine, u_exact(x_fine, eps), label="Exact", linewidth=2)
    plt.plot(x, u, 'o-', label="FEM")
    plt.title(f"FEM solution, eps={eps}, N={N}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(folder, f"fem_eps_{eps}_N_{N}.png")
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
            x, u = solve(N, eps)
            e_inf, e_l2 = compute_errors(x, u, eps)

            print(f"{eps:10.1e} {N:10d} {e_inf:15.8e} {e_l2:15.8e}")

        # збережемо один графік для кожного eps
        save_plot(x, u, eps, N_values[-1])