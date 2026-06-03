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
    return x * (1.0 - x) * (1.0 - np.exp(-(1.0 - x) / eps))

def f(x, eps):
    h = 1e-5
    u_xx = (u_exact(x + h, eps) - 2.0 * u_exact(x, eps) + u_exact(x - h, eps)) / h**2
    u_x = (u_exact(x + h, eps) - u_exact(x - h, eps)) / (2.0 * h)
    return -eps * u_xx + b * u_x + c * u_exact(x, eps)

# -------------------------------
# СІТКА
# -------------------------------
def create_mesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    h = 1.0 / N
    return x, h

# -------------------------------
# ДОПОМІЖНІ ФУНКЦІЇ
# -------------------------------
def safe_bernoulli(z):
    """
    Функція Бернуллі:
    B(z) = z / (exp(z) - 1)

    Для малих z використовуємо розклад,
    щоб уникнути втрати точності.
    """
    if abs(z) < 1e-8:
        return 1.0 - z / 2.0 + z**2 / 12.0
    return z / (np.exp(z) - 1.0)

# -------------------------------
# ЛОКАЛЬНА МАТРИЦЯ FITTED-СХЕМИ
# -------------------------------
def local_matrix_fitted(h, eps):
    """
    Експоненційно-узгоджена (fitted) локальна матриця
    для дифузійно-адвективної частини.

    Використовується стандартна fitted-апроксимація:
        eps/h * [[B(-Pe*2), -B(Pe*2)],
                 [-B(-Pe*2),  B(Pe*2)]]

    Оскільки:
        z = b*h/eps
    """

    z = b * h / eps

    bp = safe_bernoulli(z)
    bm = safe_bernoulli(-z)

    # fitted внесок для дифузія + адвекція
    a_fit = (eps / h) * np.array([
        [bm, -bp],
        [-bm, bp]
    ])

    # стандартний масовий внесок для реакції
    a_react = c * h / 6.0 * np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ])

    return a_fit + a_react

# -------------------------------
# ЛОКАЛЬНИЙ ВЕКТОР
# -------------------------------
def local_vector(x1, x2, eps):
    """
    Найпростіша квадратура середньою точкою.
    """
    h = x2 - x1
    xm = 0.5 * (x1 + x2)
    val = f(xm, eps)
    return val * h / 2.0 * np.array([1.0, 1.0])

# -------------------------------
# ЗБІРКА СИСТЕМИ
# -------------------------------
def assemble_fitted(N, eps):
    x, h = create_mesh(N)

    A = np.zeros((N + 1, N + 1))
    F = np.zeros(N + 1)

    for k in range(N):
        x1, x2 = x[k], x[k + 1]

        A_loc = local_matrix_fitted(h, eps)
        F_loc = local_vector(x1, x2, eps)

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
    # u(0) = 0
    A[0, :] = 0.0
    A[:, 0] = 0.0
    A[0, 0] = 1.0
    F[0] = 0.0

    # u(1) = 0
    A[-1, :] = 0.0
    A[:, -1] = 0.0
    A[-1, -1] = 1.0
    F[-1] = 0.0

    return A, F

# -------------------------------
# РОЗВ'ЯЗАННЯ
# -------------------------------
def solve_fitted(N, eps):
    A, F, x = assemble_fitted(N, eps)
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
def save_plot(x, u, eps, N, folder="plots_fitted"):
    os.makedirs(folder, exist_ok=True)

    x_fine = np.linspace(0.0, 1.0, 1000)
    plt.figure(figsize=(8, 5))
    plt.plot(x_fine, u_exact(x_fine, eps), label="Exact", linewidth=2)
    plt.plot(x, u, "o-", label="Fitted")
    plt.title(f"Fitted solution, eps={eps}, N={N}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(folder, f"fitted_eps_{eps}_N_{N}.png")
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
            x, u = solve_fitted(N, eps)
            e_inf, e_l2 = compute_errors(x, u, eps)
            print(f"{eps:10.1e} {N:10d} {e_inf:15.8e} {e_l2:15.8e}")

        save_plot(x, u, eps, N_values[-1])