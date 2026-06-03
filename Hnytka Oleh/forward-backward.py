import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметри моделі
s, L, alpha0, lam2, pi1, phi0 = 0.8, 50.0, 0.0031, 0.0007, 0.03, 0.4
r, K, pi_par, lam, lam0, phi, gamma1, pi2 = 0.5, 100.0, 0.004, 0.007, 0.4, 0.006, 0.0002, 0.09

# Ваги цільового функціоналу
omega1, omega2 = 1.0, 10.0

# Часова дискретизація
T, n = 100.0, 200
t = np.linspace(0, T, n)
h = T / (n - 1)

# Початкові умови
y0 = [20.0, 10.0, 5.0, 1.0]

# Кількість сегментів для кусково-сталого керування
m = 5
seg_size = n // m

def state_rhs(t, state, u):
    """Права частина системи стану."""
    B, N, P, E = state
    dB = s*B*(1 - B/L) - alpha0*(1 - u)*B*N - lam2*B**2*P + pi1*phi0*E
    dN = r*N*(1 - N/K) + pi_par*alpha0*(1 - u)*B*N
    dP = lam*N - lam0*P - pi2*gamma1*P*E
    dE = phi*(L - B) - phi0*E - gamma1*P*E
    return [dB, dN, dP, dE]

def adjoint_rhs(t, psi, B, N, P, E, u):
    """Права частина спряженої системи."""
    ps1, ps2, ps3, ps4 = psi
    d1 = (-omega1
          - ps1*(s*(1 - 2*B/L) - alpha0*(1-u)*N - 2*lam2*B*P)
          - ps2*pi_par*alpha0*(1-u)*N
          + ps4*phi)
    d2 = (ps1*alpha0*(1-u)*B
          - ps2*(r*(1 - 2*N/K) + pi_par*alpha0*(1-u)*B)
          - ps3*lam)
    d3 = (ps1*lam2*B**2
          + ps3*(lam0 + pi2*gamma1*E)
          + ps4*gamma1*E)
    d4 = (-ps1*pi1*phi0
          + ps3*pi2*gamma1*P
          + ps4*(phi0 + gamma1*P))
    return [d1, d2, d3, d4]

def optimal_u(B, N, ps1, ps2):
    """Аналітичне керування з умови максимуму."""
    u = alpha0 * B * N * (ps1 - pi_par * ps2) / (2 * omega2)
    return np.clip(u, 0, 1)

def compute_J(B_arr, u_arr, h):
    """Обчислення цільового функціоналу методом трапецій."""
    return np.sum((omega1 * B_arr - omega2 * u_arr**2) * h)

def solve_forward(t_eval, u_interp):
    """Інтегрування прямої системи за допомогою solve_ivp (RK45)."""
    def f(t, state):
        u_val = u_interp(t)
        return state_rhs(t, state, u_val)
    sol = solve_ivp(f, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method='RK45')
    if not sol.success:
        raise RuntimeError(f"Forward integration failed: {sol.message}")
    return sol.t, sol.y.T

def solve_backward(t_eval, B_arr, N_arr, P_arr, E_arr, u_interp):
    """Інтегрування спряженої системи в зворотному напрямку."""
    psi0 = [0.0, 0.0, 0.0, 0.0]
    def g(t, psi):
        B = np.interp(t, t_eval, B_arr)
        N = np.interp(t, t_eval, N_arr)
        P = np.interp(t, t_eval, P_arr)
        E = np.interp(t, t_eval, E_arr)
        u_val = u_interp(t)
        return adjoint_rhs(t, psi, B, N, P, E, u_val)
    sol = solve_ivp(g, [t_eval[-1], t_eval[0]], psi0, t_eval=t_eval[::-1], method='RK45')
    if not sol.success:
        raise RuntimeError(f"Backward integration failed: {sol.message}")
    psi_arr = sol.y.T[::-1]
    return psi_arr

def u_interp_factory(u_arr, t_vals):
    """Створює інтерполятор для керування (кусково-постійний за сегментами)."""
    return lambda t: np.interp(t, t_vals, u_arr, left=u_arr[0], right=u_arr[-1])

# --- Forward-Backward Sweep ---
u = np.ones(n) * 0.5
max_iter, tol = 500, 1e-6

for it in range(max_iter):
    u_old = u.copy()
    u_interp = u_interp_factory(u, t)

    sol_t, sol_y = solve_forward(t, u_interp)
    B_arr = np.interp(t, sol_t, sol_y[:, 0])
    N_arr = np.interp(t, sol_t, sol_y[:, 1])
    P_arr = np.interp(t, sol_t, sol_y[:, 2])
    E_arr = np.interp(t, sol_t, sol_y[:, 3])

    psi_arr = solve_backward(t, B_arr, N_arr, P_arr, E_arr, u_interp)

    u_new = optimal_u(B_arr, N_arr, psi_arr[:, 0], psi_arr[:, 1])
    u = 0.5 * u_old + 0.5 * u_new

    err = np.max(np.abs(u - u_old))
    if it % 20 == 0:
        J_cur = compute_J(B_arr, u, h)
        print(f"Ітерація {it:3d} | J = {J_cur:.2f} | B(T) = {B_arr[-1]:.2f} | err = {err:.2e}")
    if err < tol:
        print(f"\nЗбіжність досягнута на ітерації {it}")
        break

# --- Кусково-стале представлення керування ---
u_piecewise = np.zeros(n)
u_seg_vals = []
for k in range(m):
    start = k * seg_size
    end = (k + 1) * seg_size if k < m - 1 else n
    u_mean = np.mean(u[start:end])
    u_piecewise[start:end] = u_mean
    u_seg_vals.append(u_mean)

J_final = compute_J(B_arr, u, h)
print(f"\nФінальний J (неперервне u)   = {J_final:.4f}")
print(f"B(T) = {B_arr[-1]:.4f}")
print(f"\nКусково-стале u* по {m} сегментах:")
for k, uv in enumerate(u_seg_vals):
    t_start = k * (T / m)
    t_end   = (k + 1) * (T / m)
    print(f"  [{t_start:.0f}, {t_end:.0f}): u = {uv:.4f}")

# --- Графік: B(t) + кусково-стале u*(t) на одному полі з двома осями ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Ліва вісь: лісові ресурси
ax1.plot(t, B_arr, 'g-', linewidth=2, label='B(t)')
ax1.set_xlabel('Час t')
ax1.set_ylabel('B(t)', color='g')
ax1.tick_params(axis='y', labelcolor='g')
ax1.grid(True, alpha=0.3)

# Права вісь: кусково-стале керування
ax2 = ax1.twinx()
ax2.step(t, u_piecewise, 'r-', linewidth=2.5, where='post', label=f'u*(t) кусково-стале (m={m})')
ax2.set_ylabel('u*(t)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(-0.05, 1.05)

# Об'єднання легенд
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('Динаміка лісових ресурсів та оптимальне кусково-стале керування')
plt.tight_layout()
plt.savefig('pontryagin_sweep_piecewise_RK45.png', dpi=150)
plt.show()