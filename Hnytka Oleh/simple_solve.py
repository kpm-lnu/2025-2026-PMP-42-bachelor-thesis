import numpy as np
import matplotlib.pyplot as plt

# Коефіцієнти моделі згідно з Kusum Lata & Arvind Kumar Misra (2017)
s = 0.8       
L = 50.0      
alpha0 = 0.0031 
lam2 = 0.0007  
pi1 = 0.03     
phi0 = 0.4    
r = 0.5         
K = 100.0   
pi = 0.004      
lam = 0.007     
lam0 = 0.4      
phi = 0.006  
gamma1 = 0.0002 
pi2 = 0.09      

def system_dynamics(t, state, u):
    B, N, P, E = state
    dB = s*B*(1 - B/L) - alpha0*(1 - u)*B*N - lam2*B**2*P + pi1*phi0*E
    dN = r*N*(1 - N/K) + pi*alpha0*(1 - u)*B*N
    dP = lam*N - lam0*P - pi2*gamma1*P*E
    dE = phi*(L - B) - phi0*E - gamma1*P*E
    return np.array([dB, dN, dP, dE])

def rk4(func, y0, t, u):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = func(t[i], y[i], u)
        k2 = func(t[i] + h/2, y[i] + h/2 * k1, u)
        k3 = func(t[i] + h/2, y[i] + h/2 * k2, u)
        k4 = func(t[i] + h, y[i] + h * k3, u)
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return y

# Час та початкові умови
t = np.linspace(0, 100, 1000)
y0 = [20, 10, 5, 1]  # B, N, P, E

# Моделювання: 1. Без керування (u=0), 2. З фіксованим (u=0.5), 3. (u=1)
res_no_ctrl = rk4(system_dynamics, y0, t, u=0)
res_fixed_ctrl = rk4(system_dynamics, y0, t, u=0.5)
res_fixed_ctrl_1 = rk4(system_dynamics, y0, t, u=1)

# ========== ГРАФІК 1: Порівняння динаміки лісових ресурсів ==========
plt.figure(figsize=(10, 6))
plt.plot(t, res_no_ctrl[:, 0], 'r--', label='без керування (u=0)')
plt.plot(t, res_fixed_ctrl[:, 0], 'g-', label='помірне керування (u=0.5)')
plt.plot(t, res_fixed_ctrl_1[:, 0], 'b-', label='максимальне керування (u=1)')
plt.title('Порівняння динаміки лісових ресурсів $B(t)$')
plt.xlabel('Час $t$')
plt.ylabel('$B(t)$')
plt.legend()
plt.grid(True)
plt.show()

# ========== ГРАФІК 2: Розв'язок задачі Коші (4 окремі графіки для кожної змінної) ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Розв\'язок задачі Коші', fontsize=16)

# Графік для B(t) - лісові ресурси
axes[0, 0].plot(t, res_no_ctrl[:, 0], 'r-', linewidth=2)
axes[0, 0].set_title('$B(t)$ - Лісові ресурси')
axes[0, 0].set_xlabel('Час $t$')
axes[0, 0].set_ylabel('$B(t)$')
axes[0, 0].grid(True)
axes[0, 0].legend()

# Графік для N(t) - населення
axes[0, 1].plot(t, res_no_ctrl[:, 1], 'g-', linewidth=2)
axes[0, 1].set_title('$N(t)$ - Чисельність населення')
axes[0, 1].set_xlabel('Час $t$')
axes[0, 1].set_ylabel('$N(t)$')
axes[0, 1].grid(True)
axes[0, 1].legend()

# Графік для P(t) - тиск населення
axes[1, 0].plot(t, res_no_ctrl[:, 2], 'b-', linewidth=2)
axes[1, 0].set_title('$P(t)$ - Тиск населення')
axes[1, 0].set_xlabel('Час $t$')
axes[1, 0].set_ylabel('$P(t)$')
axes[1, 0].grid(True)
axes[1, 0].legend()

# Графік для E(t) - зусилля збереження
axes[1, 1].plot(t, res_no_ctrl[:, 3], 'm-', linewidth=2)
axes[1, 1].set_title('$E(t)$ - Економічні зусилля')
axes[1, 1].set_xlabel('Час $t$')
axes[1, 1].set_ylabel('$E(t)$')
axes[1, 1].grid(True)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# ========== ВИВЕДЕННЯ КІНЦЕВИХ ЗНАЧЕНЬ ЗМІННИХ ==========
print("=" * 60)
print("РОЗВ'ЯЗОК ЗАДАЧІ КОШІ (u = 0, T = 100)")
print("=" * 60)
print(f"B(T) = {res_no_ctrl[-1, 0]:.4f}  (лісові ресурси)")
print(f"N(T) = {res_no_ctrl[-1, 1]:.4f}  (чисельність населення)")
print(f"P(T) = {res_no_ctrl[-1, 2]:.4f}  (тиск населення)")
print(f"E(T) = {res_no_ctrl[-1, 3]:.4f}  (зусилля збереження)")
print("=" * 60)
print(f"Точка рівноваги F* ≈ ({res_no_ctrl[-1, 0]:.2f}, {res_no_ctrl[-1, 1]:.2f}, "
      f"{res_no_ctrl[-1, 2]:.3f}, {res_no_ctrl[-1, 3]:.3f})")
print("=" * 60)

omega1 = 1.0
omega2 = 10.0

def calculate_functional(B_values, u_value, t_values):
    integrand = omega1 * B_values - omega2 * (u_value**2)
    return np.trapezoid(integrand, t_values)

J_0 = calculate_functional(res_no_ctrl[:, 0], 0, t)
J_05 = calculate_functional(res_fixed_ctrl[:, 0], 0.5, t)
J_1 = calculate_functional(res_fixed_ctrl_1[:, 0], 1, t)

print("\n--- Значення цільового функціоналу J(u) ---")
print(f"При u = 0.0 (без керування):     J = {J_0:.2f}")
print(f"При u = 0.5 (помірне втручання): J = {J_05:.2f}")
print(f"При u = 1.0 (максимальне):       J = {J_1:.2f}")