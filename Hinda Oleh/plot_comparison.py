"""
Порівняння FEM та аналітичних напружень в перерізі z=0.5
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================
# ДАНІ ДЛЯ ЦИКЛУ 3 (з виводу програми)
# ============================================

# Для перерізу z = 0.5
r_fem = [1.0, 1.125, 1.25, 1.5, 2.0]
u_r_fem = [103.85, 94.51, 87.27, 77.06, 66.21]
sigma_rr_fem = [-86.23, -68.21, -47.62, -22.69, -18.39]
sigma_tt_fem = [170.75, 138.78, 119.23, 93.02, 58.33]

# Аналітичні значення (задача Ляме)
a, b, p, E, nu = 1.0, 2.0, 100, 1.82, 0.3
r_ana = np.linspace(1.0, 2.0, 100)

def u_r_ana(r):
    return (p * a**2) / (E * (b**2 - a**2)) * ((1 - nu) * r + (1 + nu) * b**2 / r)

def sigma_rr_ana(r):
    return (p * a**2) / (b**2 - a**2) * (1 - b**2 / r**2)

def sigma_tt_ana(r):
    return (p * a**2) / (b**2 - a**2) * (1 + b**2 / r**2)

# Побудова графіків
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(r_ana, u_r_ana(r_ana), 'b-', label='Analytical', linewidth=2)
axes[0].plot(r_fem, u_r_fem, 'ro', label='FEM (cycle 3)', markersize=8)
axes[0].set_xlabel('Radius r')
axes[0].set_ylabel('u_r')
axes[0].set_title('Radial displacement $u_r$')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(r_ana, sigma_rr_ana(r_ana), 'b-', label='Analytical', linewidth=2)
axes[1].plot(r_fem, sigma_rr_fem, 'ro', label='FEM (cycle 3)', markersize=8)
axes[1].set_xlabel('Radius r')
axes[1].set_ylabel('σ_rr')
axes[1].set_title('Radial stress $\\sigma_{rr}$')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(r_ana, sigma_tt_ana(r_ana), 'b-', label='Analytical', linewidth=2)
axes[2].plot(r_fem, sigma_tt_fem, 'ro', label='FEM (cycle 3)', markersize=8)
axes[2].set_xlabel('Radius r')
axes[2].set_ylabel('σ_φφ')
axes[2].set_title('Hoop stress $\\sigma_{\\varphi\\varphi}$')
axes[2].legend()
axes[2].grid(True)

plt.suptitle('Comparison of FEM (cycle 3) and analytical solution at section z = 0.5', fontsize=14)
plt.tight_layout()
plt.savefig('comparison_all.png', dpi=300, bbox_inches='tight')
plt.show()

# Виведення похибок
print("\n" + "=" * 60)
print("ERRORS AT SECTION z = 0.5 (CYCLE 3)")
print("=" * 60)
for i, r in enumerate(r_fem):
    u_err = abs(u_r_fem[i] - u_r_ana(r)) / u_r_ana(r) * 100
    srr_err = abs(sigma_rr_fem[i] - sigma_rr_ana(r)) / abs(sigma_rr_ana(r)) * 100 if sigma_rr_ana(r) != 0 else 0
    stt_err = abs(sigma_tt_fem[i] - sigma_tt_ana(r)) / sigma_tt_ana(r) * 100
    print(f"r={r:.3f}: u_r error={u_err:.2f}%, σ_rr error={srr_err:.2f}%, σ_φφ error={stt_err:.2f}%")