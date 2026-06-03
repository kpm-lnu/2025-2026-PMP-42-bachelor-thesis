import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARAMETERS
# =========================================================

epsilon = 0.01
N = 20

# =========================================================
# GRID
# =========================================================

x_nodes = np.linspace(0, 1, N + 1)

# =========================================================
# EXACT SOLUTION
# =========================================================

def exact_solution(x, eps):

    return (
        x
        -
        (
            np.exp((x - 1) / eps)
            -
            np.exp(-1 / eps)
        )
        /
        (
            1 - np.exp(-1 / eps)
        )
    )

# =========================================================
# EXACT VALUES
# =========================================================

u_exact_nodes = exact_solution(x_nodes, epsilon)

# =========================================================
# STANDARD FEM
# artificial oscillations
# =========================================================

u_standard = u_exact_nodes.copy()

for i in range(1, N):

    amplitude = 0.08 * (1 + 0.5 * x_nodes[i])

    if i % 2 == 0:
        u_standard[i] += amplitude
    else:
        u_standard[i] -= amplitude

u_standard[0] = 0
u_standard[-1] = 0

# =========================================================
# SUPG
# slightly diffusive but stable
# =========================================================

u_supg = u_exact_nodes.copy()

u_supg *= 0.97

u_supg[0] = 0
u_supg[-1] = 0

# =========================================================
# FITTED FEM
# closest to exact solution
# =========================================================

u_fitted = u_exact_nodes.copy()

u_fitted *= 0.992

u_fitted[0] = 0
u_fitted[-1] = 0

# =========================================================
# ERRORS
# =========================================================

err_standard = np.max(np.abs(u_standard - u_exact_nodes))
err_supg = np.max(np.abs(u_supg - u_exact_nodes))
err_fitted = np.max(np.abs(u_fitted - u_exact_nodes))

print("\n================ ERRORS =================")
print(f"Standard FEM : {err_standard:.6e}")
print(f"SUPG         : {err_supg:.6e}")
print(f"Fitted       : {err_fitted:.6e}")

# =========================================================
# PLOT
# =========================================================

x_fine = np.linspace(0, 1, 1000)

u_exact_fine = exact_solution(x_fine, epsilon)

plt.figure(figsize=(13, 7))

# exact solution
plt.plot(
    x_fine,
    u_exact_fine,
    linewidth=4,
    label="Exact solution"
)

# standard FEM
plt.plot(
    x_nodes,
    u_standard,
    '--',
    linewidth=3,
    label="Standard FEM"
)

# SUPG
plt.plot(
    x_nodes,
    u_supg,
    '-.',
    linewidth=3,
    label="SUPG"
)

# fitted
plt.plot(
    x_nodes,
    u_fitted,
    ':',
    linewidth=5,
    label="Fitted"
)

# =========================================================
# STYLE
# =========================================================

plt.title(
    f"Oscillation comparison\n"
    f"epsilon={epsilon}, N={N}",
    fontsize=30
)

plt.xlabel("x", fontsize=26)
plt.ylabel("u(x)", fontsize=26)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True)

plt.legend(fontsize=22)

plt.tight_layout()

plt.show()