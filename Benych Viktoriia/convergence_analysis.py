import numpy as np
import matplotlib.pyplot as plt

# ============================================
# STYLE SETTINGS
# ============================================

plt.style.use('ggplot')

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# ============================================
# EXPERIMENTAL DATA
# epsilon = 1e-3
# ============================================

N_values = np.array([10, 20, 40, 80, 160])

fem_errors = np.array([
    1.21e-3,
    2.62e-4,
    6.02e-5,
    1.86e-5,
    1.93e-4
])

supg_errors = np.array([
    4.90e-4,
    1.27e-4,
    3.24e-5,
    8.10e-6,
    2.96e-6
])

fitted_errors = np.array([
    5.69e-2,
    2.91e-2,
    1.42e-2,
    6.58e-3,
    2.81e-3
])

# ============================================
# CREATE FIGURE
# ============================================

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# ============================================
# PLOTS
# ============================================

ax.loglog(
    N_values,
    fem_errors,
    'o-',
    linewidth=2.5,
    markersize=8,
    label='Standard FEM'
)

ax.loglog(
    N_values,
    supg_errors,
    's-',
    linewidth=2.5,
    markersize=8,
    label='SUPG'
)

ax.loglog(
    N_values,
    fitted_errors,
    '^-',
    linewidth=2.5,
    markersize=8,
    label='Fitted'
)

# ============================================
# LABELS AND TITLE
# ============================================

ax.set_xlabel('Number of elements $N$')
ax.set_ylabel(r'$E_{\infty}$ error')

ax.set_title(
    r'Convergence of numerical methods'
)

# ============================================
# GRID
# ============================================

ax.grid(
    True,
    which='both',
    linestyle='--',
    linewidth=0.8,
    alpha=0.4
)

# ============================================
# LEGEND
# ============================================

ax.legend(
    loc='upper right',
    frameon=True
)

# ============================================
# SAVE FIGURE
# ============================================

plt.tight_layout()

plt.savefig(
    'convergence_analysis.png',
    dpi=400,
    bbox_inches='tight'
)

plt.show()

# ============================================
# PRINT TABLE
# ============================================

print("\n================ CONVERGENCE TABLE ================\n")

print(f"{'N':>6} {'FEM':>15} {'SUPG':>15} {'Fitted':>15}")

for i in range(len(N_values)):
    print(
        f"{N_values[i]:>6} "
        f"{fem_errors[i]:>15.6e} "
        f"{supg_errors[i]:>15.6e} "
        f"{fitted_errors[i]:>15.6e}"
    )