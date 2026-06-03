import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PARAMETERS
# ============================================

x = np.linspace(0, 1, 1000)

eps_values = [1e-1, 1e-2, 1e-3]

# ============================================
# EXACT SOLUTION
# ============================================

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

# ============================================
# STYLE
# ============================================

plt.style.use('ggplot')

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "legend.fontsize": 12
})

# ============================================
# PLOT
# ============================================

fig, ax = plt.subplots(figsize=(8.5, 5.5))

for eps in eps_values:

    u = exact_solution(x, eps)

    Pe = 1 / eps

    ax.plot(
        x,
        u,
        linewidth=2.5,
        label=rf'$\varepsilon={eps:.0e},\quad Pe={Pe:.0f}$'
    )

# ============================================
# LABELS
# ============================================

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x)$')

ax.set_title('Influence of Peclet number on the solution')

# ============================================
# GRID
# ============================================

ax.grid(
    True,
    linestyle='--',
    linewidth=0.8,
    alpha=0.4
)

# ============================================
# LEGEND
# ============================================

ax.legend()

# ============================================
# SAVE
# ============================================

plt.tight_layout()

plt.savefig(
    'peclet_study.png',
    dpi=400,
    bbox_inches='tight'
)

plt.show()

# ============================================
# PRINT INFO
# ============================================

print("\n============= PECLET NUMBERS =============\n")

for eps in eps_values:

    Pe = 1 / eps

    print(f"epsilon = {eps:.0e}  --->  Pe = {Pe:.0f}")