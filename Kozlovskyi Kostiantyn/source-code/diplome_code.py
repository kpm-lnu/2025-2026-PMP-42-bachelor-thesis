import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 120,
})


P1 = {
    'label'  : 'Example 1 — advection-diffusion, no source',
    'D'      : 0.1,
    'v'      : 1.0,
    'mu'     : 0.0,
    'L'      : 1.0,
    'c0'     : 1.0,
    'cL'     : 0.0,
    'f0'     : 0.0,
    'grids'  : [5, 10, 20, 40, 80],
    'N_plot' : 200,
    'mu_list': None,
}

P2 = {
    'label'  : 'Example 2 — full model: source + decay',
    'D'      : 0.1,
    'v'      : 0.5,
    'mu'     : 1.0,
    'L'      : 1.0,
    'c0'     : 1.0,
    'cL'     : 0.0,
    'f0'     : 2.0,
    'grids'  : [5, 10, 20, 40, 80],
    'N_plot' : 200,
    'mu_list': [0.0, 0.5, 1.0, 2.0, 5.0],
}

P3 = {
    'label'  : 'Example 3 — oil spill (high Pe=30, stability test)',
    'D'      : 0.05,
    'v'      : 0.3,
    'mu'     : 0.1,
    'L'      : 5.0,
    'c0'     : 10.0,
    'cL'     : 0.0,
    'f0'     : 0.0,
    'grids'  : [10, 20, 40, 80, 160],
    'N_plot' : 200,
    'mu_list': None,
}

P4 = {
    'label'  : 'Example 4 — fertilizer in root zone (bell-shaped profile)',
    'D'      : 0.2,
    'v'      : 0.1,
    'mu'     : 2.0,
    'L'      : 1.0,
    'c0'     : 0.0,
    'cL'     : 0.0,
    'f0'     : 5.0,
    'grids'  : [5, 10, 20, 40, 80],
    'N_plot' : 200,
    'mu_list': [0.5, 1.0, 2.0, 5.0, 10.0],
}

ALL_EXAMPLES = [P1, P2, P3, P4]


def fem_solve(D, v, mu, L, N, c0, cL, f_func=None):
    h = L / N
    n = N - 1

    k11 =  D/h - v/2 + mu*h/3
    k12 = -D/h + v/2 + mu*h/6
    k21 = -D/h - v/2 + mu*h/6
    k22 =  D/h + v/2 + mu*h/3

    d  = k22 + k11
    u  = k12
    lo = k21

    Pe_h = abs(v) * h / (2 * D)
    if Pe_h > 1.0:
        print(f"  [!] Warning: Pe_h = {Pe_h:.2f} > 1 — oscillations possible "
              f"(N={N}, h={h:.4f})")

    x_nodes = np.linspace(0, L, N + 1)
    F = np.zeros(n)

    if f_func is not None:
        for i in range(n):
            xi = x_nodes[i + 1]
            F[i] = h * f_func(xi)

    F[0]   -= lo * c0
    F[n-1] -= u  * cL

    c_inner = thomas_algorithm(lo, d, u, F, n)
    c = np.concatenate([[c0], c_inner, [cL]])
    return x_nodes, c


def thomas_algorithm(a, b, c_coef, F, n):
    alpha = np.zeros(n)
    beta  = np.zeros(n)

    alpha[0] = -c_coef / b
    beta[0]  =  F[0]   / b

    for i in range(1, n):
        denom    = b + a * alpha[i-1]
        alpha[i] = -c_coef / denom
        beta[i]  = (F[i] - a * beta[i-1]) / denom

    C = np.zeros(n)
    C[n-1] = beta[n-1]
    for i in range(n-2, -1, -1):
        C[i] = alpha[i] * C[i+1] + beta[i]
    return C


def exact_no_source(x, D, v, mu, L, c0, cL):
    if abs(mu) < 1e-12:
        Pe = v * L / D
        if abs(Pe) < 1e-12:
            return c0 + (cL - c0) * x / L
        exp_Pe = np.exp(Pe)
        B = (cL - c0) / (exp_Pe - 1.0)
        A = c0 - B
        return A + B * np.exp(Pe * x / L)
    else:
        disc = v**2 + 4 * D * mu
        r1 = (v + np.sqrt(disc)) / (2 * D)
        r2 = (v - np.sqrt(disc)) / (2 * D)
        e1 = np.exp(r1 * L)
        e2 = np.exp(r2 * L)
        mat = np.array([[1.0, 1.0], [e1, e2]])
        rhs = np.array([c0, cL])
        A, B = np.linalg.solve(mat, rhs)
        return A * np.exp(r1 * x) + B * np.exp(r2 * x)


def exact_with_source(x, D, v, mu, L, c0, cL, f0):
    lam = np.pi / L
    disc = v**2 + 4 * D * mu
    r1 = (v + np.sqrt(disc)) / (2 * D)
    r2 = (v - np.sqrt(disc)) / (2 * D)

    a11 =  D * lam**2 + mu
    a12 = -v * lam
    a21 =  v * lam
    a22 =  D * lam**2 + mu
    det = a11 * a22 - a12 * a21

    P_coef =  a22 * f0 / det
    Q_coef = -a21 * f0 / det

    e1 = np.exp(r1 * L)
    e2 = np.exp(r2 * L)
    mat = np.array([[1.0, 1.0], [e1, e2]])
    rhs = np.array([c0 - Q_coef, cL + Q_coef])
    A_coef, B_coef = np.linalg.solve(mat, rhs)

    return (A_coef * np.exp(r1 * x) +
            B_coef * np.exp(r2 * x) +
            P_coef * np.sin(lam * x) +
            Q_coef * np.cos(lam * x))


def get_exact(P):
    D, v, mu = P['D'], P['v'], P['mu']
    L, c0, cL, f0 = P['L'], P['c0'], P['cL'], P['f0']
    if abs(f0) < 1e-12:
        return lambda x: exact_no_source(x, D, v, mu, L, c0, cL)
    else:
        return lambda x: exact_with_source(x, D, v, mu, L, c0, cL, f0)


def get_f_func(P):
    if abs(P['f0']) < 1e-12:
        return None
    return lambda x: P['f0'] * np.sin(np.pi * x / P['L'])


def compute_errors(P):
    D, v, mu = P['D'], P['v'], P['mu']
    L, c0, cL = P['L'], P['c0'], P['cL']
    f_func = get_f_func(P)
    exact  = get_exact(P)

    h_vals, err_L2, err_H1 = [], [], []

    for N in P['grids']:
        h = L / N
        x_num, c_num = fem_solve(D, v, mu, L, N, c0, cL, f_func)
        c_ex = exact(x_num)

        diff = c_num - c_ex
        L2   = np.sqrt(np.trapezoid(diff**2, x_num))

        dc_num = np.gradient(c_num, x_num)
        dc_ex  = np.gradient(c_ex,  x_num)
        H1     = np.sqrt(np.trapezoid((dc_num - dc_ex)**2, x_num))

        h_vals.append(h)
        err_L2.append(L2)
        err_H1.append(H1)

    h_vals = np.array(h_vals)
    err_L2 = np.array(err_L2)
    err_H1 = np.array(err_H1)

    ord_L2 = np.log(err_L2[:-1] / err_L2[1:]) / np.log(h_vals[:-1] / h_vals[1:])
    ord_H1 = np.log(err_H1[:-1] / err_H1[1:]) / np.log(h_vals[:-1] / h_vals[1:])

    return {'N': P['grids'], 'h': h_vals,
            'L2': err_L2, 'H1': err_H1,
            'ord_L2': ord_L2, 'ord_H1': ord_H1}


def print_error_table(res, title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  {'N':>5}  {'h':>8}  {'||e||_L2':>12}  {'ord':>6}  "
          f"{'||e||_H1':>12}  {'ord':>6}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*12}  {'-'*6}  {'-'*12}  {'-'*6}")
    for i, N in enumerate(res['N']):
        ord_L2 = f"{res['ord_L2'][i-1]:.3f}" if i > 0 else "  —  "
        ord_H1 = f"{res['ord_H1'][i-1]:.3f}" if i > 0 else "  —  "
        print(f"  {N:>5}  {res['h'][i]:>8.5f}  "
              f"{res['L2'][i]:>12.2e}  {ord_L2:>6}  "
              f"{res['H1'][i]:>12.2e}  {ord_H1:>6}")
    print(f"{'='*65}")


def plot_example(P, idx):
    D, v, mu = P['D'], P['v'], P['mu']
    L, c0, cL, f0 = P['L'], P['c0'], P['cL'], P['f0']
    Pe = v * L / D if abs(v) > 1e-12 else 0.0

    print(f"\n>>> {P['label']}")
    print(f"    D={D}, v={v}, μ={mu}, L={L}, c(0)={c0}, c(L)={cL}, f0={f0}")
    print(f"    Peclet number Pe = v·L/D = {Pe:.1f}")

    f_func = get_f_func(P)
    exact  = get_exact(P)

    res = compute_errors(P)
    print_error_table(res, f"Example {idx} — error table")

    has_mu_list = P['mu_list'] is not None
    ncols = 2 if has_mu_list else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if not has_mu_list:
        axes = [axes]

    fig.suptitle(f'{P["label"]}  (Pe = {Pe:.1f})', fontsize=13, fontweight='bold')

    ax = axes[0]
    x_fine = np.linspace(0, L, 500)
    ax.plot(x_fine, exact(x_fine), 'k-', lw=2.5, label='Exact solution')

    if f_func is not None:
        ax.fill_between(x_fine, 0,
                        f_func(x_fine) / f0 * 0.15 + exact(x_fine).min() * 0.5,
                        alpha=0.08, color='orange', label='Source f(x) [scaled]')

    colors = ['#e74c3c', '#3498db', '#2ecc71']
    ns_plot = [P['grids'][0], P['grids'][len(P['grids'])//2], P['grids'][-1]]
    for N, col in zip(ns_plot, colors):
        x_num, c_num = fem_solve(D, v, mu, L, N, c0, cL, f_func)
        ax.plot(x_num, c_num, 'o--', color=col, lw=1.5, ms=5, label=f'FEM  N={N}')

    ax.set_xlabel('x  [cm]')
    ax.set_ylabel('c(x)  [g/cm³]')
    ax.set_title('FEM vs exact — convergence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

    if has_mu_list:
        ax2 = axes[1]
        cmap = plt.cm.viridis
        mu_list = P['mu_list']
        for i, mu_i in enumerate(mu_list):
            col = cmap(i / (len(mu_list) - 1))
            if abs(f0) < 1e-12:
                ex_i = lambda x, m=mu_i: exact_no_source(x, D, v, m, L, c0, cL)
            else:
                ex_i = lambda x, m=mu_i: exact_with_source(x, D, v, m, L, c0, cL, f0)
            ax2.plot(x_fine, ex_i(x_fine), '-', color=col, lw=2, label=f'μ = {mu_i}')

        ax2.set_xlabel('x  [cm]')
        ax2.set_ylabel('c(x)  [g/cm³]')
        ax2.set_title('Effect of decay coefficient μ')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, L)

    plt.tight_layout()
    plt.show()


def plot_heatmap(P, idx):
    D, v, mu = P['D'], P['v'], P['mu']
    L, c0, cL, f0 = P['L'], P['c0'], P['cL'], P['f0']
    Pe = v * L / D if abs(v) > 1e-12 else 0.0

    f_func = get_f_func(P)
    exact  = get_exact(P)

    N_fine = 300
    x_fine = np.linspace(0, L, N_fine + 1)
    x_num, c_num = fem_solve(D, v, mu, L, N_fine, c0, cL, f_func)

    fig, ax = plt.subplots(figsize=(5, 8))
    fig.suptitle(f'{P["label"]}\nPe = {Pe:.1f},  μ = {mu}',
                 fontsize=12, fontweight='bold')

    width = 40
    c_2d = np.tile(c_num[:, np.newaxis], (1, width))

    im = ax.imshow(
        c_2d,
        aspect='auto',
        cmap='YlOrBr',
        origin='upper',
        extent=[0, 1, L, 0],
        vmin=0,
        vmax=max(c0, cL, c_num.max()),
        interpolation='bicubic',
    )

    c_max_val = max(c0, cL, c_num.max())
    c_norm = c_num / c_max_val if c_max_val > 0 else c_num
    ax.plot(c_norm, x_num, 'w-',  lw=2.5, label='FEM profile', zorder=5)
    ax.plot(exact(x_fine) / c_max_val if c_max_val > 0 else exact(x_fine),
            x_fine, 'k--', lw=1.5, alpha=0.7, label='Exact', zorder=4)

    if f_func is not None:
        x_src = np.linspace(0.02, 0.98, 60)
        for xi in x_src:
            fi = f_func(xi) / f0
            ax.plot(fi * 0.12, xi, 's', color='cyan', ms=2, alpha=0.5, zorder=7)
        ax.text(0.07, L * 0.5, 'f(x)', color='cyan',
                fontsize=9, fontweight='bold', va='center', zorder=8)

    ax.annotate('', xy=(0.88, L * 0.72), xytext=(0.88, L * 0.38),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.2, mutation_scale=18),
                zorder=6)
    ax.text(0.90, L * 0.55, f'v={v}', color='white',
            fontsize=9, fontweight='bold', va='center', zorder=6)

    soil_bounds  = [0.0, 0.20, 0.45, 0.72, 1.0]
    soil_names   = ['Humus', 'Topsoil', 'Subsoil', 'Parent\nmaterial']
    layer_depths = [0.10, 0.32, 0.58, 0.86]
    for name, depth in zip(soil_names, layer_depths):
        yd = depth * L
        ax.text(0.02, yd, name, color='white', fontsize=8,
                alpha=0.9, va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.3))

    ax.axhline(0, color='lime', lw=2, linestyle='-', alpha=0.85, zorder=9)
    ax.axhline(L, color='deepskyblue', lw=2, linestyle='--', alpha=0.85, zorder=9)
    ax.text(0.5, -0.012 * L, f'Surface  c(0)={c0} g/cm³',
            ha='center', va='top', fontsize=8, color='lime',
            transform=ax.get_xaxis_transform(), zorder=10)
    ax.text(0.5, L + 0.012 * L, f'Groundwater  c(L)={cL} g/cm³',
            ha='center', va='bottom', fontsize=8, color='deepskyblue',
            transform=ax.get_xaxis_transform(), zorder=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Concentration [g/cm³]', fontsize=10)

    ax.set_xlabel('← less    more →', fontsize=9)
    ax.set_ylabel('Depth x [cm]', fontsize=11)
    ax.set_xticks([])
    ax.set_ylim(L, 0)
    ax.legend(loc='lower right', fontsize=8,
              facecolor='black', labelcolor='white', framealpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*65)
    print("  RUNNING CALCULATIONS")
    print("="*65)

    for idx, P in enumerate(ALL_EXAMPLES, start=1):
        plot_example(P, idx)
        plot_heatmap(P, idx)

    print("\n" + "="*65)
    print("  CALCULATIONS COMPLETE")
    print("="*65)