import numpy as np
import matplotlib.pyplot as plt
import base_functions_2d as b2f
import finite_element_2d as f2e
import mesh_2d as m2d
import system_2d as s2d
import matplotlib.animation as animation
from scipy.interpolate import griddata

def animate_solution(U_history, time_points, NL, title="Реакція-дифузія",
                     interval=120, exact_fn=None, save_filename=None):
    x = NL[:, 0]
    y = NL[:, 1]

    x_lin = np.linspace(0, 1, 80)
    y_lin = np.linspace(0, 1, 80)
    X, Y = np.meshgrid(x_lin, y_lin)

    all_vals = np.concatenate(U_history)
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1a')

    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)
    ax2d.set_facecolor('#0f0f1a')

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap='inferno', norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2d, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.set_label('u(x,y,t)', color='white')

    def draw_frame(frame):
        ax3d.clear()
        ax2d.clear()
        ax2d.set_facecolor('#0f0f1a')

        Z = griddata((x, y), U_history[frame], (X, Y), method='linear')
        t_val = time_points[frame]

        ax3d.plot_surface(X, Y, Z, cmap='inferno', vmin=vmin, vmax=vmax,
                          edgecolor='none', alpha=0.95)
        ax3d.set_zlim(vmin - 0.05, vmax + 0.05)
        ax3d.set_xlabel('x', color='white', labelpad=6)
        ax3d.set_ylabel('y', color='white', labelpad=6)
        ax3d.set_zlabel('u', color='white', labelpad=6)
        ax3d.tick_params(colors='white')
        ax3d.set_facecolor('#0f0f1a')
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.set_title(f'{title}\nt = {t_val:.3f}', color='white', fontsize=11)

        ax2d.contourf(X, Y, Z, levels=40, cmap='inferno', vmin=vmin, vmax=vmax)
        ax2d.contour(X, Y, Z, levels=10, colors='white',
                     linewidths=0.4, alpha=0.5)
        ax2d.set_xlabel('x', color='white')
        ax2d.set_ylabel('y', color='white')
        ax2d.tick_params(colors='white')
        ax2d.set_title('Теплова карта', color='white', fontsize=11)

        fig.suptitle(f'Розповсюдження: t = {t_val:.3f}', color='white',
                     fontsize=13, fontweight='bold')

    draw_frame(0)
    fig.tight_layout()

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(U_history),
                                   interval=interval, blit=False)
    if save_filename:
        print(f"Збереження анімації у файл {save_filename} (це може зайняти хвилину)...")
        ani.save(save_filename, writer='pillow', fps=1000//interval)
        print(f"Успішно збережено: {save_filename}")
    else:
        plt.show()
    return ani
    
def plot_statistics(start, end, statistics, u_exact, n_points=100):
    abscissas = np.linspace(0, 1, n_points)
    xs = [(1 - t) * start[0] + t * end[0] for t in abscissas]
    ys = [(1 - t) * start[1] + t * end[1] for t in abscissas]
    u_values = [u_exact(x, y) for x, y in zip(xs, ys)]

    plt.plot(abscissas, u_values, color='red', label='u exact')

    for i, (x_approx, u_approx) in enumerate(statistics):
        plt.plot(x_approx, u_approx, label=f'approx {i}')

    plt.xlabel('normalized distance along line')
    plt.ylabel('u')
    plt.title('Exact vs Approx Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_errors(errors):
    plt.plot([i for i in range(len(errors))], errors, color='green')
    plt.xlabel('degree of nodes')
    plt.ylabel('errors')
    plt.grid(True)
    plt.show()


def interpolate_solution(x, y, u, NL, EL, ap):
    from scipy.optimize import root

    num_nodes = 4 if ap == 1 else (9 if ap == 2 else 16)

    for element in EL:
        nodes_coords = [NL[i] for i in element]
        x_coords = [pt[0] for pt in nodes_coords]
        y_coords = [pt[1] for pt in nodes_coords]

        def equations(p):
            ksi, eta = p
            x_mapped = sum(b2f.N(i, ksi, eta, ap) * x_coords[i] for i in range(num_nodes))
            y_mapped = sum(b2f.N(i, ksi, eta, ap) * y_coords[i] for i in range(num_nodes))
            return [x_mapped - x, y_mapped - y]

        sol = root(equations, [0, 0])

        if sol.success:
            ksi, eta = sol.x
            if -1 <= ksi <= 1 and -1 <= eta <= 1:
                u_local = [u[i] for i in element]
                u_val = sum(u_local[i] * b2f.N(i, ksi, eta, ap) for i in range(num_nodes))
                return u_val

    return 0


def plot_2d_solution(u, NL, EL, exact_solution=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = NL[:, 0]
    y = NL[:, 1]
    z = u

    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)

    Z = griddata((x, y), z, (X, Y), method='linear')

    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none', label='Наближений розв\'язок')

    if exact_solution:
        Z_exact = exact_solution(X, Y)
        ax.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.4, label='Точний розв\'язок')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Наближений розв\'язок у 2D-просторі')
    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    plt.show()


def calculate_L2_error(u_exact, u_values, nodes, elements, base, degree):
    from numpy.polynomial.legendre import leggauss
    dN_dksi_list, dN_deta_list = s2d.compute_partial_derivatives(degree)
    nq = degree + 1
    quad_points, quad_weights = leggauss(nq)

    error_squared = 0

    for element in elements:
        element_coords = [nodes[i] for i in element]
        u_local = [u_values[i] for i in element]

        for i in range(nq):
            for j in range(nq):
                ξ = quad_points[i]
                η = quad_points[j]
                w = quad_weights[i] * quad_weights[j]

                x = sum(base[k](ξ, η) * element_coords[k][0] for k in range(len(element)))
                y = sum(base[k](ξ, η) * element_coords[k][1] for k in range(len(element)))

                u_h_val = sum(u_local[k] * base[k](ξ, η) for k in range(len(element)))
                u_ex_val = u_exact(x, y)

                J_val = m2d.compute_jacobian(ξ, η, [coord[0] for coord in element_coords],
                                             [coord[1] for coord in element_coords], dN_dksi_list, dN_deta_list)
                error_squared += (u_ex_val - u_h_val) ** 2 * w * J_val
    return np.sqrt(error_squared)

def compute_H1_error(u_exact, u_values, nodes, elements, ap):


    if ap == 1:
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1.0, 1.0]
    elif ap == 2:
        gauss_points = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
        gauss_weights = [5/9, 8/9, 5/9]
    elif ap == 3:
        gauss_points = [
            -np.sqrt((3 + 2*np.sqrt(6/5)) / 7),
            -np.sqrt((3 - 2*np.sqrt(6/5)) / 7),
             np.sqrt((3 - 2*np.sqrt(6/5)) / 7),
             np.sqrt((3 + 2*np.sqrt(6/5)) / 7),
        ]
        gauss_weights = [
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36,
        ]


    dN_dksi_list, dN_deta_list = s2d.compute_partial_derivatives(ap)

    err2_L2  = 0.0
    err2_H1  = 0.0

    for noe in elements:
        x_coords = [nodes[i][0] for i in noe]
        y_coords = [nodes[i][1] for i in noe]
        u_local  = [u_values[i] for i in noe]
        n = len(noe)

        for ki, ksi in enumerate(gauss_points):
            for ei, eta in enumerate(gauss_points):
                w = gauss_weights[ki] * gauss_weights[ei]

                x_gp = sum(b2f.N(k, ksi, eta, ap) * x_coords[k] for k in range(n))
                y_gp = sum(b2f.N(k, ksi, eta, ap) * y_coords[k] for k in range(n))

                u_h  = sum(u_local[k] * b2f.N(k, ksi, eta, ap) for k in range(n))
                u_ex = u_exact(x_gp, y_gp)

                J    = s2d.compute_jacobian(ksi, eta, x_coords, y_coords,
                                            dN_dksi_list, dN_deta_list)
                detJ = abs(np.linalg.det(J))
                J_inv = np.linalg.inv(J)

                err2_L2 += (u_ex - u_h)**2 * detJ * w

                grad_u_h = np.zeros(2)
                for k in range(n):
                    dN_ref = np.array([dN_dksi_list[k](ksi, eta),
                                       dN_deta_list[k](ksi, eta)])
                    grad_N = J_inv @ dN_ref
                    grad_u_h += u_local[k] * grad_N

                eps = 1e-7
                du_dx = (u_exact(x_gp + eps, y_gp) - u_exact(x_gp - eps, y_gp)) / (2*eps)
                du_dy = (u_exact(x_gp, y_gp + eps) - u_exact(x_gp, y_gp - eps)) / (2*eps)
                grad_u_ex = np.array([du_dx, du_dy])

                grad_err = grad_u_ex - grad_u_h
                err2_H1 += np.dot(grad_err, grad_err) * detJ * w

    err_L2  = np.sqrt(max(err2_L2, 0.0))
    err_H1  = np.sqrt(max(err2_H1, 0.0))
    err_W12 = np.sqrt(max(err2_L2 + err2_H1, 0.0))

    return err_L2, err_H1, err_W12


def plot_2d_solution2(u, NL, EL, exact_solution=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = NL[:, 0]
    y = NL[:, 1]
    z = u

    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = griddata((x, y), z, (X, Y), method='linear')

    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none', label='Наближений розв\'язок')

    if exact_solution:
        Z_exact = exact_solution(X, Y)
        Z_computed = griddata((x, y), z, (X, Y), method='linear')

        ax.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.4, edgecolor='none')

        mask = ~np.isnan(Z_computed) & ~np.isnan(Z_exact)
        Zc = Z_computed[mask]
        Ze = Z_exact[mask]

        l2_error = np.sqrt(np.mean((Zc - Ze) ** 2))
        max_error = np.max(np.abs(Zc - Ze))

        ax.text2D(0.02, 0.95, f"L2 похибка: {l2_error:.2e}", transform=ax.transAxes, fontsize=12)
        ax.text2D(0.02, 0.91, f"Max похибка: {max_error:.2e}", transform=ax.transAxes, fontsize=12)
        ax.text2D(0.02, 0.87, "Фіолетовий – точний,\nЗелений – FEM", transform=ax.transAxes, fontsize=10)
        print(f"L2 похибка: {l2_error:.2e}")
        print(f"Max похибка: {max_error:.2e}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Наближене vs Точне розв\'язання')

    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    plt.show()


def plot_2d_solution_exact(exact_solution, NL):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = NL[:, 0]
    y = NL[:, 1]

    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z_exact = exact_solution(X, Y)

    ax.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.8, edgecolor='none', label='Точний розв\'язок')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Точний розв\'язок')

    plt.colorbar(ax.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.8, edgecolor='none'))
    plt.show()


def plot_2d_solution_difference(u, NL, exact_solution=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = NL[:, 0]
    y = NL[:, 1]

    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)

    Z_exact = exact_solution(X, Y)

    Z_approx = griddata((x, y), u, (X, Y), method='linear')

    difference = Z_approx - Z_exact

    surface = ax.plot_surface(X, Y, difference, cmap='coolwarm', alpha=0.8, edgecolor='none',
                              label='Різниця між точним і наближеним')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Різниця')
    ax.set_title('Різниця між точним і наближеним розв\'язком')

    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    plt.show()


def generate_static_plots(U_history, time_history, NL, times_to_save, prefix="exp", title="Еволюція"):
    """
    Автоматично знаходить потрібні моменти часу в історії та зберігає їх як PNG.
    """
    import os
    x = NL[:, 0]
    y = NL[:, 1]
    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)

    all_vals = np.concatenate(U_history)
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    print(f"\n--- Генерація статичних фігур для: {title} ---")
    
    for target_t in times_to_save:
        idx = np.argmin(np.abs(np.array(time_history) - target_t))
        real_t = time_history[idx]
        U_n = U_history[idx]
        
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        Z = griddata((x, y), U_n, (X, Y), method='linear')
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', 
                                edgecolor='none', alpha=0.9, vmin=vmin, vmax=vmax)
        
        ax.set_zlim(vmin - 0.05, vmax + 0.05)
        ax.set_xlabel('Координата x', labelpad=10)
        ax.set_ylabel('Координата y', labelpad=10)
        ax.set_zlabel('Концентрація u(x,y,t)', labelpad=10)
        ax.set_title(f"{title}\nt = {real_t:.1f}", fontsize=14, pad=20)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        plt.colorbar(surface, ax=ax, shrink=0.6, aspect=15, pad=0.1)
        
        safe_t_str = f"{target_t:.1f}".replace('.', '_')
        filename = f"{prefix}_t{safe_t_str}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Збережено: {filename}")