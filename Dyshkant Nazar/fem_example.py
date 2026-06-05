import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from solve import solve_fem
from mesh_generator import create_beam_mesh

E   = 210e9
nu  = 0.3
L   = 1.0
H   = 0.2
P   = 1000.0          # total tip load [N]  (distributed over right-edge nodes)
tol = 1e-9

nodes, elements = create_beam_mesh(L=L, H=H, nx=20, ny=5)

x_coords = nodes[:, 0]
y_coords = nodes[:, 1]

x_min = np.min(x_coords)
x_max = np.max(x_coords)

left_nodes  = np.where(np.abs(x_coords - x_min) < tol)[0]
right_nodes = np.where(np.abs(x_coords - x_max) < tol)[0]

fixed_dofs = []
for n in left_nodes:
    fixed_dofs.extend([2*n, 2*n + 1])

load_per_node = -P / len(right_nodes)
loads = [[n, 1, load_per_node] for n in right_nodes]

U, K, F = solve_fem(
    nodes, elements, E, nu,
    loads=loads,
    fixed_dofs=fixed_dofs,
    element_type="Q8"
)

ux_fem = U[0::2]
uy_fem = U[1::2]

def analytical_solution(x, y, L, H, P, E):
    I = H**3 / 12
    ux      =  P * y * (2*L - x) * x / (2*E*I)
    uy      = -P * (3*L*x**2 - x**3) / (6*E*I)   # downward → negative
    sigma_xx = -P * (L - x) * y / I
    return ux, uy, sigma_xx

ux_ana, uy_ana, sxx_ana = analytical_solution(x_coords, y_coords, L, H, P, E)

def plot_deformation(nodes, elements, U, scale=10, element_type="Q8"):
    fig, ax = plt.subplots(figsize=(12, 4))
    order = [0, 1, 2, 3, 4, 5, 6, 7, 0] if element_type.upper() == "Q8" else [0, 1, 2, 3, 0]

    for elem in elements:
        pts           = nodes[elem]
        u             = U[2 * np.array(elem)]
        v             = U[2 * np.array(elem) + 1]
        displaced_pts = pts + scale * np.column_stack((u, v))

        ax.plot(pts[order, 0],           pts[order, 1],           color='gray', lw=0.8, ls='--')
        ax.plot(displaced_pts[order, 0], displaced_pts[order, 1], color='blue', lw=1.5)

    ax.set_aspect('equal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Deformed mesh  (scale × {scale})")
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

plot_deformation(nodes, elements, U, scale=4000, element_type="Q8")


fig, axes = plt.subplots(3, 3, figsize=(20, 12))
fig.suptitle("FEM (Q8) vs Analytical — Cantilever Beam", fontsize=14, fontweight='bold')

def to_grid(x, y, z, nx=200, ny=50):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    return Xi, Yi, Zi

fields = [
    (ux_fem,  ux_ana,  r"$u_x$  [m]"),
    (uy_fem,  uy_ana,  r"$u_y$  [m]"),
    (np.zeros_like(ux_fem), sxx_ana, r"$\sigma_{xx}$  [Pa]"),   # FEM σ computed below
]

from shape_functions import shape_functions_q8

sigma_xx_fem = np.zeros(len(nodes))
count        = np.zeros(len(nodes))

D = E / (1 - nu**2) * np.array([
    [1,  nu, 0],
    [nu, 1,  0],
    [0,  0,  (1 - nu) / 2]
])

s  = 1.0 / np.sqrt(3)
s3 = np.sqrt(3)

gauss_pts = [(-s, -s), (s, -s), (s, s), (-s, s)]

node_extrap = {
    0: (-s3, -s3),   # corner, maps to GP0 region
    2: ( s3, -s3),   # corner, maps to GP1 region
    4: ( s3,  s3),   # corner, maps to GP2 region
    6: (-s3,  s3),   # corner, maps to GP3 region
    1: (0.0, -s3),   # mid-side between nodes 0 and 2
    3: ( s3,  0.0),  # mid-side between nodes 2 and 4
    5: (0.0,  s3),   # mid-side between nodes 4 and 6
    7: (-s3,  0.0),  # mid-side between nodes 6 and 0
}

def bilinear_extrap(xi, eta, sg):
    n0 = 0.25 * (1 - xi) * (1 - eta)
    n1 = 0.25 * (1 + xi) * (1 - eta)
    n2 = 0.25 * (1 + xi) * (1 + eta)
    n3 = 0.25 * (1 - xi) * (1 + eta)
    return n0*sg[0] + n1*sg[1] + n2*sg[2] + n3*sg[3]

for elem in elements:
    coords = nodes[elem]

    u_elem = np.zeros(16)
    for a, n in enumerate(elem):
        u_elem[2*a]     = U[2*n]
        u_elem[2*a + 1] = U[2*n + 1]

    sigma_gp = []
    for xi, eta in gauss_pts:
        N, dN = shape_functions_q8(xi, eta)
        J     = dN.T @ coords
        invJ  = np.linalg.inv(J)
        dN_dx = dN @ invJ

        B = np.zeros((3, 16))
        for a in range(8):
            B[0, 2*a]     = dN_dx[a, 0]
            B[1, 2*a + 1] = dN_dx[a, 1]
            B[2, 2*a]     = dN_dx[a, 1]
            B[2, 2*a + 1] = dN_dx[a, 0]

        sigma_gp.append((D @ B @ u_elem)[0])

    for local_idx, global_n in enumerate(elem):
        xi_e, eta_e = node_extrap[local_idx]
        sigma_xx_fem[global_n] += bilinear_extrap(xi_e, eta_e, sigma_gp)
        count[global_n] += 1

sigma_xx_fem /= np.where(count > 0, count, 1)

interior = (x_coords > 0.2*L) & (x_coords < 0.8*L)

err_sxx_interior = (np.linalg.norm(sigma_xx_fem[interior] - sxx_ana[interior]) /
                    (np.linalg.norm(sxx_ana[interior]) + 1e-30)) * 100
print(f"  σ_xx interior only: {err_sxx_interior:.3f} %")

fields[2] = (sigma_xx_fem, sxx_ana, r"$\sigma_{xx}$  [Pa]")

cmap_field = 'RdBu_r'
cmap_error = 'OrRd'

for row, (fem_vals, ana_vals, label) in enumerate(fields):
    Xf, Yf, Zf   = to_grid(x_coords, y_coords, fem_vals)
    _,  _,  Za   = to_grid(x_coords, y_coords, ana_vals)
    _,  _,  Zerr = to_grid(x_coords, y_coords, np.abs(fem_vals - ana_vals))

    vmin = min(np.nanmin(Zf), np.nanmin(Za))
    vmax = max(np.nanmax(Zf), np.nanmax(Za))

    ax = axes[row, 0]
    im = ax.contourf(Xf, Yf, Zf, levels=20, cmap=cmap_field, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{label}  —  FEM")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect('equal')

    ax = axes[row, 1]
    im = ax.contourf(Xf, Yf, Za, levels=20, cmap=cmap_field, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{label}  —  Analytical")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect('equal')

    ax = axes[row, 2]
    im = ax.contourf(Xf, Yf, Zerr, levels=20, cmap=cmap_error)
    plt.colorbar(im, ax=ax, shrink=0.8)
    rel = np.linalg.norm(fem_vals - ana_vals) / (np.linalg.norm(ana_vals) + 1e-30) * 100
    ax.set_title(f"|FEM − Analytical|  —  {label}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect('equal')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Tip cross-section  (x = L,  all y)", fontsize=13, fontweight='bold')

tip_mask = np.abs(x_coords - x_max) < tol
y_tip    = y_coords[tip_mask]
sort_idx = np.argsort(y_tip)
y_s      = y_tip[sort_idx]

for ax, (fem_v, ana_v, lbl) in zip(
        axes,
        [(uy_fem[tip_mask][sort_idx], uy_ana[tip_mask][sort_idx], r"$u_y$ at $x=L$"),
         (ux_fem[tip_mask][sort_idx], ux_ana[tip_mask][sort_idx], r"$u_x$ at $x=L$")]):
    ax.plot(y_s, ana_v, 'k-',  lw=2,   label="Analytical")
    ax.plot(y_s, fem_v, 'r--', lw=1.5, label="FEM (Q8)", marker='o', ms=4)
    ax.set_xlabel("y [m]")
    ax.set_ylabel(lbl)
    ax.set_title(lbl)
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

plt.tight_layout()
plt.show()

err_ux  = np.linalg.norm(ux_fem - ux_ana) / (np.linalg.norm(ux_ana) + 1e-30)
err_uy  = np.linalg.norm(uy_fem - uy_ana) / (np.linalg.norm(uy_ana) + 1e-30)
err_sxx = np.linalg.norm(sigma_xx_fem - sxx_ana) / (np.linalg.norm(sxx_ana) + 1e-30)

print("\n=== Relative L2 errors ===")
print(f"  u_x  :  {err_ux  * 100:.3f} %")
print(f"  u_y  :  {err_uy  * 100:.3f} %")
print(f"  σ_xx :  {err_sxx * 100:.3f} %")

tip_uy_fem = uy_fem[np.abs(x_coords - x_max) < tol].mean()
tip_uy_ana = uy_ana[np.abs(x_coords - x_max) < tol].mean()
print(f"\n=== Tip deflection (mean over right edge) ===")
print(f"  FEM        : {tip_uy_fem:.6e} m")
print(f"  Analytical : {tip_uy_ana:.6e} m")
print(f"  Error      : {abs(tip_uy_fem - tip_uy_ana)/abs(tip_uy_ana)*100:.3f} %")