"""
mortar_example.py
=================
Demonstrates non-conforming mesh coupling on the cantilever beam.

The beam is split at x = 0.5 into two subdomains:
  • left  subdomain: coarse mesh  (nx=8,  ny=3)
  • right subdomain: fine   mesh  (nx=16, ny=5)

The two meshes do NOT share nodes at the interface — mortar.py
enforces displacement continuity weakly via Lagrange multipliers.

The coupled result is compared against the single-mesh Q8 reference
and the analytical solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from mesh_generator import create_beam_mesh
from assemble import assemble_q8
from mortar import MortarInterface, couple_meshes

E      = 210e9
nu     = 0.3
L      = 1.0
H      = 0.2
P      = 1000.0
x_split = 0.5
tol    = 1e-9

# ── Build two non-conforming meshes ──────────────────────────────────────────
#
#   Left  subdomain: x ∈ [0, 0.5],  coarse (nx=8, ny=3)
#   Right subdomain: x ∈ [0.5, 1],  fine   (nx=12, ny=5)
#
#   Different ny → interface nodes do NOT match.

nodes1, elems1 = create_beam_mesh(L=x_split, H=H, nx=8,  ny=3)
nodes2, elems2 = create_beam_mesh(L=x_split, H=H, nx=12, ny=5)

# Shift mesh 2 so it starts at x = x_split
nodes2[:, 0] += x_split

print(f"Mesh 1 (left) : {len(nodes1)} nodes, {len(elems1)} elements")
print(f"Mesh 2 (right): {len(nodes2)} nodes, {len(elems2)} elements")

x1 = nodes1[:, 0];  y1 = nodes1[:, 1]
x2 = nodes2[:, 0];  y2 = nodes2[:, 1]

slave_iface  = np.where(np.abs(x1 - x_split) < tol)[0]
master_iface = np.where(np.abs(x2 - x_split) < tol)[0]

print(f"Interface: {len(slave_iface)} slave nodes, {len(master_iface)} master nodes")

left_nodes1 = np.where(np.abs(x1 - 0.0) < tol)[0]
fixed_dofs1 = []
for n in left_nodes1:
    fixed_dofs1.extend([2*n, 2*n+1])

right_nodes2     = np.where(np.abs(x2 - L) < tol)[0]
load_per_node    = -P / len(right_nodes2)

print("Assembling K1 …")
K1, F1 = assemble_q8(nodes1, elems1, E, nu)

print("Assembling K2 …")
K2, F2 = assemble_q8(nodes2, elems2, E, nu)

for n in right_nodes2:
    F2[2*n + 1] += load_per_node

print("Building mortar interface …")
iface = MortarInterface(
    nodes1, slave_iface,
    nodes2, master_iface,
    n_gauss=4
)

print("Solving coupled system …")
U1, U2, lam = couple_meshes(
    K1, F1, nodes1, slave_iface,
    K2, F2, nodes2, master_iface,
    iface,
    fixed_dofs1=fixed_dofs1,
    fixed_dofs2=None
)

gap = iface.interface_error(U1, U2)
print(f"\nInterface gap (L2): {gap:.3e}  (should be small)")

n_nodes1 = len(nodes1)
n_nodes2 = len(nodes2)
nodes_combined = np.vstack([nodes1, nodes2])
U_combined = np.zeros(2 * (n_nodes1 + n_nodes2))
U_combined[0:2*n_nodes1] = U1
U_combined[2*n_nodes1:] = U2

ux_mortar = np.zeros(len(nodes_combined))
uy_mortar = np.zeros(len(nodes_combined))
ux_mortar[0:n_nodes1] = U1[0::2]
ux_mortar[n_nodes1:] = U2[0::2]
uy_mortar[0:n_nodes1] = U1[1::2]
uy_mortar[n_nodes1:] = U2[1::2]

def analytical_uy(x, L, H, P, E):
    I = H**3 / 12
    return -P * (3*L*x**2 - x**3) / (6*E*I)

def analytical_ux(x, y, L, H, P, E):
    I = H**3 / 12
    return P * y * (2*L - x) * x / (2*E*I)

def analytical_sxx(x, y, L, H, P, E):
    I = H**3 / 12
    return -P * (L - x) * y / I

x_combined = nodes_combined[:, 0]
y_combined = nodes_combined[:, 1]
ux_ana = analytical_ux(x_combined, y_combined, L, H, P, E)
uy_ana = analytical_uy(x_combined, L, H, P, E)
sxx_ana = analytical_sxx(x_combined, y_combined, L, H, P, E)

from shape_functions import shape_functions_q8

sigma_xx_mortar = np.zeros(len(nodes_combined))
count = np.zeros(len(nodes_combined))

for elem in elems1:
    coords = nodes1[elem]
    N, dN = shape_functions_q8(0.0, 0.0)
    J = dN.T @ coords
    invJ = np.linalg.inv(J)
    dN_dx = dN @ invJ

    B = np.zeros((3, 16))
    for a in range(8):
        B[0, 2*a] = dN_dx[a, 0]
        B[1, 2*a+1] = dN_dx[a, 1]
        B[2, 2*a] = dN_dx[a, 1]
        B[2, 2*a+1] = dN_dx[a, 0]

    u_elem = np.zeros(16)
    for a, n in enumerate(elem):
        u_elem[2*a] = U1[2*n]
        u_elem[2*a+1] = U1[2*n+1]

    D = E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    stress = D @ B @ u_elem

    for a, n in enumerate(elem):
        sigma_xx_mortar[n] += stress[0]
        count[n] += 1

for elem in elems2:
    coords = nodes2[elem]
    N, dN = shape_functions_q8(0.0, 0.0)
    J = dN.T @ coords
    invJ = np.linalg.inv(J)
    dN_dx = dN @ invJ

    B = np.zeros((3, 16))
    for a in range(8):
        B[0, 2*a] = dN_dx[a, 0]
        B[1, 2*a+1] = dN_dx[a, 1]
        B[2, 2*a] = dN_dx[a, 1]
        B[2, 2*a+1] = dN_dx[a, 0]

    u_elem = np.zeros(16)
    for a, n in enumerate(elem):
        u_elem[2*a] = U2[2*n]
        u_elem[2*a+1] = U2[2*n+1]

    D = E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    stress = D @ B @ u_elem

    for a, n in enumerate(elem):
        idx = n + n_nodes1
        sigma_xx_mortar[idx] += stress[0]
        count[idx] += 1

sigma_xx_mortar /= np.where(count > 0, count, 1)

def plot_coupled_deformation(nodes1, elems1, U1,
                             nodes2, elems2, U2,
                             scale=4000):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(f"Coupled deformed mesh  (scale × {scale})", fontweight='bold')

    mesh_data = [
        (nodes1, elems1, U1, 'steelblue', 'Mesh 1 (coarse)'),
        (nodes2, elems2, U2, 'tomato', 'Mesh 2 (fine)')
    ]

    for nodes, elems, U, color, label in mesh_data:
        for i, elem in enumerate(elems):
            order = [0, 1, 2, 3, 4, 5, 6, 7, 0]
            pts   = nodes[elem]
            u     = U[2*np.array(elem)]
            v     = U[2*np.array(elem) + 1]
            def_pts = pts + scale * np.column_stack((u, v))

            ax.plot(pts[order,0],     pts[order,1],     color='gray',  lw=0.7, ls='--')
            # Add label only for the first element of each mesh
            elem_label = label if i == 0 else ""
            ax.plot(def_pts[order,0], def_pts[order,1], color=color, lw=1.5, label=elem_label)

    ax.axvline(x_split, color='k', ls=':', lw=1.5, label=f'Interface x={x_split}')
    ax.set_aspect('equal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    
plot_coupled_deformation(nodes1, elems1, U1, nodes2, elems2, U2)

def to_grid(x, y, z, nx=200, ny=50):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    return Xi, Yi, Zi

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
fig.suptitle("Mortar Coupling (Q8) vs Analytical — Cantilever Beam", fontsize=14, fontweight='bold')

fields = [
    (ux_mortar, ux_ana, r"$u_x$  [m]"),
    (uy_mortar, uy_ana, r"$u_y$  [m]"),
    (sigma_xx_mortar, sxx_ana, r"$\sigma_{xx}$  [Pa]")
]

cmap_field = 'RdBu_r'
cmap_error = 'OrRd'

for row, (fem_vals, ana_vals, label) in enumerate(fields):
    Xf, Yf, Zf   = to_grid(nodes_combined[:, 0], nodes_combined[:, 1], fem_vals)
    _,  _,  Za   = to_grid(nodes_combined[:, 0], nodes_combined[:, 1], ana_vals)
    _,  _,  Zerr = to_grid(nodes_combined[:, 0], nodes_combined[:, 1], np.abs(fem_vals - ana_vals))

    vmin = min(np.nanmin(Zf), np.nanmin(Za))
    vmax = max(np.nanmax(Zf), np.nanmax(Za))

    ax = axes[row, 0]
    im = ax.contourf(Xf, Yf, Zf, levels=20, cmap=cmap_field, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{label}  —  Mortar")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect('equal')
    ax.axvline(x_split, color='k', ls='--', lw=1, alpha=0.5)

    ax = axes[row, 1]
    im = ax.contourf(Xf, Yf, Za, levels=20, cmap=cmap_field, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{label}  —  Analytical")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect('equal')
    ax.axvline(x_split, color='k', ls='--', lw=1, alpha=0.5)

    ax = axes[row, 2]
    im = ax.contourf(Xf, Yf, Zerr, levels=20, cmap=cmap_error)
    plt.colorbar(im, ax=ax, shrink=0.8)
    rel = np.linalg.norm(fem_vals - ana_vals) / (np.linalg.norm(ana_vals) + 1e-30) * 100
    ax.set_title(f"|Mortar − Analytical|  —  {label}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect('equal')
    ax.axvline(x_split, color='k', ls='--', lw=1, alpha=0.5)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Tip cross-section  (x = L,  all y)", fontsize=13, fontweight='bold')

tip_mask2 = np.abs(nodes2[:, 0] - L) < tol
y_tip2 = nodes2[tip_mask2, 1]
sort_idx2 = np.argsort(y_tip2)
y_s2 = y_tip2[sort_idx2]

tip_ana_mask = np.abs(x_combined - L) < tol
y_ana_tip = y_combined[tip_ana_mask]
sort_ana = np.argsort(y_ana_tip)

for ax, (fem_v_field, ana_v_field, lbl) in zip(
        axes,
        [("uy", uy_mortar, uy_ana),
         ("ux", ux_mortar, ux_ana)],
        [r"$u_y$ at $x=L$", r"$u_x$ at $x=L$"]):

    field_name, fem_full, ana_full = fem_v_field
    fem_tip = fem_full[tip_ana_mask][sort_ana] if field_name == "uy" else fem_full[tip_ana_mask][sort_ana]
    ana_tip = ana_full[tip_ana_mask][sort_ana]

    ax.plot(y_ana_tip[sort_ana], ana_tip, 'k-', lw=2, label="Analytical")
    ax.plot(y_s2, fem_tip[:len(y_s2)], 'r--', lw=1.5, label="Mortar (Q8)", marker='o', ms=4)
    ax.set_xlabel("y [m]")
    ax.set_ylabel(lbl)
    ax.set_title(lbl)
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Centreline deflection — Mortar vs Analytical", fontweight='bold')

for ax, field_name in zip(axes, ['uy', 'ux']):
    # Mortar results
    mask = np.abs(y_combined) < 1e-8
    xs = x_combined[mask]
    sort = np.argsort(xs)
    xs = xs[sort]

    if field_name == 'uy':
        vals_mortar = uy_mortar[mask][sort]
        title = r"$u_y$ (vertical deflection) [m]"
    else:
        vals_mortar = ux_mortar[mask][sort]
        title = r"$u_x$ (horizontal displacement) [m]"

    ax.plot(xs, vals_mortar, 'ro-', ms=3, lw=1.5, label='Mortar coupling')

    x_ana = np.linspace(0, L, 300)
    if field_name == 'uy':
        ana_vals = analytical_uy(x_ana, L, H, P, E)
    else:
        ana_vals = analytical_ux(x_ana, 0.0, L, H, P, E)

    ax.plot(x_ana, ana_vals, 'k--', lw=2, label='Analytical')
    ax.axvline(x_split, color='gray', ls=':', lw=1, label=f'Interface x={x_split}')
    ax.set_xlabel("x [m]")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, ls='--', alpha=0.4)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f"Interface continuity at x = {x_split}", fontweight='bold')

y_slave  = nodes1[slave_iface,  1]
y_master = nodes2[master_iface, 1]

sort_s = np.argsort(y_slave)
sort_m = np.argsort(y_master)

uy_slave  = U1[2*slave_iface  + 1][sort_s]
uy_master = U2[2*master_iface + 1][sort_m]
ux_slave  = U1[2*slave_iface    ][sort_s]
ux_master = U2[2*master_iface   ][sort_m]

for ax, vs, vm, ttl in zip(
        axes,
        [uy_slave,  ux_slave ],
        [uy_master, ux_master],
        [r"$u_y$ at interface", r"$u_x$ at interface"]):

    ax.plot(y_slave [sort_s], vs, 'o-', color='steelblue', ms=5, label='Slave  (mesh 1)')
    ax.plot(y_master[sort_m], vm, 's--', color='tomato',  ms=5, label='Master (mesh 2)')
    ax.set_xlabel("y [m]")
    ax.set_ylabel(ttl)
    ax.set_title(ttl)
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)

plt.tight_layout()
plt.show()

err_ux = np.linalg.norm(ux_mortar - ux_ana) / (np.linalg.norm(ux_ana) + 1e-30)
err_uy = np.linalg.norm(uy_mortar - uy_ana) / (np.linalg.norm(uy_ana) + 1e-30)
err_sxx = np.linalg.norm(sigma_xx_mortar - sxx_ana) / (np.linalg.norm(sxx_ana) + 1e-30)

print("\n" + "="*50)
print("Mortar Coupling Results")
print("="*50)
print("\n=== Relative L2 errors (vs Analytical) ===")
print(f"  u_x  :  {err_ux  * 100:.3f} %")
print(f"  u_y  :  {err_uy  * 100:.3f} %")
print(f"  σ_xx :  {err_sxx * 100:.3f} %")

tip_uy_mortar = uy_mortar[np.abs(x_combined - L) < tol].mean()
tip_uy_ana = uy_ana[np.abs(x_combined - L) < tol].mean()
print(f"\n=== Tip deflection (mean over right edge) ===")
print(f"  Mortar     : {tip_uy_mortar:.6e} m")
print(f"  Analytical : {tip_uy_ana:.6e} m")
print(f"  Error      : {abs(tip_uy_mortar - tip_uy_ana)/abs(tip_uy_ana)*100:.3f} %")

print(f"\n=== Interface Quality ===")
print(f"  Interface gap (L2 norm): {gap:.3e} m")
print(f"  Max displacement jump   : {np.max(np.abs(uy_master - uy_slave)):.3e} m (uy)")
print(f"  Max displacement jump   : {np.max(np.abs(ux_master - ux_slave)):.3e} m (ux)")
