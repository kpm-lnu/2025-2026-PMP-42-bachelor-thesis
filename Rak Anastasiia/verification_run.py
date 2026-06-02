
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata

import mesh_2d as m2d
import system_2d as s2d
import finite_element_2d as f2e
import graph2d as g2d

BETA_VAL = 0.5
K1       = lambda x, y: 1.0
K2       = lambda x, y: 1.0
BETA     = lambda x, y: BETA_VAL

def ug_zero(x, y):
    return 0.0

def exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

def f_space(x, y):
    return (2 * np.pi**2 - 1 + BETA_VAL) * np.sin(np.pi * x) * np.sin(np.pi * y)

def run_verification(ap, p=10, m=10, dt=0.01, t_end=1.0, save_every=5,
                     save_gif=True, show_plots=True):
    print(f"\n{'='*55}")
    print(f"  ВЕРИФІКАЦІЯ   ap = {ap}   сітка {p}×{m}   Δt = {dt}")
    print(f"{'='*55}")

    vertices = [(0,0),(1,0),(1,1),(0,1)]

    ug = [
        [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # γ₁ – нижнє  (y=0)
        [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # γ₂ – праве  (x=1)
        [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # γ₃ – верхнє (y=1)
        [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # γ₄ – ліве   (x=0)
    ]

    NL, EL = m2d.uniform_mesh_with_vertices(vertices, p, m, 'D2QU4N', ap)
    N_dof   = (ap*p + 1) * (ap*m + 1)

    print("  Збірка матриць...")
    K_loc = s2d.compute_element_stiffness(EL, NL, ap, K1, K2)
    M_loc = s2d.compute_element_mass_matrix(EL, NL, ap)
    R_loc = s2d.compute_element_reaction_matrix(EL, NL, ap, BETA)

    K = s2d.assemble_global_matrix(NL, EL, p, m, K_loc, ap)
    M = s2d.assemble_global_matrix(NL, EL, p, m, M_loc, ap)
    R = s2d.assemble_global_matrix(NL, EL, p, m, R_loc, ap)

    KR = K + R

    theta     = 0.5
    num_steps = int(round(t_end / dt))

    A = M + theta * dt * KR
    A = f2e.apply_boundary_conditions_matrix(A, p, m, ug, ap)

    print("  Обчислення вектора навантаження...")
    F_base = s2d.set_up_vector(f_space, NL, EL, p, m, ap)

    U_n = np.array([exact(x, y, 0.0) for x, y in NL])

    print(f"  Інтегрування: {num_steps} кроків...")
    U_history   = [U_n.copy()]
    time_history = [0.0]
    current_t    = 0.0

    right_M = M - (1 - theta) * dt * KR

    for step in range(num_steps):
        t_n       = current_t
        t_n1      = current_t + dt

        F_n   = F_base * np.exp(-t_n)
        F_n1  = F_base * np.exp(-t_n1)

        b = right_M @ U_n + dt * (theta * F_n1 + (1 - theta) * F_n)
        b = f2e.apply_boundary_conditions_vector(b, p, m, NL, ug, ap,
                                                  current_time=t_n1)
        U_n = np.linalg.solve(A, b)

        current_t = t_n1

        if (step + 1) % save_every == 0:
            U_history.append(U_n.copy())
            time_history.append(current_t)

        if (step + 1) % (num_steps // 5) == 0:
            print(f"    крок {step+1:4d}/{num_steps}   t = {current_t:.3f}")

    print("  Інтегрування завершено.")

    exact_final = lambda x, y: exact(x, y, t_end)
    err_L2, err_H1semi, err_W12 = g2d.compute_H1_error(
        exact_final, U_n, NL, EL, ap)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Похибки при t = {t_end:.1f}  (ap = {ap})         │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  L₂-норма          : {err_L2:.4e}      │")
    print(f"  │  H¹-семінорма      : {err_H1semi:.4e}      │")
    print(f"  │  W₂¹ (H¹)-норма   : {err_W12:.4e}      │")
    print(f"  └─────────────────────────────────────┘")

    if show_plots:
        g2d.plot_2d_solution2(U_n, NL, EL, exact_solution=exact_final)

    gif_name = f"verification_ap{ap}_dt{dt}.gif" if save_gif else None
    g2d.animate_solution(
        U_history, time_history, NL,
        title=f"Реакція-дифузія  ap={ap}",
        interval=120,
        save_filename=gif_name
    )

    return {"ap": ap, "L2": err_L2, "H1semi": err_H1semi, "W12": err_W12}


def print_summary(results):
    print("\n")
    print("╔══════╦══════════════╦══════════════╦══════════════╗")
    print("║  ap  ║   L₂-норма   ║ H¹-семінорма ║  W₂¹-норма  ║")
    print("╠══════╬══════════════╬══════════════╬══════════════╣")
    for r in results:
        print(f"║  {r['ap']}   ║  {r['L2']:.4e}  ║  {r['H1semi']:.4e}  ║  {r['W12']:.4e}  ║")
    print("╚══════╩══════════════╩══════════════╩══════════════╝")

    if len(results) >= 2:
        print("\n  Зниження H¹-семінорми:")
        for i in range(1, len(results)):
            ratio = results[i-1]['H1semi'] / results[i]['H1semi']
            print(f"    ap={results[i-1]['ap']} → ap={results[i]['ap']}:  "
                  f"{results[i-1]['H1semi']:.4e} / {results[i]['H1semi']:.4e} = {ratio:.2f}×")


if __name__ == "__main__":
    results = []
    for ap in [1, 2, 3]:
        r = run_verification(
            ap        = ap,
            p         = 10,
            m         = 10,
            dt        = 0.01,
            t_end     = 1.0,
            save_every= 5,
            save_gif  = True,
            show_plots= True
        )
        results.append(r)

    print_summary(results)