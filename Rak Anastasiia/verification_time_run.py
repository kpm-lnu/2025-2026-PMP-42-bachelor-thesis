

import numpy as np
import matplotlib.pyplot as plt
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
    """Просторова частина джерела"""
    return (2 * np.pi**2 - 1 + BETA_VAL) * np.sin(np.pi * x) * np.sin(np.pi * y)

def run_time_convergence():
    ap = 2
    p, m = 20, 20
    t_end = 1.0
    theta = 0.5

    print(f"\n{'='*60}")
    print(f" ДОСЛІДЖЕННЯ ЗБІЖНОСТІ ЗА ЧАСОМ (ap={ap}, сітка {p}×{m})")
    print(f"{'='*60}")

    vertices = [(0,0),(1,0),(1,1),(0,1)]
    ug = [
        [f2e.TypeOfBoundCond.DIRICHLET, ug_zero], 
        [f2e.TypeOfBoundCond.DIRICHLET, ug_zero], 
        [f2e.TypeOfBoundCond.DIRICHLET, ug_zero], 
        [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  
    ]

    print("\n1. Генерація сітки та просторових матриць...")
    print("   (Це робиться лише 1 раз, зачекайте 1-2 хвилини)")
    NL, EL = m2d.uniform_mesh_with_vertices(vertices, p, m, 'D2QU4N', ap)
    
    K_loc = s2d.compute_element_stiffness(EL, NL, ap, K1, K2)
    M_loc = s2d.compute_element_mass_matrix(EL, NL, ap)
    R_loc = s2d.compute_element_reaction_matrix(EL, NL, ap, BETA)

    K = s2d.assemble_global_matrix(NL, EL, p, m, K_loc, ap)
    M = s2d.assemble_global_matrix(NL, EL, p, m, M_loc, ap)
    R = s2d.assemble_global_matrix(NL, EL, p, m, R_loc, ap)
    
    KR = K + R
    F_base = s2d.set_up_vector(f_space, NL, EL, p, m, ap)
    U_0 = np.array([exact(x, y, 0.0) for x, y in NL])

    dt_values = [0.1, 0.05, 0.025, 0.01]
    results = []

    print("\n2. Починаємо інтегрування за часом для різних dt:")

    for dt in dt_values:
        num_steps = int(round(t_end / dt))
        print(f"\n  >>> Запуск: dt = {dt:.3f} ({num_steps} кроків)")

        A = M + theta * dt * KR
        A = f2e.apply_boundary_conditions_matrix(A, p, m, ug, ap)
        right_M = M - (1 - theta) * dt * KR

        U_n = U_0.copy()
        current_t = 0.0
        
        save_every = max(1, num_steps // 20)
        U_history = [U_n.copy()]
        time_history = [0.0]

        for step in range(num_steps):
            t_n  = current_t
            t_n1 = current_t + dt

            F_n  = F_base * np.exp(-t_n)
            F_n1 = F_base * np.exp(-t_n1)

            b = right_M @ U_n + dt * (theta * F_n1 + (1 - theta) * F_n)
            b = f2e.apply_boundary_conditions_vector(b, p, m, NL, ug, ap, current_time=t_n1)
            
            U_n = np.linalg.solve(A, b)
            current_t = t_n1

            if (step + 1) % save_every == 0 or (step + 1) == num_steps:
                U_history.append(U_n.copy())
                time_history.append(current_t)

        exact_final = lambda x, y: exact(x, y, t_end)
        err_L2, _, _ = g2d.compute_H1_error(exact_final, U_n, NL, EL, ap)
        
        print(f"      Похибка L2 = {err_L2:.6e}")
        results.append({"dt": dt, "steps": num_steps, "L2": err_L2})

        if dt == 0.01:
            gif_name = f"time_convergence_dt_{dt}.gif"
            print(f"      Збереження анімації у {gif_name} ...")
            g2d.animate_solution(
                U_history, time_history, NL,
                title=f"Кранк-Ніколсон (dt={dt})",
                interval=100,
                save_filename=gif_name
            )

    print("\n\n╔════════════╦══════════════╦══════════════════╦════════════════════╗")
    print("║ Крок (Δt)  ║ К-сть кроків ║ Похибка L₂       ║ Відношення похибок ║")
    print("╠════════════╬══════════════╬══════════════════╬════════════════════╣")
    
    for i in range(len(results)):
        dt_val = results[i]["dt"]
        steps  = results[i]["steps"]
        err    = results[i]["L2"]
        
        if i == 0:
            ratio_str = "-"
        else:
            ratio = results[i-1]["L2"] / err
            ratio_str = f"{ratio:.2f} разів"
            
        print(f"║ {dt_val:<10.3f} ║ {steps:<12d} ║ {err:.4e}     ║ {ratio_str:<18} ║")
    print("╚════════════╩══════════════╩══════════════════╩════════════════════╝")

if __name__ == "__main__":
    run_time_convergence()