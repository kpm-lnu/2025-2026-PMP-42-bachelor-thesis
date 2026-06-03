"""


Розв'язання двовимірного нестаціонарного рівняння реакції-дифузії
методом скінченних елементів зі схемою Кранка–Ніколсона за часом.

У цьому файлі залишено ЛИШЕ перший тест (Exp1_Active_Source)
та апроксимацію 1 (лінійні скінченні елементи, ap=1).

Запуск:
    python main.py

Результат:
    - Exp1_Active_Source.gif — анімація розв'язку
    - Exp1_Active_Source_t0_0.png, *_t1_0.png, *_t3_0.png — статичні зрізи
"""

import numpy as np
import mesh_2d as m2d
import system_2d as s2d
import finite_element_2d as f2e
import graph2d as g2d


def ug_zero(x, y):
    """Нульова гранична умова (повний карантин)."""
    return 0.0


# Граничні умови першого тесту: Діріхле = 0 на всіх чотирьох межах
ug_strict_quarantine = [
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # y = 0
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # x = 1
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # y = 1
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero]   # x = 0
]


# --- Параметри першого тесту (Exp1_Active_Source) ---
def k_exp1(x, y):
    """Коефіцієнт дифузії."""
    return 0.01


def beta_exp1(x, y):
    """Коефіцієнт реакції."""
    return 0.02


def init_exp1(x, y):
    """Початкова умова — локалізований осередок."""
    return 2.0 * np.exp(-30 * ((x - 0.3)**2 + (y - 0.5)**2))


def f_exp1(x, y):
    """Зовнішнє джерело."""
    return 5.0 * np.exp(-50 * ((x - 0.7)**2 + (y - 0.5)**2))


def run_simulation(exp_name, k_func, beta_func, init_func, f_func,
                   boundary_conds, t_end, dt, times_to_save, ap=1):
    print(f"\n{'='*50}\nЗАПУСК СИМУЛЯЦІЇ: {exp_name}\n{'='*50}")

    # ap=1 — лінійна апроксимація (білінійні чотирикутні елементи D2QU4N)
    p, m = 20, 20
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    NL, EL = m2d.uniform_mesh_with_vertices(vertices, p, m, 'D2QU4N', ap)

    print("Обчислення матриць...")
    K_loc = s2d.compute_element_stiffness(EL, NL, ap, k1=k_func, k2=k_func)
    M_loc = s2d.compute_element_mass_matrix(EL, NL, ap)
    R_loc = s2d.compute_element_reaction_matrix(EL, NL, ap, beta_func)

    K = s2d.assemble_global_matrix(NL, EL, p, m, K_loc, ap)
    M = s2d.assemble_global_matrix(NL, EL, p, m, M_loc, ap)
    R = s2d.assemble_global_matrix(NL, EL, p, m, R_loc, ap)

    KR = K + R
    num_steps = int(t_end / dt)
    theta = 0.5  # схема Кранка–Ніколсона

    save_every = max(1, num_steps // 40)
    U_history, time_history = [], []

    A_matrix = M + theta * dt * KR
    A_matrix = f2e.apply_boundary_conditions_matrix(A_matrix, p, m, boundary_conds, ap)

    num_total_nodes = (ap * p + 1) * (ap * m + 1)

    print("Встановлення початкових умов...")
    U_n = np.zeros(num_total_nodes)
    for i, (x_val, y_val) in enumerate(NL):
        U_n[i] = init_func(x_val, y_val)

    U_history.append(U_n.copy())
    time_history.append(0.0)

    if f_func is not None:
        print("Формування вектора зовнішніх джерел...")
        F_base = s2d.set_up_vector(f_func, NL, EL, p, m, ap)
    else:
        F_base = np.zeros(num_total_nodes)

    print(f"Інтегрування по часу (кроків: {num_steps})...")
    current_t = 0.0
    for step in range(num_steps):
        t_n1 = current_t + dt

        right_matrix = M - (1 - theta) * dt * KR
        # Додаємо вплив зовнішнього джерела
        b_vector = right_matrix @ U_n + dt * F_base

        b_vector = f2e.apply_boundary_conditions_vector(
            b_vector, p, m, NL, boundary_conds, ap, current_time=t_n1)

        U_n = np.linalg.solve(A_matrix, b_vector)
        current_t = t_n1

        if (step + 1) % save_every == 0 or step == num_steps - 1:
            U_history.append(U_n.copy())
            time_history.append(current_t)

    print("Інтегрування завершено!")

    gif_filename = f"{exp_name}.gif"
    g2d.animate_solution(U_history, time_history, NL,
                         title=exp_name.replace('_', ' ').capitalize(),
                         interval=100, save_filename=gif_filename)

    g2d.generate_static_plots(U_history, time_history, NL, times_to_save,
                              prefix=exp_name,
                              title=exp_name.replace('_', ' ').capitalize())


def main():
    run_simulation(
        exp_name="Exp1_Active_Source",
        k_func=k_exp1,
        beta_func=beta_exp1,
        init_func=init_exp1,
        f_func=f_exp1,
        boundary_conds=ug_strict_quarantine,
        t_end=3.0,
        dt=0.05,
        times_to_save=[0.0, 1.0, 3.0],
        ap=1  # апроксимація 1
    )


if __name__ == '__main__':
    main()