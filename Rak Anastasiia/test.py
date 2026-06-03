
import numpy as np
import mesh_2d as m2d
import system_2d as s2d
import finite_element_2d as f2e
import graph2d as g2d
def ug_zero(x, y):
    return 0.0

ug_strict_quarantine = [
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # y = 0
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # x = 1
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # y = 1
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero]   # x = 0
]

ug_open_border_right = [
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # y = 0
    [f2e.TypeOfBoundCond.NEUMANN, ug_zero],    # x = 1
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero],  # y = 1
    [f2e.TypeOfBoundCond.DIRICHLET, ug_zero]   # x = 0
]

def k_exp1(x, y): return 0.01
def beta_exp1(x, y): return 0.02
def init_exp1(x, y):
    return 2.0 * np.exp(-30 * ((x - 0.3)**2 + (y - 0.5)**2))
def f_exp1(x, y):
    return 5.0 * np.exp(-50 * ((x - 0.7)**2 + (y - 0.5)**2))

def k_exp2(x, y):
    if 0.45 <= x <= 0.55 and 0.2 <= y <= 0.8:
        return 0.0001  # Стіна
    return 0.015       

def beta_exp2(x, y): return 0.01
def init_exp2(x, y):
    return 2.5 * np.exp(-30 * ((x - 0.2)**2 + (y - 0.5)**2))

def k_exp3(x, y): return 0.015
def beta_exp3(x, y):
    if x >= 0.5: return 0.15   
    return 0.005               
def init_exp3(x, y):
    return 2.5 * np.exp(-30 * ((x - 0.3)**2 + (y - 0.5)**2))

def run_simulation(exp_name, k_func, beta_func, init_func, f_func, boundary_conds, t_end, dt, times_to_save):
    print(f"\n{'='*50}\nЗАПУСК СИМУЛЯЦІЇ: {exp_name}\n{'='*50}")
    
    p, m, ap = 20, 20, 2
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
    theta = 0.5        

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
        
        b_vector = f2e.apply_boundary_conditions_vector(b_vector, p, m, NL, boundary_conds, ap, current_time=t_n1)
        
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
                    
    g2d.generate_static_plots(U_history, time_history, NL, times_to_save, prefix=exp_name, title=exp_name.replace('_', ' ').capitalize())

def main():
    run_simulation(
        exp_name="Exp1_Active_Source",
        k_func=k_exp1, beta_func=beta_exp1, init_func=init_exp1, f_func=f_exp1, boundary_conds=ug_strict_quarantine,
        t_end=3.0, dt=0.05, times_to_save=[0.0, 1.0, 3.0]
    )
    run_simulation(
        exp_name="Exp2_Barrier_Neumann",
        k_func=k_exp2, beta_func=beta_exp2, init_func=init_exp2, f_func=None, boundary_conds=ug_open_border_right,
        t_end=2.5, dt=0.05, times_to_save=[0.0, 1.0, 2.5]
    )
    run_simulation(
        exp_name="Exp3_Quarantine",
        k_func=k_exp3, beta_func=beta_exp3, init_func=init_exp3, f_func=None, boundary_conds=ug_strict_quarantine,
        t_end=2.5, dt=0.05, times_to_save=[0.0, 1.0, 2.5]
    )

if __name__ == '__main__':
    main()