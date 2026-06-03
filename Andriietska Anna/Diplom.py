import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =====================================================================
# МАТЕМАТИЧНА ЧАСТИНА 
# =====================================================================

def initial_condition(x: np.ndarray) -> np.ndarray:
    return np.exp(-100.0 * (x - 0.2) ** 2)

def source_term(x: np.ndarray, t: float) -> np.ndarray:
    return np.zeros_like(x)

def apply_dirichlet_bc(matrix: np.ndarray,
                       rhs: np.ndarray,
                       left_value: float,
                       right_value: float) -> tuple[np.ndarray, np.ndarray]:
    A = matrix.copy()
    b = rhs.copy()
    n = A.shape[0]
    A[0, :] = 0.0
    A[:, 0] = 0.0
    A[0, 0] = 1.0
    b[0] = left_value
    A[n - 1, :] = 0.0
    A[:, n - 1] = 0.0
    A[n - 1, n - 1] = 1.0
    b[n - 1] = right_value
    return A, b

def assemble_fem_matrices(n_elements: int,
                          length: float,
                          velocity: float,
                          diffusion: float,
                          reaction: float = 0.0,
                          use_supg: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = n_elements + 1
    x = np.linspace(0.0, length, n_nodes)
    h = length / n_elements
    M = np.zeros((n_nodes, n_nodes))
    K = np.zeros((n_nodes, n_nodes))
    M_loc = (h / 6.0) * np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ])
    K_diff = (diffusion / h) * np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ])
    K_adv = (velocity / 2.0) * np.array([
        [-1.0, 1.0],
        [-1.0, 1.0]
    ])
    K_reac = reaction * M_loc
    if use_supg:
        abs_u = abs(velocity)
        if abs_u > 1e-14:
            pe = abs_u * h / (2.0 * diffusion) if diffusion > 0 else np.inf
            if diffusion > 0:
                coth_pe = np.cosh(pe) / np.sinh(pe) if pe < 50 else 1.0
                tau = h / (2.0 * abs_u) * (coth_pe - 1.0 / pe) if pe > 1e-12 else 0.0
            else:
                tau = h / (2.0 * abs_u)
            K_supg = (tau * velocity * velocity / h) * np.array([
                [1.0, -1.0],
                [-1.0, 1.0]
            ])
            M_supg = tau * velocity * np.array([
                [-0.5, -0.5],
                [0.5, 0.5]
            ])
        else:
            K_supg = np.zeros((2, 2))
            M_supg = np.zeros((2, 2))
    else:
        K_supg = np.zeros((2, 2))
        M_supg = np.zeros((2, 2))
    for e in range(n_elements):
        nodes = [e, e + 1]
        M_e = M_loc + M_supg
        K_e = K_diff + K_adv + K_reac + K_supg
        for i_local in range(2):
            for j_local in range(2):
                M[nodes[i_local], nodes[j_local]] += M_e[i_local, j_local]
                K[nodes[i_local], nodes[j_local]] += K_e[i_local, j_local]
    return M, K, x

def assemble_load_vector(n_elements: int,
                         length: float,
                         time_value: float,
                         source_function) -> np.ndarray:
    n_nodes = n_elements + 1
    h = length / n_elements
    b = np.zeros(n_nodes)
    gauss_points = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gauss_weights = np.array([1.0, 1.0])
    for e in range(n_elements):
        x_left = e * h
        x_right = (e + 1) * h
        local_b = np.zeros(2)
        for gp, w in zip(gauss_points, gauss_weights):
            xi = gp
            x_phys = 0.5 * (x_left + x_right) + 0.5 * h * xi
            phi = np.array([
                0.5 * (1.0 - xi),
                0.5 * (1.0 + xi)
            ])
            f_val = source_function(np.array([x_phys]), time_value)[0]
            local_b += w * f_val * phi * (h / 2.0)
        nodes = [e, e + 1]
        b[nodes[0]] += local_b[0]
        b[nodes[1]] += local_b[1]
    return b

def solve_advection_diffusion_1d(n_elements: int = 80,
                                 length: float = 1.0,
                                 final_time: float = 0.4,
                                 dt: float = 0.002,
                                 velocity: float = 2.0,
                                 diffusion: float = 1e-3,
                                 reaction: float = 0.0,
                                 left_bc: float = 0.0,
                                 right_bc: float = 0.0,
                                 use_supg: bool = True) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    M, K, x = assemble_fem_matrices(
        n_elements=n_elements, length=length, velocity=velocity,
        diffusion=diffusion, reaction=reaction, use_supg=use_supg
    )
    n_steps = int(np.round(final_time / dt))
    c = initial_condition(x)
    c[0] = left_bc
    c[-1] = right_bc
    snapshots = [c.copy()]
    times = [0.0]
    system_matrix = M + dt * K
    for step in range(1, n_steps + 1):
        t_new = step * dt
        f_vec = assemble_load_vector(
            n_elements=n_elements, length=length, time_value=t_new, source_function=source_term
        )
        rhs = M @ c + dt * f_vec
        A_bc, rhs_bc = apply_dirichlet_bc(system_matrix, rhs, left_value=left_bc, right_value=right_bc)
        c = np.linalg.solve(A_bc, rhs_bc)
        snapshots.append(c.copy())
        times.append(t_new)
    return x, c, snapshots, times

# =====================================================================
# ГРАФІЧНИЙ ІНТЕРФЕЙС (GUI)
# =====================================================================

class AdvectionDiffusionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("МСЕ: Задача адвекції-дифузії (Керування числом Пекле)")
        self.geometry("1100x750")
        self.configure(padx=10, pady=10)

        # --- Змінні для полів вводу ---
        self.n_elements_var = tk.IntVar(value=100)
        self.length_var = tk.DoubleVar(value=1.0)
        self.final_time_var = tk.DoubleVar(value=0.3)
        self.dt_var = tk.DoubleVar(value=0.001)
        self.velocity_var = tk.DoubleVar(value=3.0)
        self.diffusion_var = tk.DoubleVar(value=0.001)
        self.peclet_var = tk.DoubleVar(value=15.0) # Нова змінна для Пекле
        self.reaction_var = tk.DoubleVar(value=0.0)
        self.left_bc_var = tk.DoubleVar(value=0.0)
        self.right_bc_var = tk.DoubleVar(value=0.0)
        self.use_supg_var = tk.BooleanVar(value=True)
        
        # Режим автообчислення
        self.auto_calc_var = tk.StringVar(value="Число Пекле (Pe)")

        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.LabelFrame(self, text="Параметри задачі", padding=(10, 10))
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Базові параметри
        self.add_input_row(control_frame, "Кількість елементів (N):", self.n_elements_var, 0)
        self.add_input_row(control_frame, "Довжина області (L):", self.length_var, 1)
        self.add_input_row(control_frame, "Кінцевий час (T):", self.final_time_var, 2)
        self.add_input_row(control_frame, "Крок за часом (dt):", self.dt_var, 3)

        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)

        # Режим зв'язку параметрів u, D, Pe
        ttk.Label(control_frame, text="Розраховувати автоматично:").grid(row=5, column=0, sticky="w", pady=5)
        self.mode_cb = ttk.Combobox(control_frame, textvariable=self.auto_calc_var,
                                    values=["Число Пекле (Pe)", "Коефіцієнт дифузії (D)", "Швидкість потоку (u)"],
                                    state="readonly", width=18)
        self.mode_cb.grid(row=5, column=1, sticky="e", pady=5)
        self.mode_cb.bind("<<ComboboxSelected>>", self.toggle_inputs)

        self.u_entry = self.add_input_row(control_frame, "Швидкість потоку (u):", self.velocity_var, 6)
        self.D_entry = self.add_input_row(control_frame, "Коефіцієнт дифузії (D):", self.diffusion_var, 7)
        self.Pe_entry = self.add_input_row(control_frame, "Число Пекле (Pe):", self.peclet_var, 8)

        ttk.Separator(control_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky='ew', pady=10)

        self.add_input_row(control_frame, "Коефіцієнт реакції (r):", self.reaction_var, 10)
        self.add_input_row(control_frame, "Ліва гр. умова (x=0):", self.left_bc_var, 11)
        self.add_input_row(control_frame, "Права гр. умова (x=L):", self.right_bc_var, 12)

        supg_check = ttk.Checkbutton(control_frame, text="Увімкнути SUPG стабілізацію", variable=self.use_supg_var)
        supg_check.grid(row=13, column=0, columnspan=2, pady=10, sticky="w")

        run_btn = ttk.Button(control_frame, text="Розрахувати та Побудувати", command=self.run_simulation)
        run_btn.grid(row=14, column=0, columnspan=2, pady=20, sticky="ew")

        # Права панель для графіків
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.figure.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Ініціалізація полів
        self.toggle_inputs()
        self.run_simulation()

    def add_input_row(self, parent, label_text, variable, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=5, padx=5)
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=1, sticky="e", pady=5, padx=5)
        return entry

    def toggle_inputs(self, event=None):
        mode = self.auto_calc_var.get()
        # Розблокувати всі
        self.Pe_entry.config(state="normal")
        self.D_entry.config(state="normal")
        self.u_entry.config(state="normal")

        # Заблокувати те, що рахується автоматично
        if mode == "Число Пекле (Pe)":
            self.Pe_entry.config(state="readonly")
        elif mode == "Коефіцієнт дифузії (D)":
            self.D_entry.config(state="readonly")
        elif mode == "Швидкість потоку (u)":
            self.u_entry.config(state="readonly")

    def run_simulation(self):
        try:
            n_el = self.n_elements_var.get()
            L = self.length_var.get()
            T = self.final_time_var.get()
            dt = self.dt_var.get()
            r = self.reaction_var.get()
            l_bc = self.left_bc_var.get()
            r_bc = self.right_bc_var.get()
            supg = self.use_supg_var.get()
            
            mode = self.auto_calc_var.get()
            h = L / n_el
            
            # Математика автообчислення
            if mode == "Число Пекле (Pe)":
                u = self.velocity_var.get()
                D = self.diffusion_var.get()
                pe = abs(u) * h / (2.0 * D) if D > 0 else np.inf
                self.peclet_var.set(round(pe, 3))
            elif mode == "Коефіцієнт дифузії (D)":
                u = self.velocity_var.get()
                pe = self.peclet_var.get()
                D = abs(u) * h / (2.0 * pe) if pe > 0 else 0.0
                self.diffusion_var.set(round(D, 6))
            elif mode == "Швидкість потоку (u)":
                D = self.diffusion_var.get()
                pe = self.peclet_var.get()
                u = (2.0 * pe * D) / h
                self.velocity_var.set(round(u, 3))

            # Оновлюємо значення після обчислень
            u = self.velocity_var.get()
            D = self.diffusion_var.get()

        except tk.TclError:
            return 
        
        # Виклик розрахунку
        x, c_final, snapshots, times = solve_advection_diffusion_1d(
            n_elements=n_el, length=L, final_time=T, dt=dt,
            velocity=u, diffusion=D, reaction=r,
            left_bc=l_bc, right_bc=r_bc, use_supg=supg
        )

        selected_indices = [0, len(times) // 4, len(times) // 2, 3 * len(times) // 4, len(times) - 1]

        self.ax1.clear()
        self.ax2.clear()

        for idx in selected_indices:
            self.ax1.plot(x, snapshots[idx], label=f"t = {times[idx]:.3f}")
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("c(x,t)")
        self.ax1.set_title(f"Зрізи розв'язку у часі (Pe = {self.peclet_var.get()})")
        self.ax1.grid(True)
        self.ax1.legend()

        self.ax2.plot(x, c_final, linewidth=2, color='tab:blue')
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel(f"c(x, {T})")
        self.ax2.set_title("Кінцевий розв'язок")
        self.ax2.grid(True)

        self.canvas.draw()

if __name__ == "__main__":
    app = AdvectionDiffusionApp()
    app.mainloop()