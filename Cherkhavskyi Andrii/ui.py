import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as patches
from planner import ObstacleAvoidancePathPlanner
import time

# Імпорти для роботи з геометрією
from shapely.geometry import Polygon, Point  
from shapely.ops import unary_union

# Імпорти для збереження/завантаження у форматі JSON
import json
import tkinter as tk
from tkinter import filedialog

class InteractiveMapCreator:
    def __init__(self):
        self.stage = 0  
        self.map_size = 10
        self.num_points = 300
        self.safety_dist = 0.4
        self.start_point = None
        self.goal_point = None
        self.obstacles = []
        self.current_obstacle = []

        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_axes([0.05, 0.08, 0.65, 0.85])
        
        self.start_marker = None
        self.goal_marker = None
        
        self.setup_ui()
        self.setup_event_handlers()
        
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.update_title()
        
        plt.show()

    def setup_ui(self):
        self.txt = {}
        # Загальні налаштування
        self.txt['size'] = TextBox(plt.axes([0.82, 0.85, 0.12, 0.05]), 'Розмір карти: ', initial=str(self.map_size))
        self.txt['pts'] = TextBox(plt.axes([0.82, 0.77, 0.12, 0.05]), 'К-сть точок: ', initial=str(self.num_points))
        self.txt['dist'] = TextBox(plt.axes([0.82, 0.69, 0.12, 0.05]), 'Відступ: ', initial=str(self.safety_dist))
        
        # Поля для вводу координат перешкод, старту і фінішу
        self.txt['obs_x'] = TextBox(plt.axes([0.82, 0.58, 0.12, 0.05]), 'Точка X: ', initial='')
        self.txt['obs_y'] = TextBox(plt.axes([0.82, 0.51, 0.12, 0.05]), 'Точка Y: ', initial='')
        
        # Кнопки керування координатами
        self.add_pt_btn = Button(plt.axes([0.75, 0.44, 0.2, 0.05]), 'Додати точку', color='#e6e6e6', hovercolor='#cccccc')
        self.close_obs_btn = Button(plt.axes([0.75, 0.37, 0.2, 0.05]), 'Замкнути', color='#ffdb99', hovercolor='#ffb84d')

        # Основні кнопки керування алгоритмом
        self.clear_btn = Button(plt.axes([0.75, 0.27, 0.2, 0.05]), 'Очистити', color='#ffcccc', hovercolor='#ff9999')
        self.next_btn = Button(plt.axes([0.75, 0.19, 0.2, 0.06]), 'Далі', color='#ccffcc', hovercolor='#99ff99')

        # Кнопки для збереження та завантаження JSON
        self.save_btn = Button(plt.axes([0.75, 0.11, 0.09, 0.05]), 'Зберегти', color='#e6f2ff', hovercolor='#b3d9ff')
        self.load_btn = Button(plt.axes([0.86, 0.11, 0.09, 0.05]), 'Відкрити', color='#e6f2ff', hovercolor='#b3d9ff')

    def setup_event_handlers(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.clear_btn.on_clicked(self.clear_map)
        self.next_btn.on_clicked(lambda e: self.next_stage())
        self.txt['size'].on_submit(self.update_map_size)
        
        self.add_pt_btn.on_clicked(self.add_point_from_text)
        self.close_obs_btn.on_clicked(self.close_obstacle_from_text)
        
        self.save_btn.on_clicked(self.save_to_file)
        self.load_btn.on_clicked(self.load_from_file)

    def save_to_file(self, event):
        root = tk.Tk()
        root.withdraw() 
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Зберегти карту як..."
        )
        if not file_path: return
        
        data = {
            "map_size": self.map_size,
            "start": self.start_point,
            "goal": self.goal_point,
            "obstacles": self.obstacles
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"\n[УСПІХ] Карту збережено у файл: {file_path}")
        except Exception as e:
            print(f"\n[ПОМИЛКА] Не вдалося зберегти файл: {e}")

    def load_from_file(self, event):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Відкрити карту..."
        )
        if not file_path: return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.clear_map(None)
            
            self.map_size = data.get("map_size", 10)
            self.txt['size'].set_val(str(self.map_size))
            self.ax.set_xlim(0, self.map_size)
            self.ax.set_ylim(0, self.map_size)
            
            if data.get("start"): self.start_point = tuple(data["start"])
            if data.get("goal"): self.goal_point = tuple(data["goal"])
            
            if data.get("obstacles"):
                self.obstacles = data["obstacles"]
                for obs in self.obstacles:
                    self.ax.add_patch(patches.Polygon(obs, color='red', alpha=0.3))
            
            self.draw_markers()
            
            self.stage = 3
            self.update_title()
            self.fig.canvas.draw_idle()
            
            print(f"\n[УСПІХ] Карту завантажено з файлу: {file_path}")
        except Exception as e:
            print(f"\n[ПОМИЛКА] Не вдалося завантажити файл: {e}")

    def update_map_size(self, text):
        if self.stage == 0:
            try:
                self.map_size = float(text)
                self.ax.set_xlim(0, self.map_size)
                self.ax.set_ylim(0, self.map_size)
                self.fig.canvas.draw_idle()
            except ValueError: 
                pass

    def update_title(self):
        titles = [
            "Крок 1: Введіть розмір карти справа та натисніть 'Далі'", 
            "Крок 2: Малюйте (ЛКМ) АБО вводьте координати. ПКМ - замкнути", 
            "Крок 3: Встановіть Старт та Фініш (ЛКМ АБО координати) і 'Далі'", 
            "Крок 4: Вкажіть Відступ та Точки справа, натисніть 'Далі'"
        ]
        if self.stage < len(titles):
            self.ax.set_title(titles[self.stage], fontsize=13, fontweight='bold', color='black')
            self.fig.canvas.draw_idle()

    def add_point_from_text(self, event):
        try:
            x = float(self.txt['obs_x'].text.replace(',', '.'))
            y = float(self.txt['obs_y'].text.replace(',', '.'))
            
            if self.stage == 1:
                # Додавання точок для перешкод
                self.current_obstacle.append((x, y))
                self.ax.plot(x, y, 'r.', markersize=8)
                self.fig.canvas.draw_idle()
                
                print(f"Точку перешкоди додано: X={x}, Y={y}")
                self.txt['obs_x'].set_val('')
                self.txt['obs_y'].set_val('')
                
            elif self.stage == 2:
                # Додавання точок для Старту і Фінішу
                pt = Point(x, y)
                in_obstacle = False
                for obs in self.obstacles:
                    if len(obs) >= 3:
                        poly = Polygon(obs)
                        if poly.contains(pt):
                            in_obstacle = True
                            break
                
                if in_obstacle:
                    self.ax.set_title("ПОМИЛКА: Точка на перешкоді! Виберіть вільне місце.", color='red', fontweight='bold')
                    self.fig.canvas.draw_idle()
                    return
                
                self.update_title()

                if self.start_point is None: 
                    self.start_point = (x, y)
                    print(f"Встановлено СТАРТ: ({x}, {y})")
                elif self.goal_point is None: 
                    self.goal_point = (x, y)
                    print(f"Встановлено ФІНІШ: ({x}, {y})")
                    
                self.draw_markers()
                self.txt['obs_x'].set_val('')
                self.txt['obs_y'].set_val('')
                
        except ValueError:
            pass

    def close_obstacle_from_text(self, event):
        if self.stage == 1 and len(self.current_obstacle) >= 3:
            self.obstacles.append(self.current_obstacle)
            self.ax.add_patch(patches.Polygon(self.current_obstacle, color='red', alpha=0.3))
            
            rounded_obs = [(round(float(px), 2), round(float(py), 2)) for px, py in self.current_obstacle]
            print(f"Створено перешкоду: {rounded_obs}")
            
            self.current_obstacle = []
            self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        
        if self.stage == 1:
            if event.button == 1: 
                self.current_obstacle.append((event.xdata, event.ydata))
                self.ax.plot(event.xdata, event.ydata, 'r.', markersize=8)
                print(f"Точку додано: X={round(float(event.xdata), 2)}, Y={round(float(event.ydata), 2)}")
            elif event.button == 3 and len(self.current_obstacle) >= 3:
                self.obstacles.append(self.current_obstacle)
                self.ax.add_patch(patches.Polygon(self.current_obstacle, color='red', alpha=0.3))
                self.current_obstacle = []
                
        elif self.stage == 2:
            if event.button == 1:
                pt = Point(event.xdata, event.ydata)
                in_obstacle = False
                for obs in self.obstacles:
                    if len(obs) >= 3:
                        poly = Polygon(obs)
                        if poly.contains(pt):
                            in_obstacle = True
                            break
                
                if in_obstacle:
                    self.ax.set_title("ПОМИЛКА: Точка на перешкоді! Виберіть вільне місце.", color='red', fontweight='bold')
                    self.fig.canvas.draw_idle()
                    return
                
                self.update_title()

                if self.start_point is None: 
                    self.start_point = (event.xdata, event.ydata)
                    print(f"Встановлено СТАРТ: ({round(float(event.xdata), 2)}, {round(float(event.ydata), 2)})")
                elif self.goal_point is None: 
                    self.goal_point = (event.xdata, event.ydata)
                    print(f"Встановлено ФІНІШ: ({round(float(event.xdata), 2)}, {round(float(event.ydata), 2)})")
                self.draw_markers()
        self.fig.canvas.draw_idle()

    def draw_markers(self):
        for m in [self.start_marker, self.goal_marker]:
            if m: 
                try: m.remove()
                except: pass
        if self.start_point: self.start_marker = self.ax.plot(*self.start_point, 'yo', markersize=12, markeredgecolor='black')[0]
        if self.goal_point: self.goal_marker = self.ax.plot(*self.goal_point, 'mo', markersize=12, markeredgecolor='black')[0]
        self.fig.canvas.draw_idle()

    def next_stage(self):
        if self.stage == 0: 
            try:
                self.map_size = float(self.txt['size'].text)
                self.ax.set_xlim(0, self.map_size)
                self.ax.set_ylim(0, self.map_size)
                self.fig.canvas.draw_idle()
            except ValueError:
                pass
            self.stage += 1
            self.update_title()
            
        elif self.stage == 1:
            self.stage += 1
            self.update_title()
            
        elif self.stage == 2: 
            if self.start_point is None or self.goal_point is None:
                self.ax.set_title("ПОМИЛКА: Спочатку встановіть Старт і Фініш!", color='red', fontweight='bold')
                self.fig.canvas.draw_idle()
                return
            self.stage += 1
            self.update_title()
            
        elif self.stage == 3:
            self.run_planner()

    def clear_map(self, event):
        self.stage = 0  
        self.start_point = None
        self.goal_point = None
        self.obstacles = []
        self.current_obstacle = []
        self.start_marker = None
        self.goal_marker = None
        
        self.ax.clear()
        self.fig.texts.clear()
        self.txt['obs_x'].set_val('')
        self.txt['obs_y'].set_val('')
        
        try:
            self.map_size = float(self.txt['size'].text)
        except ValueError:
            self.map_size = 10
            
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.update_title()

    def run_planner(self):
        try:
            self.num_points = int(self.txt['pts'].text)
            self.safety_dist = float(self.txt['dist'].text)
        except ValueError:
            pass
            
        valid_obstacles_check = []
        for obs_coords in self.obstacles:
            if len(obs_coords) >= 3:
                poly = Polygon(obs_coords)
                if not poly.is_valid: poly = poly.buffer(0)
                if not poly.is_empty: valid_obstacles_check.append(poly)
                
        obs_union = unary_union(valid_obstacles_check) if valid_obstacles_check else Polygon()
        safe_buffer = obs_union.buffer(self.safety_dist, join_style=1) if not obs_union.is_empty else Polygon()
        
        if safe_buffer.contains(Point(self.start_point)):
            self.ax.set_title("ПОМИЛКА: Старт у зоні відступу! Зменшіть відступ.", color='red', fontweight='bold')
            self.fig.canvas.draw_idle()
            return
            
        if safe_buffer.contains(Point(self.goal_point)):
            self.ax.set_title("ПОМИЛКА: Фініш у зоні відступу! Зменшіть відступ.", color='red', fontweight='bold')
            self.fig.canvas.draw_idle()
            return
        
        self.update_title()

        print("\n" + "="*40)
        print("ЗАПУСК ОБЧИСЛЕННЯ МАРШРУТУ")
        print("="*40)
        print(f"Розмір карти: {self.map_size} x {self.map_size}")
        print(f"Кількість точок (N): {self.num_points}")
        print(f"Відступ: {self.safety_dist}")
        
        self.ax.set_title("Обчислюю маршрут... Будь ласка, зачекайте!", fontsize=14, color='blue', fontweight='bold')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        start_t = time.time()
        
        planner = ObstacleAvoidancePathPlanner(
            self.start_point, self.goal_point, self.obstacles, 
            self.map_size, self.num_points, self.safety_dist,
            fig=self.fig, ax=self.ax
        )
        
        path, points, triangles, exists, dist, timing_info = planner.plan_path()
        calc_time = time.time() - start_t
        
        print(f"\n--- Деталізація часу виконання ---")
        for step_name, step_time in timing_info.items():
            print(f"{step_name}: {step_time:.4f} секунд")
        print("----------------------------------")
        
        print(f"Обчислення завершено за {calc_time:.3f} секунд.")
        if exists: print(f"Шлях ЗНАЙДЕНО. Загальна довжина: {dist:.2f}")
        else: print("Шлях НЕ ЗНАЙДЕНО.")
            
        planner.visualize(path, points, triangles, exists, dist, calc_time)