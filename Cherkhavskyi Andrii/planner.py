import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import time # Додано для заміру часу

class ObstacleAvoidancePathPlanner:
    def __init__(self, start, goal, obstacles, map_size=15, num_points=10, safety_dist=0.3, fig=None, ax=None):
        self.start = start
        self.goal = goal
        self.map_size = map_size
        self.num_points = num_points
        self.safety_dist = safety_dist
        
        valid_obstacles = []
        for obs in obstacles:
            if len(obs) >= 3:
                poly = Polygon(obs)
                if not poly.is_valid:
                    poly = poly.buffer(0) 
                if not poly.is_empty:
                    valid_obstacles.append(poly)
                    
        self.obstacles = valid_obstacles
        self.obstacles_union = unary_union(self.obstacles) if self.obstacles else Polygon()
        
        if not self.obstacles_union.is_empty:
            self.safety_buffer = self.obstacles_union.buffer(self.safety_dist, join_style=1)
        else:
            self.safety_buffer = Polygon()
        
        if fig and ax:
            self.fig, self.ax = fig, ax
        else:
            self.fig = plt.figure(figsize=(14, 8))
            self.ax = self.fig.add_axes([0.05, 0.08, 0.65, 0.85])

    def is_point_in_free_space(self, point):
        return not self.safety_buffer.contains(Point(point))

    def generate_sample_points(self):
        points = []
        if not self.safety_buffer.is_empty:
            adaptive_radius = self.map_size * 0.1
            buffer_zone = self.safety_buffer.buffer(adaptive_radius)
            valid_buffer = buffer_zone.difference(self.safety_buffer)
        else:
            valid_buffer = Polygon()
            
        num_buffer_points = int(self.num_points * 0.7) if not valid_buffer.is_empty else 0
        num_uniform_points = self.num_points - num_buffer_points

        uniform_points = []
        attempts = 0
        while len(uniform_points) < num_uniform_points and attempts < num_uniform_points * 50:
            x, y = np.random.uniform(0, self.map_size, 2)
            if self.is_point_in_free_space((x, y)):
                uniform_points.append([x, y])
            attempts += 1

        adaptive_points = []
        attempts = 0
        while len(adaptive_points) < num_buffer_points and attempts < num_buffer_points * 50:
            x, y = np.random.uniform(0, self.map_size, 2)
            if valid_buffer.contains(Point(x, y)):
                adaptive_points.append([x, y])
            attempts += 1

        points = uniform_points + adaptive_points
        points.extend([self.start, self.goal])
        return np.array(points)

    def build_triangulation_and_graph(self, points):
        tri = Delaunay(points)
        valid_mask = []
        for simplex in tri.simplices:
            triangle = Polygon(points[simplex])
            valid_mask.append(not triangle.intersects(self.safety_buffer))
            
        valid_triangles, centroids, old_to_new = [], [], {}
        new_idx = 0
        
        for old_idx, is_valid in enumerate(valid_mask):
            if is_valid:
                old_to_new[old_idx] = new_idx
                valid_triangles.append(tri.simplices[old_idx])
                centroids.append(np.mean(points[tri.simplices[old_idx]], axis=0))
                new_idx += 1
                
        G = nx.Graph()
        for i, c in enumerate(centroids): G.add_node(i, pos=c)
            
        for old_idx, is_valid in enumerate(valid_mask):
            if is_valid:
                u = old_to_new[old_idx]
                for neighbor_old in tri.neighbors[old_idx]:
                    if neighbor_old != -1 and valid_mask[neighbor_old]:
                        v = old_to_new[neighbor_old]
                        if u < v:
                            dist = np.linalg.norm(centroids[u] - centroids[v])
                            G.add_edge(u, v, weight=dist)
                            
        return np.array(valid_triangles), G, centroids

    def smooth_path(self, path):
        if not path or len(path) <= 2: return path
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            if LineString([simplified[-1], path[i+1]]).intersects(self.safety_buffer):
                simplified.append(path[i])
        simplified.append(path[-1])
        
        if len(simplified) <= 2: return simplified

        turn_radius = 2.0
        corner_path = [np.array(simplified[0])]
        for i in range(1, len(simplified) - 1):
            p_prev, p_curr, p_next = map(np.array, [simplified[i-1], simplified[i], simplified[i+1]])
            v_in = p_curr - p_prev
            v_out = p_next - p_curr
            d_in, d_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
            actual_turn = min(turn_radius, d_in * 0.4, d_out * 0.4)
            corner_path.extend([p_curr - (v_in/d_in)*actual_turn, p_curr, p_curr + (v_out/d_out)*actual_turn])
        corner_path.append(np.array(simplified[-1]))

        dense_path = []
        for i in range(len(corner_path)-1):
            p1, p2 = corner_path[i], corner_path[i+1]
            num = max(int(np.linalg.norm(p2-p1)/0.5), 2)
            for j in range(num): dense_path.append(p1 + (p2-p1)*(j/num))
        dense_path.append(corner_path[-1])
        
        unique_pts = [dense_path[0]]
        for p in dense_path[1:]:
            if np.linalg.norm(p - unique_pts[-1]) > 1e-4: unique_pts.append(p)
            
        ux, uy = zip(*unique_pts)
        try:
            tck, u = splprep([ux, uy], s=len(ux)*0.005, k=min(len(ux)-1, 3))
            xs, ys = splev(np.linspace(0, 1, 300), tck)
            if not LineString(zip(xs, ys)).intersects(self.obstacles_union):
                return list(zip(xs, ys))
        except: pass
        return simplified
    
    def plan_path(self):
        timing_info = {} # Словник для зберігання часу кожного етапу
        
        # 1. Генерація точок
        t0 = time.time()
        points = self.generate_sample_points()
        timing_info['Генерація точок'] = time.time() - t0
        
        # 2. Тріангуляція та побудова графа
        t1 = time.time()
        triangles, G, centroids = self.build_triangulation_and_graph(points)
        timing_info['Тріангуляція та граф'] = time.time() - t1
        
        # 3. Пошук шляху (А*)
        t2 = time.time()
        start_idx = np.argsort([np.linalg.norm(np.array(self.start) - np.array(c)) for c in centroids])
        goal_idx = np.argsort([np.linalg.norm(np.array(self.goal) - np.array(c)) for c in centroids])
        
        path = None
        for s in start_idx[:5]:
            for g in goal_idx[:5]:
                try:
                    path_nodes = nx.astar_path(G, s, g, heuristic=lambda n1, n2: np.linalg.norm(centroids[n1]-centroids[n2]), weight='weight')
                    path = [centroids[i] for i in path_nodes]; break
                except: continue
            if path: break
        timing_info['Пошук маршруту (A*)'] = time.time() - t2
            
        if not path: 
            return None, points, triangles, False, 0, timing_info
            
        # 4. Згладжування маршруту
        t3 = time.time()
        smoothed = self.smooth_path([self.start] + path + [self.goal])
        timing_info['Згладжування (сплайни)'] = time.time() - t3
        
        dist = sum(np.linalg.norm(np.array(smoothed[i]) - np.array(smoothed[i+1])) for i in range(len(smoothed)-1))
        return smoothed, points, triangles, True, dist, timing_info

    def visualize(self, path, points, triangles, exists, dist, calc_time):
        self.ax.clear(); self.fig.texts.clear()
        if not self.safety_buffer.is_empty:
            polys = [self.safety_buffer] if isinstance(self.safety_buffer, Polygon) else self.safety_buffer.geoms
            for p in polys: self.ax.fill(*p.exterior.xy, alpha=0.2, fc='gray', ec='none')
        for p in self.obstacles:
            for geom in (p.geoms if p.geom_type == 'MultiPolygon' else [p]):
                self.ax.fill(*geom.exterior.xy, alpha=0.7, fc='red', ec='black')
        
        from matplotlib.collections import PolyCollection
        triangle_verts = [points[tri] for tri in triangles]
        
        pc = PolyCollection(
            triangle_verts,
            facecolors='#B3D9FF',
            edgecolors='#004C97',
            linewidths=1.0,
            alpha=0.2,
            closed=True,
            rasterized=True
        )
        self.ax.add_collection(pc)

        self.ax.scatter(points[:, 0], points[:, 1], s=2, c='blue', alpha=0.3, rasterized=True)

        if exists:
            px, py = zip(*path)
            self.ax.plot(px, py, '-g', linewidth=3.5)
            self.fig.text(0.37, 0.02, f"Довжина: {dist:.2f} | Загальний час: {calc_time:.3f} с", fontsize=12, color='green', fontweight='bold', ha='center', transform=self.fig.transFigure)
        
        self.ax.plot(*self.start, 'yo', markersize=10, markeredgecolor='black')
        self.ax.plot(*self.goal, 'mo', markersize=10, markeredgecolor='black')
        self.ax.set_xlim(0, self.map_size); self.ax.set_ylim(0, self.map_size); self.ax.set_aspect('equal')
        self.fig.canvas.draw_idle()