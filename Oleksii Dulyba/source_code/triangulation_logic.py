from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
import triangle as tr
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    
    orig_f = original.astype(np.float64)
    rec_f = reconstructed.astype(np.float64)
    return peak_signal_noise_ratio(orig_f, rec_f, data_range=255.0)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    
    orig_f = original.astype(np.float64)
    rec_f = reconstructed.astype(np.float64)
    return mean_squared_error(orig_f, rec_f)


def initialize_points(image: np.ndarray) -> np.ndarray:

    h, w = image.shape[:2]
    points = np.array([
        [0,     0    ],
        [w - 1, 0    ],
        [0,     h - 1],
        [w - 1, h - 1],
        [w // 2, 0   ],
        [w // 2, h - 1],
        [0,      h // 2],
        [w - 1,  h // 2],
    ], dtype=np.float64)
    return points


def triangulate_points(points: np.ndarray, max_area: float) -> dict:

    pts = {"vertices": points}
    
    opts = f"qDza{max_area:.1f}"
    result = tr.triangulate(pts, opts)
    return result


def render_triangulation(image: np.ndarray, tri_result: dict) -> np.ndarray:

    from matplotlib.path import Path

    h, w = image.shape[:2]
    vertices = tri_result["vertices"]
    triangles = tri_result["triangles"]

    if image.ndim == 3:
        rendered = np.zeros((h, w, image.shape[2]), dtype=np.float64)
    else:
        rendered = np.zeros((h, w), dtype=np.float64)

    
    ys, xs = np.mgrid[0:h, 0:w]
    pixel_coords = np.column_stack([xs.ravel(), ys.ravel()])

    for tri in triangles:
        pts = vertices[tri] 
        path = Path(pts)
        mask = path.contains_points(pixel_coords, radius=0.5)
        mask = mask.reshape(h, w)
        pixel_ys, pixel_xs = np.where(mask)

        if len(pixel_ys) == 0:
           
            cx = int(np.clip(np.mean(pts[:, 0]), 0, w - 1))
            cy = int(np.clip(np.mean(pts[:, 1]), 0, h - 1))
            if image.ndim == 3:
                rendered[cy, cx] = image[cy, cx].astype(np.float64)
            else:
                rendered[cy, cx] = float(image[cy, cx])
            continue

        if image.ndim == 3:
            mean_color = image[pixel_ys, pixel_xs].astype(np.float64).mean(axis=0)
            rendered[mask] = mean_color
        else:
            mean_val = image[pixel_ys, pixel_xs].astype(np.float64).mean()
            rendered[mask] = mean_val

    return np.clip(rendered, 0, 255).astype(np.uint8)


def compute_error_map(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:

    orig_f = original.astype(np.float64)
    rec_f = reconstructed.astype(np.float64)

    if orig_f.ndim == 3:
        diff = orig_f - rec_f
        error_map = (diff ** 2).mean(axis=2)
    else:
        error_map = (orig_f - rec_f) ** 2

    return error_map


def find_worst_triangles(
    error_map: np.ndarray,
    tri_result: dict,
    n: int = 5,
) -> list:

    from matplotlib.path import Path

    h, w = error_map.shape
    vertices  = tri_result["vertices"]
    triangles = tri_result["triangles"]

    ys, xs = np.mgrid[0:h, 0:w]
    pixel_coords = np.column_stack([xs.ravel(), ys.ravel()])
    flat_error   = error_map.ravel()

    tri_errors = []
    for idx, tri in enumerate(triangles):
        pts      = vertices[tri]
        centroid = pts.mean(axis=0)

        path   = Path(pts)
        mask   = path.contains_points(pixel_coords, radius=0.5)
        inside = flat_error[mask]

        score = inside.mean() * len(inside) if len(inside) > 0 else 0.0

        tri_errors.append((idx, score, centroid))

    tri_errors.sort(key=lambda x: x[1], reverse=True)
    return tri_errors[:n]


def add_points_in_worst_triangles(
    points: np.ndarray,
    tri_result: dict,
    worst: list,
    error_map: np.ndarray,
    points_per_tri: int = 3
) -> np.ndarray:

    h, w = error_map.shape
    vertices = tri_result["vertices"]
    triangles = tri_result["triangles"]
    new_pts = list(points)

    from matplotlib.path import Path
    ys, xs = np.mgrid[0:h, 0:w]
    pixel_coords = np.column_stack([xs.ravel(), ys.ravel()])
    flat_error = error_map.ravel()

    for idx, mean_err, centroid in worst:
        tri = triangles[idx]
        pts = vertices[tri]
        path = Path(pts)
        mask = path.contains_points(pixel_coords, radius=0.5)
        inside_idx = np.where(mask.ravel())[0]

        if len(inside_idx) == 0:
            new_pts.append(centroid)
            continue

        
        best_pi = inside_idx[np.argmax(flat_error[inside_idx])]
        new_pts.append([float(pixel_coords[best_pi, 0]), float(pixel_coords[best_pi, 1])])

        
        cx = float(np.clip(centroid[0], 0, w - 1))
        cy = float(np.clip(centroid[1], 0, h - 1))
        new_pts.append([cx, cy])

        
        added = 2
        for _ in range(points_per_tri - 2):
            
            r1, r2 = sorted([np.random.random(), np.random.random()])
            bary = np.array([r1, r2 - r1, 1 - r2])
            pt = (bary[:, None] * pts).sum(axis=0)
            new_pts.append([float(np.clip(pt[0], 0, w-1)), float(np.clip(pt[1], 0, h-1))])
            added += 1
            if added >= points_per_tri:
                break

    arr = np.array(new_pts, dtype=np.float64)
    
    arr = np.unique(arr, axis=0)
    return arr


class AdaptiveTriangulation:


    def __init__(
        self,
        image: np.ndarray,
        max_area: float,
        psnr_threshold: float | None = None,
        mse_threshold: float | None = None,
        max_iterations: int = 20,
        worst_tri_per_iter: int = 5,
        points_per_tri: int = 3,
    ):
        assert psnr_threshold is not None or mse_threshold is not None, \
            "Provide at least one of psnr_threshold or mse_threshold"

        self.image = image
        self.max_area = max_area
        self.psnr_threshold = psnr_threshold
        self.mse_threshold = mse_threshold
        self.max_iterations = max_iterations
        self.worst_tri_per_iter = worst_tri_per_iter
        self.points_per_tri = points_per_tri

        self.history: list[dict] = []  

    def quality_ok(self, psnr: float, mse: float) -> bool:
        
        psnr_ok = (self.psnr_threshold is None) or (psnr >= self.psnr_threshold)
        mse_ok  = (self.mse_threshold  is None) or (mse  <= self.mse_threshold)
        return psnr_ok and mse_ok

    def run(self) -> list[dict]:
   
        points = initialize_points(self.image)

        for iteration in range(self.max_iterations):
            print(f"[Iter {iteration}] Points: {len(points)}", end="  ")

            tri_result = triangulate_points(points, self.max_area)
            rendered   = render_triangulation(self.image, tri_result)
            error_map  = compute_error_map(self.image, rendered)
            psnr       = compute_psnr(self.image, rendered)
            mse        = compute_mse(self.image, rendered)

            print(f"PSNR={psnr:.2f} dB  MSE={mse:.4f}")

            snapshot = {
                "iteration":   iteration,
                "points":      points.copy(),
                "tri_result":  tri_result,
                "rendered":    rendered.copy(),
                "error_map":   error_map.copy(),
                "psnr":        psnr,
                "mse":         mse,
            }
            self.history.append(snapshot)

            if self.quality_ok(psnr, mse):
                print("  → Quality threshold reached!")
                break

            worst = find_worst_triangles(
                error_map, tri_result, n=self.worst_tri_per_iter
            )
            points = add_points_in_worst_triangles(
                points, tri_result, worst, error_map,
                points_per_tri=self.points_per_tri
            )

        return self.history
    
    
    
    
    
    
    
    
    

def save_tri(snapshot: dict, source_path: str, result_dir: str):
    import os
    tri_result = snapshot["tri_result"]
    rendered   = snapshot["rendered"]
    h, w       = rendered.shape[:2]
    vertices   = tri_result["vertices"]
    triangles  = tri_result["triangles"]

    mask_img = np.zeros((h, w), dtype=np.uint8)
    rows = []

    for tri in triangles:
        pts     = vertices[tri]
        contour = pts.astype(np.int32).reshape((-1, 1, 2))
        mask_img[:] = 0
        cv2.fillPoly(mask_img, [contour], 1)
        mask  = mask_img.astype(bool)
        color = rendered[mask].mean(axis=0) if mask.any() else np.zeros(3)

        row = np.array([
            pts[0,0], pts[0,1],
            pts[1,0], pts[1,1],
            pts[2,0], pts[2,1],
            color[0], color[1], color[2]
        ], dtype=np.uint16)
        rows.append(row)

    basename = os.path.splitext(os.path.basename(source_path))[0]
    out_path = os.path.join(result_dir, f"{basename}.tri")

    data = np.array(rows, dtype=np.uint16)
    np.savez_compressed(out_path,
        triangles = data,
        size      = np.array([h, w], dtype=np.uint16)
    )

    size_kb = os.path.getsize(out_path + '.npz') / 1024
    print(f"  Збережено .tri: {out_path}.npz  ({size_kb:.1f} КБ)")
    return out_path

def load_tri(path: str) -> np.ndarray:
    data      = np.load(path)
    rows      = data["triangles"]  # shape (N, 9)
    h, w      = data["size"]

    rendered = np.zeros((h, w, 3), dtype=np.uint8)
    mask_img = np.zeros((h, w),    dtype=np.uint8)

    for row in rows:
        # розпаковуємо рядок назад
        pts = np.array([
            [row[0], row[1]],
            [row[2], row[3]],
            [row[4], row[5]],
        ], dtype=np.int32)
        color = np.array([row[6], row[7], row[8]], dtype=np.uint8)

        contour = pts.reshape((-1, 1, 2))
        mask_img[:] = 0
        cv2.fillPoly(mask_img, [contour], 1)
        rendered[mask_img.astype(bool)] = color

    return rendered