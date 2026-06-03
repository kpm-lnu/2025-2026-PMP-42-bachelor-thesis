from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable




def draw_triangulation_overlay(
    ax: plt.Axes,
    image: np.ndarray,
    tri_result: dict,
    points: np.ndarray,
    title: str = "Point distribution & error",
    error_map: np.ndarray | None = None,
    alpha_tri: float = 0.6,
):
 
    vertices  = tri_result["vertices"]
    triangles = tri_result["triangles"]

    ax.imshow(image, origin="upper", aspect="equal")

    
    for tri in triangles:
        pts = vertices[tri]
        xs = np.append(pts[:, 0], pts[0, 0])
        ys = np.append(pts[:, 1], pts[0, 1])
        ax.plot(xs, ys, color="cyan", linewidth=0.4, alpha=alpha_tri)

    
    if error_map is not None:
        vals = []
        for p in points:
            px = int(np.clip(round(p[0]), 0, error_map.shape[1] - 1))
            py = int(np.clip(round(p[1]), 0, error_map.shape[0] - 1))
            vals.append(error_map[py, px])
        vals = np.array(vals)
        norm = Normalize(vmin=vals.min(), vmax=max(vals.max(), 1e-6))
        cmap = plt.cm.hot
        colors = cmap(norm(vals))
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=8, zorder=5)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Local error")
    else:
        ax.scatter(points[:, 0], points[:, 1], c="lime", s=8, zorder=5)

    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_error_profile(
    ax: plt.Axes,
    history: list[dict],
    metric: str = "psnr",
    psnr_threshold: float | None = None,
    mse_threshold: float | None = None,
):
    
    iters  = [s["iteration"] for s in history]
    values = [s[metric]      for s in history]

    color = "#2196F3" if metric == "psnr" else "#F44336"
    ax.plot(iters, values, marker="o", color=color, linewidth=2, markersize=5)

    if metric == "psnr" and psnr_threshold is not None:
        ax.axhline(psnr_threshold, color="green", linestyle="--",
                   linewidth=1.5, label=f"PSNR threshold ({psnr_threshold} dB)")
        ax.legend(fontsize=8)

    if metric == "mse" and mse_threshold is not None:
        ax.axhline(mse_threshold, color="green", linestyle="--",
                   linewidth=1.5, label=f"MSE threshold ({mse_threshold})")
        ax.legend(fontsize=8)

    ylabel = "PSNR (dB)" if metric == "psnr" else "MSE"
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{ylabel} per iteration", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)




def plot_snapshot(
    snapshot: dict,
    original: np.ndarray,
    psnr_threshold: float | None = None,
    mse_threshold: float | None = None,
    history_so_far: list[dict] | None = None,
    save_path: str | None = None,
    show: bool = True,
):

    it        = snapshot["iteration"]
    rendered  = snapshot["rendered"]
    tri_res   = snapshot["tri_result"]
    points    = snapshot["points"]
    error_map = snapshot["error_map"]
    psnr      = snapshot["psnr"]
    mse       = snapshot["mse"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Iteration {it} | Points: {len(points)} | "
        f"PSNR: {psnr:.2f} dB | MSE: {mse:.4f}",
        fontsize=12, fontweight="bold"
    )

    axes[0].imshow(original, origin="upper", aspect="equal")
    axes[0].set_title("Original image", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(rendered, origin="upper", aspect="equal")
    axes[1].set_title("Triangulated approximation", fontsize=9)
    axes[1].axis("off")

    draw_triangulation_overlay(
        axes[2], original, tri_res, points,
        title="Points distribution & local error",
        error_map=error_map,
    )

    
    hist = history_so_far or [snapshot]
    
    if psnr_threshold is not None:
        draw_error_profile(axes[3], hist, "psnr", psnr_threshold, mse_threshold)
    elif mse_threshold is not None:
        draw_error_profile(axes[3], hist, "mse", psnr_threshold, mse_threshold)
    else:
        draw_error_profile(axes[3], hist, "psnr")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)




def plot_summary(
    history: list[dict],
    original: np.ndarray,
    psnr_threshold: float | None = None,
    mse_threshold: float | None = None,
    save_path: str | None = None,
    show: bool = True,
):
 
    first = history[0]
    last  = history[-1]

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        f"Adaptive Delaunay Triangulation — Summary\n"
        f"Iterations: {len(history)} | "
        f"Final points: {len(last['points'])} | "
        f"Final PSNR: {last['psnr']:.2f} dB | "
        f"Final MSE: {last['mse']:.4f}",
        fontsize=12, fontweight="bold"
    )

    gs = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.3)

    
    ax_orig_f   = fig.add_subplot(gs[0, 0])
    ax_tri_f    = fig.add_subplot(gs[0, 1])
    ax_pts_f    = fig.add_subplot(gs[0, 2])

    
    ax_orig_l   = fig.add_subplot(gs[1, 0])
    ax_tri_l    = fig.add_subplot(gs[1, 1])
    ax_pts_l    = fig.add_subplot(gs[1, 2])

    
    ax_psnr = fig.add_subplot(gs[0, 3:])
    ax_mse  = fig.add_subplot(gs[1, 3:])

    def _fill_row(ax_o, ax_t, ax_p, snap, label):
        ax_o.imshow(original, origin="upper"); ax_o.axis("off")
        ax_o.set_title(f"[{label}] Original", fontsize=8)

        ax_t.imshow(snap["rendered"], origin="upper"); ax_t.axis("off")
        ax_t.set_title(f"[{label}] Triangulated", fontsize=8)

        draw_triangulation_overlay(
            ax_p, original, snap["tri_result"], snap["points"],
            title=f"[{label}] Points & error",
            error_map=snap["error_map"],
        )

    _fill_row(ax_orig_f, ax_tri_f, ax_pts_f, first, f"Iter 0")
    _fill_row(ax_orig_l, ax_tri_l, ax_pts_l, last,  f"Iter {last['iteration']}")

    draw_error_profile(ax_psnr, history, "psnr", psnr_threshold, mse_threshold)
    draw_error_profile(ax_mse,  history, "mse",  psnr_threshold, mse_threshold)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Summary saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)