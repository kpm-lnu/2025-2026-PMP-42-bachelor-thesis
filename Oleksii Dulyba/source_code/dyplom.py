from __future__ import annotations

import os
import sys
import time
import numpy as np
from PIL import Image

from triangulation_logic import AdaptiveTriangulation
from visualization_utils import plot_snapshot, plot_summary





IMAGE_PATH = "/Users/oleksijduliba/Desktop/main.py/dyplom/exaple_1 copy.bmp"

MAX_AREA = 500

PSNR_THRESHOLD = 40.0

MSE_THRESHOLD = None

MAX_ITERATIONS = 6

WORST_TRI_PER_ITER = 300

POINTS_PER_TRI = 20

SAVE_DIR = None

SHOW = True

RESULT_DIR = "result_duplom"


def load_image(path: str) -> np.ndarray:
    
    img = Image.open(path)
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def validate_config():
    
    if not os.path.isfile(IMAGE_PATH):
        print(f"ПОМИЛКА: Файл не знайдено: {IMAGE_PATH}", file=sys.stderr)
        print("Перевір змінну IMAGE_PATH у main.py", file=sys.stderr)
        sys.exit(1)

    if PSNR_THRESHOLD is None and MSE_THRESHOLD is None:
        print("ПОМИЛКА: Задай хоча б один поріг — PSNR_THRESHOLD або MSE_THRESHOLD",
              file=sys.stderr)
        sys.exit(1)

    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)

    os.makedirs(RESULT_DIR, exist_ok=True)


def save_result_image(rendered: np.ndarray, source_path: str):
    
    basename = os.path.splitext(os.path.basename(source_path))[0]
    out_path = os.path.join(RESULT_DIR, f"{basename}_triangulated.png")
    Image.fromarray(rendered).save(out_path)
    print(f"  Збережено стиснене зображення: {out_path}")
    return out_path


def main():
    validate_config()

    # Завантаження зображення
    print(f"Завантаження зображення: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    print(f"  Ширина  : {w} px")
    print(f"  Висота  : {h} px")
    print(f"  Канали  : {channels}")
    print(f"  Тип     : {image.dtype}")

    # Запуск адаптивної тріангуляції
    print("\nЗапуск адаптивної тріангуляції ...")
    start_time = time.time()
    print(f"  max_area={MAX_AREA}  PSNR>={PSNR_THRESHOLD}  MSE<={MSE_THRESHOLD}")
    print(f"  max_iter={MAX_ITERATIONS}  worst_tri={WORST_TRI_PER_ITER}  pts_per_tri={POINTS_PER_TRI}\n")

    engine = AdaptiveTriangulation(
        image=image,
        max_area=MAX_AREA,
        psnr_threshold=PSNR_THRESHOLD,
        mse_threshold=MSE_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
        worst_tri_per_iter=WORST_TRI_PER_ITER,
        points_per_tri=POINTS_PER_TRI,
    )
    history = engine.run()
    elapsed = time.time() - start_time

    print("\nГенерація фігур по ітераціях ...")
    for snap in history:
        it = snap["iteration"]
        save_path = None
        if SAVE_DIR:
            save_path = os.path.join(SAVE_DIR, f"iter_{it:03d}.png")

        plot_snapshot(
            snapshot=snap,
            original=image,
            psnr_threshold=PSNR_THRESHOLD,
            mse_threshold=MSE_THRESHOLD,
            history_so_far=history[: it + 1],
            save_path=save_path,
            show=SHOW,
        )

    print("\nГенерація підсумкової фігури ...")
    summary_path = os.path.join(SAVE_DIR, "summary.png") if SAVE_DIR else None

    plot_summary(
        history=history,
        original=image,
        psnr_threshold=PSNR_THRESHOLD,
        mse_threshold=MSE_THRESHOLD,
        save_path=summary_path,
        show=SHOW,
    )
    print("\nЗбереження результату ...")
    final = history[-1]
    save_result_image(final["rendered"], IMAGE_PATH)

    print("\n--- Результат ------------------------------------------")
    print(f"  Ітерацій        : {len(history)}")
    print(f"  Всього точок    : {len(final['points'])}")
    print(f"  Трикутників     : {len(final['tri_result']['triangles'])}")
    print(f"  PSNR            : {final['psnr']:.4f} дБ")
    print(f"  MSE             : {final['mse']:.6f}")
    print(f"  Час виконання   : {elapsed:.2f} сек")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    main()