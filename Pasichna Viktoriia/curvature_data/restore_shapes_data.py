import json
import numpy as np
import cv2
from pathlib import Path

INPUT_JSON_DIR = Path("curvature_resault")
OUTPUT_IMAGE_DIR = Path("curvature_resault_data")

DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256

LINE_THICKNESS = 2

def restore_from_json(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data.get("height", DEFAULT_HEIGHT)
    width = data.get("width", DEFAULT_WIDTH)

    canvas = np.full((height, width), 255, dtype=np.uint8)

    figures = data.get("figures", data)

    for figure_name, figure_data in figures.items():
        if "points" not in figure_data or "curvature" not in figure_data:
            continue

        points = np.array(figure_data["points"], dtype=np.float32)
        curvature = np.array(figure_data["curvature"], dtype=np.float32)

        if len(points) < 2 or len(curvature) < 2:
            continue

        xs = points[:, 0].astype(np.int32)
        ys = points[:, 1].astype(np.int32)

        k_min = np.min(curvature)
        k_max = np.max(curvature)

        curvature_norm = (
            255 * (curvature - k_min)
            / (k_max - k_min + 1e-8)
        )

        curvature_norm = 255 - curvature_norm.astype(np.uint8)

        for i in range(min(len(xs), len(curvature_norm)) - 1):
            intensity = int(curvature_norm[i])

            cv2.line(
                canvas,
                (xs[i], ys[i]),
                (xs[i + 1], ys[i + 1]),
                intensity,
                LINE_THICKNESS
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

def main():
    for split in ["train", "test", "val"]:
        input_split_dir = INPUT_JSON_DIR / split
        output_split_dir = OUTPUT_IMAGE_DIR / split

        if not input_split_dir.exists():
            print(f"[SKIP] Немає папки: {input_split_dir}")
            continue

        json_files = list(input_split_dir.rglob("*.json"))

        print(f"\n[{split}] Знайдено JSON файлів: {len(json_files)}")

        for json_path in json_files:
            relative_path = json_path.relative_to(input_split_dir)
            output_path = output_split_dir / relative_path.with_suffix(".png")

            restore_from_json(json_path, output_path)

            print(f"[OK] {json_path} -> {output_path}")

    print("\nГотово!")
    print(f"Відновлені зображення збережено у: {OUTPUT_IMAGE_DIR}")

if __name__ == "__main__":
    main()