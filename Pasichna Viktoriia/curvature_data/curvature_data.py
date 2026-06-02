import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json
from pathlib import Path

from methods import METHODS

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("curvature_resault")

method_name = "gaussian_laplacian"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def normalize(img):
    img = np.array(img)

    if img.ndim == 3:
        img = img.mean(axis=2)

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def make_threshold(processed, method_name):
    edge_methods = {
        "canny",
        "gaussian_canny",
        "sobel",
        "roberts",
        "laplacian",
        "dog",
        "log",
        "gaussian_sobel",
        "gaussian_laplacian"
    }

    if method_name in edge_methods:
        processed_blur = cv2.GaussianBlur(processed, (3, 3), 0)

        _, thresh = cv2.threshold(
            processed_blur,
            30,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return thresh

    _, thresh = cv2.threshold(
        processed,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return thresh


def compute_curvature(contour, sigma=4):
    points = contour[:, 0, :].astype(np.float64)

    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    points = points[np.insert(dists > 0, 0, True)]

    if len(points) < 5:
        return None, None, None

    x, y = points[:, 0], points[:, 1]

    x_s = gaussian_filter1d(x, sigma, mode="wrap")
    y_s = gaussian_filter1d(y, sigma, mode="wrap")

    dx = np.gradient(x_s)
    dy = np.gradient(y_s)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    numerator = dx * ddy - dy * ddx
    denominator = (dx ** 2 + dy ** 2) ** 1.5

    curvature = np.abs(numerator) / (denominator + 1e-8)

    return curvature, x_s, y_s


def process_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[SKIP] Не вдалося завантажити: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method_name not in METHODS:
        raise ValueError(
            f"Невідомий метод: {method_name}. "
            f"Доступні методи: {list(METHODS.keys())}"
        )

    processed = METHODS[method_name](gray)
    processed = normalize(processed)

    thresh = make_threshold(processed, method_name)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    img_h, img_w = gray.shape
    total_area = img_h * img_w

    valid_figures = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 200 or area > total_area * 0.9:
            continue

        valid_figures.append(cnt)

    result_json = {
        "image": image_path.name,
        "method": method_name,
        "figures_count": len(valid_figures),
        "figures": {}
    }

    for i, cnt in enumerate(valid_figures, start=1):
        area = cv2.contourArea(cnt)

        kappa, xs, ys = compute_curvature(cnt, sigma=4)

        if kappa is None:
            continue

        result_json["figures"][f"figure_{i}"] = {
            "area": float(area),
            "curvature": kappa.tolist(),
            "points": [
                [float(x), float(y)]
                for x, y in zip(xs, ys)
            ]
        }

    return result_json


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Папку не знайдено: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    total_images = 0
    processed_images = 0

    for split in ["train", "test", "val"]:
        input_split_dir = INPUT_DIR / split
        output_split_dir = OUTPUT_DIR / split

        if not input_split_dir.exists():
            print(f"[SKIP] Немає папки: {input_split_dir}")
            continue

        output_split_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            p for p in input_split_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\n[{split}] Знайдено зображень: {len(image_files)}")

        for image_path in image_files:
            total_images += 1

            result = process_image(image_path)

            if result is None:
                continue

            relative_path = image_path.relative_to(input_split_dir)
            output_json_path = output_split_dir / relative_path.with_suffix(".json")

            output_json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            processed_images += 1

            print(
                f"[OK] {image_path} -> {output_json_path} | "
                f"фігур: {result['figures_count']}"
            )

    print("\nГотово!")
    print(f"Усього знайдено зображень: {total_images}")
    print(f"Оброблено зображень: {processed_images}")
    print(f"JSON збережено у папці: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()