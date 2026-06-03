import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

from src.preprocess_manual import apply_method
from src.config import DEFAULT_MODEL, RESULTS_DIR

def run_inference(image_path: str, method: str = "original", model_name: str = DEFAULT_MODEL):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    processed = apply_method(image, method)
    model = YOLO(model_name)
    results = model(processed)
    annotated = results[0].plot()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{Path(image_path).stem}_{method}_detected.jpg"
    cv2.imwrite(str(output_path), annotated)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO inference with manual preprocessing methods.")
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument("--method", default="original", help="Preprocessing method")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model")
    args = parser.parse_args()
    print(run_inference(args.image, args.method, args.model))