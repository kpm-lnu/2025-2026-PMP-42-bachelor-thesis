import csv
from pathlib import Path
from ultralytics import YOLO
from .config import RESULTS_DIR

def evaluate(data_yaml: str, method: str, model_path: str, imgsz: int = 640):
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        project="yolo/evaluation",
        name=method,
        exist_ok=True
    )

    precision = float(getattr(results.box, "mp", 0.0))
    recall = float(getattr(results.box, "mr", 0.0))
    map50 = float(getattr(results.box, "map50", 0.0))
    map5095 = float(getattr(results.box, "map", 0.0))
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = metrics_path.exists()

    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["method", "precision", "recall", "map50", "map50_95", "f1-score"])

        writer.writerow([method, precision, recall, map50, map5095, f1_score])

    return {
        "method": method,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map5095,
        "f1-score": f1_score,
    }