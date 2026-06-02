from pathlib import Path
from src.evaluate import evaluate

BASE_DIR = Path(__file__).resolve().parent

methods = [
    "original",
    "canny",
    "dog",
    "gaussian",
    "gaussian_canny",
    "gray",
    "laplacian",
    "log",
    "median",
    "roberts",
    "sobel",
    "curvature_resault_data"
]

def run_all():
    for m in methods:
        model_path = BASE_DIR / f"runs/detect/results/{m}/weights/best.pt"

        if m == "original":
            data_yaml = BASE_DIR / "data/original/data.yaml"

        elif m == "curvature_resault_data":
            data_yaml = BASE_DIR / "curvature_resault_data/data.yaml"

        else:
            data_yaml = BASE_DIR / f"data/variants/{m}/data.yaml"

        if not model_path.exists():
            print(f"Немає моделі для {m}")
            continue

        if not data_yaml.exists():
            print(f"Немає data.yaml для {m}")
            continue

        try:
            result = evaluate(str(data_yaml), m, str(model_path))
            print(result)
        except Exception as e:
            print(f"Помилка для {m}: {e}")

    print("\nБудуємо графік")

    metrics_path = BASE_DIR / "results/metrics.csv"

    if not metrics_path.exists():
        print("metrics.csv не створено")
        return

if __name__ == "__main__":
    run_all()