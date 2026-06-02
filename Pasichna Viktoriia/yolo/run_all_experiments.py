from pathlib import Path
from src.dataset_variant_builder import build_variant
from src.train_yolo import train

METHODS = [
    "original",
    "gray",
    "gaussian",
    "median",
    "sobel",
    "roberts",
    "laplacian",
    "dog",
    "log",
    "canny",
    "gaussian_canny",
]

def run():
    
    original_yaml = Path("data/original/data.yaml").as_posix()
    print("[1/{}] original".format(len(METHODS)))
    train(data_yaml=original_yaml, epochs=70, imgsz=256, project="results", name="original")

    for idx, method in enumerate(METHODS[1:], start=2):
        print(f"[{idx}/{len(METHODS)}] {method}")
        variant_root = build_variant(method)
        data_yaml = (variant_root / "data.yaml").as_posix()
        train(data_yaml=data_yaml, epochs=70, imgsz=256, project="results", name=method)

if __name__ == "__main__":
    run()