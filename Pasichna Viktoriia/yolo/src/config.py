from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATASET_DIR = PROJECT_ROOT / "data" / "original"
VARIANTS_DIR = PROJECT_ROOT / "data" / "variants"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

DEFAULT_MODEL = str(PROJECT_ROOT / "yolov8n.pt")
DEFAULT_EPOCHS = 70
DEFAULT_IMGSZ = 256
DEFAULT_BATCH = 16

CLASS_NAMES = ["circle", "square", "triangle"]

SUPPORTED_METHODS = [
    "original",
    "gray",
    "mean",
    "gaussian",
    "median",
    "sobel",
    "roberts",
    "laplacian",
    "dog",
    "log",
    "canny",
    "gaussian_canny",
    "gray_gaussian",
    "gaussian_sobel",
    "gaussian_laplacian",
]