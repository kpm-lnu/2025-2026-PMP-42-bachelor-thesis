from src.train_yolo import train
from src.config import PROJECT_ROOT

data_yaml = PROJECT_ROOT / "curvature_resault_data" / "data.yaml"

train(
    data_yaml=str(data_yaml),
    epochs=70,
    imgsz=256,
    batch=16,
    project=str(PROJECT_ROOT / "runs" / "detect" / "results"),
    name="curvature_resault_data",
)