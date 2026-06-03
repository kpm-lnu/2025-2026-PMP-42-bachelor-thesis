from ultralytics import YOLO
from .config import DEFAULT_MODEL, DEFAULT_EPOCHS, DEFAULT_IMGSZ, DEFAULT_BATCH

def train(data_yaml: str, epochs: int = DEFAULT_EPOCHS, imgsz: int = DEFAULT_IMGSZ, batch: int = DEFAULT_BATCH, model_name: str = DEFAULT_MODEL, project: str = "results", name: str = "train_run"):
    model = YOLO(model_name)
    return model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
    )

if __name__ == "__main__":
    train("data/original/data.yaml")