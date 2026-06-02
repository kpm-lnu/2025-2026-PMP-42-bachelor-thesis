import shutil
from pathlib import Path
import cv2
from .config import ORIGINAL_DATASET_DIR, VARIANTS_DIR
from .preprocess_manual import apply_method

def _copy_labels(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for txt in src_dir.glob("*.txt"):
        shutil.copy2(txt, dst_dir / txt.name)

def build_variant(method: str) -> Path:
    variant_root = VARIANTS_DIR / method
    if variant_root.exists():
        shutil.rmtree(variant_root)
    for split in ["train", "val", "test"]:
        img_src = ORIGINAL_DATASET_DIR / "images" / split
        lbl_src = ORIGINAL_DATASET_DIR / "labels" / split
        img_dst = variant_root / "images" / split
        lbl_dst = variant_root / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        _copy_labels(lbl_src, lbl_dst)

        for image_path in img_src.glob("*.jpg"):
            image = cv2.imread(str(image_path))
            processed = apply_method(image, method)
            cv2.imwrite(str(img_dst / image_path.name), processed)

    yaml_text = f"""path: {variant_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: circle
  1: square
  2: triangle
"""
    (variant_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return variant_root

if __name__ == "__main__":
    for m in ["gray", "gaussian", "median", "sobel", "roberts", "laplacian", "dog", "log", "canny", "gaussian_canny"]:
        print("Building", m)
        print(build_variant(m))