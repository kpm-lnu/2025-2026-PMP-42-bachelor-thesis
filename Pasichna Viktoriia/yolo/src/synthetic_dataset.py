from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W = 256
H = 256
CLASSES = ["circle", "square", "triangle"]
COLORS = [(220,30,30),(30,180,60),(40,90,220),(240,180,20),(160,60,200),(30,200,200)]
BG_COLORS = [(245,245,245),(230,238,250),(245,235,225),(235,245,235),(250,250,230)]

def bbox_to_yolo(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H)

def add_noise(img):
    arr = np.array(img).astype(np.int16)
    arr = np.clip(arr + np.random.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def draw_shape(draw, cls, box, color):
    x1,y1,x2,y2 = box
    if cls == "circle":
        draw.ellipse([x1,y1,x2,y2], fill=color, outline=(0,0,0), width=2)
    elif cls == "square":
        draw.rectangle([x1,y1,x2,y2], fill=color, outline=(0,0,0), width=2)
    else:
        pts = [((x1+x2)//2,y1),(x2,y2),(x1,y2)]
        draw.polygon(pts, fill=color, outline=(0,0,0))

def make_sample(img_path: Path, lbl_path: Path):
    bg = Image.new("RGB", (W,H), random.choice(BG_COLORS))
    draw = ImageDraw.Draw(bg)
    labels = []
    for _ in range(random.randint(1,4)):
        cls_id = random.randint(0, 2)
        size = random.randint(35, 90)
        x1 = random.randint(5, W-size-5)
        y1 = random.randint(5, H-size-5)
        box = (x1, y1, x1+size, y1+size)
        draw_shape(draw, CLASSES[cls_id], box, random.choice(COLORS))
        xc, yc, bw, bh = bbox_to_yolo(box)
        labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    if random.random() < 0.25:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    bg = add_noise(bg)
    bg.save(img_path, quality=95)
    lbl_path.write_text("\n".join(labels), encoding="utf-8")

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "data" / "original"
    splits = {"train": 180, "val": 45, "test": 45}
    for split, count in splits.items():
        for i in range(count):
            make_sample(
                base / "images" / split / f"{split}_{i:04d}.jpg",
                base / "labels" / split / f"{split}_{i:04d}.txt",
            )
    print("Dataset regenerated.")