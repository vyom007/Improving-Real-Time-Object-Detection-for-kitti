import os
import cv2
from tqdm import tqdm
import random
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
BASE_DIR = (SCRIPT_DIR.parent / "KITTI" / "training").resolve()  
SOURCE_IMAGE_DIR = BASE_DIR / "image_2"
SOURCE_LABEL_DIR = BASE_DIR / "label_2"

IMAGES_DIR = BASE_DIR / "images"       
LABELS_DIR = BASE_DIR / "labels"       

TRAIN_TXT = BASE_DIR / "train.txt"
VAL_TXT = BASE_DIR / "val.txt"

CLASS_NAMES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IGNORED_CLASSES = {"DontCare"}


def kitti_to_yolo(label_path: Path, img_w: int, img_h: int):
    """
    Convert one KITTI label file to YOLO format (normalized xywh).
    Returns list of 'cls x y w h' strings. Skips DontCare and unknown classes.
    """
    yolo_lines = []

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            cls_name = parts[0]
            if cls_name in IGNORED_CLASSES:
                continue
            if cls_name not in CLASS_TO_ID:
                continue

            cls_id = CLASS_TO_ID[cls_name]

            left, top, right, bottom = map(float, parts[4:8])

            x_center = ((left + right) / 2.0) / img_w
            y_center = ((top + bottom) / 2.0) / img_h
            w = (right - left) / img_w
            h = (bottom - top) / img_h

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            yolo_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

    return yolo_lines


def prepare_data(train_ratio: float = 0.8, seed: int = 42):
    print(f"Using KITTI base directory: {BASE_DIR}")
    print(f"Source images: {SOURCE_IMAGE_DIR}")
    print(f"Source labels: {SOURCE_LABEL_DIR}")

    assert SOURCE_IMAGE_DIR.exists(), f"Missing {SOURCE_IMAGE_DIR}"
    assert SOURCE_LABEL_DIR.exists(), f"Missing {SOURCE_LABEL_DIR}"

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)  
    LABELS_DIR.mkdir(parents=True, exist_ok=True)   

    image_paths = sorted(
        p for p in SOURCE_IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    print(f"Found {len(image_paths)} images in {SOURCE_IMAGE_DIR}")

    valid_image_paths = []

    for img_path in tqdm(image_paths, desc="Converting KITTI → YOLO"):
        label_path = SOURCE_LABEL_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        yolo_lines = kitti_to_yolo(label_path, img_w, img_h)
        if not yolo_lines:
            continue

        out_img_path = IMAGES_DIR / img_path.name
        if not out_img_path.exists():
            shutil.copy2(str(img_path), str(out_img_path))

        out_label_path = LABELS_DIR / f"{img_path.stem}.txt"
        with out_label_path.open("w") as f:
            f.write("\n".join(yolo_lines))

        valid_image_paths.append(str(out_img_path.resolve()))

    print(f"Valid images with labels: {len(valid_image_paths)}")

    if not valid_image_paths:
        raise RuntimeError("No valid images with labels were found. Check paths and KITTI labels.")

    random.seed(seed)
    random.shuffle(valid_image_paths)

    split_idx = int(len(valid_image_paths) * train_ratio)
    train_files = valid_image_paths[:split_idx]
    val_files = valid_image_paths[split_idx:]

    print(f"Training images: {len(train_files)}, Validation images: {len(val_files)}")

    with TRAIN_TXT.open("w") as f:
        f.write("\n".join(train_files))

    with VAL_TXT.open("w") as f:
        f.write("\n".join(val_files))

    print(f"train.txt written to: {TRAIN_TXT}")
    print(f"val.txt written to  : {VAL_TXT}")
    print("Data preparation complete.")


if __name__ == "__main__":
    prepare_data()
