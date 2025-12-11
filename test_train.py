import os
import sys
from pathlib import Path

import yaml
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parent                       
YOLO_ROOT = ROOT / "yolov5"

if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))      
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import check_dataset, check_img_size, colorstr
from yolov5.utils.plots import plot_images   

def main():
    print("Starting KITTI dataloader + GT visualization test...")

    data_yaml = ROOT / "kitti.yaml"
    assert data_yaml.exists(), f"kitti.yaml not found at {data_yaml}"

    data_dict = check_dataset(str(data_yaml))
    train_path, val_path = data_dict["train"], data_dict["val"]
    class_names = data_dict.get("names", None)

    print(f"Train list: {train_path}")
    print(f"Val   list: {val_path}")
    print(f"Classes ({data_dict['nc']}): {class_names}")

    imgsz = 640
    batch_size = 8
    workers = 4
    single_cls = False

    hyp_path = YOLO_ROOT / "data" / "hyps" / "hyp.scratch-low.yaml"
    with hyp_path.open("r") as f:
        hyp = yaml.safe_load(f)

    hyp["mosaic"] = 0.0      
    hyp["mixup"] = 0.0      
    hyp["perspective"] = 0.0  
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)

    print("Creating dataloaders...")

    train_loader, train_dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        stride,
        single_cls,
        hyp=hyp,
        augment=False, 
        cache=None,
        rect=False,
        rank=-1,
        workers=workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("train: "),
        shuffle=True,
    )


    val_loader, val_dataset = create_dataloader(
        val_path,
        imgsz,
        batch_size,
        stride,
        single_cls,
        hyp=hyp,
        augment=False,
        cache=None,
        rect=True,
        rank=-1,
        workers=workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("val: "),
        shuffle=False,
    )

    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    save_dir = ROOT / "runs" / "kitti_debug"     
    save_dir.mkdir(parents=True, exist_ok=True)
    imgs, targets, paths, _ = next(iter(train_loader))

    gt_plot_path = save_dir / "train_batch0_labels.jpg"
    plot_images(
        imgs,
        targets,
        paths=paths,
        fname=str(gt_plot_path),
        names=class_names,               
    )

    print(f"Saved GT bounding-box visualization to: {gt_plot_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

if __name__ == "__main__":
    main()
