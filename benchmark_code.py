import os
import sys
from pathlib import Path

import torch
import yaml
torch.use_deterministic_algorithms(False)

ROOT = Path(__file__).resolve().parent
YOLO_ROOT = ROOT / "yolov5"
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

from yolov5.train import train, parse_opt
import yolov5.val as validate
from yolov5.utils.general import check_dataset, increment_path, LOGGER  
from yolov5.utils.torch_utils import select_device
from yolov5.utils.callbacks import Callbacks


models = ["yolov5s_kitti", "yolov5s_kitti_ca_only", "yolov5s_kitti_ghost_only", "yolov5s_kitti_ghost_ca"]

def run_all():
    for currentModel in models:

        opt = parse_opt(known=True)

        opt.data = "kitti.yaml"
        opt.cfg = f"yolov5/models/{currentModel}.yaml"
        opt.weights = f"runs/train/{currentModel}/weights/best.pt"
        opt.batch_size = 16
        opt.epochs = 50
        opt.imgsz = 640
        opt.device = "0" if torch.cuda.is_available() else "cpu"
        opt.project = "runs/train"
        opt.name = f"{currentModel}"
        opt.exist_ok = False
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        
        
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)
        print(f"{opt.weights}   {currentModel}")
        print(f"Device: {opt.device}")
        print(f"Options: {opt}")

        
        data_dict = check_dataset(opt.data)

        
        device = select_device(opt.device, batch_size=opt.batch_size)

        
        callbacks = Callbacks()

        print("\n=== Final validation (after training) ===")
        results_after, maps_after, _ = validate.run(
            opt.data,
            weights=Path(opt.weights),
            batch_size=opt.batch_size * 2,
            imgsz=opt.imgsz,
            device=opt.device,
            workers=opt.workers,
            single_cls=opt.single_cls,
            save_dir=Path(opt.save_dir) / "val_after",
            plots=True,
            callbacks=callbacks,
        )

        print("Final validation complete.")
        print("Final results:", results_after)
        print("Final mAPs:", maps_after)

        print("\nAll done. Check these folders for visuals:")
        print(f"  - Training run   : {Path(opt.save_dir)}")   



if __name__ == "__main__":  
    run_all()
