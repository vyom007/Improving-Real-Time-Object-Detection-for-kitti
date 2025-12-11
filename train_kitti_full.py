
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

models = [
    "yolov5s_kitti_ghost_only",
    "yolov5s_kitti_ghost_ca"
]
import os
from datetime import datetime
import sys
import atexit
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR,
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        return False

_log_fp = open(LOG_FILE, "a", buffering=1, encoding="utf-8", errors="replace")
sys.stdout = _Tee(sys.stdout, _log_fp)
sys.stderr = _Tee(sys.stderr, _log_fp)

@atexit.register
def _close_log():
    try:
        _log_fp.flush()
        _log_fp.close()
    except Exception:
        pass

print(f"[logger] Writing full session log to: {LOG_FILE}")


def run_all():
    for currentModel in models:

        opt = parse_opt(known=True)

        opt.data = "kitti.yaml"
        opt.cfg = f"yolov5/models/{currentModel}.yaml"
        opt.weights = "yolov5s.pt"  
        opt.batch_size = 16
        opt.epochs = 500            
        opt.imgsz = 640
        opt.device = "0" if torch.cuda.is_available() else "cpu"
        opt.project = "runs/new_train"
        opt.name = f"{currentModel}"
        opt.exist_ok = True         
        opt.save_dir = str(increment_path(
            Path(opt.project) / opt.name,
            exist_ok=opt.exist_ok
        ))

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)

        print(f"\n========== TRAINING {currentModel} ==========")
        print(f"Device: {opt.device}")
        print(f"Options: {opt}")

        data_dict = check_dataset(opt.data)

        device = select_device(opt.device, batch_size=opt.batch_size)

        callbacks = Callbacks()

        print(f"\n=== Starting training {currentModel} for 500 epochs ===")
        results_train = train(
            hyp,
            opt,
            device,
            callbacks
        )
        print("Training finished.")
        print("Training results:", results_train)

        best_weights = Path(opt.save_dir) / "weights" / "best.pt"
        print(f"Best weights saved at: {best_weights}")

        print("\n=== Validation after training ===")
        results_after, maps_after, _ = validate.run(
            opt.data,
            weights=str(best_weights),
            batch_size=opt.batch_size * 2,
            imgsz=opt.imgsz,
            device=opt.device,
            workers=opt.workers,
            single_cls=opt.single_cls,
            save_dir=Path(opt.save_dir) / "val_after",
            plots=True,
            callbacks=callbacks,
        )

        print(f"\n=== DONE {currentModel} ===")
        print("Final results:", results_after)
        print("Final mAPs:", maps_after)
        print("\nOutputs saved to:")
        print(f"  - {Path(opt.save_dir)}")

if __name__ == "__main__":
    run_all()
