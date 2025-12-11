import os

models = [
    "yolov5s_kitti",
    "yolov5s_kitti_ca_only",
    "yolov5s_kitti_ghost_only",
    "yolov5s_kitti_ghost_ca"
]

VIDEO = "driving_car.mp4"    
IMG_SIZE = 640
CONF = 0.25
DEVICE = "0"

for model in models:
    weights_path = f"./runs/train/{model}/weights/best.pt"
    name_param = f"{model}"   

    cmd = (
        f"python yolov5/detect.py "
        f"--weights {weights_path} "
        f"--source {VIDEO} "
        f"--data ./yolov5/models/{model}.yaml "
        f"--img {IMG_SIZE} "
        f"--conf {CONF} "
        f"--device {DEVICE} "
        f"--name {name_param}"
    )

    print("\n--------------------------------------------------")
    print(f"Running detection for model: {model}")
    print(f"Command: {cmd}")
    print("--------------------------------------------------\n")

    os.system(cmd)
