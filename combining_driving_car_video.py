import cv2
import numpy as np
from pathlib import Path

print(">>> Starting combine_videos_2x2.py")  

BASE_DIR = Path(__file__).resolve().parent
print(f">>> BASE_DIR = {BASE_DIR}")  

models = [
    "yolov5s_kitti",
    "yolov5s_kitti_ca_only",
    "yolov5s_kitti_ghost_only",
    "yolov5s_kitti_ghost_ca",
]

video_paths = [
    BASE_DIR / "yolov5" / "runs" / "detect" / m / "driving_car.mp4"
    for m in models
]

print(">>> Expecting videos at:")  
for m, p in zip(models, video_paths):
    print(f"    {m}: {p}")

caps = []
for path in video_paths:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Missing video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Could not open video: {path}")
    caps.append(cap)

print(">>> All videos opened successfully.")  

frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = caps[0].get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0  

print(f">>> Single video size: {frame_width}x{frame_height}, FPS={fps}")  

out_width = frame_width * 2
out_height = frame_height * 2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = BASE_DIR / "combined_driving_car_2x2.mp4"
out = cv2.VideoWriter(str(out_path), fourcc, fps, (out_width, out_height))

if not out.isOpened():
    raise RuntimeError("[ERROR] Failed to open VideoWriter. Check codec/fourcc and permissions.")

print(f">>> Writing combined video to: {out_path}")  

label_positions = [
    (30, 50),  
    (30, 50),  
    (30, 50),  
    (30, 50),  
]

frame_count = 0  
end = False
while True:
    frames = []

    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            end = True
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        frames.append(frame)

    if end:
        print(">>> One of the videos ended. Stopping.")  
        break

    for frame, label, pos in zip(frames, models, label_positions):
        cv2.putText(
            frame,
            label,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    combined = np.vstack((top_row, bottom_row))

    out.write(combined)
    frame_count += 1
    if frame_count % 50 == 0:
        print(f">>> Wrote {frame_count} frames...")  

for cap in caps:
    cap.release()
out.release()

print(f">>> Done! Total frames written: {frame_count}")  
print(f">>> Output file: {out_path}")
