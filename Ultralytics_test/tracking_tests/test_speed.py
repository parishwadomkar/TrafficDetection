import cv2
import numpy as np
import pandas as pd
import math
import time
from ultralytics import YOLO


model = YOLO("yolov8n.pt")
names = model.model.names
cap = cv2.VideoCapture(r"C:\Users\omkarp\Downloads\Elin_Code\Code\sample2_short.mp4")
assert cap.isOpened(), "Error reading video file"

# video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 3840
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 2160
fps = cap.get(cv2.CAP_PROP_FPS)              # 30

# annotated video output
video_writer = cv2.VideoWriter(
    "speed_estimation.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# demarcate region: horizontal band where vehicles pass the line
line_region_y_min = 1495
line_region_y_max = 1500
region_pts = [(0, 1500), (w, 1500), (w, 1495), (0, 1495)]

def draw_region(img, pts, color=(0, 255, 0), thickness=2):
    pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=thickness)

# pixels per meter- if 3840 pixels cover ~30 m → ~128 ppm
ppm = 128

# Lanes
num_lanes = 6
lane_width = w / num_lanes


start_positions = {}  # {track_id: (start_center_x, start_center_y, start_time)}
recorded_speed = {}   # {track_id: speed_kmh}
# We'll also record the lane for each vehicle when speed is recorded.
recorded_lane = {}    # {track_id: lane_number}
speed_records = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count / fps  # timestamp sec

    # tracking- YOLOv8 tracker
    results = model.track(frame, persist=True, show=False)
    if not results or results[0].boxes is None:
        video_writer.write(frame)
        continue

    draw_region(frame, region_pts, color=(0, 255, 0), thickness=2)
    boxes = results[0].boxes.xyxy  # shape: (n, 4)
    cls_vals = results[0].boxes.cls  # class IDs
    try:
        ids = results[0].boxes.id  # track IDs (if available)
    except Exception:
        ids = np.arange(len(boxes))

    for i in range(len(boxes)):
        # track ID
        try:
            track_id = int(ids[i].item())
        except Exception:
            track_id = i
        bbox = boxes[i]
        x1, y1, x2, y2 = bbox
        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)

        # lane number based on center_x
        lane_number = int(center_x // lane_width) + 1  # lanes 1 to num_lanes
        # vehicle type from class
        cls_val = cls_vals[i]
        cls_id = int(cls_val.item()) if hasattr(cls_val, "item") else int(cls_val)
        vehicle_type = names.get(cls_id, "unknown")

        # vehicle start position and time
        if track_id not in start_positions:
            start_positions[track_id] = (center_x, center_y, current_time)

        if (line_region_y_min <= center_y <= line_region_y_max) and (track_id not in recorded_speed):
            start_x, start_y, start_time = start_positions[track_id]
            dt = current_time - start_time
            if dt >= 0.5:
                # vertical displacement - Euclidean distance
                displacement_pixels = abs(center_y - start_y)
                displacement_m = displacement_pixels / ppm
                speed_m_s = displacement_m / dt
                speed_kmh = speed_m_s * 3.6
                recorded_speed[track_id] = speed_kmh
                recorded_lane[track_id] = lane_number

                # record once for this vehicle
                speed_records.append({
                    'Vehicle_ID': track_id,
                    'Time_s': current_time,
                    'Vehicle_Type': vehicle_type,
                    'Center_X': center_x,
                    'Center_Y': center_y,
                    'Speed_km_h': speed_kmh,
                    'Lane': lane_number
                })

        # label for annotation
        label = f"{vehicle_type}"
        if track_id in recorded_speed:
            label += f" {recorded_speed[track_id]:.1f} km/h, Lane {recorded_lane[track_id]}"
        else:
            label += f", Lane {lane_number}"

        # bounding box and label
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    video_writer.write(frame)
    print(f"Frame {frame_count}: Processed {len(boxes)} tracks.")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
df_speed = pd.DataFrame(speed_records)
excel_filename = 'vehicle_tracking.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df_speed.to_excel(writer, sheet_name='Speed', index=False)
print(f"Results saved to '{excel_filename}' in sheet 'Speed'.")
