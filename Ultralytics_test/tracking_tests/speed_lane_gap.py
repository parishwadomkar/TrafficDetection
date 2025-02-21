import cv2
import numpy as np
import pandas as pd
import math
import time
from ultralytics import YOLO
from collections import defaultdict


model = YOLO("yolov8n.pt")
names = model.model.names
cap = cv2.VideoCapture(r"C:\Users\MorinE\Desktop\Videos\sample2_short.mp4") #define what video to analyse
assert cap.isOpened(), "Error reading video file"

# video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 3840
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 2160
fps = cap.get(cv2.CAP_PROP_FPS)              # 30



# annotated video output, creates a audio video interleave
video_writer = cv2.VideoWriter(
    "speed_estimation.avi", 
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# demarcate region: horizontal band where vehicles pass the line
line_region_y_min = 1495
line_region_y_max = 1550
line_gap = 1495

region_pts = [(0, 1550), (w, 1550), (w, 1495), (0, 1495)]
def draw_region(img, pts, color=(0, 255, 0), thickness=2):
    pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=thickness)

# pixels per meter- if 3840 pixels cover ~30 m → ~128 ppm
ppm = 128
start_positions = {}  # {track_id: (start_center_x, start_center_y, start_time)}, stores the positon when the vehicle first is detected
recorded_speed = {}   # {track_id: speed_kmh}, stores the speed
# We'll also record the lane for each vehicle when speed is recorded.
recorded_lane = {}    # {track_id: lane_number}, stores the lane which the vehicles is detected in
speed_records = [] #stores the data about the vehicle
frame_count = 0 # stores the frames processed

last_detection_times_per_lane = defaultdict(lambda: -1)   # To store last detection times for each lane in the area, new
gap_times_records=[] # # To store gap times between vehicles
recorded_gap_vehicles = set()

while True:
    ret, frame = cap.read()
    if not ret:  break
    frame_count += 1
    current_time = frame_count / fps  # timestamp sec
    # tracking- YOLOv8 tracker
    results = model.track(frame, persist=True, show=False) #runs the YOLO tracking model
    if not results or results[0].boxes is None:
        video_writer.write(frame)
        continue
    draw_region(frame, region_pts, color=(0, 255, 0), thickness=2)
    boxes = results[0].boxes.xyxy  # shape: (n, 4), n = number of detected objects
    cls_vals = results[0].boxes.cls  # class IDs, if it is a car and so on
    try:  ids = results[0].boxes.id  # track IDs (if available)
    except Exception:  ids = np.arange(len(boxes)) #auto generates id if no id is avalible

    for i in range(len(boxes)): #loops for each detected object
        # track ID
        try:  track_id = int(ids[i].item()) #convert id to interger 
        except Exception:
            track_id = i #if converisn fails use index
        bbox = boxes[i] #determine the location of the vehicle
        x1, y1, x2, y2 = bbox
        center_x = float((x1 + x2) / 2) 
        center_y = float((y1 + y2) / 2)
        # lane number based on center_x, added
        if center_x >= 1450 and center_x <= 1616:
            lane_number = 1
        elif center_x >= 1617 and center_x <= 1776:
            lane_number = 2
        elif center_x >= 1777 and center_x <= 1957:
            lane_number = 3
        elif center_x >= 2188 and center_x <= 2339:
            lane_number = 4
        elif center_x >= 2340 and center_x <= 2534:
            lane_number = 5
        elif center_x >= 2535 and center_x <= 2715:
            lane_number = 6
        else:
            lane_number = 0

        # vehicle type from class
        cls_val = cls_vals[i]
        cls_id = int(cls_val.item()) if hasattr(cls_val, "item") else int(cls_val)
        vehicle_type = names.get(cls_id, "unknown")

        # vehicle start position and time
        if track_id not in start_positions: start_positions[track_id] = (center_x, center_y, current_time)


        if (line_region_y_min <= center_y <= line_region_y_max) and (track_id not in recorded_speed):
            start_x, start_y, start_time = start_positions[track_id]
            dt = current_time - start_time #elapsed time
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
        #Calcualtion of time gap bwtween vehicles in each lane
        if abs(center_y - line_gap) < 5 and (track_id not in recorded_gap_vehicles):
            time_when_line_passed = frame_count / fps 
            gap_time = time_when_line_passed - last_detection_times_per_lane[lane_number]
            gap_times_records.append({'Lane': lane_number, 'GapTimes_s': gap_time
            })

            last_detection_times_per_lane[lane_number] = time_when_line_passed
            recorded_gap_vehicles.add(track_id)

        # label for annotation
        label = f"{vehicle_type}"
        if track_id in recorded_speed:
            label += f" {recorded_speed[track_id]:.1f} km/h, Lane {recorded_lane[track_id]}"
        else: label += f", Lane {lane_number}"
        # bounding box and label
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    video_writer.write(frame)
    print(f"Frame {frame_count}: Processed {len(boxes)} tracks.")


cap.release()
video_writer.release()
cv2.destroyAllWindows()
df_speed = pd.DataFrame(speed_records)
excel_filename = 'vehicle_tracking.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer: df_speed.to_excel(writer, sheet_name='Speed', index=False)
print(f"Results saved to '{excel_filename}' in sheet 'Speed'.")



if gap_times_records != []:
    df_gaps_times = pd.DataFrame(gap_times_records)
else:
    print("Inga gap-tider har registrerats! Skapar en tom Excel-fil.")
    df_gaps_times = pd.DataFrame(columns=['Lane', 'GapTime_s'])  

excel_filename = 'gap_tracking.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df_gaps_times.to_excel(writer, sheet_name='Gap Times', index=False)

print(f"Results saved to '{excel_filename}' in sheet 'Gap Times'.")
