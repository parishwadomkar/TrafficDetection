import cv2
import numpy as np
import pandas as pd
import math
import time
from ultralytics import YOLO


VIDEO_PATH = r"C:\Users\omkarp\Downloads\enhanced_colour.avi"
OUTPUT_VIDEO = "speed_estimation.avi"
EXCEL_FILENAME = "vehicle_tracking.xlsx"

# Detection settings
# Lower conf threshold = more detections with false positives
CONF_THRESHOLD = 0.15
# classes=None - detect all 80 COCO classes (car=2, truck=7, bus=5...)
DETECT_CLASSES =  [2, 7]  # car, truck

# video properties
FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
FPS = 30

# Horizontal band for capturing speed- with a wider range
LINE_REGION_Y_MIN = 1000
LINE_REGION_Y_MAX = 2800
REGION_PTS = [
    (0, LINE_REGION_Y_MAX),
    (FRAME_WIDTH, LINE_REGION_Y_MAX),
    (FRAME_WIDTH, LINE_REGION_Y_MIN),
    (0, LINE_REGION_Y_MIN)
]

# Calibration: 3840 px ~ 30 m => ~128 px/m
PPM = 128

# Lane Setup (4 total lanes)- revise for accuracy to consider the exact lane- pixels region
NUM_LANES = 4
LANE_WIDTH = FRAME_WIDTH / NUM_LANES

# Minimum time tracked before speed is recorded- increase to detect more accurate speeds
MIN_TRACK_TIME = 0.1  # seconds

# SELECTIVE VIDEO ENHANCEMENT (dakk- avoid headlight flicker)
# We split the frame into left half [0 : MID_X] and right half [MID_X : FRAME_WIDTH].
# apply separate brightness/contrast/gamma
MID_X = FRAME_WIDTH // 2

# Left side (dark side) heavier enhancement
ALPHA_LEFT = 0.9    # contrast
BETA_LEFT  = 30     # brightness
GAMMA_LEFT = 1.2    # gamma correction
# Right side (brighter side) lighter enhancement
ALPHA_RIGHT = 1.2
BETA_RIGHT  = 20
GAMMA_RIGHT = 1.1

def adjust_brightness_contrast(frame, alpha=1.2, beta=30):
    """
    Adjust brightness/contrast using OpenCV's convertScaleAbs.
    alpha > 1.0 => more contrast, beta > 0 => more brightness
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def adjust_gamma(frame, gamma=1.2):
    """
    Apply gamma correction. gamma > 1.0 => brightens, < 1.0 => darkens
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(frame, table)

def enhance_side_by_side(frame):
    """
    Split the frame into left and right halves,
    apply different brightness/contrast/gamma to each,
    then recombine horizontally.
    """
    left_half = frame[:, :MID_X]
    right_half = frame[:, MID_X:]
    # Heavier enhancement on left half
    left_enh = adjust_brightness_contrast(left_half, alpha=ALPHA_LEFT, beta=BETA_LEFT)
    left_enh = adjust_gamma(left_enh, gamma=GAMMA_LEFT)
    # Lighter enhancement on right half
    right_enh = adjust_brightness_contrast(right_half, alpha=ALPHA_RIGHT, beta=BETA_RIGHT)
    right_enh = adjust_gamma(right_enh, gamma=GAMMA_RIGHT)
    return np.hstack([left_enh, right_enh])

def draw_region(img, pts, color=(0, 255, 0), thickness=2):
    """
    Draws a polygon (in this case, a horizontal band) on the frame.
    """
    pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=thickness)

def main():
    model = YOLO("yolov8n.pt")
    model.overrides['conf'] = CONF_THRESHOLD
    model.overrides['classes'] = DETECT_CLASSES 
    names = model.model.names
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Error reading video file: {VIDEO_PATH}"

    # annotated output
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    start_positions = {}   # track_id - (start_x, start_y, start_time)
    recorded_speed = {}    # track_id - speed_kmh
    recorded_lane = {}     # track_id - lane_number
    speed_records = []     # final data logs
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # If the video resolution is not 3840x2160- resize:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_count += 1
        current_time = frame_count / FPS  # seconds
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)) # If not 3840x2160,resize
        frame = enhance_side_by_side(frame)  #selective enhance
        results = model.track(frame, persist=True, show=False)
        if not results or results[0].boxes is None:
            video_writer.write(frame)
            continue

        # horizontal band region
        draw_region(frame, REGION_PTS, color=(0, 255, 0), thickness=2)

        # bounding boxes, class IDs, track IDs
        boxes = results[0].boxes.xyxy
        cls_vals = results[0].boxes.cls
        try: ids = results[0].boxes.id
        except Exception: ids = np.arange(len(boxes))

        for i in range(len(boxes)):
            try: track_id = int(ids[i].item())        # set unique track ID
            except Exception: track_id = i

            x1, y1, x2, y2 = boxes[i]
            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)

            # Lane calculation
            lane_number = int(center_x // LANE_WIDTH) + 1  # from 1..NUM_LANES

            #vehicle type
            cls_val = cls_vals[i]
            cls_id = int(cls_val.item()) if hasattr(cls_val, "item") else int(cls_val)
            vehicle_type = names.get(cls_id, "unknown")

            # If first time track, store position/time
            if track_id not in start_positions:
                start_positions[track_id] = (center_x, center_y, current_time)

            # If center crosses the band & speed not recorded yet
            if (LINE_REGION_Y_MIN <= center_y <= LINE_REGION_Y_MAX) and (track_id not in recorded_speed):
                start_x, start_y, start_time = start_positions[track_id]
                dt = current_time - start_time
                if dt >= MIN_TRACK_TIME:
                    disp_pixels = abs(center_y - start_y)       # Use vertical displacement
                    disp_m = disp_pixels / PPM
                    speed_m_s = disp_m / dt
                    speed_kmh = speed_m_s * 3.6
                    recorded_speed[track_id] = speed_kmh
                    recorded_lane[track_id] = lane_number
                    speed_records.append({
                        'Vehicle_ID': track_id,
                        'Time_s': current_time,
                        'Vehicle_Type': vehicle_type,
                        'Center_X': center_x,
                        'Center_Y': center_y,
                        'Speed_km_h': speed_kmh,
                        'Lane': lane_number
                    })

            label = f"{vehicle_type}"
            if track_id in recorded_speed:
                label += f" {recorded_speed[track_id]:.1f} km/h, Lane {recorded_lane[track_id]}"
            else: label += f", Lane {lane_number}"

            # annotated vid- bounding box & label
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
    with pd.ExcelWriter(EXCEL_FILENAME, engine='openpyxl') as writer:
        df_speed.to_excel(writer, sheet_name='Speed', index=False)
    print(f"Results saved to '{EXCEL_FILENAME}' in sheet 'Speed'.")

if __name__ == "__main__":
    main()
