import cv2
from ultralytics import YOLO, solutions
import pandas as pd

model = YOLO("yolov8n.pt")
names = model.model.names
cap = cv2.VideoCapture(r"C:\Users\omkarp\Downloads\Elin_Code\Code\sample2_short.mp4")
assert cap.isOpened(), "Error reading video file"

#  video properties
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH, 
    cv2.CAP_PROP_FRAME_HEIGHT, 
    cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(
    "speed_estimation.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (w, h)
)

# region- set pixels (used by SpeedEstimator)
line_pts = [(0, 1500), (3840, 1500), (3840, 1495), (0, 1495)]
# (Note: ppm and camera_angle_deg are no longer passed to estimate_speed)
speed_obj = solutions.SpeedEstimator(region=line_pts, names=names, view_img=True)

# ----- lane and car-following analysis -----

#  lane boundaries (set now for a 3840-pixel wide image, 3 lanes)
lane_boundaries = [0, 1280, 2560, 3840]
# Threshold (in pixels) to decide if one car is following immediately the car ahead
following_threshold = 50

#previous lane of each track for lane change detection
track_history = {}

#results for each detection in every frame
results = []
frame_count = 0
# --------------------------------------------------------------

while cap.isOpened():
    frame_count += 1
    success, im0 = cap.read()
    if not success:
        break

    # tracking (DeepSORT)
    tracks = model.track(im0, persist=True, show=False)
    # Estimate speed; from image
    im0 = speed_obj.estimate_speed(im0)
    frame_tracks = []

    # Iterate over each detected track
    for track in tracks:
        # --- Extract basic info from the track ---
        # Adjust attribute names if needed. Here we assume:
        #   - track.id: the unique track ID
        #   - track.speed: the estimated speed
        #   - track.boxes.xyxy: bounding box coordinates as [x1, y1, x2, y2]
        try:
            track_id = track.id  # if available
        except AttributeError:
            track_id = track[1]  # else if using indexing

        speed = track.speed

        # if detection has one bounding box; extract its coordinates
        bbox = track.boxes.xyxy[0]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # --- check lane based on center_x ---
        lane = None
        for i in range(len(lane_boundaries) - 1):
            if lane_boundaries[i] <= center_x < lane_boundaries[i + 1]:
                lane = i + 1  # Lane numbering starts at 1
                break

        # --- check lane change using history ---
        if track_id in track_history:
            previous_lane = track_history[track_id]['lane']
            lane_change = (previous_lane != lane)
        else:
            lane_change = False

        # take history for this track
        track_history[track_id] = {'lane': lane, 'center_x': center_x, 'center_y': center_y}

        detection = {
            'Frame': frame_count,
            'ID': track_id,
            'Speed': speed,
            'Lane': lane,
            'Lane_Change': lane_change,
            'Center_X': center_x,
            'Center_Y': center_y,
            'Following_ID': None  # placeholder
        }
        frame_tracks.append(detection)

    # --- Car Following Analysis ---
    # group detections by lane for this frame
    lane_tracks = {}
    for det in frame_tracks:
        if det['Lane'] is None:
            continue
        lane_tracks.setdefault(det['Lane'], []).append(det)

    # For each lane, sort vehicles by center_y
    for lane, det_list in lane_tracks.items():
        det_list.sort(key=lambda d: d['Center_Y'])
        # cars (except the first) check the gap with the one ahead
        for i in range(1, len(det_list)):
            gap = det_list[i]['Center_Y'] - det_list[i - 1]['Center_Y']
            if gap < following_threshold:
                det_list[i]['Following_ID'] = det_list[i - 1]['ID']
            else:
                det_list[i]['Following_ID'] = None
        # first vehicle in a lane is not following anyone
        if det_list:
            det_list[0]['Following_ID'] = None

    # all detections
    results.extend(frame_tracks)
    video_writer.write(im0)
    print(f"Frame {frame_count}: Processed {len(frame_tracks)} tracks")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
df = pd.DataFrame(results)
excel_filename = 'vehicle_tracking.xlsx'
df.to_excel(excel_filename, index=False, engine='openpyxl')
print(f"Results saved to {excel_filename}")