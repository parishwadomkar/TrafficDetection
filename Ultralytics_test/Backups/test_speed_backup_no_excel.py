import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture(r"C:\Ultralytics_test\ultralytics\sample2.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (
cv2.CAP_PROP_FRAME_WIDTH, 
cv2.CAP_PROP_FRAME_HEIGHT, 
cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(
"speed_estimation.avi",
cv2.VideoWriter_fourcc(*"mp4v"),
fps, (w, h))

line_pts = [(0, 1500), (3840, 1500), (3840, 1520), (0, 1520)]

speed_obj = solutions.SpeedEstimator(region=line_pts, names=names, view_img=True,)

while cap.isOpened():
  success, im0 = cap.read()
  if not success:        
    break
  tracks = model.track(im0, persist=True, show=False)
  im0 = speed_obj.estimate_speed(im0) #, tracks (removed)
  video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()