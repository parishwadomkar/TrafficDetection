# Vehicle Speed Estimation & Lane Identification using YOLOv8

This repository contains a Python script that detects, tracks, and estimates the speed of vehicles as they cross a predefined region in a video. In addition, the script computes the approximate lane number for each vehicle based on its horizontal position. The results are visualized in an annotated video and logged in an Excel file for further analysis.

> **Note:**  
> • This code is tailored for right-side traffic (vehicles passing through a designated region).  
> • Calibration is critical—adjust the pixels-per-meter (ppm) and demarcated region parameters to match your scene.  
> • The detection and tracking are powered by [Ultralytics YOLOv8](https://github.com/ultralytics).

---

## Features

- **Real-Time Detection & Tracking:** Utilizes YOLOv8 for detecting vehicles and assigning persistent track IDs.
- **Speed Estimation:** Computes vehicle speed (in km/h) using vertical displacement when a vehicle’s center crosses a specified horizontal band.
- **Lane Identification:** Divides the frame into 6 lanes (3 per side) and determines the lane number based on the center x-coordinate.
- **Output Generation:** Produces an annotated video with bounding boxes and labels, and exports a unique record per vehicle in an Excel file (sheet "Speed").

---

## Prerequisites

- Python 3.8 or later
- OpenCV, NumPy, Pandas, math, and time modules
- [Ultralytics YOLOv8](https://github.com/ultralytics) (Install via `pip install ultralytics`)
- A video file for processing (`sample2_short.mp4`)

---

## Calibration & Setup
- **Main setup:** To run the speed estimation program please check the main [file](https://github.com/parishwadomkar/ObjectDetection/blob/main/Ultralytics_test/tracking_tests/test_speed.py).
- **Frame Dimensions:** Assumes a resolution of 3840x2160.
- **FPS:** The video is assumed to be 30 frames per second.
- **Pixels Per Meter (ppm):**  
  For example, if the 3840‑pixel width represents approximately 30 meters, use:  
  `ppm = 3840 / 30 ≈ 128`
- **Demarcated Region:**  
  The script records speed when the vehicle’s center passes between y = 1495 and 1500.
- **Lane Division:**  
  The frame is divided equally into 6 lanes (each ~3840/6 ≈ 640 pixels wide).

---

## Code Overview

1. **Video Capture & Setup:**  
   Opens the video file, extracts properties (width, height, fps), and sets up a video writer for the annotated output.

2. **Region Definition:**  
   A horizontal band (y = 1495 to 1500) is defined where vehicles are “recorded.” This region is drawn on every frame.

3. **Detection & Tracking:**  
   YOLOv8 is used to detect vehicles and track them across frames. Bounding boxes, class IDs, and track IDs are extracted.

4. **Speed Computation:**  
   - For each vehicle, the first observed center position and timestamp are stored.
   - When a vehicle’s center enters the demarcated region and has been tracked for at least 0.5 seconds, the vertical displacement is measured.
   - The displacement is converted from pixels to meters using the calibration factor (`ppm`), and the speed is calculated:
     
     ```
     Speed (km/h) = (displacement (m) / time (s)) * 3.6
     ```
     
5. **Lane Calculation:**  
   The lane number is determined by dividing the x-coordinate of the vehicle’s center by the lane width.

6. **Annotation & Logging:**  
   - Each bounding box is drawn and labeled with the vehicle type, speed (if computed), and lane number.
   - A unique record (per vehicle) is stored in a list and exported to an Excel file (sheet "Speed").

---

## Running the Code

1. **Configure Calibration:**  
   Adjust `ppm`, demarcated region, and lane divisions as needed for your scene.

2. **Execute the Script:**  
   Run the code with:
   ```bash
   python your_script.py

## Outputs

- **Annotated Video:**  
  Saved as `speed_estimation.avi` with bounding boxes drawn around each detected vehicle. Each box is labeled with the vehicle type, the computed speed (in km/h), and the lane number.

- **Excel File:**  
  Saved as `vehicle_tracking.xlsx` (sheet "Speed") with the following columns:
  - **Vehicle_ID:** A unique identifier for each tracked vehicle. This is acquired from the YOLOv8 tracker.
  - **Time_s:** The timestamp (in seconds) when the vehicle's speed is recorded. It is computed from the frame count and FPS.
  - **Vehicle_Type:** The detected class (car) based on the YOLOv8 model predictions.
  - **Center_X:** The x-coordinate of the vehicle's bounding box center, used to determine its lateral position and lane.
  - **Center_Y:** The y-coordinate of the vehicle's bounding box center, used to trigger speed computation when passing the demarcated line.
  - **Speed_km_h:** The computed speed of the vehicle (in km/h) based on vertical displacement over time, using the calibrated pixels-per-meter value.
  - **Lane:** The approximate lane number (1–6) calculated from the horizontal (x-axis) position of the vehicle's center.

## Improving Accuracy

- **Calibration:**  
  Use a reference object or a known distance in your scene to determine an accurate `ppm` (pixels per meter) value.
- **Temporal Averaging:**  
  Consider averaging the displacement over several frames to minimize noise and improve the stability of speed measurements.
- **Perspective Correction:**  
  For increased accuracy, apply a perspective transformation to map image pixel coordinates to real-world coordinates.
- **Filtering:**  
  Increase the minimum time threshold before computing speed to reduce the impact of momentary fluctuations or noise.

## Role of Ultralytics YOLOv8

This code leverages [Ultralytics YOLOv8](https://github.com/ultralytics) for its robust real-time object detection and tracking capabilities. YOLOv8 provides the necessary bounding boxes, class predictions, and persistent track IDs that serve as the basis for our manual computation of vehicle speed and lane identification.

## Contributing

Contributions to improve calibration, enhance the speed estimation algorithm, and add additional features (such as multi-region tracking) are highly welcome.  
Feel free to fork this repository, submit pull requests, and share your improvements with the community.
