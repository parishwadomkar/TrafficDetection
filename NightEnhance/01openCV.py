import cv2
import numpy as np

def adjust_brightness_contrast(frame, alpha=1.2, beta=30):
    """
    Adjust brightness and contrast using OpenCV's convertScaleAbs.
    Args:
        frame (np.ndarray): Input BGR image.
        alpha (float): Contrast control (1.0 = no change).
        beta (float): Brightness control (0 = no change).
    Returns:
        np.ndarray: Brightness/contrast-adjusted image.
    """
    # convertScaleAbs applies alpha * frame + beta to each pixel
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return adjusted

def adjust_gamma(frame, gamma=1.2):
    """
    Apply gamma correction to the input frame.
    Args:
        frame (np.ndarray): Input BGR image.
        gamma (float): Gamma value (>1.0 brightens the image, <1.0 darkens).
    Returns:
        np.ndarray: Gamma-corrected image.
    """
    # Build a lookup table mapping [0..255] -> gamma-corrected values
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(frame, table)

def main():
    input_video = r"C:\Users\omkarp\Downloads\Elin_Code\Videos\03-06_4_colour.mp4"
    output_video = "enhanced_colour.avi"
    cap = cv2.VideoCapture(input_video)
    assert cap.isOpened(), "Error opening input video file."

    # video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # enhancement parameters
    alpha = 1.2   # contrast multiplier (>1.0 increases contrast)
    beta = 30     # brightness offset
    gamma_val = 1.2  # gamma correction (>1.0 brightens overall)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        bc_adjusted = adjust_brightness_contrast(frame, alpha=alpha, beta=beta)        # brightness/contrast
        gamma_adjusted = adjust_gamma(frame, gamma=gamma_val)        #gamma correction
        # combine them (first gamma, then brightness/contrast)
        combined = adjust_brightness_contrast(gamma_adjusted, alpha=alpha, beta=beta)
        enhanced_frame = bc_adjusted
        out.write(enhanced_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Enhanced video saved to {output_video}")
    print(f"Processed {frame_count} frames at {fps} fps.")

if __name__ == "__main__":
    main()
