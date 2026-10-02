import cv2
import numpy as np
import os

# Find the main Module 6 folder automatically
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output paths
video_path = os.path.join(project_folder, "videos", "tramway.MOV")
output_path = os.path.join(project_folder, "results", "tramway_optical_flow.mp4")

# Open video
cap = cv2.VideoCapture(video_path)

# Read first frame
ret, first_frame = cap.read()

if not ret:
    print("Error: Could not read the video.")
    exit()

# Resize first frame for faster processing
first_frame = cv2.resize(first_frame, (640, 360))

# Convert first frame to grayscale
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Output video settings
width = 640
height = 360
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# Process video frame by frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 360))

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    # Separate horizontal and vertical flow
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # Create HSV image for optical flow visualization
    hsv = np.zeros_like(frame)

    # Maximum saturation
    hsv[..., 1] = 255

    # Hue represents direction of motion
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Brightness represents magnitude of motion
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert HSV visualization to BGR
    optical_flow_frame = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    # Write frame to output video
    out.write(optical_flow_frame)

    # Current frame becomes previous frame
    previous_gray = gray

# Release video files
cap.release()
out.release()

print("Optical flow video created successfully!")
print("Saved to:", output_path)
