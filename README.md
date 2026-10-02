# CSC 8830 - Computer Vision Assignment 6

**Student:** Mishun Miller  
**Course:** CSC 8830 - Computer Vision

## Overview
This repository contains the code and supporting files for Assignment 6. The project includes:

1. Dense optical flow visualization for two motion videos.
2. Consecutive-frame extraction and Lucas-Kanade feature tracking.
3. A planar Structure-from-Motion / multi-view geometry experiment using four viewpoints of a book cover.

## Project Structure

```text
ComputerVision_Module6/
├── code/
│   ├── optical_flow.py
│   ├── extract_frames.py
│   ├── track_points.py
│   ├── structure_from_motion.py
│   └── sfm_manual_corners.py
├── videos/
│   ├── patio_day.MOV
│   └── tramway.MOV
├── frames/
│   ├── patio_frame1.png
│   ├── patio_frame2.png
│   ├── tramway_frame1.png
│   └── tramway_frame2.png
├── images_sfm/
│   ├── view1.jpeg
│   ├── view2.jpeg
│   ├── view3.jpeg
│   └── view4.jpeg
└── results/
    ├── patio_optical_flow.mp4
    ├── tramway_optical_flow.mp4
    ├── patio_tracking.png
    ├── tramway_tracking.png
    ├── manual_sfm_view2.png
    ├── manual_sfm_view3.png
    ├── manual_sfm_view4.png
    └── sfm_corner_coordinates.txt
```

## Requirements

- Python 3
- OpenCV
- NumPy

Install dependencies with:

```bash
python3 -m pip install opencv-python numpy
```

## How to Run

From the main `ComputerVision_Module6` folder:

### 1. Dense Optical Flow

Set the desired input/output filenames near the top of `code/optical_flow.py`, then run:

```bash
python3 code/optical_flow.py
```

### 2. Extract Consecutive Frames

```bash
python3 code/extract_frames.py
```

### 3. Track Feature Points

```bash
python3 code/track_points.py
```

### 4. Initial Automatic Homography Experiment

```bash
python3 code/structure_from_motion.py
```

### 5. Final Manual-Corner Planar SfM Experiment

```bash
python3 code/sfm_manual_corners.py
```

For each of the four images, click the book corners in this order:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

Press **Enter** after selecting all four corners in each image.

## Methods

### Optical Flow
Dense motion is estimated using Farneback optical flow. Motion direction is represented by hue and motion magnitude by brightness in the generated HSV visualization.

### Point Tracking
Strong image features are detected in the first frame and tracked into the next frame using pyramidal Lucas-Kanade optical flow. The script reports the original and new pixel locations along with horizontal and vertical displacement.

### Planar Multi-View Geometry
The book is treated as a planar object. Four manually selected corresponding corners are used to estimate a 3 x 3 homography between View 1 and each additional viewpoint. The calculated transform is used to project the book boundary into the target views.

## Main Results

- Both videos produced working dense optical-flow visualizations.
- 50 feature points were tracked in each consecutive-frame pair.
- Patio motion was predominantly leftward at approximately 6 pixels per frame for the first ten tracked points.
- The tramway clip produced more varied motion across the frame.
- Homographies successfully mapped the four book corners from View 1 into Views 2, 3, and 4.

## References

- Farneback, G. (2003). *Two-frame motion estimation based on polynomial expansion.*
- Lucas, B. D., & Kanade, T. (1981). *An iterative image registration technique with an application to stereo vision.*
- OpenCV Optical Flow Tutorial: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- OpenCV Homography Tutorial: https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html
