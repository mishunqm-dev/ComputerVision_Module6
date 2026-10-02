import cv2
import numpy as np
import os

# ---------------------------------------------------------
# CSC 8830 - Module 6
# Track feature points between two consecutive frames
# using Lucas-Kanade Optical Flow
# ---------------------------------------------------------

# Find the main Module 6 folder automatically
project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

frames_folder = os.path.join(project_folder, "frames")
results_folder = os.path.join(project_folder, "results")

os.makedirs(results_folder, exist_ok=True)


def track_points(frame1_name, frame2_name, output_name, label):
    # Build full file paths
    frame1_path = os.path.join(frames_folder, frame1_name)
    frame2_path = os.path.join(frames_folder, frame2_name)

    # Load images
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    if frame1 is None or frame2 is None:
        print(f"ERROR: Could not load frames for {label}")
        return

    # Convert both images to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect strong feature points in Frame 1
    points1 = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7
    )

    if points1 is None:
        print(f"ERROR: No feature points detected for {label}")
        return

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    # Track points from Frame 1 to Frame 2
    points2, status, error = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray2,
        points1,
        None,
        **lk_params
    )

    # Keep only successfully tracked points
    good_old = points1[status == 1]
    good_new = points2[status == 1]

    # Create output image
    output = frame2.copy()

    print("\n-------------------------------------------")
    print(f"{label} TRACKING RESULTS")
    print("-------------------------------------------")
    print("Point | Frame 1 (x, y) | Frame 2 (x, y) | dx | dy")
    print("-------------------------------------------")

    # Limit printed results to first 10 tracked points
    count = min(10, len(good_old))

    for i in range(count):
        old = good_old[i]
        new = good_new[i]

        x1, y1 = old.ravel()
        x2, y2 = new.ravel()

        dx = x2 - x1
        dy = y2 - y1

        print(
            f"{i+1:>5} | "
            f"({x1:7.2f}, {y1:7.2f}) | "
            f"({x2:7.2f}, {y2:7.2f}) | "
            f"{dx:7.2f} | {dy:7.2f}"
        )

        # Draw motion line
        cv2.line(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        # Draw point in Frame 2
        cv2.circle(
            output,
            (int(x2), int(y2)),
            5,
            (0, 0, 255),
            -1
        )

    # Save result image
    output_path = os.path.join(results_folder, output_name)
    cv2.imwrite(output_path, output)

    print("-------------------------------------------")
    print(f"Tracked points: {len(good_old)}")
    print(f"Saved visualization: {output_path}")


# ---------------------------------------------------------
# Patio tracking
# ---------------------------------------------------------

track_points(
    "patio_frame1.png",
    "patio_frame2.png",
    "patio_tracking.png",
    "PATIO VIDEO"
)

# ---------------------------------------------------------
# Tramway tracking
# ---------------------------------------------------------

track_points(
    "tramway_frame1.png",
    "tramway_frame2.png",
    "tramway_tracking.png",
    "TRAMWAY VIDEO"
)

print("\nTracking analysis completed successfully!")
