import cv2
import numpy as np
import os

# ---------------------------------------------------------
# CSC 8830 - Module 6
# Structure from Motion example using 4 viewpoints
# Planar object: book cover
# ---------------------------------------------------------

# Find project folder
project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

images_folder = os.path.join(project_folder, "images_sfm")
results_folder = os.path.join(project_folder, "results")

os.makedirs(results_folder, exist_ok=True)

# Input image names
image_names = [
    "view1.jpeg",
    "view2.jpeg",
    "view3.jpeg",
    "view4.jpeg"
]

# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

images = []

for name in image_names:
    path = os.path.join(images_folder, name)
    img = cv2.imread(path)

    if img is None:
        print(f"ERROR: Could not load {name}")
        exit()

    images.append(img)

print("All four images loaded successfully.")

# ---------------------------------------------------------
# Detect ORB keypoints and descriptors
# ---------------------------------------------------------

orb = cv2.ORB_create(nfeatures=2000)

keypoints = []
descriptors = []

for i, img in enumerate(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = orb.detectAndCompute(gray, None)

    keypoints.append(kp)
    descriptors.append(des)

    print(
        f"View {i + 1}: "
        f"{len(kp)} keypoints detected"
    )

# ---------------------------------------------------------
# Match each image to View 1
# ---------------------------------------------------------

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

reference_image = images[0]
reference_kp = keypoints[0]
reference_des = descriptors[0]

for i in range(1, 4):

    matches = matcher.match(
        reference_des,
        descriptors[i]
    )

    # Sort best matches first
    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    # Keep strongest matches
    best_matches = matches[:80]

    print(
        f"View 1 to View {i + 1}: "
        f"{len(best_matches)} matches used"
    )

    # -----------------------------------------------------
    # Build corresponding point arrays
    # -----------------------------------------------------

    src_points = np.float32(
        [
            reference_kp[m.queryIdx].pt
            for m in best_matches
        ]
    ).reshape(-1, 1, 2)

    dst_points = np.float32(
        [
            keypoints[i][m.trainIdx].pt
            for m in best_matches
        ]
    ).reshape(-1, 1, 2)

    # -----------------------------------------------------
    # Estimate homography
    # -----------------------------------------------------

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        print(
            f"ERROR: Homography could not be "
            f"estimated for View {i + 1}"
        )
        continue

    print(
        f"\nHomography: View 1 -> View {i + 1}"
    )

    print(H)

    # -----------------------------------------------------
    # Estimate reference image boundary in new viewpoint
    # -----------------------------------------------------

    h, w = reference_image.shape[:2]

    corners = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ]
    ).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(
        corners,
        H
    )

    print(
        f"Projected boundary in View {i + 1}:"
    )

    for j, point in enumerate(transformed_corners):
        x, y = point[0]

        print(
            f"Corner {j + 1}: "
            f"({x:.2f}, {y:.2f})"
        )

    # -----------------------------------------------------
    # Draw estimated boundary
    # -----------------------------------------------------

    output = images[i].copy()

    corner_int = np.int32(
        transformed_corners
    )

    cv2.polylines(
        output,
        [corner_int],
        True,
        (0, 255, 0),
        6
    )

    # Draw matching points
    match_image = cv2.drawMatches(
        reference_image,
        reference_kp,
        images[i],
        keypoints[i],
        best_matches[:30],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save boundary result
    boundary_path = os.path.join(
        results_folder,
        f"sfm_boundary_view{i + 1}.png"
    )

    cv2.imwrite(
        boundary_path,
        output
    )

    # Save feature matches
    match_path = os.path.join(
        results_folder,
        f"sfm_matches_view1_view{i + 1}.png"
    )

    cv2.imwrite(
        match_path,
        match_image
    )

    print(
        f"Saved boundary image: "
        f"{boundary_path}"
    )

    print(
        f"Saved match image: "
        f"{match_path}"
    )

print("\nStructure from Motion analysis completed successfully!")
