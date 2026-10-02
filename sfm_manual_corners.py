import cv2
import numpy as np
import os

# ---------------------------------------------------------
# CSC 8830 - Module 6
# Manual Corner Selection for Planar Structure from Motion
# Object: Book Cover
# ---------------------------------------------------------

# Find main project folder
project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

images_folder = os.path.join(project_folder, "images_sfm")
results_folder = os.path.join(project_folder, "results")

os.makedirs(results_folder, exist_ok=True)

image_names = [
    "view1.jpeg",
    "view2.jpeg",
    "view3.jpeg",
    "view4.jpeg"
]

all_points = []

# ---------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------

current_points = []
display_image = None


def click_corner(event, x, y, flags, param):
    global current_points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(current_points) < 4:
            current_points.append((x, y))

            # Draw point
            cv2.circle(
                display_image,
                (x, y),
                10,
                (0, 0, 255),
                -1
            )

            # Label point number
            cv2.putText(
                display_image,
                str(len(current_points)),
                (x + 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.imshow("Select Book Corners", display_image)


# ---------------------------------------------------------
# Collect 4 book corners from every view
# ---------------------------------------------------------

for index, name in enumerate(image_names):

    image_path = os.path.join(images_folder, name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: Could not load {name}")
        exit()

    # Resize image only for easier clicking
    original_height, original_width = image.shape[:2]

    scale = 0.25

    display_width = int(original_width * scale)
    display_height = int(original_height * scale)

    display_image = cv2.resize(
        image,
        (display_width, display_height)
    )

    current_points = []

    window_name = "Select Book Corners"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        click_corner
    )

    print("\n-------------------------------------------")
    print(f"VIEW {index + 1}: {name}")
    print("-------------------------------------------")
    print("Click the FOUR book corners in this order:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("")
    print("After clicking all 4 corners, press ENTER.")

    while True:

        cv2.imshow(
            window_name,
            display_image
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(current_points) == 4:
            break

    cv2.destroyAllWindows()

    # Convert clicked coordinates back to original resolution
    original_points = []

    for x, y in current_points:

        original_x = x / scale
        original_y = y / scale

        original_points.append(
            [original_x, original_y]
        )

    original_points = np.float32(
        original_points
    )

    all_points.append(
        original_points
    )

    print(
        f"\nRecorded corners for View {index + 1}:"
    )

    for j, point in enumerate(original_points):

        print(
            f"Corner {j + 1}: "
            f"({point[0]:.2f}, {point[1]:.2f})"
        )


# ---------------------------------------------------------
# Use View 1 as reference
# ---------------------------------------------------------

reference_points = all_points[0]

for i in range(1, 4):

    target_points = all_points[i]

    # Calculate homography
    H, status = cv2.findHomography(
        reference_points,
        target_points,
        method=0
    )

    print("\n===========================================")
    print(f"HOMOGRAPHY: VIEW 1 -> VIEW {i + 1}")
    print("===========================================")

    print(H)

    # Project View 1 corners into target view
    projected_points = cv2.perspectiveTransform(
        reference_points.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    print(
        f"\nProjected boundary in View {i + 1}:"
    )

    for j, point in enumerate(projected_points):

        print(
            f"Corner {j + 1}: "
            f"({point[0]:.2f}, {point[1]:.2f})"
        )

    # Load target image
    target_image = cv2.imread(
        os.path.join(
            images_folder,
            image_names[i]
        )
    )

    # Draw measured boundary
    measured = np.int32(
        target_points.reshape(-1, 1, 2)
    )

    cv2.polylines(
        target_image,
        [measured],
        True,
        (0, 255, 0),
        12
    )

    # Draw projected points
    for j, point in enumerate(projected_points):

        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            target_image,
            (x, y),
            18,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            target_image,
            f"P{j + 1}",
            (x + 20, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3
        )

    # Save result
    output_path = os.path.join(
        results_folder,
        f"manual_sfm_view{i + 1}.png"
    )

    cv2.imwrite(
        output_path,
        target_image
    )

    print(
        f"\nSaved result: {output_path}"
    )


# ---------------------------------------------------------
# Save all corner coordinates to text file
# ---------------------------------------------------------

coordinates_path = os.path.join(
    results_folder,
    "sfm_corner_coordinates.txt"
)

with open(
    coordinates_path,
    "w"
) as file:

    for i, points in enumerate(all_points):

        file.write(
            f"VIEW {i + 1}\n"
        )

        for j, point in enumerate(points):

            file.write(
                f"Corner {j + 1}: "
                f"({point[0]:.2f}, "
                f"{point[1]:.2f})\n"
            )

        file.write("\n")


print("\n===========================================")
print("Manual Structure from Motion completed!")
print("===========================================")

print(
    "Corner coordinates saved to:",
    coordinates_path
)
