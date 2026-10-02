import cv2
import os

# ---------------------------------------------------------
# CSC 8830 - Module 6
# Extract two consecutive frames from each original video
# ---------------------------------------------------------

# Find the main ComputerVision_Module6 folder
project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

videos_folder = os.path.join(project_folder, "videos")
frames_folder = os.path.join(project_folder, "frames")

# Make sure frames folder exists
os.makedirs(frames_folder, exist_ok=True)


def extract_two_frames(video_filename, output_prefix, time_seconds=5):
    """
    Extract two consecutive frames from a video.

    video_filename:
        Name of video inside the videos folder.

    output_prefix:
        Name used for saved frame files.

    time_seconds:
        Approximate point in the video where frames are selected.
    """

    video_path = os.path.join(videos_folder, video_filename)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_filename}")
        return

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Select a frame around 5 seconds into the video
    first_frame_number = int(fps * time_seconds)
    second_frame_number = first_frame_number + 1

    # Make sure selected frames exist
    if second_frame_number >= total_frames:
        print(f"ERROR: Selected frames exceed video length for {video_filename}")
        cap.release()
        return

    # -----------------------------------------------------
    # Extract first frame
    # -----------------------------------------------------

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_number)

    ret1, frame1 = cap.read()

    if not ret1:
        print(f"ERROR: Could not read first frame from {video_filename}")
        cap.release()
        return

    # -----------------------------------------------------
    # Extract immediately following frame
    # -----------------------------------------------------

    ret2, frame2 = cap.read()

    if not ret2:
        print(f"ERROR: Could not read second frame from {video_filename}")
        cap.release()
        return

    # Output filenames
    frame1_path = os.path.join(
        frames_folder,
        f"{output_prefix}_frame1.png"
    )

    frame2_path = os.path.join(
        frames_folder,
        f"{output_prefix}_frame2.png"
    )

    # Save frames
    cv2.imwrite(frame1_path, frame1)
    cv2.imwrite(frame2_path, frame2)

    # Release video
    cap.release()

    # Print useful information
    print("-------------------------------------------")
    print(f"Video: {video_filename}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"Frame 1 Number: {first_frame_number}")
    print(f"Frame 2 Number: {second_frame_number}")
    print(f"Saved: {frame1_path}")
    print(f"Saved: {frame2_path}")


# ---------------------------------------------------------
# Extract frames from Patio video
# ---------------------------------------------------------

extract_two_frames(
    "patio_day.MOV",
    "patio",
    time_seconds=5
)

# ---------------------------------------------------------
# Extract frames from Tramway video
# ---------------------------------------------------------

extract_two_frames(
    "tramway.MOV",
    "tramway",
    time_seconds=5
)

print("-------------------------------------------")
print("Frame extraction completed successfully!")
