import cv2
import multiprocessing as mp
import numpy as np
from rich.progress import Progress, track
from joblib import Parallel, delayed
from .printing import *
from .utils import *


def compute_background(handler, num_images=30):
    # Use the stabilized video to compute the background
    cap = cv2.VideoCapture(handler.path_to_save / f"Stabilized_{handler.N}.MOV")
    if not cap.isOpened():
        print_error("Error: Could not open video.")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index_list = np.linspace(total_frames * 0.1, total_frames*0.9, num_images, dtype=int)

    frames = []

    for i in track(index_list, description="Reading frames for background"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    frames = np.array(frames)
    background_max = 0

    background_path = handler.path_to_save / "background.png"
    cv2.imwrite(background_path, cv2.cvtColor(background_max, cv2.COLOR_RGB2BGR))

    print_success(f"Background computed and saved to {background_path}")
    return background_max


def remove_background(handler):
    background = cv2.imread(handler.path_to_save / "background.png")
    if background is None:
        print_error("Error: Background image not found.")
        return False

    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    # Open the video (Stabilized) to remove the background
    cap = cv2.VideoCapture(handler.path_to_save / f"Stabilized_{handler.N}.MOV")
    if not cap.isOpened():
        print_error("Error: Could not open video.")
        return False
    else:
        print_success("Video opened successfully")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    (width, height) = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Prepare the video writer to save the output video
    writer = cv2.VideoWriter(handler.path_to_save / f"Backgrounded_{handler.N}.MOV",
                             cv2.VideoWriter_fourcc(*'avc1'), 
                             fps,
                             (width, height))
    ret, frame = cap.read()
    for i in track(range(num_frames), description="Removing background"):
        ret, frame = cap.read()
        if not ret:
            print_error(f"Error reading frame {i}")
            break
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        diff = frame.astype(np.int16) - background.astype(np.int16)
        diff = np.abs(diff).astype(np.uint8)

        # Write the frame to the output video
        writer.write(cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
    
    cap.release()
    writer.release()

    print_success(f"Background removed and saved to {handler.path_to_save / f'Backgrounded_{handler.N}.MOV'}")
    return True
