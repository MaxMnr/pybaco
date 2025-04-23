import os
import cv2
import multiprocessing as mp
import time
import numpy as np
import shutil  
from rich.progress import Progress
from joblib import Parallel, delayed
from .printing import *
from .utils import *



def video_to_images(path_to_video, path_to_save, num_frames_to_extract=30, name="frame_"):
    os.makedirs(path_to_save, exist_ok=True)

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        print_error("Error: Could not open video.")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames_to_extract < 0:
        num_frames_to_extract = frame_count
    if num_frames_to_extract > frame_count:
        num_frames_to_extract = frame_count
        print_warning(f"num_frames_to_extract is greater than the number of frames in the video. Setting it to {frame_count}.")
    if num_frames_to_extract == 0:
        print_warning("num_frames_to_extract is 0. No frames will be extracted.")
        return False

    step = max(1, frame_count // num_frames_to_extract)
    indices = list(range(0, frame_count, step))[:num_frames_to_extract]

    for i, idx in track(enumerate(indices), description="Extracting frames", total=len(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print_error(f"Error: Could not read frame at index {idx}.")
            continue
        frame_filename = os.path.join(path_to_save, f"{name}{i:06d}.png")
        cv2.imwrite(frame_filename, frame)

    cap.release()
    print_success(f"Frames saved to {path_to_save}.")
    return True


def compute_background(path_to_video, path_to_save, num_images=20):
    path_to_bg_frames = os.path.join(path_to_save, "frames_for_background")

    if os.path.exists(path_to_bg_frames):
        shutil.rmtree(path_to_bg_frames)
    os.makedirs(path_to_bg_frames)

    video_to_images(path_to_video, path_to_bg_frames, num_frames_to_extract=num_images, name="frame_")

    frame_files = sorted(os.listdir(path_to_bg_frames))
    frames = []

    for frame_file in frame_files:
        frame_path = os.path.join(path_to_bg_frames, frame_file)
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    frames_array = np.array(frames)
    print_title(f"Frame stack shape: {frames_array.shape}")

    background_max = np.max(frames_array, axis=0).astype(np.uint8)
    background_path = os.path.join(path_to_save, "background.png")
    cv2.imwrite(background_path, cv2.cvtColor(background_max, cv2.COLOR_RGB2BGR))

    print_success(f"Background computed and saved to {background_path}")
    return background_max


def remove_background_frame(path_to_video, background, frame_index, path_to_save):
    cap = cv2.VideoCapture(path_to_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    if not ret:
        print_error(f"Error reading frame {frame_index}")
        return False

    diff = frame.astype(np.int16) - background.astype(np.int16)
    diff = np.abs(diff).astype(np.uint8)

    diff_filename = os.path.join(path_to_save, f"frame_{frame_index:06d}.png")
    cv2.imwrite(diff_filename, cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))  # No color conversion
    return True


def remove_background(path_to_video, path_to_save):
    background = compute_background(path_to_video, path_to_save)

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        print_error("Error: Could not open video.")
        return False

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    path_to_save_frames = os.path.join(path_to_save, "frames_no_background")
    os.makedirs(path_to_save_frames, exist_ok=True)

    Parallel(n_jobs=100)(
        delayed(remove_background_frame)(path_to_video, background, i, path_to_save_frames)
        for i in track(range(num_frames), description="Removing background", total=num_frames)
    )

    print_success(f"Background removed from {num_frames} frames and saved to {path_to_save_frames}")

    # Compute a video from the frames
    video_name = path_to_save_frames.split("/")[-1]
    video_output_path = os.path.join(path_to_save, f"{video_name}_no_background.MOV")
    images_to_video(path_to_save_frames, video_output_path, fps=60)
    return True