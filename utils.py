import subprocess
from pathlib import Path
from typing import Union, List
import re
from .printing import *

def create_folder(path: Union[str, Path], overwrite: bool = False) -> Path:
    """Create a folder, optionally overwriting if it exists.
    
    Args:
        path: Path to create
        overwrite: Whether to overwrite existing folder
        
    Returns:
        Path object for the created folder
        
    Raises:
        FileExistsError: If folder exists and overwrite is False
    """
    path = Path(path)
    
    if path.exists():
        if overwrite:
            import shutil
            shutil.rmtree(path)
            path.mkdir(parents=True)
        else:
            raise FileExistsError(f"Folder '{path}' already exists.")
    else:
        path.mkdir(parents=True, exist_ok=True)
        
    return path

def check_frame_sequence(path_to_folder: Union[str, Path], pattern: str = "frame_*****.png") -> List[int]:
    """Check if all files matching a pattern with '*' are present without gaps.

    Args:
        path_to_folder: Path to folder containing frames.
        pattern: Filename pattern with '*' as the numeric placeholder. (frame_0001.png --> frame_****.png) etc.

    Returns:
        List of missing frame indices, empty if all frames are present.
    """
    path = Path(path_to_folder)

    # Convert user pattern to a regex pattern
    regex_pattern = re.escape(pattern).replace(r'\*', r'(\d+)')  # Convert * to (\d+)
    
    # Find all matching files
    files = sorted(path.glob(pattern.replace('*', '?')))  # Convert * to glob's ?
    
    # Extract numbers based on pattern
    numbers = []
    for f in files:
        match = re.match(regex_pattern, f.name)
        if match:
            numbers.append(int(match.group(1)))

    if not numbers:
        print_error(f"No valid files found matching {pattern}\n in {path_to_folder}")
        return []

    # Find missing numbers
    expected_range = set(range(min(numbers), max(numbers) + 1))
    missing_numbers = sorted(expected_range - set(numbers))

    if missing_numbers:
        print_error(f"Missing {len(missing_numbers)} frames: {missing_numbers}")
        return False
    else:
        print_success("All frames present and in order.")
        return True

def images_to_video(
    folder_path: Union[str, Path],
    output_video: Union[str, Path],
    fps: int = 60,
    lossless: bool = False
) -> bool:
    """
    Converts PNG images into a video (MP4 or MOV), with optional lossless encoding.

    Args:
        folder_path: Path to folder with frame_XXXX.png images
        output_video: Output .mp4 or .mov file path
        fps: Frames per second
        lossless: If True, saves with lossless compression
    """
    folder_path = Path(folder_path)
    output_video = Path(output_video)

    if not folder_path.exists():
        print_error(f"Folder does not exist: {folder_path}")
        return False

    frame_files = sorted(folder_path.glob("frame_*.png"))
    if not frame_files:
        print_error("No frame images found.")
        return False

    match = re.match(r"frame_(\d+)\.png", frame_files[0].name)
    if not match:
        print_error("Could not parse frame number.")
        return False

    digit_count = len(match.group(1))
    start_number = int(match.group(1))
    input_pattern = str(folder_path / f"frame_%0{digit_count}d.png")

    ext = output_video.suffix.lower()
    if ext not in [".mp4", ".mov"]:
        print_error(f"Unsupported output format: {ext}")
        return False

    codec = "libx264"
    crf = "0" if lossless else "18"

    codec_args = [
        "-crf", crf,
        "-preset", "veryslow" if lossless else "slow",
        "-pix_fmt", "yuv420p"
    ]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-framerate", str(fps),
        "-start_number", str(start_number),
        "-i", input_pattern,
        "-c:v", codec,
        *codec_args,
        "-threads", "0",
        str(output_video)
    ]

    print_info(f"Running FFmpeg: {folder_path}")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print_success(f"Video created: {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"FFmpeg failed: {e}")
        return False

def video_to_images(path_to_video, path_to_save, num_frames_to_extract=30, name="frame_"):
    import os
    import cv2
    from rich.progress import track

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

def get_os_root_dir():
    import sys
    if sys.platform == "darwin":
        return Path("/Volumes/Shared Bartololab3")
    elif sys.platform.startswith("linux"):
        return Path("/partages/Bartololab3/Shared")
    else:
        raise EnvironmentError("Unsupported operating system. This code is designed for MacOS or Linux.")