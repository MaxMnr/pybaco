from ultralytics import YOLO
import torch
import os
import cv2
import numpy as np
from pathlib import Path
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, track
import time

from pybaco.printing import *
from shapely.geometry import Polygon
import subprocess

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


class ContoursPipelineCBP:
    def __init__(self, handler, model_path):
        self.handler = handler
        self.model = YOLO(model_path)
        self.local_folder_to_save = Path("temp")
        self.local_folder_to_save.mkdir(parents=True, exist_ok=True)

    def download_video(self, username, server):
        """
        For this method to work smoothly it may be neccesary to create a ssh key between the cbp and the server.
        Otherwise, the password will be prompted every time.
        """
        path_to_video_remote = self.handler.path_to_save / f"Backgrounded_{self.handler.N}.MOV"
        scp_command = [
            "scp",
            f"{username}@{server}:{path_to_video_remote}",
            self.local_folder_to_save
        ]

        # Execute the command
        try:
            print_info(f"Downloading {path_to_video_remote} from {username}@{server}")
            subprocess.run(scp_command, check=True)
            print_success("Video downloaded")
            self.video_path = self.local_folder_to_save / f"Backgrounded_{self.handler.N}.MOV"

        except subprocess.CalledProcessError as e:
            print_error(f"Error downloading file: {e}")
            exit(1)
            return False
        return True 

    def delete_files(self):
        subprocess.run(["rm", "-rf", self.local_folder_to_save / f"Backgrounded_{self.handler.N}.MOV"], check=True)
        print_info(f"Video deleted!")

        subprocess.run(["rm", "-rf", self.local_folder_to_save / f"Contours_{self.handler.N}.MOV"], check=True)
        print_info(f"Final Video deleted!")

        subprocess.run(["rm", "-rf", self.local_folder_to_save / f"raw_contours.npy"], check=True)
        print_info(f"Contours deleted!")


    def send_files(self, username, server):
        # Send the two contours folders and the final video to the server
        scp_command = [
            "scp",
            "-r",
            self.local_folder_to_save / "raw_contours.npy",
            f"{username}@{server}:{self.handler.path_to_save}"
        ]
        subprocess.run(scp_command, check=True)
        print_success("Contours sent to server")
        scp_command = [
            "scp",
            "-r",
            self.local_folder_to_save / f"Contours_{self.handler.N}.MOV",
            f"{username}@{server}:{self.handler.path_to_save}"
        ]
        subprocess.run(scp_command, check=True)
        print_success("Final video sent to server")
        
    def process_video(self, num_frames=None):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        print_title("Processing video with YOLOv11", title=f"Running with {device}")

        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_frames is None or num_frames > total_frames:
            num_frames = total_frames

        self.raw_contours = []

        with progress:
            task = progress.add_task("Looking for contour only once!", total=num_frames)

            frame_idx = 0
            while frame_idx < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(
                    frame,
                    verbose=False,
                    save=False,
                    device=device,
                    imgsz=640,
                    retina_masks=True
                )

                result = results[0]
                if result.masks is not None and len(result.masks.xy) > 0:
                    mask = result.masks.xy[0]
                    self.raw_contours.append(mask)

                progress.update(task, advance=1)
                frame_idx += 1

        cap.release()

        print_success("Done processing video.")
        print_title("Computing Smoothed Contours", title="")

        self.raw_contours = self._resample_contours(self.raw_contours)
        np.save(self.local_folder_to_save / "raw_contours.npy", self.raw_contours)

        print_title("Creating video from contours", title="")
        self.create_video_from_contours(self.local_folder_to_save / f"Contours_{self.handler.N}.MOV")
        print_success("Video successfully created!")

    def _resample_contours(self, raw_contours):
        max_length = max(len(c) for c in raw_contours)
        resampled_contours = []
        for contour in raw_contours:
            contour = np.asarray(contour)
            if len(contour) < max_length:
                original_indices = np.linspace(0, 1, len(contour))
                target_indices = np.linspace(0, 1, max_length)
                # Interpolate x and y separately
                x_interp = np.interp(target_indices, original_indices, contour[:, 0])
                y_interp = np.interp(target_indices, original_indices, contour[:, 1])
                resampled_contour = np.stack((x_interp, y_interp), axis=1)
            else:
                resampled_contour = contour
            resampled_contours.append(resampled_contour)
        return np.stack(resampled_contours)

    def create_video_from_contours(self, output_path):
        cap = cv2.VideoCapture(self.video_path)
        width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for index in track(range(len(self.raw_contours)), total=len(self.raw_contours), description="Creating video from contours"):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw raw contour
            raw_contour = np.array(self.raw_contours[index], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [raw_contour], isClosed=True, color=(42, 22, 171), thickness=6, lineType=cv2.LINE_AA)

            # Frame number (top left)
            cv2.putText(frame, f"Frame: {index}", (200, 250),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 6, cv2.LINE_AA)

            out.write(frame)

        cap.release()
        out.release()
        print_success("Video created from contours.")
        print_success("Video saved to:" + str(output_path))

