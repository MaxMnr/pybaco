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


class ContoursPipeline:
    def __init__(self, handler, model_path):
        self.handler = handler
        self.model = YOLO(model_path)

        self.video_path = self.handler.path_to_save / f"Stabilized_{self.handler.N}.MOV"

        self.main_dir = self.handler.path_to_save
        self.contours_dir = self.main_dir / "contours"
        self.contours_smooth_dir = self.main_dir / "contours_smooth"
        self.videos_dir = self.main_dir

        os.makedirs(self.main_dir, exist_ok=True)
        os.makedirs(self.contours_dir, exist_ok=True)
        os.makedirs(self.contours_smooth_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

    def process_video(self, num_frames=None):

            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            
            print_title("Processing video with YOLOv?", title=f"Running with {device}")
            
            results = self.model(self.video_path, stream=True, verbose=False, save=False, device=device, imgsz=1024, retina_masks=True)

            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap = None

            if num_frames is None or num_frames > total_frames:
                num_frames = total_frames
        
            for index, result in track(enumerate(results), total=num_frames, description="Looking for contour only once!"):
                if index >= num_frames:
                    break
                if result.masks is not None and len(result.masks.xy) > 0:
                    mask = result.masks.xy[0]
                    np.save(self.contours_dir / f"frame_{index:06d}.npy", mask)
            
            print_success("Done processing video.")
            print_success("Contours saved to:" + str(self.contours_dir))
            print_title("Computing Smoothed Contours", title="")
            
            raw_contours = sorted([f for f in os.listdir(self.contours_dir) if not f.startswith('.') and f.endswith('.npy')])
            raw_contours = [np.load(os.path.join(self.contours_dir, f)) for f in raw_contours]
            self.smooth_contours(raw_contours, num_frames_to_average=61)
            print_success("Smoothed contours saved to:" + str(self.contours_smooth_dir))

            print_title("Creating video from contours", title="")
            self.create_video_from_contours(self.videos_dir / "final_video.MOV")
            print_success("Video succesfully created! \nSaved to:" + str(self.videos_dir / "final_video.MOV"))

    def resample_contours(self, raw_contours):
        max_length = max(len(c) for c in raw_contours)
        resampled_contours = []

        for contour in raw_contours:
            contour = np.asarray(contour)

            if len(contour) < max_length:
                # Generate uniform spacing for interpolation
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

    def align_contours_to_first(self, resampled_contours):
        aligned_contours = [resampled_contours[0]]
        reference = resampled_contours[0]

        for i in range(1, len(resampled_contours)):
            contour = resampled_contours[i]
            dists = np.linalg.norm(contour - reference[0], axis=1)
            idx = np.argmin(dists)
            aligned = np.roll(contour, -idx, axis=0)
            aligned_contours.append(aligned)
            reference = aligned  
        return np.stack(aligned_contours)

    def smooth_contours(self, raw_contours, num_frames_to_average=31):
        
        # Get the contours from the saved .npy files get by YOLO
        resampled_contours = self.resample_contours(raw_contours)
        aligned_contours = self.align_contours_to_first(resampled_contours)
        
        smoothed_contours = []
        num_contours = len(aligned_contours)

        for mid_index in range(num_frames_to_average // 2, num_contours - num_frames_to_average // 2):
            start_index = mid_index - num_frames_to_average // 2
            end_index = mid_index + num_frames_to_average // 2 + 1 
            averaged_contour = np.mean(aligned_contours[start_index:end_index], axis=0)
            smoothed_contours.append(averaged_contour)
            # Save the smoothed contour in Cartesian coordinates
            np.save(self.contours_smooth_dir / f"frame_{mid_index:06d}.npy", averaged_contour)
    
        self.smoothed_contours = np.stack(smoothed_contours)

    def create_video_from_contours(self, output_path):
        cap = cv2.VideoCapture(self.video_path)
        width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get the cartesian contour files
        raw_files = sorted([f for f in os.listdir(self.contours_dir) if not f.startswith('.') and f.endswith('.npy')])
        cartesian_files = sorted([f for f in os.listdir(self.contours_smooth_dir) if not f.startswith('.') and f.endswith('.npy')])
        first_index = cartesian_files[0].split("_")[1].split(".")[0]
        
        # Set the camera to the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_index))
        
        for index, (raw_file, contour_file) in track(enumerate(zip(raw_files, cartesian_files)), total=len(cartesian_files), description="Creating video from contours"):
            ret, frame = cap.read()
            if not ret:
                break

            # Load raw contour
            raw_contour = np.load(os.path.join(self.contours_dir, raw_file))
            raw_contour = np.array(raw_contour, dtype=np.int32)
            raw_contour = raw_contour.reshape((-1, 1, 2))

            # Load cartesian contour
            contour = np.load(os.path.join(self.contours_smooth_dir, contour_file))
            contour = np.array(contour, dtype=np.int32)
            contour = contour.reshape((-1, 1, 2))
            
    
        
            # Draw cartesian contour
            cv2.polylines(frame, [contour], isClosed=True, color=(28, 29, 23), thickness=12)
            
            # Draw raw contour
            cv2.polylines(frame, [raw_contour], isClosed=True, color=(244, 233, 239), thickness=6)


            # Add frame number
            cv2.putText(frame, f"Frame: {index}", (200, 250), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 3)
            
            # Add legend
            cv2.putText(frame, "Raw", (width-700, height-150), cv2.FONT_HERSHEY_COMPLEX, 1.5, (244, 233, 239), 3)
            cv2.line(frame, (width-400, height-150), (width-200, height-150), (244, 233, 239), 5)

            cv2.putText(frame, "Cartesian", (width-700, height-100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (28, 29, 23), 3)
            cv2.line(frame, (width-400, height-100), (width-200, height-100), (28, 29, 23), 5)
        
            out.write(frame)

        out.release()
        cap.release()
        print_success("Video created from contours.")
        print_success("Video saved to:" + str(output_path))

class ContoursPipelineCBP:
    def __init__(self, handler, model_path):
        self.handler = handler
        self.model = YOLO(model_path)
        self.local_folder_to_save = Path("temp")

    def download_video(self, username, server):
        """
        For this method to work smoothly it may be neccesary to create a ssh key between the cbp and the server.
        Otherwise, the password will be prompted every time.
        """
        path_to_video_remote = self.handler.path_to_save / f"Stabilized_{self.handler.N}.MOV"
        #path_to_video_local = Path(__file__).resolve().parent
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
            self.video_path = self.local_folder_to_save / f"Stabilized_{self.handler.N}.MOV"

        except subprocess.CalledProcessError as e:
            print_error(f"Error downloading file: {e}")
            exit(1)
            return False
        return True 

    def delete_files(self):
        subprocess.run(["rm", "-rf", self.local_folder_to_save / f"Stabilized_{self.handler.N}.MOV"], check=True)
        print_info(f"Video deleted!")

        subprocess.run(["rm", "-rf", self.local_folder_to_save / f"Contours_{self.handler.N}.MOV"], check=True)
        print_info(f"Final Video deleted!")

        subprocess.run(["rm", "-rf", self.local_folder_to_save / f"raw_contours.npy"], check=True)
        print_info(f"Contours deleted!")
        
        subprocess.run(["rm", "-rf", self.local_folder_to_save / f"smooth_contours.npy"], check=True)
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
            self.local_folder_to_save / "smooth_contours.npy",
            f"{username}@{server}:{self.handler.path_to_save}"
        ]
        subprocess.run(scp_command, check=True)
        print_success("Contours smooth sent to server")
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
            
            print_title("Processing video with YOLOv?", title=f"Running with {device}")
            
            results = self.model(self.video_path, stream=True, verbose=False, save=False, device=device, imgsz=1024, retina_masks=True)

            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap = None

            if num_frames is None or num_frames > total_frames:
                num_frames = total_frames
            raw_contours = []
            with progress:
                task = progress.add_task("Looking for contour only once!", total=num_frames)
                
                for index, result in enumerate(results):
                    if index >= num_frames:
                        break
                    if result.masks is not None and len(result.masks.xy) > 0:
                        mask = result.masks.xy[0]
                        #np.save(self.contours_dir / f"frame_{index:06d}.npy", mask)
                        raw_contours.append(mask)
                    progress.update(task, advance=1)


            print_success("Done processing video.")
            print_title("Computing Smoothed Contours", title="")
        
            self.raw_contours = self._resample_contours(raw_contours)
            self.smooth_contours = self._smooth_contours(self.raw_contours, num_frames_to_average=31)
            
            np.save(self.local_folder_to_save / "raw_contours.npy", self.raw_contours)
            np.save(self.local_folder_to_save / "smooth_contours.npy", self.smooth_contours)   

            print_title("Creating video from contours", title="")
            self.create_video_from_contours(self.local_folder_to_save / f"Contours_{self.handler.N}.MOV")
            print_success("Video succesfully created!")

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

    def _align_contours_to_first(self, resampled_contours):
        aligned_contours = [resampled_contours[0]]
        reference = resampled_contours[0]

        for i in range(1, len(resampled_contours)):
            contour = resampled_contours[i]
            dists = np.linalg.norm(contour - reference[0], axis=1)
            idx = np.argmin(dists)
            aligned = np.roll(contour, -idx, axis=0)
            aligned_contours.append(aligned)
            reference = aligned  
        return np.stack(aligned_contours)

    def _smooth_contours(self, raw_contours, num_frames_to_average=31):
        
        resampled_contours = raw_contours # Already resaloked at the end of process_video
        aligned_contours = self._align_contours_to_first(resampled_contours)
        
        smoothed_contours = []
        num_contours = len(aligned_contours)

        for mid_index in range(num_frames_to_average // 2, num_contours - num_frames_to_average // 2):
            start_index = mid_index - num_frames_to_average // 2
            end_index = mid_index + num_frames_to_average // 2 + 1 
            averaged_contour = np.mean(aligned_contours[start_index:end_index], axis=0)
            smoothed_contours.append(averaged_contour)
            # Save the smoothed contour in Cartesian coordinates
            #np.save(self.contours_smooth_dir / f"frame_{mid_index:06d}.npy", averaged_contour)
        return np.stack(smoothed_contours)

    def create_video_from_contours(self, output_path):

        if self.smooth_contours is None:
            print_error("No smoothed contours found. Please run the smooth_contours method first.")
            return

        cap = cv2.VideoCapture(self.video_path)
        width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        first_index = 0
        
        # Set the camera to the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_index))
        
        for index in track(range(len(self.smooth_contours)), total=len(self.smooth_contours), description="Creating video from contours"):
            ret, frame = cap.read()
            if not ret:
                break

            # Load raw contour
            raw_contour = self.raw_contours[index]
            raw_contour = np.array(raw_contour, dtype=np.int32)
            raw_contour = raw_contour.reshape((-1, 1, 2))

            # Load cartesian contour
            contour = self.smooth_contours[index]
            contour = np.array(contour, dtype=np.int32)
            contour = contour.reshape((-1, 1, 2))
            
    
        
            # Draw cartesian contour
            cv2.polylines(frame, [contour], isClosed=True, color=(28, 29, 23), thickness=12)
            
            # Draw raw contour
            cv2.polylines(frame, [raw_contour], isClosed=True, color=(244, 233, 239), thickness=6)


            # Add frame number
            cv2.putText(frame, f"Frame: {index}", (200, 250), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 3)
            
            # Add legend
            cv2.putText(frame, "Raw", (width-700, height-150), cv2.FONT_HERSHEY_COMPLEX, 1.5, (244, 233, 239), 3)
            cv2.line(frame, (width-400, height-150), (width-200, height-150), (244, 233, 239), 5)

            cv2.putText(frame, "Cartesian", (width-700, height-100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (28, 29, 23), 3)
            cv2.line(frame, (width-400, height-100), (width-200, height-100), (28, 29, 23), 5)
            

            out.write(frame)
            
        out.release()
        cap.release()
        print_success("Video created from contours.")
        print_success("Video saved to:" + str(output_path))