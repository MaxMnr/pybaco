import numpy as np
import cv2
from shapely.geometry import Polygon
import multiprocessing as mp
from ultralytics import YOLO

from typing import Optional, List
import logging

import os
from pathlib import Path
import shutil

from .handler import VideoHandler
from .stabilization import *
from .utils import *
from .printing import *
from .progress import ProgressTracker
from rich.progress import track

"""
============= This class is deprecated. =============
It is not guaranted to work fine as I continue to modify 
other classes and functions while not maintaining this one. 
=====================================================
"""


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='/partages/Bartololab3/Shared/Maxime/video_correction.log',    
    filemode='a',                    
    level=logging.DEBUG,             
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ProcessingPipeline:
    """
    A pipeline for processing video frames with averaging and stabilization using workers and queues.
    """
    def __init__(self, handler: VideoHandler, first_frame_index: int = 0, total_frames: Optional[int] = 60, 
                 num_frames_to_average: int = 21, num_workers: int = 10, num_transform_workers: int = None,
                 do_averaging: bool = True, do_stabilization: bool = True):
        self.handler = handler
        
        # Video parameters
        self.first_frame_index = int(first_frame_index)
        self.total_frames = int(total_frames)
        
        # Processing options
        self.num_frames_to_average = int(num_frames_to_average) if do_averaging else 1 
        self.num_workers = int(num_workers)
        # Set num_transform_workers to 2*num_workers if not specified
        self.num_transform_workers = int(num_transform_workers) if num_transform_workers is not None else int(num_workers) * 2
        self.do_averaging = bool(do_averaging)
        self.do_stabilization = bool(do_stabilization)

        # Paths setup
        self.path_to_bad_contour = self.handler.path_to_save / "bad_contours"
        self.averaging_path = self.handler.path_to_save / f"Averaged_{self.num_frames_to_average}"
        self.transformations_path = self.handler.path_to_save / f"Transformation_Matrices_{self.num_frames_to_average}"
        self.stabilizing_path = self.handler.path_to_save / f"Stabilized_{self.num_frames_to_average}"

        if self.handler.path_to_save.exists():
            shutil.rmtree(self.handler.path_to_save)

        # Create directories
        for path in [self.handler.path_to_save, self.averaging_path, self.transformations_path, self.stabilizing_path, self.path_to_bad_contour]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Transformation data
        self.frame_to_frame_transforms = None
        self.smoothed_transforms = None

    def _find_contours(self, path_to_model):
        """
        Find contours in the video using YOLO and save them to the specified directory.
        """
        # Load YOLO model
        model = YOLO(path_to_model, verbose=False)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.path_to_bad_contour, exist_ok=True)
        # Open video file
        cap = cv2.VideoCapture(str(self.handler.path_to_video))
        
        if not cap.isOpened():
            logger.error(f"Error opening video file: {self.handler.path_to_video}")
            return False
        
        # Process each frame
        for frame_index in track(range(self.first_frame_index, self.first_frame_index + self.total_frames), description="Finding Bad Contours"):
            # Only process one frame every second
            if (frame_index - self.first_frame_index) % 60 == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Error reading frame {frame_index} from video")
                    break
                
                # Run YOLO model on the frame
                results = model(frame, imgsz=640, retina_masks=True, verbose=False)
                result = results[0]
                if result.masks is not None and len(result.masks.xy) > 0:
                    mask = result.masks.xy[0]                        

            np.save(self.path_to_bad_contour / f"frame_{frame_index:06d}.npy", mask)
        
        cap.release()
        print_success(f"Contours saved to {self.path_to_bad_contour}")
        

    def run(self):
        """Main method to run the entire pipeline."""
        logger.info(f"Starting processing: {self.handler.event_name} {self.handler.day} {self.handler.number} {self.handler.version}")
        print_title(f"Starting processing: {self.handler.event_name} {self.handler.day} DJI_{self.handler.number} {self.handler.version}", title="")
        print_info(f"Gonna process {self.total_frames} frames\n")
        
        # STEP 0: Create bad contours for stabilization 
        if self.do_stabilization:
            print_info(f"Creating contours in {self.handler.path_to_save / "bad_contours"}")
            self._find_contours(path_to_model="./yolo_models/banc_best.pt")

        # STEP 1: Process frames (average and compute transformations)
        success = self.run_phase_one()
        if not success:
            logger.error("Frame processing failed")
            return False
        
        print_success("Phase 1 was succesfull!\n")

        # STEP 2: Apply stabilization if needed
        if self.do_stabilization:
            success = self.run_phase_two()
            if not success:
                logger.error("Stabilization failed")
                return False
        
        print_success("Phase 2 was succesfull!\n")

        # STEP 3: Convert frames to video for averaged and stabilized frames
        try: 
            images_to_video(
                self.averaging_path, 
                f"{self.handler.path_to_save}/Averaged_{self.num_frames_to_average}.MOV", 
                fps=60, 
                lossless=False
            )
            images_to_video(
                self.stabilizing_path, 
                f"{self.handler.path_to_save}/Stabilized_{self.num_frames_to_average}.MOV", 
                fps=60, 
                lossless=False    
            )
            logger.info(f"Videos succesfully created!")
            print_success("Video creations was succesfull!\n")

            if self.do_stabilization:
                self._cleanup_frames_folders()
                logger.info(f"Removed {self.averaging_path}")
                logger.info(f"Removed {self.stabilizing_path}")
        except:
            logger.error(f"Error creating videos: {self.averaging_path} or {self.stabilizing_path}")
            print_error(f"Error creating videos: {self.averaging_path} or {self.stabilizing_path}")
            return False
        return True
    
    def run_phase_one(self):
        """
        Run the first phase: Average frames and compute transformations.
        Uses worker processes and a queue system.
        """
        try:
            # Create frame list to process
            frame_indices = list(range(
                self.first_frame_index + self.num_frames_to_average // 2 + 1, 
                self.first_frame_index + self.total_frames - self.num_frames_to_average // 2 - 1
            ))
            
            if self.do_stabilization:
                # Check wether all contours from frame_indices exist
                contours = sorted([f for f in os.listdir(self.path_to_bad_contour) if f.endswith(".npy")])

                for index in frame_indices:
                    expected_file = f"frame_{index:06d}.npy"
                    if expected_file not in contours:
                        logger.error(f"File {expected_file} not found in the folder {self.path_to_bad_contour}")
                        return False
                
            # Set up progress tracking
            tasks = {"Averaging frames": len(frame_indices)}
            if self.do_stabilization:
                tasks["Computing transformations"] = len(frame_indices)
            
            progress = ProgressTracker(tasks)

            # Create a queue for passing data between processes
            frame_queue = mp.Queue(maxsize=0)

            # Get reference frame for stabilization
            cap = cv2.VideoCapture(str(self.handler.path_to_video))
            if not cap.isOpened():
                logger.error(f"Error opening video file: {self.handler.path_to_video}")
                return False
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_frame_index)
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read reference frame")
                cap.release()
                return False
            ref_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # get bad contour for the reference frame
            contour_file = self.path_to_bad_contour / f"frame_{self.first_frame_index:06d}.npy"
            polygon = Polygon(np.load(contour_file))
            mask = polygon_to_mask(polygon, ref_img.shape)
            ref_img = cv2.bitwise_and(ref_img, ref_img, mask=mask.astype(np.uint8))
            cap.release()

            # Calculate chunks for frame averaging workers based on num_workers
            chunk_size = max(1, len(frame_indices) // self.num_workers)
            
            # Start the progress tracker
            progress.start()
            
            # Start workers
            workers = self._start_phase_one_workers(
                frame_indices, chunk_size, self.handler.path_to_video, self.averaging_path,
                self.transformations_path, self.path_to_bad_contour, frame_queue,
                progress, ref_img
            )
            
            # Wait for workers to finish
            success = self._monitor_workers(workers)
            
            # Stop the progress tracker
            progress.stop()
            
            return success
            
        except Exception as e:
            logger.error(f"Error in phase one: {e}")
            return False
    
    def _start_phase_one_workers(
            self, frame_indices, chunk_size, video_path_str, averaging_path_str,
            transform_path_str, contours_path_str, frame_queue, progress, ref_img):
        """Start the workers for phase one and return the list of worker processes."""
        workers = []
        
        # Start averaging workers based on self.num_workers
        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_workers - 1 else len(frame_indices)
            chunk = frame_indices[start_idx:end_idx]
            
            worker = mp.Process(
                target=self._frame_averaging_worker,
                args=(
                    video_path_str,
                    averaging_path_str,
                    chunk,
                    self.num_frames_to_average,
                    frame_queue,
                    progress.counters["Averaging frames"],
                    self.do_stabilization
                )
            )
            workers.append(worker)
            worker.start()
        
        # Start transformation workers if needed - using self.num_transform_workers
        if self.do_stabilization:
            for _ in range(self.num_transform_workers):
                worker = mp.Process(
                    target=self._transformation_worker,
                    args=(
                        frame_queue,
                        ref_img,
                        progress.counters["Computing transformations"],
                        contours_path_str,
                        transform_path_str
                    )
                )
                workers.append(worker)
                worker.start()
        if self.do_stabilization and self.num_transform_workers > self.num_workers:
            for _ in range(self.num_transform_workers - self.num_workers):
                frame_queue.put(None)
        
        return workers
    def run_phase_two(self):
        """
        Run the second phase: Apply stabilization to averaged frames.
        Uses worker processes but no queue.
        """
        try:
            # Check if we have all required files
            if not (check_frame_sequence(self.averaging_path, pattern="frame_******.png") and
                    check_frame_sequence(self.transformations_path, pattern="frame_******.npy")):
                logger.error("Missing averaged frames or transformation matrices")
                return False
            
            # Load all transformation matrices
            transform_files = sorted(self.transformations_path.glob("frame_*.npy"))
            all_transforms = np.array([np.load(path) for path in transform_files])
            
            # Process transformation matrices
            self.frame_to_frame_transforms = get_frame_to_frame_transforms(all_transforms)
            smoothed_f2f = smooth_trajectory_parameters(self.frame_to_frame_transforms, radius=15)
            self.smoothed_transforms = np.array(reconstruct_reference_transforms(smoothed_f2f))
            
            # Get image paths
            image_paths = sorted(self.averaging_path.glob("frame_*.png"))
            
            # Set up progress tracking
            progress = ProgressTracker({"Applying stabilization": len(image_paths)})
            
            # Split work among workers
            chunk_size = max(1, len(image_paths) // (self.num_workers))
            
            # Start the progress tracker
            progress.start()
            
            # Start stabilization workers
            workers = self._start_phase_two_workers(
                image_paths, chunk_size, progress.counters["Applying stabilization"]
            )
            
            # Wait for workers to finish
            success = self._monitor_workers(workers)
            
            # Stop the progress tracker
            progress.stop()
            
            return success
            
        except Exception as e:
            logger.error(f"Error in phase two: {e}")
            return False
    
    def _start_phase_two_workers(self, image_paths, chunk_size, counter):
        """Start the workers for phase two and return the list of worker processes."""
        workers = []
        
        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_workers - 1 else len(image_paths)
            
            # Only process up to the number of transforms we have
            end_idx = min(end_idx, len(self.smoothed_transforms))
            
            if start_idx >= end_idx:
                continue
            
            worker = mp.Process(
                target=self._stabilize_frames_worker,
                args=(
                    image_paths[start_idx:end_idx],
                    self.smoothed_transforms[start_idx:end_idx],
                    self.first_frame_index + start_idx,
                    str(self.stabilizing_path),
                    counter
                )
            )
            workers.append(worker)
            worker.start()
        
        return workers
    
    def _monitor_workers(self, workers):
        """Monitor workers and handle exceptions."""
        try:
            # Wait for all workers to finish
            for worker in workers:
                worker.join()
            
            # Check if any workers failed
            for worker in workers:
                if worker.exitcode != 0:
                    logger.error(f"Worker failed with exit code {worker.exitcode}")
                    return False
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            return False
        except Exception as e:
            logger.error(f"Error monitoring workers: {e}")
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            return False

    
    @staticmethod # Using static method to ensure pickability for multiprocess stuff 
    def _frame_averaging_worker(
            video_path: str, 
            averaging_path: str,
            frame_indices: List[int],
            num_frames_to_average: int,
            queue: mp.Queue,
            counter: mp.Value,
            do_stabilization: bool
        ):
        """Worker process to average frames, save them, and send to queue."""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Error opening video in worker: {video_path}")
                return
            
            # Process each frame in the chunk
            for frame_index in frame_indices:
                # Position at correct location
                start_pos = max(0, frame_index - num_frames_to_average // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
                
                # Collect frames for averaging
                frames = []
                for _ in range(num_frames_to_average):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame.astype(np.float32))
                
                # Skip if we couldn't get enough frames
                if len(frames) != num_frames_to_average:
                    logger.warning(f"Skipping frame {frame_index}: Could not get {num_frames_to_average} frames")
                    continue
                
                # Average the frames
                avg_frame = np.mean(frames, axis=0).astype(np.uint8)
                
                # Save the averaged frame
                output_path = os.path.join(averaging_path, f"frame_{frame_index:06d}.png")
                cv2.imwrite(output_path, avg_frame)
                
                # Send to queue for transformation calculation if needed
                if do_stabilization:
                    queue.put((frame_index, avg_frame))
                
                # Update counter
                with counter.get_lock():
                    counter.value += 1
            
            # Signal completion to queue
            if do_stabilization:
                queue.put(None)
            
            # Close video
            cap.release()
            
        except Exception as e:
            logger.error(f"Error in averaging worker: {e}")
   
    @staticmethod # Using static method to ensure pickability for multiprocess stuff      
    def _transformation_worker(
            queue: mp.Queue,
            ref_img: np.ndarray,
            counter: mp.Value,
            contours_path: str,
            transform_path: str
        ):
        """Worker process to calculate transformations from frames in the queue."""
        try:
            while True:
                # Get data from queue
                data = queue.get()
                
                # Check for termination signal
                if data is None:
                    break
                
                frame_index, avg_frame = data
                
                # Find contour file
                try:
                    contour_file = os.path.join(contours_path, f"frame_{frame_index:06d}.npy")
                    polygon = Polygon(np.load(contour_file))
                    # Log an error if the exact contour was not found
                except:
                    logger.error(f"Exact contour file not found for frame {frame_index}. Trying fallback.")
                    # Try fallback: searching 120 frames around the current one
                    found_contour = False
                    for i in range(-60, 60):  # Checking +/- 60 frames
                        fallback_file = os.path.join(contours_path, f"frame_{frame_index + i:06d}.npy")
                        if os.path.exists(fallback_file):
                            polygon = Polygon(np.load(fallback_file))
                            found_contour = True
                            logger.warning(f"Using fallback contour from frame {frame_index + i}")
                            break
                    if not found_contour:
                        # Critical error: no contour found for the current frame
                        logger.critical(f"No contour found for frame {frame_index}. Terminating process.")
                        raise ValueError(f"Missing contour for frame {frame_index}, stopping worker.")
                
                # Calculate transformation matrix
                avg_frame_gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
                M = get_transformation(ref_img, avg_frame_gray, polygon)
                
                # Save transformation matrix
                transform_output = os.path.join(transform_path, f"frame_{frame_index:06d}.npy")
                np.save(transform_output, M)
                
                # Update counter
                with counter.get_lock():
                    counter.value += 1
                
        except Exception as e:
            logger.error(f"Error in transformation worker: {e}")
            raise  # Re-raise the exception to terminate the worker
    
    @staticmethod  # Using static method to ensure pickability for multiprocess stuff 
    def _stabilize_frames_worker(
            image_paths: List[Path],
            transforms: np.ndarray,
            start_frame_index: int,
            output_path: str,
            counter: mp.Value
        ):
        """Worker process to apply stabilization to averaged frames."""
        try:
            for i, img_path in enumerate(image_paths):
                # Load the image
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue
                
                h, w = img.shape[:2]
                
                # Get transformation matrix
                M = transforms[i]
                
                # Calculate inverse transform
                A = M[:, :2]
                t = M[:, 2]
                
                A_inv = np.linalg.inv(A)
                t_inv = -A_inv @ t
                M_inv = np.hstack([A_inv, t_inv.reshape(2, 1)])
                
                # Apply the transformation
                corrected_img = cv2.warpAffine(img, M_inv, (w, h))
            
                # Save the stabilized image
                frame_index = start_frame_index + i
                output_file = os.path.join(output_path, f"frame_{frame_index:06d}.png")
                cv2.imwrite(output_file, corrected_img)
                
                # Update counter
                with counter.get_lock():
                    counter.value += 1
                
        except Exception as e:
            logger.error(f"Error in stabilization worker: {e}")
    
    def _cleanup_frames_folders(self):
        """Remove frames to save disk space (frames are really heavy)."""
        try:
            if self.averaging_path.exists():
                shutil.rmtree(self.averaging_path)
                
            if self.stabilizing_path.exists():
                shutil.rmtree(self.stabilizing_path)
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")