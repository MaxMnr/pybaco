import numpy as np
import cv2
from shapely.geometry import Polygon
import multiprocessing as mp
from ultralytics import YOLO
from typing import Optional, List, Union
import logging
import os
from pathlib import Path
import shutil
import sys

from .handler import VideoHandler
from .stabilization import *
from .utils import *
from .printing import *
from .progress import ProgressTracker
from rich.progress import track, Progress

logger = logging.getLogger(__name__)

class VideoAverager:
    """Simple video averaging class."""
    
    def __init__(self, handler: VideoHandler, num_frames_to_average: int = 21, num_workers: int = 8):
        self.handler = handler
        self.num_frames_to_average = num_frames_to_average
        self.num_workers = num_workers
        self.output_path = None
        
    def average_video(self, first_frame: int = 0, total_frames: int = 60) -> bool:
        """Average frames from video and create output video."""
        
        # Setup output directory
        self.output_path = self.handler.path_to_save / f"Averaged_{self.num_frames_to_average}"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Get frame indices to process
        frame_indices = list(range(
            first_frame + self.num_frames_to_average // 2 + 1,
            first_frame + total_frames - self.num_frames_to_average // 2 - 1
        ))
        
        if not frame_indices:
            logger.warning("No frames to average")
            return False
        
        print_info(f"Averaging {len(frame_indices)} frames with window size {self.num_frames_to_average}")
        
        # Setup progress tracking
        progress = ProgressTracker({"Averaging": len(frame_indices)})
        progress.start()
        
        try:
            # Process frames in parallel
            success = self._process_frames_parallel(frame_indices, progress.counters["Averaging"])
            
            if success:
                # Create video from averaged frames
                self._create_averaged_video()
                print_success(f"Video averaging completed! Saved to: {self.handler.path_to_save}")
            
            return success
            
        finally:
            progress.stop()
    
    def _process_frames_parallel(self, frame_indices: List[int], counter) -> bool:
        """Process frames in parallel using multiprocessing."""
        chunk_size = max(1, len(frame_indices) // self.num_workers)
        workers = []
        
        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_workers - 1 else len(frame_indices)
            chunk = frame_indices[start_idx:end_idx]
            
            if chunk:
                worker = mp.Process(
                    target=self._process_chunk,
                    args=(chunk, counter)
                )
                workers.append(worker)
                worker.start()
        
        # Wait for all workers to complete
        try:
            for worker in workers:
                worker.join(timeout=300)  # 5 minute timeout
                if worker.is_alive():
                    worker.terminate()
            
            return all(worker.exitcode == 0 for worker in workers)
            
        except KeyboardInterrupt:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            return False
    
    def _process_chunk(self, frame_indices: List[int], counter):
        """Process a chunk of frames for averaging."""
        cap = cv2.VideoCapture(str(self.handler.path_to_video))
        if not cap.isOpened():
            logger.error(f"Error opening video: {self.handler.path_to_video}")
            return
        
        try:
            for frame_index in frame_indices:
                start_pos = max(0, frame_index - self.num_frames_to_average // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
                
                frames = []
                for _ in range(self.num_frames_to_average):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame.astype(np.float32))
                
                if len(frames) != self.num_frames_to_average:
                    logger.warning(f"Skipping frame {frame_index}: Could not get {self.num_frames_to_average} frames")
                    continue
                
                avg_frame = np.mean(frames, axis=0).astype(np.uint8)
                
                output_path = self.output_path / f"frame_{frame_index:06d}.png"
                cv2.imwrite(str(output_path), avg_frame)
                
                with counter.get_lock():
                    counter.value += 1
        
        except Exception as e:
            logger.error(f"Error in averaging worker: {e}")
        finally:
            cap.release()
    
    def _create_averaged_video(self):
        """Create video from averaged frames."""
        try:
            output_video_path = self.handler.path_to_save / f"Averaged_{self.num_frames_to_average}.MOV"
            images_to_video(
                self.output_path,
                str(output_video_path),
                fps=60,
                lossless=False
            )
            
            # Clean up frame directory
            shutil.rmtree(self.output_path)
            print_success(f"Averaged video saved: {output_video_path}")
            
        except Exception as e:
            logger.error(f"Error creating averaged video: {e}")
            print_error(f"Error creating averaged video: {e}")

class TransformationCalculator:
    """Calculate transformation matrices for video stabilization."""
    
    def __init__(self, handler: VideoHandler, path_to_model: str = "./yolo_models/banc_best.pt", num_workers: int = 8, detection_interval: int = 60):
        self.handler = handler
        self.path_to_model = path_to_model
        self.num_workers = num_workers
        self.detection_interval = detection_interval
        self.contour_path = None
        self.transform_path = None
        self.ref_img = None
        
    def calculate_transformations(self, video_path: str, first_frame: int = 0, total_frames: int = 60) -> bool:
        """Calculate transformation matrices for stabilization."""
        
        self.video_path = video_path
        
        # Setup output directories
        self.contour_path = self.handler.path_to_save / "bad_contours"
        self.transform_path = self.handler.path_to_save / "Transformation_Matrices"
        self.contour_path.mkdir(parents=True, exist_ok=True)
        self.transform_path.mkdir(parents=True, exist_ok=True)
        
        print_info("Starting transformation calculation...")
        
        # Detect contours
        print_info("Detecting contours...")
        if not self._detect_contours(first_frame, total_frames):
            return False
        
        # Setup reference image
        print_info("Setting up reference image...")
        if not self._setup_reference_image():
            return False
        
        # Calculate transformations
        print_info("Calculating transformations...")
        if not self._calculate_transforms(video_path, first_frame, total_frames):
            return False
        
        print_success(f"Transformation calculation completed! Matrices saved to: {self.transform_path}")
        return True
    
    def _detect_contours(self, first_frame: int, total_frames: int) -> bool:
        """Detect contours using YOLO model."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {self.video_path}")
            return False

        model = YOLO(self.path_to_model, verbose=False)
        detected_contour = None
        
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Detecting contours...", total=total_frames)
                
                for frame_index in range(first_frame, first_frame + total_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect contours every 'detection_interval' frames
                    if (frame_index - first_frame) % self.detection_interval == 0:
                        results = model(frame, imgsz=640, retina_masks=True, verbose=False)
                        result = results[0]
                        
                        if result.masks is not None and len(result.masks.xy) > 0:
                            detected_contour = result.masks.xy[0]

                    # Save the detected contour for the current frame (if there's one)
                    if detected_contour is not None:
                        output_path = self.contour_path / f"frame_{frame_index:06d}.npy"
                        np.save(output_path, detected_contour)

                    progress.update(task, advance=1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error detecting contours: {e}")
            return False
        finally:
            cap.release()
    
    def _setup_reference_image(self) -> bool:
        """Setup reference image with contour mask."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("Could not open video for reference image")
            return False
            
        try:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read first frame")
                return False
                
            self.ref_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contour_file = self.contour_path / f"frame_{0:06d}.npy"
            
            if contour_file.exists():
                polygon = Polygon(np.load(contour_file))
                mask = polygon_to_mask(polygon, self.ref_img.shape)
                self.ref_img = cv2.bitwise_and(self.ref_img, self.ref_img, mask=mask.astype(np.uint8))
            else:
                logger.warning("No contour found for reference frame, using full frame")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up reference image: {e}")
            return False
        finally:
            cap.release()
    
    def _calculate_transforms(self, video_path: str, first_frame: int, total_frames: int) -> bool:
        """Calculate transformation matrices for each frame."""
        
        # Get frame indices (use all frames in range)
        frame_indices = list(range(first_frame, first_frame + total_frames))
        
        if not frame_indices:
            logger.warning("No frames to process")
            return False
        
        # Setup progress tracking
        progress = ProgressTracker({"Transformations": len(frame_indices)})
        progress.start()
        
        try:
            # Process frames in parallel
            success = self._process_transforms_parallel(video_path, frame_indices, progress.counters["Transformations"])
            return success
            
        finally:
            progress.stop()
    
    def _process_transforms_parallel(self, video_path: str, frame_indices: List[int], counter) -> bool:
        """Process transformation calculation in parallel."""
        chunk_size = max(1, len(frame_indices) // self.num_workers)
        workers = []
        
        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_workers - 1 else len(frame_indices)
            chunk = frame_indices[start_idx:end_idx]
            
            if chunk:
                worker = mp.Process(
                    target=self._process_transform_chunk,
                    args=(video_path, chunk, counter)
                )
                workers.append(worker)
                worker.start()
        
        # Wait for all workers
        try:
            for worker in workers:
                worker.join(timeout=300)
                if worker.is_alive():
                    worker.terminate()
            
            return all(worker.exitcode == 0 for worker in workers)
            
        except KeyboardInterrupt:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            return False
    
    def _process_transform_chunk(self, video_path: str, frame_indices: List[int], counter):
        """Process a chunk of frames for transformation calculation."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {video_path}")
            return
        
        try:
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Find contour with fallback
                contour_file = self._find_contour_file(frame_index)
                if not contour_file:
                    logger.warning(f"No contour found for frame {frame_index}")
                    continue
                
                try:
                    polygon = Polygon(np.load(contour_file))
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    M = get_transformation(self.ref_img, frame_gray, polygon)
                    
                    output_path = self.transform_path / f"frame_{frame_index:06d}.npy"
                    np.save(output_path, M)
                    
                    with counter.get_lock():
                        counter.value += 1
                        
                except Exception as e:
                    logger.error(f"Error processing frame {frame_index}: {e}")
        
        except Exception as e:
            logger.error(f"Error in transformation worker: {e}")
        finally:
            cap.release()
    
    def _find_contour_file(self, frame_index):
        """Find contour file with fallback search."""
        contour_file = self.contour_path / f"frame_{frame_index:06d}.npy"
        if contour_file.exists():
            return contour_file
            
        # Fallback search
        for i in range(-60, 61):
            fallback_file = self.contour_path / f"frame_{frame_index + i:06d}.npy"
            if fallback_file.exists():
                return fallback_file
        return None




class VideoStabilizer:
    """Apply stabilization transformations to video."""
    
    def __init__(self, handler: VideoHandler, num_workers: int = 8):
        self.handler = handler
        self.num_workers = num_workers
        self.transform_path = None
        self.smoothed_transforms = None
        
    def stabilize_video(self, video_path: str, first_frame: int = 0, total_frames: int = 60) -> bool:
        """Stabilize video using pre-calculated transformation matrices."""
        
        self.video_path = video_path
        self.transform_path = self.handler.path_to_save / "Transformation_Matrices"
        
        if not self.transform_path.exists():
            logger.error(f"Transformation matrices not found at: {self.transform_path}")
            print_error("Please run transformation calculation first!")
            return False
        
        print_info("Starting video stabilization...")
        
        # Load and smooth transformations
        if not self._load_and_smooth_transforms():
            return False
        
        # Setup output directory
        output_path = self.handler.path_to_save / "Stabilized"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get frame indices
        frame_indices = list(range(first_frame, first_frame + total_frames))
        frame_indices = frame_indices[:len(self.smoothed_transforms)]  # Limit to available transforms
        
        if not frame_indices:
            logger.warning("No frames to stabilize")
            return False
        
        print_info(f"Stabilizing {len(frame_indices)} frames...")
        
        # Setup progress tracking
        progress = ProgressTracker({"Stabilization": len(frame_indices)})
        progress.start()
        
        try:
            # Process frames in parallel
            success = self._process_stabilization_parallel(frame_indices, output_path, progress.counters["Stabilization"])
            
            if success:
                # Create stabilized video
                self._create_stabilized_video(output_path)
                print_success(f"Video stabilization completed! Saved to: {self.handler.path_to_save}")
            
            return success
            
        finally:
            progress.stop()
    
    def _load_and_smooth_transforms(self) -> bool:
        """Load transformation matrices and apply smoothing."""
        try:
            transform_files = sorted(self.transform_path.glob("frame_*.npy"))
            if not transform_files:
                logger.error("No transformation files found")
                return False
            
            print_info(f"Loading {len(transform_files)} transformation matrices...")
            all_transforms = np.array([np.load(path) for path in transform_files])
            
            # Apply smoothing
            frame_to_frame_transforms = get_frame_to_frame_transforms(all_transforms)
            smoothed_f2f = smooth_trajectory_parameters(frame_to_frame_transforms, radius=15)
            self.smoothed_transforms = np.array(reconstruct_reference_transforms(smoothed_f2f))
            
            print_info(f"Smoothed {len(self.smoothed_transforms)} transformations")
            return True
            
        except Exception as e:
            logger.error(f"Error loading transformations: {e}")
            return False
    
    def _process_stabilization_parallel(self, frame_indices: List[int], output_path: Path, counter) -> bool:
        """Process stabilization in parallel."""
        chunk_size = max(1, len(frame_indices) // self.num_workers)
        workers = []
        
        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_workers - 1 else len(frame_indices)
            chunk = frame_indices[start_idx:end_idx]
            
            if chunk:
                worker = mp.Process(
                    target=self._process_stabilization_chunk,
                    args=(chunk, output_path, counter)
                )
                workers.append(worker)
                worker.start()
        
        # Wait for all workers
        try:
            for worker in workers:
                worker.join(timeout=300)
                if worker.is_alive():
                    worker.terminate()
            
            return all(worker.exitcode == 0 for worker in workers)
            
        except KeyboardInterrupt:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            return False
    
    def _process_stabilization_chunk(self, frame_indices: List[int], output_path: Path, counter):
        """Process a chunk of frames for stabilization."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {self.video_path}")
            return
        
        try:
            for i, frame_index in enumerate(frame_indices):
                if i >= len(self.smoothed_transforms):
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                h, w = frame.shape[:2]
                M = self.smoothed_transforms[i]
                
                # Calculate inverse transform
                A = M[:, :2]
                t = M[:, 2]
                A_inv = np.linalg.inv(A)
                t_inv = -A_inv @ t
                M_inv = np.hstack([A_inv, t_inv.reshape(2, 1)])
                
                # Apply transformation
                corrected_frame = cv2.warpAffine(frame, M_inv, (w, h))
                
                output_file = output_path / f"frame_{frame_index:06d}.png"
                cv2.imwrite(str(output_file), corrected_frame)
                
                with counter.get_lock():
                    counter.value += 1
        
        except Exception as e:
            logger.error(f"Error in stabilization worker: {e}")
        finally:
            cap.release()
    
    def _create_stabilized_video(self, output_path: Path):
        """Create video from stabilized frames."""
        try:
            output_video_path = self.handler.path_to_save / "Stabilized.MOV"
            images_to_video(
                output_path,
                str(output_video_path),
                fps=60,
                lossless=False
            )
            
            # Clean up frame directory
            shutil.rmtree(output_path)
            print_success(f"Stabilized video saved: {output_video_path}")
            
        except Exception as e:
            logger.error(f"Error creating stabilized video: {e}")
            print_error(f"Error creating stabilized video: {e}")

