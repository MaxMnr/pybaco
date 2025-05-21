import cv2
import numpy as np
from pathlib import Path
from typing import Union
from rich.progress import track
from .printing import print_error, print_success, print_info


class BackgroundPipeline:
    """
    Class to handle background computation and removal for drone videos.
    Works with the VideoHandler class to process stabilized videos.
    """
    def __init__(self, handler):
        """
        Initialize the BackgroundProcessor with a VideoHandler instance.
        --> handler: A VideoHandler instance with properly configured paths
        """
        self.handler = handler
        self.background_path = self.handler.path_to_save / "background.png"
    
    def compute_background(self, num_images: int = 30) -> Union[np.ndarray, bool]:
        """
        Compute a background by taking the maximum value of pixels across frames.
        --> num_images: Number of frames to sample for background computation
        """
        cap = cv2.VideoCapture(str(self.handler.path_to_save / f"Stabilized_{self.handler.N}.MOV"))
        if not cap.isOpened():
            print_error("Error: Could not open video.")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        index_list = np.linspace(total_frames * 0.1, total_frames * 0.9, num_images, dtype=int)
        background_max = None

        for i in track(index_list, description="Computing Background..."):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if background_max is None:
                background_max = frame_rgb.astype(np.uint8)
            else:
                np.maximum(background_max, frame_rgb, out=background_max)

        cap.release()

        if background_max is None:
            print_error("No valid frames were read to compute background.")
            return False
            
        # Save the background image
        cv2.imwrite(str(self.background_path), cv2.cvtColor(background_max, cv2.COLOR_RGB2BGR))
        print_success(f"Background computed and saved to {self.background_path}")
        return True
    
    def remove_background(self) -> bool:
        """
        Remove background from each frame of the stabilized video.
        """
        if self.background_path.exists():
            self.background = cv2.imread(str(self.background_path))
        else:
            print_error("Background image not found. Run compute_background first.")
            return False

        # Open the stabilized video
        video_path = self.handler.path_to_save / f"Stabilized_{self.handler.N}.MOV"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print_error(f"Could not open video at {video_path}")
            return False
        
        print_success("Video opened successfully")

        # Get video properties
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare the output video file
        output_path = self.handler.path_to_save / f"Backgrounded_{self.handler.N}.MOV"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'avc1'), 
            fps,
            (width, height)
        )
        
        for i in track(range(num_frames), description="Removing background..."):
            ret, frame = cap.read()
            if not ret:
                print_error(f"Error reading frame {i}")
                break
                
            # Convert the frame to RGB and calculate difference (assure no negative values or wrong wrapping)
            diff = np.abs(frame.astype(np.int16) - self.background.astype(np.int16)).astype(np.uint8)
            writer.write(diff)
        
        # Clean up
        cap.release()
        writer.release()
        print_success(f"Background removed and saved to {output_path}")
        return True
        
    def run(self, num_images: int = 30) -> bool:
        try:
            success = self.compute_background(num_images)
            if not success:
                return False
            success = self.remove_background()
            return success
             
        except Exception as e:
            print_error(f"Process failed with error: {str(e)}")
            return False