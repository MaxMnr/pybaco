from pathlib import Path
from typing import Union, Dict, Any
import cv2
import numpy as np
from .printing import *
import os
import pandas as pd
import pysrt
import re
import datetime
from typing import Union

class VideoHandler:
    """Handles video paths (video IO and metadata) for drone recordings."""

    def __init__(self, year: Union[int, float], month: str, day: Union[int, float], number: Union[int, float], version: Union[int, float], N: int, config: bool = True, main_path: str = "/Maxime/Corrected"):
        """
        Carefull params must be strictly in the format:
        year: int - 20XX
        month: str - ["February", "March", "November", "December"]
        day: int - [1, 2, 3, ...]
        number: int - [from 0 to 9999]
        version: int - [0, 1, 2, ...]
        N: int - Number of frames used to average while removing the ripples
        """
        self.year = year
        self.month = month
        self.day = f"Day{int(float(day))}"
        self.number = str(int(float(number))).zfill(4)
        self.version = str(int(float(version)))
        self.N = int(float(N))



        self.event_name = Path(f"Abaco_{self.month}_{self.year}")
        self.path_to_raw_videos = Path("Fish/Data")
        self.path_to_corrected_videos = Path(main_path)

        if config:
            self._configure_paths()

    def _configure_paths(self):
        import sys
        # Detect if the user is on a Mac or Linux system to adapt the path
        if sys.platform == "darwin":
            # MacOS
            main_path = Path("/Volumes/Shared Bartololab3/")
        elif sys.platform.startswith("linux"):
            # Linux
            main_path = Path("/partages/Bartololab3/Shared/")
        else:
            raise EnvironmentError("Unsupported operating system. This code is designed for MacOS or Linux. If you are using Windows it's still time to upgrade ;)")
        
        """Configure file paths based on parameters."""
        path_to_raw_videos = main_path / self.path_to_raw_videos
        path_to_corrected_videos = main_path / self.path_to_corrected_videos
        path_to_bad_contours = main_path / "Maxime/Polygons_Contours"
    
        # Configure video path and output directory
        self.path_to_video = path_to_raw_videos / self.event_name / self.day / f"DJI_{self.number}.MOV"
        self.path_to_save = path_to_corrected_videos / self.event_name / self.day / f"DJI_{self.number}_{self.version}"

        # Configure contour directory for stabilization (Gaspard's Contours)
        contour_dir_name = f"{str(self.year)[2:]}_{self.month}_{self.day}_{self.number}"
        if self.version:
            contour_dir_name += f"_{self.version}"
        self.path_to_contour = path_to_bad_contours / contour_dir_name / "Cartesian"

        # Check if the contour directory exists
        if not self.path_to_contour.exists():
            print_error(f"Contour directory does not exist: {self.path_to_contour}")
            # Create the output directory if it doesn't exist
            self.path_to_contour.mkdir(parents=True, exist_ok=True)







    def get_video_info(self) -> Dict[str, Any]:
        """Get video information like frame count, fps, width, and height."""
        if not self.path_to_video.exists():
            raise FileNotFoundError(f"Video file not found: {self.path_to_video}")
        cap = cv2.VideoCapture(str(self.path_to_video))
        info = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        cap.release()
        return info
    
    def save_data(self, filename):
        data = {"year": self.year,
                "month": self.month,
                "day": self.day,
                "number": self.number,
                "version": self.version,
                "N": self.N}
        np.save(filename, data)
        print_success(f"Data saved to {filename}")

    def get_srt_info(self):
        """Get subtitle information from the SRT file."""
        parser = ParserSRT(self.path_to_video.with_suffix('.SRT'))
        return parser.data
        

class ParserSRT:
    """
    Just a class to parse the srt files to get a pandas dataframe with every info for each frame
    """
    def __init__(self, path_to_srt):
        """
        Given the path to a srt file get the data
        """
        self.subs = pysrt.open(path_to_srt)
        self.data = self.parse()

    def parse(self):
        columns = ["frame_count", "diff_time", "timestamp", "iso", "shutter", 
                   "fnum", "ev", "ct", "color_md", "focal_len", 
                   "latitude", "longitude", "rel_alt", "abs_alt"]
        data = []

        for sub in self.subs:
            text = sub.text.replace('<font size="28">', '').replace('</font>', '').strip()
            lines = text.split('\n')
            if len(lines) < 3:
                continue
            frame_info = lines[0]
            timestamp = lines[1].strip()
            metadata = lines[2:]

            try:
                frame_count = int(re.search(r'FrameCnt:\s*(\d+)', frame_info).group(1))
                diff_time = re.search(r'DiffTime:\s*([\d.]+ms)', frame_info).group(1)

                timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                
                values = {
                    "frame_count": frame_count,
                    "diff_time": diff_time,
                    "timestamp": timestamp
                }

                matches = re.findall(r'\[([^\]]+)\]', " ".join(metadata))
                for match in matches:
                    if "rel_alt" in match and "abs_alt" in match:
                        alt_match = re.match(r'rel_alt:\s*([-.\d]+)\s+abs_alt:\s*([-.\d]+)', match)
                        if alt_match:
                            values["rel_alt"] = float(alt_match.group(1))
                            values["abs_alt"] = float(alt_match.group(2))
                    else:
                        key_val = match.split(":")
                        if len(key_val) == 2:
                            key = key_val[0].strip()
                            val = key_val[1].strip()
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                            values[key] = val

                data.append(values)
            except Exception as e:
                print(f"Error parsing subtitle: {e}")

        df = pd.DataFrame(data, columns=columns)
        return df

    def __item__(self, idx):
        return self.data.iloc[idx]