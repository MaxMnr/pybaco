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

    def __init__(self, year: Union[int, float], month: str, day: Union[int, float], number: Union[int, float], version: Union[int, float], N: int, config: bool = True):
        """
        year: int or float - 20XX
        month: str - ["February", "March", "November", "December"]
        day: int or float - Day number of a given campaign
        number: int or float - Number of the video (DJI_XXXX)
        version: int or float - Sub Index for a video sequence
        N: int or float - Number of frames used to average while removing the ripples
        """
        # Convert all parameters to expected format and to strings 
        verified_params = check_params_format(year, month, day, number, version, N)
        self.year = verified_params["year"]
        self.month = verified_params["month"]
        self.day = verified_params["day"]
        self.number = verified_params["number"]
        self.version = verified_params["version"]
        self.N = verified_params["N"]

        self.event_name = Path(f"Abaco_{self.month}_{self.year}")
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
        path_to_raw_videos = main_path / "Fish/Data"
        path_to_corrected_videos = main_path / "Maxime/Corrected"
        path_to_bad_contours = main_path / "Maxime/Polygons_Contours"
    
        # Configure video path and output directory
        self.path_to_video = path_to_raw_videos / self.event_name / self.day / f"DJI_{self.number}.MOV"
        self.path_to_save = path_to_corrected_videos / self.event_name / self.day / f"DJI_{self.number}_{self.version}"

        # Configure contour directory for stabilization (Gaspard's Contours)
        contour_dir_name = f"{self.year[2:]}_{self.month}_{self.day}_{self.number}"
        if self.version:
            contour_dir_name += f"_{self.version}"
        self.path_to_contour = path_to_bad_contours / contour_dir_name / "Cartesian"

        # Check if the contour directory exists
        if not self.path_to_contour.exists():
            print_error(f"Contour directory does not exist: {self.path_to_contour}")

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
        

def check_params_format(year: Union[int, float], month: str, day: Union[int, float, str], number: Union[int, float, str], version: Union[int, float, str], N: int):
    try:
        year = int(float(year))
        if not 2000 <= year <= 2100:
            raise ValueError
    except:
        raise ValueError(f"Year must be convertible to int between 2000 and 2100, got {year}")

    if not isinstance(month, str) or month not in {"February", "March", "November", "December"}:
        raise ValueError(f"Month must be one of ['February', 'March', 'November', 'December'], got {month}")

    try:
        d = str(day).strip()
        if d.lower().startswith("day"):
            d = d[3:]
        day = f"Day{int(float(d))}"
    except:
        raise ValueError(f"Day must be convertible to format 'DayX', got {day}")

    try:
        number = str(int(float(number))).zfill(4)
        if len(number) != 4:
            raise ValueError
    except:
        raise ValueError(f"Number must be convertible to 4-digit string, got {number}")

    try:
        version = str(int(float(version)))
        if len(version) != 1:
            raise ValueError
    except:
        raise ValueError(f"Version must be convertible to 1-digit string, got {version}")

    try: 
        N = int(float(N))
        if N < 1:
            raise ValueError
    except:
        raise ValueError(f"N must be convertible to int greater than 0, got {N}")

    # Returns a dictionary with the verified parameters
    return {
        "year": str(year),
        "month": str(month),
        "day": str(day),
        "number": str(number),
        "version": str(version),
        "N": str(N)
    }


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