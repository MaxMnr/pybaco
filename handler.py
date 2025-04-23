from pathlib import Path
from typing import Union, Dict, Any
import cv2

class VideoHandler:
    """Handles video paths (video IO and metadata) for drone recordings."""

    def __init__(self, year: Union[int, float], month: str, day: Union[int, float], number: Union[int, float], version: Union[int, float], N: int, config: bool = True):
        # Validate and convert inputs
        self._validate_inputs(year, month, day, number, version)

        # Convert values to the correct type
        self.year = str(int(year))
        self.month = month
        self.day = f"Day{int(day)}"
        self.number = str(int(number)).zfill(4)
        self.version = str(int(version)) if version >= 0 else None 
        self.N = N

        self.event_name = Path(f"Abaco_{self.month}_{self.year}")
        if config:
            self._configure_paths()

    def _validate_inputs(self, year, month, day, number, version):
        """Validate and convert parameters to correct types."""
        def convert_to_int(value):
            try:
                return int(value)
            except:
                raise TypeError(f"Expected int or float, got {type(value).__name__}")

        # Convert inputs to integers
        year = convert_to_int(year)
        day = convert_to_int(day)
        number = convert_to_int(number)
        version = convert_to_int(version)

        # Validate month is a string and within the expected values
        if not isinstance(month, str):
            raise TypeError(f"'month' must be a string, got {type(month).__name__}")
        if month not in ["February", "March", "November", "December"]:
            raise ValueError(f"Invalid month '{month}'. Expected one among [February, March, November, December]")

    def _configure_paths(self):
        """Configure file paths based on parameters."""
        base_data_path = Path("/partages/Bartololab3/Shared/Fish/Data")
        corrected_path = Path("/partages/Bartololab3/Shared/Maxime/Corrected")
        contours_root = Path("/partages/Bartololab3/Shared/Maxime/Polygons_Contours")

        # Configure video path and output directory
        self.video_path = base_data_path / self.event_name / self.day / f"DJI_{self.number}.MOV"
        version_str = self.version if self.version else "None"
        self.output_dir = corrected_path / self.event_name / self.day / f"DJI_{self.number}_{version_str}"

        # Configure contour directory
        contour_dir_name = f"{self.year[2:]}_{self.month}_{self.day}_{self.number}"
        if self.version:
            contour_dir_name += f"_{self.version}"
        self.contour_dir = contours_root / contour_dir_name / "Cartesian"

        # Check if the contour directory exists
        if not self.contour_dir.exists():
            raise FileNotFoundError(f"Contour directory not found: {self.contour_dir}")

    def get_video_info(self) -> Dict[str, Any]:
        """Get video information like frame count, fps, width, and height."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        info = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        cap.release()
        return info


