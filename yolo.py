from ultralytics import YOLO
import torch
import os
import cv2
import numpy as np
from pathlib import Path
from rich.progress import track
from pybaco.printing import *
from shapely.geometry import Polygon

class ContoursPipeline:
    def __init__(self, handler, model_path, video_path):
        self.handler = handler
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.main_dir = Path("processed") / self.handler.event_name / self.handler.day / f"DJI_{self.handler.number}_{self.handler.version}"
        self.contours_dir = self.main_dir / "contours"
        self.contours_smooth_dir = self.main_dir / "contours_smooth"
        self.contours_smooth_polar_dir = self.main_dir / "contours_smooth_polar"
        self.videos_dir = self.main_dir / "videos"

        os.makedirs(self.main_dir, exist_ok=True)
        os.makedirs(self.contours_dir, exist_ok=True)
        os.makedirs(self.contours_smooth_dir, exist_ok=True)
        os.makedirs(self.contours_smooth_polar_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

    def process_video(self, num_frames=None):
            
            print_title("Processing video with YOLOv?", title="")

            # TEMPORARY: SAVE THE AREA OF THE FLOCK FOR DEVELOPMENT PURPOSES
            areas = []

            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
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
                    polygon = Polygon(mask)

                    # Calculate the area of the mask
                    area = polygon.area
                    areas.append(area)

            cap.release()


            plot_area(areas, save_path=self.main_dir / "areas_raw.png")
            np.save(self.main_dir / "areas.npy", areas)
            
            print_success("Done processing video.")
            print_success("Contours saved to:" + str(self.contours_dir))


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
        
        # TEMPORARY: SAVE THE AREA OF THE FLOCK FOR DEVELOPMENT PURPOSES
        areas = []
        
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
            
            # Calculate the centroid of the polygon
            polygon = Polygon(averaged_contour)
            centroid = np.array(polygon.centroid.coords[0])
            
            # Calculate polar coordinates relative to the centroid
            translated_contour = averaged_contour - centroid
            complex_contour = np.array([complex(x, y) for x, y in translated_contour])
            polar_contour = np.array([(np.abs(c), np.angle(c)) for c in complex_contour])
            
            # Save both polar coordinates and centroid
            polar_data = {'polar': polar_contour, 'centroid': centroid}
            np.save(self.contours_smooth_polar_dir / f"frame_{mid_index:06d}.npy", polar_data)

            # Calculate the area of the mask
            area = polygon.area
            areas.append(area)

        self.smoothed_contours = np.stack(smoothed_contours)

        plot_area(areas, save_path=self.main_dir / "areas_smooth.png")
        np.save(self.main_dir / "areas_smooth.npy", areas)

    def create_video_from_contours(self, output_path):
        cap = cv2.VideoCapture(self.video_path)
        width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get the cartesian contour files
        raw_files = sorted([f for f in os.listdir(self.contours_dir) if not f.startswith('.') and f.endswith('.npy')])
        cartesian_files = sorted([f for f in os.listdir(self.contours_smooth_dir) if not f.startswith('.') and f.endswith('.npy')])
        polar_files = sorted([f for f in os.listdir(self.contours_smooth_polar_dir) if not f.startswith('.') and f.endswith('.npy')])
        first_index = cartesian_files[0].split("_")[1].split(".")[0]
        
        # Set the camera to the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_index))
        
        for index, (raw_file, contour_file, polar_file) in track(enumerate(zip(raw_files, cartesian_files, polar_files)), total=len(cartesian_files), description="Creating video from contours"):
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
            
            # Load corresponding polar contour and convert back to cartesian for display
            polar_data = np.load(os.path.join(self.contours_smooth_polar_dir, polar_file), allow_pickle=True).item()
        
            polar_contour = polar_data['polar']
            centroid = polar_data['centroid']
            
            # Convert polar back to cartesian for visualization
            cartesian_from_polar = np.zeros((len(polar_contour), 2))
            for i, (r, theta) in enumerate(polar_contour):
                cartesian_from_polar[i, 0] = r * np.cos(theta) + centroid[0]
                cartesian_from_polar[i, 1] = r * np.sin(theta) + centroid[1]
            
            polar_display = np.array(cartesian_from_polar, dtype=np.int32)
            polar_display = polar_display.reshape((-1, 1, 2))
            
            # Draw cartesian contour
            cv2.polylines(frame, [contour], isClosed=True, color=(28, 29, 23), thickness=12)
            
            # Draw polar contour
            for i in range(0, len(polar_display), 10):
                center = tuple(polar_display[i][0])
                rad = 6
                cv2.circle(frame, center, rad, (197, 186, 22), -1)  # Orange color

            # Draw raw contour
            cv2.polylines(frame, [raw_contour], isClosed=True, color=(244, 233, 239), thickness=6)


            # Add frame number
            cv2.putText(frame, f"Frame: {index}", (200, 250), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 3)
            
            # Add legend
            cv2.putText(frame, "Raw", (width-700, height-150), cv2.FONT_HERSHEY_COMPLEX, 1.5, (244, 233, 239), 3)
            cv2.line(frame, (width-400, height-150), (width-200, height-150), (244, 233, 239), 5)

            cv2.putText(frame, "Cartesian", (width-700, height-100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (28, 29, 23), 3)
            cv2.line(frame, (width-400, height-100), (width-200, height-100), (28, 29, 23), 5)
            
            cv2.putText(frame, "Polar", (width-700, height-50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (197, 186, 22), 3)
            # Draw dashed line for legend
            for i in range(0, 200, 20):
                cv2.circle(frame, (width-400+i, height-50), 6, (197, 186, 22), -1)
            
            out.write(frame)
            
        out.release()
        cap.release()
        print_success("Video created from contours.")
        print_success("Video saved to:" + str(output_path))



def plot_area(areas, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = sns.color_palette(palette='RdBu', n_colors=10)

    # Plot the areas
    plt.figure(figsize=(5.5, 5))
    plt.plot(areas/np.max(areas), label='Area of the flock', c=cm[0], lw=2)
    plt.xlabel('Frame number')
    plt.ylabel('Normalized Area')
    plt.savefig(save_path, dpi=400)







    # def process_video(self, num_frames=None):
        
    #     print_title("Processing video with YOLOv?", title="")

    #     # TEMPORARY: SAVE THE AREA OF THE FLOCK FOR DEVELOPMENT PURPOSES
    #     areas = []

    #     device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    #     results = self.model(self.video_path, stream=True, verbose=False, save=False, device=device, imgsz=1024, retina_masks=True)

    #     cap = cv2.VideoCapture(self.video_path)
    #     width, height, fps, total_frames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(self.videos_dir / "original_yolo.MOV", fourcc, fps, (width, height))

    #     if num_frames is None:
    #         num_frames = total_frames
    
    #     for index, result in track(enumerate(results), total=num_frames, description="Processing frames"):
    #         if index >= num_frames:
    #             break
    #         if result.masks is not None and len(result.masks.xy) > 0:
    #             mask = result.masks.xy[0]
    #             np.save(self.contours_dir / f"frame_{index:06d}.npy", mask)

    #             # Calculate the area of the mask
    #             polygon = Polygon(mask)
    #             area = polygon.area
    #             areas.append(area)

    #         ret, frame = cap.read()
    #         if not ret:
    #             print_info("No more frames to read.")
    #             break
    #         contour = np.array(mask, dtype=np.int32)
    #         contour = contour.reshape((-1, 1, 2))  # Format for cv2.polylines
    #         # add a text to indicate the frame number
    #         cv2.putText(frame, f"Frame: {index}", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
    #         cv2.polylines(frame, [contour], isClosed=True, color=(0, 0, 0), thickness=10)
    #         out.write(frame)

    #     out.release()
    #     cap.release()


    #     plot_area(areas, save_path=self.main_dir / "areas_raw.png")
    #     np.save(self.main_dir / "areas.npy", areas)
        
    #     print_success("Done processing video.")
    #     print_success("Video saved to:" + str(self.videos_dir) + "/main_video.MOV")
    #     print_success("Contours saved to:" + str(self.contours_dir))