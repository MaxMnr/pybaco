import numpy as np
import cv2
from shapely.geometry import Polygon, Point
from .printing import *

import numpy as np
import cv2

def get_transformation(ref_img: np.ndarray, img_gray: np.ndarray, polygon: Polygon, grid_size: int = 100) -> np.ndarray:
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(ref_img, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Define a grid of points for tracking
    h, w = ref_img.shape
    grid_y, grid_x = np.mgrid[0:h:grid_size, 0:w:grid_size]
    grid_pts = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1).astype(np.float16)

    # Filter out points inside the subject mask (only use background)
    filtered_pts = []
    for pt in grid_pts:
        if not polygon.contains(Point(pt[0], pt[1])):
            filtered_pts.append(pt)

    filtered_pts = np.array(filtered_pts, dtype=np.float32)
    
    # Skip stabilization if not enough points
    if len(filtered_pts) < 4:
        return np.array([[1, 0, 0], [0, 1, 0]]) 

    # Compute flow for background points
    dx = flow[filtered_pts[:, 1].astype(int), filtered_pts[:, 0].astype(int), 0]
    dy = flow[filtered_pts[:, 1].astype(int), filtered_pts[:, 0].astype(int), 1]
    displaced_pts = filtered_pts + np.stack((dx, dy), axis=-1)

    # Estimate transformation using background points only
    M, _ = cv2.estimateAffinePartial2D(filtered_pts, displaced_pts)
    
    if M is None:
        print_error("Singular matrix")
        return np.array([[1, 0, 0], [0, 1, 0]])

    return M

def stabilize_frame(ref_img: np.ndarray, img_gray: np.ndarray, img: np.ndarray, polygon: Polygon, grid_size: int = 20) -> np.ndarray:
    """Stabilizes the current frame based on optical flow with the reference frame.
    
    This function computes optical flow between two frames and applies an affine
    transformation to stabilize the background while preserving subject motion.
    
    Args:
        ref_img: Reference frame in grayscale
        img_gray: Current frame in grayscale
        img: Current frame in color (BGR)
        polygon: Shapely Polygon defining the subject area to exclude
        grid_size: Grid size for sampling points (smaller = more precise but slower)
        
    Returns:
        Stabilized color frame
    """
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(ref_img, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Define a grid of points for tracking
    h, w = ref_img.shape
    grid_y, grid_x = np.mgrid[0:h:grid_size, 0:w:grid_size]
    grid_pts = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1).astype(np.float16)

    # Filter out points inside the subject mask (only use background)
    filtered_pts = []
    for pt in grid_pts:
        if not polygon.contains(Point(pt[0], pt[1])):
            filtered_pts.append(pt)

    filtered_pts = np.array(filtered_pts, dtype=np.float32)
    
    # Skip stabilization if not enough points
    if len(filtered_pts) < 4:
        return img.copy()

    # Compute flow for background points
    dx = flow[filtered_pts[:, 1].astype(int), filtered_pts[:, 0].astype(int), 0]
    dy = flow[filtered_pts[:, 1].astype(int), filtered_pts[:, 0].astype(int), 1]
    displaced_pts = filtered_pts + np.stack((dx, dy), axis=-1)

    # Estimate transformation using background points only
    M, _ = cv2.estimateAffinePartial2D(filtered_pts, displaced_pts)
    return M
    
def get_frame_to_frame_transforms(ref_to_frame_transforms):
    """
    Convert reference-to-frame transforms to frame-to-frame transforms
    """
    frame_to_frame = [None]  # First frame has no previous frame
    
    for i in range(1, len(ref_to_frame_transforms)):
        if ref_to_frame_transforms[i-1] is None or ref_to_frame_transforms[i] is None:
            frame_to_frame.append(None)
            continue
            
        # Get previous and current transforms
        prev_M = ref_to_frame_transforms[i-1]
        curr_M = ref_to_frame_transforms[i]
        
        # Calculate previous frame to current frame transform
        # M_{i-1\to i} = M_{ref \to i} * M_{ref \to i-1}^(-1)
        prev_A = prev_M[:, :2]
        prev_t = prev_M[:, 2]
        
        try:
            prev_A_inv = np.linalg.inv(prev_A)
            prev_M_inv = np.hstack([prev_A_inv, (-prev_A_inv @ prev_t).reshape(2, 1)])
            
            # Combine transformations (prev_frame -> reference -> current_frame)
            frame_to_frame_M = np.vstack([curr_M, [0, 0, 1]]) @ np.vstack([prev_M_inv, [0, 0, 1]])
            frame_to_frame.append(frame_to_frame_M[:2])
        except np.linalg.LinAlgError:
            frame_to_frame.append(None)
            
    return frame_to_frame

def smooth_trajectory_parameters(frame_to_frame_transforms, radius=15):
    """
    Extract and smooth the trajectory parameters from frame-to-frame transforms
    """
    # Extract trajectory parameters
    dx = []
    dy = []
    da = []  # angles
    ds = []  # scales
    
    for M in frame_to_frame_transforms:
        if M is None:
            # Handle None transforms
            dx.append(0)
            dy.append(0)
            da.append(0)
            ds.append(1)
            continue
        
        # Extract translation
        dx.append(M[0, 2])
        dy.append(M[1, 2])
        
        # Extract rotation and scale
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        
        scale = np.sqrt(a*a + b*b)
        angle = np.arctan2(b, a)
        
        da.append(angle)
        ds.append(scale)
    
    # Apply smoothing
    smooth_dx = moving_average(dx, radius)
    smooth_dy = moving_average(dy, radius)
    smooth_da = moving_average(da, radius)
    smooth_ds = moving_average(ds, radius)
    
    # Create smoothed frame-to-frame transforms
    smoothed_transforms = []
    
    for i in range(len(frame_to_frame_transforms)):
        if frame_to_frame_transforms[i] is None:
            smoothed_transforms.append(None)
            continue
            
        s = smooth_ds[i]
        a = smooth_da[i]
        tx = smooth_dx[i]
        ty = smooth_dy[i]
        
        # Create the smoothed matrix
        smooth_M = np.zeros((2, 3), dtype=np.float32)
        smooth_M[0, 0] = s * np.cos(a)
        smooth_M[0, 1] = s * np.sin(a)
        smooth_M[1, 0] = -s * np.sin(a)
        smooth_M[1, 1] = s * np.cos(a)
        smooth_M[0, 2] = tx
        smooth_M[1, 2] = ty
        
        smoothed_transforms.append(smooth_M)
    
    return smoothed_transforms

def reconstruct_reference_transforms(smoothed_frame_to_frame):
    """
    Reconstruct reference-to-frame transforms from smoothed frame-to-frame transforms
    """
    # First transform is identity (reference frame to itself)
    reference_transforms = [np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)]
    
    # Accumulate transformations
    for i in range(1, len(smoothed_frame_to_frame)):
        if smoothed_frame_to_frame[i] is None:
            if len(reference_transforms) > 0:
                reference_transforms.append(reference_transforms[-1].copy())
            else:
                reference_transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
            continue
            
        # Get previous accumulated transform
        prev_ref_M = reference_transforms[-1]
        
        # Get current frame-to-frame transform
        f2f_M = smoothed_frame_to_frame[i]
        
        # Combine them (reference -> prev_frame -> current_frame)
        combined_M = np.vstack([f2f_M, [0, 0, 1]]) @ np.vstack([prev_ref_M, [0, 0, 1]])
        reference_transforms.append(combined_M[:2])
    
    return reference_transforms

def moving_average(curve, radius):
    """
    Apply moving average filter to a curve
    """
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed