import numpy as np


class Analyser:
    def __init__(self, handler):
        self.handler = handler
    
    def get_cartesian_contours(self):
        contours = self.handler.get_contours()
        if contours is None:
            raise ValueError("Contours not found. Please compute contours first.")
        # Check if the contours are in the right format
        if contours.ndim != 3 or contours.shape[2] != 2:
            raise ValueError("Contours should be a 3D array with shape (N, M, 2).")
        return contours

    def get_polar_contours(self):
        # Check wether the contours (in cartesian) are already computed
        contours = self.handler.get_contours()
        if contours is None:
            raise ValueError("Contours not found. Please compute contours first.")
        # Check if the contours are in the right format
        if contours.ndim != 3 or contours.shape[2] != 2:
            raise ValueError("Contours should be a 3D array with shape (N, M, 2).")
    
        N, M, _ = contours.shape
        polar_contours = np.empty_like(contours)

        for i in range(N):
            contour = contours[i]

            # Center the contour
            centroid = np.mean(contour, axis=0)
            centered = contour - centroid

            # Convert to polar
            x, y = centered[:, 0], centered[:, 1]
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)

            # Normalize theta to [0, 2π) (don't know if it's needed but arctan2 is weird)
            theta = np.mod(theta, 2 * np.pi)

            # Sort by theta
            sort_indices = np.argsort(theta)
            r_sorted = r[sort_indices]
            theta_sorted = theta[sort_indices]

            # Store sorted polar coordinates
            polar_contours[i] = np.stack((r_sorted, theta_sorted), axis=-1)

        return polar_contours

    def polar_to_cartesian(polar_contours):
        N, M, _ = polar_contours.shape
        cartesian_contours = np.empty_like(polar_contours)

        for i in range(N):
            polar_contour = polar_contours[i]

            # Convert to Cartesian
            r, theta = polar_contour[:, 0], polar_contour[:, 1]
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Center the contour
            centroid = np.mean(np.stack((x, y), axis=-1), axis=0)
            centered = np.stack((x, y), axis=-1) - centroid

            # Store centered Cartesian coordinates
            cartesian_contours[i] = centered

        return cartesian_contours
    
    def fft_polar(self, polar_contours):
        N, M, _ = polar_contours.shape
        fft_results = np.empty((N, M), dtype=complex)

        for i in range(N):
            polar_contour = polar_contours[i]
            r = polar_contour[:, 0]
            theta = polar_contour[:, 1]

            # Step 1: Compute FFT
            fft_result = np.fft.fft(r)
            fft_results[i] = fft_result

        return fft_results
    
    def get_fourier_coefficients(polar_contours):
        fft_results = self.fft_polar(polar_contours)
        N, M = fft_results.shape
        fourier_coefficients = np.empty((N, M), dtype=complex)
        for i in range(N):
            fourier_coefficients[i] = np.fft.fftshift(fft_results[i])
