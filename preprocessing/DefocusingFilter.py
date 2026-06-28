import cv2
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit # NEW: For Gaussian fitting

# --- 1. Configuration Parameters ---
# Define your folder paths
IMAGE_DIR = r'F:\AI_Project\Process\02_Defocusing Filter\DefocusingTest\Img'
MASK_DIR = r'F:\AI_Project\Process\02_Defocusing Filter\DefocusingTest\UNETMask'
FILTERED_MASKS_DIR = r'F:\AI_Project\Process\02_Defocusing Filter\DefocusingTest\FilteredMask'
RESULTS_DIR = r'F:\AI_Project\Process\02_Defocusing Filter\DefocusingTest\FilteredMask'

# Filtering thresholds (ADJUST THESE BASED ON DEBUGGING OUTPUT)
# Area thresholds
MIN_PARTICLE_AREA = 10
MAX_PARTICLE_AREA = 1000

# Shape thresholds
DESIRED_ASPECT_RATIO_MIN = 0.5
DESIRED_ASPECT_RATIO_MAX = 2.5
DESIRED_EXTENT_MIN = 0.6
MAX_ECCENTRICITY_FOR_PARTICLE = 0.9

# Sharpness/Focus threshold
LAPLACIAN_VAR_THRESHOLD = 1

# --- Gaussian Fitting Thresholds (NEW!) ---
# A higher value means a poorer fit (more deviation from a perfect Gaussian)
MAX_GAUSSIAN_RESIDUAL_SUM = 100000.0 # Adjust: Max sum of squared residuals for a 'good' Gaussian fit
# The aspect ratio of the *fitted* Gaussian's sigmas. A value close to 1 is spherical.
MAX_GAUSSIAN_SIGMA_ASPECT_RATIO = 2.5 # Adjust: Filter out highly elongated fitted Gaussians

# --- 2. Helper Functions ---

def load_image_and_mask(image_path, mask_path):
    """
    Loads a grayscale image (preserving bit depth) and its corresponding binary mask.
    Handles 16-bit images correctly.
    """
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        raise FileNotFoundError(f"Original image not found at {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Mask not found at {mask_path}")
    
    mask = (mask > 0).astype(np.uint8) * 255
    return original_image, mask

def calculate_laplacian_variance(image_roi, mask_roi):
    """
    Calculates the Laplacian Variance for a masked region of interest.
    Normalizes 16-bit input to 0-255 range for consistent Laplacian calculation.
    """
    if image_roi.shape[0] == 0 or image_roi.shape[1] == 0:
        return 0.0

    masked_pixels = cv2.bitwise_and(image_roi, image_roi, mask=mask_roi)
    normalized_pixels = masked_pixels.astype(np.float32)
    cv2.normalize(normalized_pixels, normalized_pixels, 0, 255, cv2.NORM_MINMAX)
    
    laplacian = cv2.Laplacian(normalized_pixels, cv2.CV_32F)
    non_zero_laplacian = laplacian[mask_roi > 0]

    if non_zero_laplacian.size > 0:
        return np.var(non_zero_laplacian)
    else:
        return 0.0

# --- NEW: 2D Gaussian function for fitting ---
def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function for fitting.
    coords: tuple (x, y) coordinates
    amplitude: peak intensity
    xo, yo: center coordinates
    sigma_x, sigma_y: standard deviations in x and y (before rotation)
    theta: rotation angle of the ellipse (radians)
    offset: background offset
    """
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    
    # Pre-calculate terms for efficiency
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    # Calculate Gaussian value at each point
    g = offset + amplitude * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel() # Flatten for curve_fit

# --- 3. Core Processing Logic ---

def process_frame(original_image, unet_mask, frame_id):
    """
    Processes a single frame's image and U-Net mask to filter particles.
    """
    height, width = original_image.shape
    filtered_mask = np.zeros_like(unet_mask, dtype=np.uint8)
    kept_particles_data = []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(unet_mask, 8, cv2.CV_32S)

    print(f"\n--- Processing particles for Frame: {frame_id} ---")
    if num_labels == 1:
        print(f"  No particles detected by U-Net in this frame (mask is entirely background).")

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        centroid_x, centroid_y = centroids[i]

        current_particle_mask = (labels == i).astype(np.uint8) * 255

        if w == 0 or h == 0:
            print(f"  Particle {i}: Zero width or height bounding box. Discarded.")
            continue

        original_roi_patch = original_image[y:y+h, x:x+w]
        particle_mask_roi_patch = current_particle_mask[y:y+h, x:x+w]

        # --- Calculate Shape Metrics ---
        aspect_ratio = float(w) / h
        extent = float(area) / (w * h)

        eccentricity = 1.0
        contours, _ = cv2.findContours(particle_mask_roi_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            if len(cnt) >= 5: # Need at least 5 points to fit an ellipse
                try:
                    (x_el, y_el), (MA, ma), angle_el = cv2.fitEllipse(cnt)
                    major_axis = max(MA, ma)
                    minor_axis = min(MA, ma)
                    if major_axis != 0:
                        eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2)
                    else:
                        eccentricity = 0.0
                except cv2.error:
                    # fitEllipse can sometimes fail for weird contours, treat as high eccentricity
                    eccentricity = 1.0
            else:
                eccentricity = 1.0 # Very small or complex contours

        lap_var = calculate_laplacian_variance(original_roi_patch, particle_mask_roi_patch)

        # --- NEW: Gaussian Fitting and Metrics ---
        gaussian_fit_success = False
        fitted_sigma_aspect_ratio = 0.0
        gaussian_fit_residual_sum = float('inf') # Initialize with a high value

        # Extract coordinates and pixel values from the masked ROI for fitting
        # Normalize the pixel values (0-255 range) for stable fitting
        pixels_for_fit = original_roi_patch.astype(np.float32)
        cv2.normalize(pixels_for_fit, pixels_for_fit, 0, 255, cv2.NORM_MINMAX)
        pixels_for_fit = cv2.bitwise_and(pixels_for_fit.astype(np.uint8), pixels_for_fit.astype(np.uint8), mask=particle_mask_roi_patch) # Apply mask again

        # Get the actual coordinates and intensity values for fitting
        y_coords_in_patch, x_coords_in_patch = np.where(particle_mask_roi_patch > 0)
        intensities_in_patch = pixels_for_fit[y_coords_in_patch, x_coords_in_patch]

        if intensities_in_patch.size > 0:
            # Initial guesses for Gaussian parameters
            # amplitude: max intensity in the patch
            # xo, yo: center of the patch relative to its top-left corner
            # sigma_x, sigma_y: rough estimate based on 1/6th of width/height of patch (approx. 3 sigma covers most of distribution)
            # theta: 0 (no initial rotation guess)
            # offset: min intensity in the patch (background)
            initial_amplitude = np.max(intensities_in_patch) - np.min(intensities_in_patch)
            initial_offset = np.min(intensities_in_patch)
            initial_xo = w / 2.0
            initial_yo = h / 2.0
            initial_sigma_x = w / 6.0
            initial_sigma_y = h / 6.0

            # Ensure initial sigmas are not zero or too small
            if initial_sigma_x <= 0: initial_sigma_x = 1.0
            if initial_sigma_y <= 0: initial_sigma_y = 1.0

            p0 = [initial_amplitude, initial_xo, initial_yo, initial_sigma_x, initial_sigma_y, 0.0, initial_offset]

            # Bounds for parameters: (min, max)
            # amplitude: [0, 255] (max 8-bit value)
            # xo, yo: [0, w/h] (within patch bounds)
            # sigma_x, sigma_y: [0.5, max(w,h)*2] (must be positive, not excessively large)
            # theta: [-pi, pi]
            # offset: [0, 255]
            bounds = (
                [0, 0, 0, 0.5, 0.5, -np.pi, 0],
                [255, w, h, max(w,h)*2, max(w,h)*2, np.pi, 255]
            )

            try:
                # curve_fit expects x and y coordinates as separate arrays in the first argument
                popt, pcov = curve_fit(gaussian_2d, (x_coords_in_patch, y_coords_in_patch),
                                       intensities_in_patch, p0=p0, bounds=bounds,
                                       maxfev=5000) # Increased maxfev for more iterations if needed
                
                # Extract fitted parameters
                fitted_amplitude, fitted_xo, fitted_yo, fitted_sigma_x, fitted_sigma_y, fitted_theta, fitted_offset = popt

                # Calculate sum of squared residuals
                fitted_values = gaussian_2d((x_coords_in_patch, y_coords_in_patch), *popt)
                gaussian_fit_residual_sum = np.sum((intensities_in_patch - fitted_values)**2)

                # Calculate aspect ratio of fitted sigmas
                if fitted_sigma_x != 0 and fitted_sigma_y != 0:
                    fitted_sigma_aspect_ratio = max(fitted_sigma_x, fitted_sigma_y) / min(fitted_sigma_x, fitted_sigma_y)
                else:
                    fitted_sigma_aspect_ratio = float('inf') # Indicate a problem if sigma is zero

                gaussian_fit_success = True

            except RuntimeError:
                print(f"    Gaussian fit failed for Particle {i} (RuntimeError).")
                gaussian_fit_success = False
            except ValueError as ve:
                print(f"    Gaussian fit failed for Particle {i} (ValueError: {ve}).")
                gaussian_fit_success = False
            except Exception as e:
                print(f"    Gaussian fit failed for Particle {i} (Other Error: {e}).")
                gaussian_fit_success = False
        else:
            print(f"    Particle {i} mask has no pixels for Gaussian fit.")

        # --- DEBUGGING PRINTS AND FILTERING LOGIC ---
        print(f"  Particle {i}: Area={area}, LapVar={lap_var:.2f}, AR={aspect_ratio:.2f}, Extent={extent:.2f}, Eccentricity={eccentricity:.2f}")
        if gaussian_fit_success:
            print(f"    Gaussian Fit: Success, Residual Sum={gaussian_fit_residual_sum:.2f}, Sigma AR={fitted_sigma_aspect_ratio:.2f}")
        else:
            print(f"    Gaussian Fit: Failed.")


        # 1. Filter by Minimum Area
        if area < MIN_PARTICLE_AREA:
            print(f"    --> Discarded: Area ({area}) < MIN_PARTICLE_AREA ({MIN_PARTICLE_AREA})")
            continue

        # 2. Filter by Maximum Area
        if area > MAX_PARTICLE_AREA:
            print(f"    --> Discarded: Area ({area}) > MAX_PARTICLE_AREA ({MAX_PARTICLE_AREA})")
            continue

        # 3. Filter by Aspect Ratio (Bounding Box)
        if not (DESIRED_ASPECT_RATIO_MIN <= aspect_ratio <= DESIRED_ASPECT_RATIO_MAX):
            print(f"    --> Discarded: Aspect Ratio ({aspect_ratio:.2f}) outside [{DESIRED_ASPECT_RATIO_MIN:.2f}, {DESIRED_ASPECT_RATIO_MAX:.2f}]")
            continue

        # 4. Filter by Extent
        if extent < DESIRED_EXTENT_MIN:
            print(f"    --> Discarded: Extent ({extent:.2f}) < DESIRED_EXTENT_MIN ({DESIRED_EXTENT_MIN:.2f})")
            continue
            
        # 5. Filter by Eccentricity (Contour-based)
        if eccentricity > MAX_ECCENTRICITY_FOR_PARTICLE:
            print(f"    --> Discarded: Eccentricity ({eccentricity:.2f}) > MAX_ECCENTRICITY_FOR_PARTICLE ({MAX_ECCENTRICITY_FOR_PARTICLE:.2f})")
            continue

        # --- NEW: Gaussian Fit Filters ---
        if not gaussian_fit_success:
            print(f"    --> Discarded: Gaussian fit failed.")
            continue
            
        if gaussian_fit_residual_sum > MAX_GAUSSIAN_RESIDUAL_SUM:
            print(f"    --> Discarded: Gaussian Residual Sum ({gaussian_fit_residual_sum:.2f}) > MAX_GAUSSIAN_RESIDUAL_SUM ({MAX_GAUSSIAN_RESIDUAL_SUM:.2f})")
            continue

        if fitted_sigma_aspect_ratio > MAX_GAUSSIAN_SIGMA_ASPECT_RATIO:
            print(f"    --> Discarded: Fitted Sigma Aspect Ratio ({fitted_sigma_aspect_ratio:.2f}) > MAX_GAUSSIAN_SIGMA_ASPECT_RATIO ({MAX_GAUSSIAN_SIGMA_ASPECT_RATIO:.2f})")
            continue

        # 6. Filter by Laplacian Variance (Sharpness/Focus)
        if lap_var <= LAPLACIAN_VAR_THRESHOLD:
            print(f"    --> Discarded: Laplacian_Var ({lap_var:.2f}) <= LAPLACIAN_VAR_THRESHOLD ({LAPLACIAN_VAR_THRESHOLD})")
            continue


        # If all filters passed, the particle is considered "kept"
        print(f"    --> KEPT: Meets all criteria.")
        filtered_mask = cv2.add(filtered_mask, current_particle_mask)

        kept_particles_data.append({
            'frame_id': frame_id,
            'particle_id': i,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'bbox_x': x,
            'bbox_y': y,
            'bbox_w': w,
            'bbox_h': h,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'eccentricity': eccentricity,
            'laplacian_variance': lap_var,
            'gaussian_fit_residual_sum': gaussian_fit_residual_sum, # New data
            'fitted_sigma_aspect_ratio': fitted_sigma_aspect_ratio # New data
        })
    print(f"--- Finished processing particles for Frame: {frame_id} ---\n")
    return filtered_mask, kept_particles_data

# --- 4. Main Execution Logic ---

if __name__ == "__main__":
    os.makedirs(FILTERED_MASKS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_frames_particle_data = []

    image_filenames = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    if not image_filenames:
        print(f"No image files found in {IMAGE_DIR}. Please check the path and file extensions.")

    for img_filename in image_filenames:
        frame_id = os.path.splitext(img_filename)[0]

        image_path = os.path.join(IMAGE_DIR, img_filename)
        mask_filename = f"{frame_id}_predict_mask.png"
        mask_path = os.path.join(MASK_DIR, mask_filename)

        print(f"Attempting to process image: {img_filename} with mask: {mask_filename}")

        try:
            original_image, unet_mask = load_image_and_mask(image_path, mask_path)
            
            if np.sum(unet_mask) == 0:
                print(f"  Warning: U-Net mask for {frame_id} is completely empty (all black). Skipping particle filtering for this frame.")
                continue

            filtered_mask, kept_particles_data = process_frame(original_image, unet_mask, frame_id)

            output_mask_path = os.path.join(FILTERED_MASKS_DIR, f"{frame_id}_filtered_mask.png")
            cv2.imwrite(output_mask_path, filtered_mask)
            print(f"  Saved filtered mask to {output_mask_path}")

            all_frames_particle_data.extend(kept_particles_data)

        except FileNotFoundError as e:
            print(f"  Error: {e}. Skipping frame {frame_id}.")
        except Exception as e:
            print(f"  An unexpected error occurred while processing {img_filename}: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for full traceback during debugging


    if all_frames_particle_data:
        df_results = pd.DataFrame(all_frames_particle_data)
        results_csv_path = os.path.join(RESULTS_DIR, 'filtered_particle_data.csv')
        df_results.to_csv(results_csv_path, index=False)
        print(f"\n--- Processing Complete ---")
        print(f"All filtered particle data saved to {results_csv_path}")
    else:
        print("\n--- Processing Complete ---")
        print("No particles were filtered and kept across all frames based on current thresholds. "
              "Please review debug output and adjust ALL thresholds (Area, Aspect Ratio, Extent, Eccentricity, Laplacian Variance, Gaussian).")