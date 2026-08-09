import os
import subprocess
import shutil
import glob
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import logging
import joblib

logging.getLogger('trackpy').setLevel(logging.WARNING)

# --- Python 3.12.10

# --- Initialize potential imports to None to ensure they are always defined in global scope ---
torch = None
UNet = None
A = None
ToTensorV2 = None
CarvanaDataset = None
trackpy = None
label = None
regionprops = None

# --- Deep Learning & Image Processing Imports ---
try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    
    from unet_model import UNet
    from data_loading import CarvanaDataset
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    print("Deep learning (PyTorch, Albumentations) libraries loaded.")
except ImportError as e:
    print(f"ERROR: Deep learning libraries not found or your UNet/DataLoader modules are missing/incorrect: {e}")
    print("Please ensure 'unet_model.py', 'unet_parts.py', and 'data_loading.py' are in your project root or accessible via PYTHONPATH.")
    print("Also ensure PyTorch, Albumentations, etc. are installed.")
    torch = None
    UNet = None
    A = None
    ToTensorV2 = None
    CarvanaDataset = None

# --- Tracking Libraries Imports ---
try:
    from skimage.measure import label, regionprops
    print("Tracking (scikit-image) libraries loaded.")
except ImportError as e:
    print(f"ERROR: Tracking libraries not found. Please install scikit-image. ({e})")
    print("Defocusing filter and tracking stage will not function.")

# --- Siamese Network Imports ---
try:
    import torch
    import torch.nn.functional as F
    from SiameseNet import SiameseNet
    print("Siamese network libraries loaded.")
except ImportError as e:
    print(f"ERROR: Siamese network libraries or modules missing: {e}")
    SiameseNet = None

# --- Import the new plotting functions ---
import Illustration

# --- Configuration Parameters ---
PROJECT_ROOT = Path(__file__).parent

CONFIG = {
    'batch_mode_enabled': True,     # Set to True for batch processing for all files in input_dir or False for GUI selection

    # --- Input/Output Directories ---
    'input_dir': PROJECT_ROOT / 'input_files', 
    'temp_video_frames_dir': PROJECT_ROOT / 'temp_video_frames',
    'unet_masks_dir': PROJECT_ROOT / 'unet_masks',
    'filtered_masks_dir': PROJECT_ROOT / 'filtered_masks',
    'final_results_dir': PROJECT_ROOT / 'results',

    # --- Pipeline Control Flags ---
    'clean_previous_run_data': False, 
    'input_type': 'tif',                             # Support seq and tif files type
    'gt_mask_cache_enabled': False,                  # Enable to skip U-Net detection step allowing the use of existing binary mask in unet_masks_dir
    'gt_mask_cache_glob': '*_predict_mask.png',      
    'gt_mask_cache_require_full_match': True,

    # --- Performance Evaluation ---
    'evaluate_tracking_performance': False,
    'ground_truth_dir': PROJECT_ROOT / 'ground_truth',
    'ground_truth_filename_pattern': '_ground_truth.csv',   # Pattern to find the matching GT file

    # --- MATLAB Configuration ---
    'matlab_exe_path': Path(r'C:\Program Files\MATLAB\R2021b\bin\matlab.exe'),  # Only needed for seq file processing
    'matlab_scripts_dir': PROJECT_ROOT / 'matlab_scripts',
    'matlab_seq_to_tif_script_name': "seq_to_tif.m",
    'matlab_video_overlay_script_name': "create_overlay_video_matlab.m", 

    # --- U-Net Configuration ---
    'unet_model_path': PROJECT_ROOT / 'checkpoints' / 'R2G3B3.pth',
    'unet_img_scale': 1.0,
    'unet_threshold': 0.5,
    'fixed_norm_mean': 0.021386,
    'fixed_norm_std': 0.0230073,

    # --- Defocusing Filter & Property Extraction Configuration ---
    'disable_filter': True,
    'min_particle_area': 10,
    'max_particle_area': 1000,
    'desired_aspect_ratio_min': 0.5,
    'desired_aspect_ratio_max': 2.5,
    'desired_extent_min': 0.6,
    'max_eccentricity_for_particle': 0.9,
    'laplacian_var_threshold': 1.0,
    'max_gaussian_residual_sum': 100000.0,
    'max_gaussian_sigma_aspect_ratio': 2.5,

    # --- Siamese Network Configuration ---
    'siamese_model_path': PROJECT_ROOT / 'checkpoints' / 'S1.pth',
    'siamese_patch_size': 32, 
    'siamese_classification_threshold': 0.0949, 

    # --- Initial Linking Parameters ---
    'trackpy_search_range': 5,
    'trackpy_memory': 0,
    'min_trajectory_length': 3,

    # --- SMLM & Camera Specific Parameters  ---
    'pixelsize_nm': 35.9,       # Camera pixel size (nm/pixel)
    'sCMOS_quanteff': 0.71,     # Quantum efficiency of the detector camera
    'collect_eff': 0.015,       # Collecting efficiency of the microscope
    'camgain': 0.071,           # Camera conversion gain (electron/count)
    'frame_rate_hz': 100.0,     # Camera framerate (Hz)

    # --- Parameters for MSD Calculation ---
    'msd_max_lag_frames': 30,   # Set to any value to calculate MSD using specified length or 0 for entire trajectory 

    # --- Random-Forest Judge Configuration ---
    'rf_model_path': PROJECT_ROOT / 'checkpoints' / 'F5.pkl', 
    'judge_enabled': True,
    'judge_max_time_gap_frames': 120,      # Maximum gap between old track end and new track start
    'judge_max_start_distance_px': 30.0,   # Maximum start distance (pixel)
    'judge_min_confidence': 0.50,          


    # --- Debugging/Visualization Flags ---
    'save_debug_images_unet_stage': False,
    'show_final_trajectory_plot': False, 
    'generate_overlay_video': False, 
    'overlay_video_fps': 10,
    'overlay_video_particle_color_matlab': '[0, 0, 1]', 
    'overlay_video_trajectory_color_matlab': '[1, 0, 0]',
    'overlay_video_line_thickness_matlab': 1,
    'overlay_video_dot_radius_matlab': 2,
    'overlay_video_display_id_matlab': 1
}

# --- Utility Functions for Pipeline Orchestration ---
def setup_directories():
    """Ensures all necessary input/output/temporary directories exist."""
    print("Setting up directories...")
    dirs_to_create = [
        CONFIG['input_dir'],
        CONFIG['temp_video_frames_dir'],
        CONFIG['unet_masks_dir'],
        CONFIG['filtered_masks_dir'],
        CONFIG['final_results_dir'],
        CONFIG['matlab_scripts_dir'],
        CONFIG['ground_truth_dir']
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
    print("All necessary directories are set up.")

def clean_directories(temp_dir_to_clean, unet_mask_dir_to_clean, filtered_mask_dir_to_clean, final_results_dir_to_clean, clean_flag, preserve_unet_masks_dir: bool = False):
    """
    Conditionally clears temporary and output directories before a new run.
    This version applies to specific video subdirectories.
    """
    if clean_flag:
        print(f"Cleaning previous run data for: {temp_dir_to_clean.name}...")
        dirs_to_clean = [
            temp_dir_to_clean,
            filtered_mask_dir_to_clean,
            final_results_dir_to_clean  # <--- This is the corrected variable
        ]
        if not preserve_unet_masks_dir:
            dirs_to_clean.insert(1, unet_mask_dir_to_clean)
        else:
            print(f"  Preserving cached U-Net masks in: {unet_mask_dir_to_clean}")
        for d in dirs_to_clean:
            if d.exists():
                shutil.rmtree(d)
                print(f"  Removed: {d}")
            # Do not re-create here, let the main pipeline function create them as needed
        print("Directories cleaned.")
    else:
        print(f"Skipping cleaning of previous run data for {temp_dir_to_clean.name} as 'clean_previous_run_data' is False.")



def gt_mask_cache_is_usable(input_frames_dir: Path, unet_mask_dir: Path, mask_glob: str, require_full_match: bool = True) -> bool:
    """Return True if cached masks exist (and optionally appear complete) for this video."""
    try:
        if not unet_mask_dir.exists():
            return False

        mask_files = sorted(unet_mask_dir.glob(mask_glob))
        if not mask_files:
            return False

        if not require_full_match:
            return True

        frame_files = sorted([
            p for p in input_frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        ])
        if not frame_files:
            return True

        if mask_glob == '*_predict_mask.png':
            expected_names = {f"{p.stem}_predict_mask.png" for p in frame_files}
            existing_names = {p.name for p in mask_files}
            missing = expected_names - existing_names
            if missing:
                print(f"GT cache: Found {len(mask_files)} masks in {unet_mask_dir}, but missing {len(missing)} expected masks. Will run U-Net.")
                return False
            return True

        if len(mask_files) < len(frame_files):
            print(f"GT cache: Found {len(mask_files)} masks in {unet_mask_dir}, but expected at least {len(frame_files)}. Will run U-Net.")
            return False

        return True

    except Exception as e:
        print(f"GT cache: Error while checking cached masks ({unet_mask_dir}): {e}. Will run U-Net.")
        return False


def select_seq_file_gui():
    """Opens a GUI file dialog to let the user select a .seq or .tif file."""
    root = tk.Tk()
    root.withdraw()
    
    initial_dir_str = str(CONFIG['input_dir'])
    if not Path(initial_dir_str).exists():
        initial_dir_str = str(PROJECT_ROOT)

    file_path = filedialog.askopenfilename(
        title="Select an SMLM Video File (.seq or .tif)",
        initialdir=initial_dir_str,
        filetypes=[("Video files", "*.seq *.tif"),
                   ("All files", "*.*")]
    )
    root.destroy()
    return Path(file_path) if file_path else None

# --- Helper function for 16-bit image loading ---
def _load_image_as_normalized_numpy(filename):
    """
    Loads an image and converts it to a float32 NumPy array,
    preserving the original bit depth for TIFFs and normalizing to 0-1.
    """
    img = Image.open(filename)
    
    if img.mode == 'I;16' or img.mode == 'I':
        img_np = np.array(img, dtype=np.float32)
        max_val = 65535.0
        img_np_normalized = img_np / max_val
        return img_np_normalized
    else:
        img_np = np.array(img.convert('L'), dtype=np.float32)
        img_np_normalized = img_np / 255.0
        return img_np_normalized

# --- Stage 1: Preprocessing (Call MATLAB Script) ---
def run_matlab_preprocessing(seq_file_path, output_tif_dir, matlab_exe, matlab_script_dir, matlab_script_name):
    """
    Executes a MATLAB script to convert a .seq file into .tif frames.
    """
    print(f"\n--- Stage 1: Preprocessing '{seq_file_path.name}' with MATLAB ---")

    matlab_script_full_path = matlab_script_dir / matlab_script_name

    if not matlab_script_full_path.exists():
        print(f"ERROR: MATLAB script '{matlab_script_full_path}' not found.")
        print("Please ensure 'seq_to_tif.m' is in the 'matlab_scripts' folder.")
        return False

    command = [
        str(matlab_exe),
        '-batch',
        f"addpath('{matlab_script_dir.as_posix()}'); {Path(matlab_script_name).stem}('{seq_file_path.as_posix()}', '{output_tif_dir.as_posix()}')"
    ]

    try:
        print(f"Executing MATLAB command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        print("MATLAB Output:\n", result.stdout)
        if result.stderr:
            print("MATLAB Errors (if any):\n", result.stderr)
        print("MATLAB preprocessing completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: MATLAB script failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"ERROR: MATLAB executable not found at '{matlab_exe}'.")
        print("Please verify the 'matlab_exe_path' in the script configuration.")
        return False
    except subprocess.TimeoutExpired:
        print(f"ERROR: MATLAB script timed out after {300} seconds.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during MATLAB execution: {e}")
        return False

# --- Utility for MSD Calculation ---
def calculate_msd(linked_particles_df, pixelsize_nm, frame_rate_hz=1, msd_max_lag_frames=0):
    """
    Calculates Mean Squared Displacement (MSD) for each particle in x, y, and xy (total).

    Parameters:
    -----------
    linked_particles_df : pandas.DataFrame
        DataFrame of linked particle trajectories, from Nearest Neighbor.
        Must contain 'x', 'y', 'frame', and 'particle' columns.
    pixelsize_nm : float
        The physical size of one pixel in nanometers.
    frame_rate_hz : float, optional
        The frame rate of the video in Hz. Used to convert lag times to seconds.
        Defaults to 1, meaning lag time is in frames.
    msd_max_lag_frames : int, optional
        Maximum number of lag frames to calculate MSD for.
        If 0 (or greater than or equal to the trajectory length), MSD is calculated for all possible lag times.
        Defaults to 0.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with MSD values for each particle (MSDx, MSDy, MSDxy)
        and 'lag_time_frames', 'lag_time_s'.
    """


    print("\nCalculating Mean Squared Displacement (MSD) for X, Y, and XY...")

    all_msd_data = []

    # Iterate over each unique particle
    for particle_id, trajectory in linked_particles_df.groupby('particle'):
        # Ensure 'frame' is a column and not also an index level when sorting
        # This will reset any index levels to columns and ensure 'frame' is just a column
        trajectory = trajectory.reset_index(drop=True).sort_values(by='frame')

        # Extract coordinates and convert to nanometers
        x_coords = trajectory['x'].values * pixelsize_nm
        y_coords = trajectory['y'].values * pixelsize_nm
        # If you had a 'z' column extracted and propagated, it would be used here:
        # z_coords = trajectory['z'].values * pixelsize_nm

        # MODIFIED: Apply msd_max_lag_frames limit
        max_possible_lag_frames = len(trajectory) - 1
        
        if max_possible_lag_frames == 0:
            continue # Skip single-frame trajectories

        if msd_max_lag_frames == 0 or msd_max_lag_frames >= max_possible_lag_frames:
            # Calculate MSD for all possible lag times
            effective_max_lag_frames = max_possible_lag_frames
        else:
            # Limit MSD calculation to msd_max_lag_frames
            effective_max_lag_frames = msd_max_lag_frames
        
        lag_times_frames = np.arange(1, effective_max_lag_frames + 1)
        lag_times_s = lag_times_frames / frame_rate_hz

        msd_x_values = []
        msd_y_values = []
        msd_z_values = [] # Placeholder for Z
        msd_xy_values = []
        
        for dt in lag_times_frames:
            # Squared displacement in x for this lag time
            # Ensure the slices are correct: x_coords[dt:] are positions at time t+dt,
            # x_coords[:-dt] are positions at time t.
            dx_squared = (x_coords[dt:] - x_coords[:-dt])**2
            # Mean over all possible starting points for this lag time (dt)
            msd_x_values.append(np.mean(dx_squared))

            # Squared displacement in y for this lag time
            dy_squared = (y_coords[dt:] - y_coords[:-dt])**2
            msd_y_values.append(np.mean(dy_squared))

            # Squared displacement in xy (total) for this lag time
            dxy_squared = dx_squared + dy_squared # Sum of squared displacements
            msd_xy_values.append(np.mean(dxy_squared))

            # If you had Z-coordinates and wanted MSDz:
            # dz_squared = (z_coords[dt:] - z_coords[:-dt])**2
            # msd_z_values.append(np.mean(dz_squared))


        df_particle_msd = pd.DataFrame({
            'particle': particle_id,
            'lag_time_frames': lag_times_frames,
            'lag_time_s': lag_times_s,
            'msd_x_nm2': msd_x_values,
            'msd_y_nm2': msd_y_values,
            'msd_xy_nm2': msd_xy_values,
            # 'msd_z_nm2': msd_z_values # Uncomment if Z is available
        })
        all_msd_data.append(df_particle_msd)

    if all_msd_data:
        full_msd_df = pd.concat(all_msd_data, ignore_index=True)
        print("MSD calculation complete.")
        return full_msd_df
    else:
        print("No particles to calculate MSD for.")
        return pd.DataFrame()


# --- Stage 2: Particle Detection (U-Net Prediction) ---
def run_unet_prediction(input_tif_dir, output_mask_dir, model_path, img_scale, threshold, 
                        norm_mean, norm_std, save_debug_images_unet_stage):
    """
    Runs U-Net inference on all .tif images in input_tif_dir and saves masks.
    """
    if torch is None or UNet is None or A is None or ToTensorV2 is None or CarvanaDataset is None:
        print("Skipping U-Net prediction: Required deep learning libraries not loaded or modules missing.")
        return False

    print(f"\n--- Stage 2: Running U-Net prediction on '{input_tif_dir}' ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"U-Net will run on: {device}")

    if not model_path.exists():
        print(f"ERROR: U-Net model not found at {model_path}.")
        print("Please ensure 'unet_model_path' is correct and the model file exists.")
        return False

    try:
        net = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)
        state_dict = torch.load(model_path, map_location=device)
        state_dict.pop("mask_values", None)
        net.load_state_dict(state_dict)
        net.eval()
        print(f"U-Net model loaded from {model_path}.")
    except Exception as e:
        print(f"ERROR: Failed to load U-Net model from {model_path}: {e}")
        return False

    if A:
        inference_transform = A.Compose([
            A.Normalize(mean=(norm_mean,), std=(norm_std,)),
            ToTensorV2()
        ])
    else:
        from torchvision import transforms
        inference_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[norm_mean], std=[norm_std])
        ])
        print("WARNING: Albumentations not found, using torchvision transforms. Ensure consistency with training.")

    image_paths = sorted([p for p in input_tif_dir.iterdir() if p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')])
    if not image_paths:
        print(f"WARNING: No images found in {input_tif_dir} for U-Net prediction. Skipping.")
        return True

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="U-Net Predicting Frames"):
            img_np_float_0_1 = _load_image_as_normalized_numpy(img_path)
            orig_h, orig_w = img_np_float_0_1.shape

            if save_debug_images_unet_stage:
                debug_img_path = output_mask_dir / f"{img_path.stem}_debug_01_original.png"
                Image.fromarray((img_np_float_0_1 * 255).astype(np.uint8)).save(debug_img_path)
                
            transformed = inference_transform(image=img_np_float_0_1)
            img_tensor = transformed['image'].unsqueeze(0)
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)

            if save_debug_images_unet_stage:
                img_norm_display = img_tensor.squeeze().cpu().numpy()
                img_norm_display_scaled = (img_norm_display - img_norm_display.min()) / (img_norm_display.max() - img_norm_display.min() + 1e-8)
                debug_norm_input_path = output_mask_dir / f"{img_path.stem}_debug_03_normalized_input.png"
                Image.fromarray((img_norm_display_scaled * 255).astype(np.uint8)).save(debug_norm_input_path)

            output = net(img_tensor)
            
            output = F.interpolate(output, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            heatmap = torch.sigmoid(output).squeeze().cpu().numpy()
            
            if heatmap.ndim == 1:
                heatmap = heatmap.reshape(orig_h, orig_w)
            elif heatmap.ndim == 0:
                heatmap = np.array([[heatmap]])

            mask = (heatmap > threshold).astype(np.uint8) * 255

            mask_filename = img_path.stem + "_predict_mask.png"
            output_mask_path = output_mask_dir / mask_filename
            Image.fromarray(mask).save(output_mask_path)

    print("U-Net prediction completed successfully.")
    return True

# --- Stage 3: Defocusing Filter & Properties Extraction + Linking ---
def get_particle_embedding(original_image, centroid_x, centroid_y, patch_size, siamese_model, device):
    """
    Extracts a patch, normalizes it, and computes the Siamese embedding.
    """
    half_patch = patch_size // 2
    height, width = original_image.shape
    
    # Ensure patch coordinates are within image bounds
    x1 = int(centroid_x - half_patch)
    y1 = int(centroid_y - half_patch)
    x2 = int(centroid_x + half_patch)
    y2 = int(centroid_y + half_patch)
    
    # Pad the patch with zeros if it goes beyond the image borders
    padded_patch = np.zeros((patch_size, patch_size), dtype=original_image.dtype)
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(width, x2)
    src_y2 = min(height, y2)

    dst_x1 = max(0, -x1)
    dst_y1 = max(0, -y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        padded_patch[dst_y1:dst_y2, dst_x1:dst_x2] = original_image[src_y1:src_y2, src_x1:src_x2]
        
    # Normalize the 16-bit patch and convert to tensor
    patch_tensor = torch.from_numpy(padded_patch.astype(np.float32) / 65535.0).unsqueeze(0).unsqueeze(0)
    
    # Compute and return the embedding
    with torch.no_grad():
        embedding = siamese_model.forward_once(patch_tensor.to(device)).squeeze().cpu().numpy()
        
    return embedding


def _parse_embedding_string(embedding):
    """Robustly parse Siamese embedding from CSV string or passthrough numpy array."""
    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32)
    if embedding is None or (isinstance(embedding, float) and np.isnan(embedding)):
        return np.array([], dtype=np.float32)
    s = str(embedding).strip()
    if len(s) >= 2 and s[0] in "[(" and s[-1] in "])":
        s = s[1:-1]
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    try:
        return np.asarray([float(p) for p in parts], dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

def _rf_features_from_rows(row_end, row_start):
    """
    Build the single feature vector expected by the RF:
      siamese_euclidean_distance, position_euclidean_distance, time_difference,
      area_difference, Ibcnt_difference, fwhm_avg_difference,
      area_ratio, Ibcnt_ratio, fwhm_avg_ratio
    row_end  : last row of the earlier trajectory
    row_start: first row of the later trajectory
    """
    emb1 = _parse_embedding_string(row_end.get('siamese_embedding', None))
    emb2 = _parse_embedding_string(row_start.get('siamese_embedding', None))

    # Siamese distance (0 if missing or mismatched dims)
    if emb1.size and emb2.size and emb1.shape == emb2.shape:
        siamese_dist = float(np.linalg.norm(emb1 - emb2))
    else:
        siamese_dist = float('inf')  # strongly discourages merge when embedding missing

    # Position distance
    pos_dist = float(np.hypot(row_end['x'] - row_start['x'], row_end['y'] - row_start['y']))

    # Δt (frames)
    time_diff = int(row_start['frame'] - row_end['frame'])

    # Safe diffs/ratios
    def fdiff(a,b): 
        try: return float(abs(float(a) - float(b)))
        except: return float('inf')

    def fratio(a,b):
        try:
            b = float(b) if float(b) != 0 else 1e-6
            return float(a)/b
        except:
            return float('inf')

    area_diff  = fdiff(row_end.get('area', 0),  row_start.get('area', 0))
    ib_diff    = fdiff(row_end.get('Ibcnt', 0), row_start.get('Ibcnt', 0))
    fwhm_diff  = fdiff(row_end.get('fwhm_avg_pixels', 0), row_start.get('fwhm_avg_pixels', 0))

    area_ratio = fratio(row_end.get('area', 0),  row_start.get('area', 0))
    ib_ratio   = fratio(row_end.get('Ibcnt', 0), row_start.get('Ibcnt', 0))
    fwhm_ratio = fratio(row_end.get('fwhm_avg_pixels', 0), row_start.get('fwhm_avg_pixels', 0))

    feats = {
        'siamese_euclidean_distance': siamese_dist,
        'position_euclidean_distance': pos_dist,
        'time_difference': time_diff,
        'area_difference': area_diff,
        'Ibcnt_difference': ib_diff,
        'fwhm_avg_difference': fwhm_diff,
        'area_ratio': area_ratio,
        'Ibcnt_ratio': ib_ratio,
        'fwhm_avg_ratio': fwhm_ratio
    }
    return feats

def _build_segments(df):
    """
    From linked dataframe create a per-particle segment table with:
      particle_id, start_frame, end_frame, first_row, last_row
    """
    segs = []
    for pid, g in df.groupby('particle'):
        g_sorted = g.sort_values('frame')
        segs.append({
            'particle': pid,
            'start_frame': int(g_sorted['frame'].iloc[0]),
            'end_frame':   int(g_sorted['frame'].iloc[-1]),
            'first_row':   g_sorted.iloc[0],
            'last_row':    g_sorted.iloc[-1],
        })
    return segs

def _apply_rf_judge_and_merge(linked_df: pd.DataFrame, config: dict):
    """
    Use RF judge to merge particle IDs when a 'new' trajectory likely belongs to
    the same particle that previously disappeared.

    Returns:
      judged_df           : dataframe with corrected 'particle' IDs
      merge_log_df        : dataframe describing merges (old_id <- new_id, proba, delta_t, distance)
      num_merged_segments : count of merges performed
    """
    # ---- define & normalize 'judged' FIRST ----
    judged = linked_df.copy()

    # If 'frame' is an index level, push it to columns; avoid duplicate 'frame' on reset
    if isinstance(judged.index, pd.MultiIndex) and ('frame' in judged.index.names):
        if 'frame' in judged.columns:
            judged = judged.reset_index(drop=True)
        else:
            judged = judged.reset_index()
    elif getattr(judged.index, "name", None) == "frame":
        if 'frame' in judged.columns:
            judged = judged.reset_index(drop=True)
        else:
            judged = judged.reset_index()

    # Flatten odd multi-columns (rare)
    if isinstance(judged.columns, pd.MultiIndex):
        judged.columns = ['_'.join(map(str, c)).strip('_') for c in judged.columns]

    # Ensure single 'frame' column only
    if (judged.columns == 'frame').sum() > 1:
        keep = True
        new_cols = []
        for c in judged.columns:
            if c == 'frame':
                if keep:
                    new_cols.append(c); keep = False
                else:
                    continue
            else:
                new_cols.append(c)
        judged = judged.loc[:, new_cols]

    # dtype hygiene
    if 'frame' in judged.columns:
        judged['frame'] = pd.to_numeric(judged['frame'], errors='coerce').astype('Int64').astype(int, errors='ignore')
    if 'particle' in judged.columns:
        judged['particle'] = pd.to_numeric(judged['particle'], errors='coerce').astype('Int64').astype(int, errors='ignore')

    # ---- config / model guards ----
    if not config.get('judge_enabled', True):
        print("RF judge disabled in CONFIG. Skipping.")
        return judged, pd.DataFrame(), 0

    model_path = config.get('rf_model_path', None)
    if not model_path or not Path(model_path).exists():
        print(f"RF model not found at: {model_path}. Skipping judge step.")
        return judged, pd.DataFrame(), 0

    print("\n--- RF Judge: Loading model ---")
    rf = joblib.load(model_path)

    # Dynamically read the exact features this specific model expects
    expected_features = list(rf.feature_names_in_)


    max_gap = int(config.get('judge_max_time_gap_frames', 120))
    max_dist = float(config.get('judge_max_start_distance_px', 15.0))
    min_conf = float(config.get('judge_min_confidence', 0.5))

    merges = []
    changed = True

    # --- helper ---
    def _build_segments_local(df):
        d = df.reset_index(drop=True)
        segs = []
        for pid, g in d.groupby('particle'):
            g_sorted = g.sort_values('frame')
            segs.append({
                'particle': int(pid),
                'start_frame': int(g_sorted['frame'].iloc[0]),
                'end_frame':   int(g_sorted['frame'].iloc[-1]),
                'first_row':   g_sorted.iloc[0],
                'last_row':    g_sorted.iloc[-1],
            })
        return segs

    # --- iterative merging ---
    while changed:
        changed = False

        segs = _build_segments_local(judged)
        segs_sorted_start = sorted(segs, key=lambda s: s['start_frame'])
        segs_sorted_end   = sorted(segs, key=lambda s: s['end_frame'])

        for s_new in segs_sorted_start:
            pid_new = s_new['particle']
            start_f = s_new['start_frame']
            first_row = s_new['first_row']

            # If this pid already got merged in a previous iteration, skip
            current_pids = set(pd.to_numeric(judged['particle'], errors='coerce').dropna().astype(int).unique().tolist())
            if pid_new not in current_pids:
                continue

            # candidate previous segments
            candidates = []
            for s_old in segs_sorted_end:
                if s_old['particle'] == pid_new:
                    continue
                dt = start_f - s_old['end_frame']
                if 1 <= dt <= max_gap:
                    d = float(np.hypot(
                        s_old['last_row']['x'] - first_row['x'],
                        s_old['last_row']['y'] - first_row['y']
                    ))
                    if d <= max_dist:
                        candidates.append((s_old, dt, d))
            if not candidates:
                continue

            # score with RF
            best = None
            best_proba = -1.0
            for s_old, dt, dist in candidates:
                feats = _rf_features_from_rows(s_old['last_row'], first_row)
                
                # Filter the dictionary to only include the features the model expects
                filtered_feats = {k: feats[k] for k in expected_features}
                
                X = pd.DataFrame([filtered_feats], columns=expected_features)
                X = X.replace([np.inf, -np.inf], 1e9).fillna(1e9)
                try:
                    proba_pos = float(rf.predict_proba(X)[:, 1][0])
                except Exception as e:
                    print(f"Warning: RF prediction failed: {e}") # Print the error instead of hiding it!
                    continue
                if proba_pos > best_proba:
                    best_proba = proba_pos
                    best = (s_old, dt, dist, feats)

            if best and best_proba >= min_conf:
                s_old, dt, dist, feats = best
                pid_old = s_old['particle']
                judged.loc[judged['particle'] == pid_new, 'particle'] = int(pid_old)
                merges.append({
                    'merged_into': int(pid_old),
                    'merged_from': int(pid_new),
                    'confidence': best_proba,
                    'time_gap_frames': int(dt),
                    'start_distance_px': float(dist),
                    **feats
                })
                changed = True
                break  # rebuild segments after each merge

    merge_log_df = pd.DataFrame(merges)
    judged = judged.sort_values(['particle', 'frame']).reset_index(drop=True)
    return judged, merge_log_df, len(merges)


def simple_nearest_neighbor_linking(df: pd.DataFrame, max_displacement_px: float) -> pd.DataFrame:
    """
    Simple frame-to-frame nearest-neighbor linking that assigns integer
    'particle' IDs. It only connects detections
    between consecutive frames (no temporal gap bridging).
    """
    if df.empty:
        return df

    # Work on a sorted copy so index order is temporal
    df = df.sort_values('frame').reset_index(drop=True)
    frames = df['frame'].unique()

    # Pre-allocate particle IDs
    particle_ids = np.full(len(df), -1, dtype=int)
    next_id = 0

    # active_tracks: track_id -> last row index in df
    active_tracks = {}

    # --- Initialize first frame ---
    first_frame = frames[0]
    first_idxs = df.index[df['frame'] == first_frame].tolist()
    for idx in first_idxs:
        particle_ids[idx] = next_id
        active_tracks[next_id] = idx
        next_id += 1

    # --- Process subsequent frames ---
    frame_set = set(frames)
    for f in frames[1:]:
        prev_frame = f - 1
        if prev_frame not in frame_set:
            active_tracks = {}

        current_idxs = df.index[df['frame'] == f].tolist()
        if not current_idxs:
            continue

        candidates = []
        for tid, last_idx in active_tracks.items():
            if df.at[last_idx, 'frame'] != prev_frame:
                continue
            x1 = df.at[last_idx, 'x']
            y1 = df.at[last_idx, 'y']
            for idx in current_idxs:
                x2 = df.at[idx, 'x']
                y2 = df.at[idx, 'y']
                dist = float(np.hypot(x2 - x1, y2 - y1))
                if dist <= max_displacement_px:
                    candidates.append((dist, tid, idx))

        candidates.sort(key=lambda t: t[0])
        used_tracks = set()
        used_dets = set()
        new_active_tracks = {}

        for dist, tid, idx in candidates:
            if tid in used_tracks or idx in used_dets:
                continue
            particle_ids[idx] = tid
            used_tracks.add(tid)
            used_dets.add(idx)
            new_active_tracks[tid] = idx

        for idx in current_idxs:
            if idx in used_dets:
                continue
            particle_ids[idx] = next_id
            new_active_tracks[next_id] = idx
            next_id += 1

        active_tracks = new_active_tracks

    df = df.copy()
    df['particle'] = particle_ids.astype(int)
    return df

def run_defocusing_and_extraction_and_tracking(original_img_dir, unet_mask_dir, filtered_masks_dir, final_results_dir, config, video_base_name: str):
    """
    Filters U-Net masks for defocused particles, extracts properties,
    and then uses Nearest Neighbor to link particles across frames.
    
    Args:
        original_img_dir (Path): Directory containing original .tif frames.
        unet_mask_dir (Path): Directory containing U-Net generated masks.
        filtered_masks_dir (Path): Directory to save filtered masks.
        final_results_dir (Path): Directory to save final results (CSV, plots).
        config (dict): Configuration dictionary with pipeline parameters.
        video_base_name (str): Base name of the video file (e.g., 'my_video').
    """
    if label is None or regionprops is None:
        print("Skipping defocusing filter and tracking: Required library (scikit-image) not loaded.")
        return False, pd.DataFrame()

    print(f"\n--- Stage 3: Filtering Defocused Particles, Extracting Properties & Tracking ---")

    if SiameseNet is None:
        print("Skipping Siamese-enhanced linking: SiameseNet not loaded.")
        siamese_model = None
    else:
        print("\n--- Stage 3.1: Loading Siamese Network for Linking ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        siamese_model = SiameseNet(patch_size=CONFIG['siamese_patch_size']).to(device)
        siamese_model.load_state_dict(torch.load(CONFIG['siamese_model_path'], map_location=device))
        siamese_model.eval()
        print("Siamese network loaded successfully.")

    # Helper functions for this stage (from ParticleTagging.py)
    def load_image_and_mask_for_filter(image_path, mask_path):
        """
        Loads a grayscale image (preserving bit depth) and its corresponding binary mask.
        Handles 16-bit images correctly.
        """
        # Load original image WITHOUT normalization for intensity calculations
        original_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if original_image is None:
            raise FileNotFoundError(f"Original image not found at {image_path}")
        if mask is None:
            raise FileNotFoundError(f"U-Net mask not found at {mask_path}")
        
        mask = (mask > 0).astype(np.uint8) * 255
        return original_image, mask

    def calculate_laplacian_variance(image_roi, mask_roi):
        """
        Calculates the Laplacian Variance for a masked region of interest.
        Normalizes 16-bit input to 0-255 range for consistent Laplacian calculation.
        """
        if image_roi.shape[0] == 0 or image_roi.shape[1] == 0:
            return 0.0

        # Ensure image_roi is float32 for Laplacian
        normalized_pixels = image_roi.astype(np.float32)

        # Normalize to 0-255 for consistent Laplacian calculation, as in MATLAB's implied 8-bit operations
        # Check if it's potentially a 16-bit image before normalizing.
        if normalized_pixels.max() > 255 and normalized_pixels.max() <= 65535: # Heuristic for 16-bit
            cv2.normalize(normalized_pixels, normalized_pixels, 0, 255, cv2.NORM_MINMAX)
        
        # Apply mask AFTER normalization for accurate Laplacian on particle pixels
        masked_pixels = cv2.bitwise_and(normalized_pixels.astype(np.uint8), normalized_pixels.astype(np.uint8), mask=(mask_roi > 0).astype(np.uint8))
        
        laplacian = cv2.Laplacian(masked_pixels, cv2.CV_32F)
        
        non_zero_laplacian = laplacian[mask_roi > 0]

        if non_zero_laplacian.size > 0:
            return np.var(non_zero_laplacian)
        else:
            return 0.0

    def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = coords
        xo = float(xo)
        yo = float(yo)
        
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        
        g = offset + amplitude * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()

    def process_frame_for_filter(original_image, unet_mask, frame_num, config_params):
        """
        Processes a single frame's image and U-Net mask to filter particles.
        Returns the filtered mask and a list of particle properties for that frame.
        """
        height, width = original_image.shape
        filtered_mask = np.zeros_like(unet_mask, dtype=np.uint8)
        kept_particles_data = []

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(unet_mask, 8, cv2.CV_32S)

        if num_labels == 1: # Only background label
            return filtered_mask, kept_particles_data

        # Estimate global background (simplified from MATLAB's median of sorted image)
        global_background = np.median(original_image.flatten())

        for i in range(1, num_labels): # Iterate through detected components (skip background label 0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            centroid_x, centroid_y = centroids[i]

            current_particle_mask = (labels == i).astype(np.uint8) * 255 # Mask for this specific particle

            if w == 0 or h == 0:
                continue

            # Ensure ROI stays within image bounds for particle content and related mask
            x_end = min(x + w, width)
            y_end = min(y + h, height)
            
            original_roi_patch = original_image[y:y_end, x:x_end] # Original bbox for particle content
            particle_mask_roi_patch = current_particle_mask[y:y_end, x:x_end] # Mask within original bbox

            # Adding a small border around the detected bbox for robust background estimation (e.g., 1 pixel)
            border = 1 
            x_start_padded = max(0, x - border)
            y_start_padded = max(0, y - border)
            x_end_padded_full = min(x + w + border, width) # Use a different name to avoid confusion with original x_end
            y_end_padded_full = min(y + h + border, height) # Use a different name to avoid confusion with original y_end

            padded_original_roi = original_image[y_start_padded:y_end_padded_full, x_start_padded:x_end_padded_full]
            padded_mask_roi = np.zeros_like(padded_original_roi, dtype=np.uint8)
            # Copy particle mask into the padded region, adjusting for the new padded ROI's top-left corner
            padded_mask_roi[y - y_start_padded : y_end - y_start_padded, 
                            x - x_start_padded : x_end - x_start_padded] = particle_mask_roi_patch
            
            # Background mask: pixels in padded_roi that are NOT part of the particle
            background_mask = (padded_mask_roi == 0).astype(np.uint8)
            background_pixels = padded_original_roi[background_mask > 0]


            if original_roi_patch.size == 0 or particle_mask_roi_patch.size == 0:
                continue

            # --- Basic Morphological Properties (already existing) ---
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
                            eccentricity = 0.0 # Perfect circle, or degenerate ellipse
                    except cv2.error:
                        eccentricity = 1.0 # Default to highly eccentric
                else:
                    eccentricity = 1.0 # Not enough points, assume highly eccentric

            lap_var = calculate_laplacian_variance(original_roi_patch, particle_mask_roi_patch)

            # --- Gaussian Fit and related metrics ---
            gaussian_fit_success = False
            fitted_sigma_aspect_ratio = float('inf')
            gaussian_fit_residual_sum = float('inf') 
            fitted_amplitude = np.nan
            fitted_sigma_x = np.nan
            fitted_sigma_y = np.nan
            fwhm_x = np.nan
            fwhm_y = np.nan
            fwhm_avg = np.nan
            fitted_offset = np.nan
            r_squared_gaussian = np.nan
            rmse_gaussian = np.nan
            
            # Prepare pixels for Gaussian fit (normalize to 0-255 range if 16-bit, similar to MATLAB's cntr2dg preprocessing)
            pixels_for_fit = original_roi_patch.astype(np.float32)
            if pixels_for_fit.max() > 0:
                 if original_image.dtype == np.uint16 or original_image.max() > 255:
                     cv2.normalize(pixels_for_fit, pixels_for_fit, 0, 255, cv2.NORM_MINMAX)
            pixels_for_fit = cv2.bitwise_and(pixels_for_fit.astype(np.uint8), pixels_for_fit.astype(np.uint8), mask=particle_mask_roi_patch)

            y_coords_in_patch, x_coords_in_patch = np.where(particle_mask_roi_patch > 0)
            intensities_in_patch = pixels_for_fit[y_coords_in_patch, x_coords_in_patch]

            if intensities_in_patch.size > 0:
                # Initial guesses for Gaussian fit
                initial_amplitude = np.max(intensities_in_patch) - np.min(intensities_in_patch)
                initial_offset = np.min(intensities_in_patch)
                initial_xo = x_coords_in_patch.mean()
                initial_yo = y_coords_in_patch.mean()
                initial_sigma_x = (w / 2) / 2.355 if w > 0 else 1.0 # FWHM approx / 2.355
                initial_sigma_y = (h / 2) / 2.355 if h > 0 else 1.0 # FWHM approx / 2.355

                # Ensure initial sigmas are not too small
                initial_sigma_x = max(initial_sigma_x, 0.5)
                initial_sigma_y = max(initial_sigma_y, 0.5)
                
                p0 = [initial_amplitude, initial_xo, initial_yo, initial_sigma_x, initial_sigma_y, 0.0, initial_offset]
                bounds = (
                    [0, 0, 0, 0.1, 0.1, -np.pi, 0], 
                    [256, w, h, max(w,h), max(w,h), np.pi, 255] 
                )

                try:
                    popt, pcov = curve_fit(gaussian_2d, (x_coords_in_patch, y_coords_in_patch),
                                           intensities_in_patch, p0=p0, bounds=bounds,
                                           maxfev=5000)
                    
                    fitted_amplitude, fitted_xo, fitted_yo, fitted_sigma_x, fitted_sigma_y, fitted_theta, fitted_offset = popt
                    
                    fitted_values = gaussian_2d((x_coords_in_patch, y_coords_in_patch), *popt)
                    gaussian_fit_residual_sum = np.sum((intensities_in_patch - fitted_values)**2)

                    ss_total = np.sum((intensities_in_patch - np.mean(intensities_in_patch))**2)
                    r_squared_gaussian = 1 - (gaussian_fit_residual_sum / (ss_total + 1e-9)) if ss_total > 0 else 0.0

                    rmse_gaussian = np.sqrt(gaussian_fit_residual_sum / intensities_in_patch.size) if intensities_in_patch.size > 0 else 0.0

                    fwhm_x = fitted_sigma_x * 2.355
                    fwhm_y = fitted_sigma_y * 2.355
                    fwhm_avg = (fwhm_x + fwhm_y) / 2 

                    if fitted_sigma_x != 0 and fitted_sigma_y != 0:
                        fitted_sigma_aspect_ratio = max(fitted_sigma_x, fitted_sigma_y) / min(fitted_sigma_x, fitted_sigma_y)
                    else:
                        fitted_sigma_aspect_ratio = float('inf')

                    gaussian_fit_success = True

                except RuntimeError:
                    gaussian_fit_success = False
                except ValueError:
                    gaussian_fit_success = False
                except Exception:
                    gaussian_fit_success = False

            # --- Intensity (Photon Count) Calculations (Analogous to MATLAB's Ibcnt, Ib, Idetect, Iphelect) ---
            if background_pixels.size > 0:
                particle_background = np.median(background_pixels) # More robust than corner average
            else:
                particle_background = global_background # Fallback if no background pixels in padded ROI

            total_particle_intensity = np.sum(original_roi_patch[particle_mask_roi_patch > 0])
            
            Ibcnt = total_particle_intensity - (particle_background * area)
            Ibcnt = max(0, Ibcnt) # Ensure non-negative

            camgain = config_params['camgain']
            sCMOS_quanteff = config_params['sCMOS_quanteff']
            collect_eff = config_params['collect_eff']

            Ib = Ibcnt * camgain / sCMOS_quanteff / collect_eff 
            Idetect = Ibcnt * camgain / sCMOS_quanteff 
            Iphelect = Ibcnt * camgain 

            
             # --- Filtering Logic ---
            if not config_params.get('disable_filter', False):
                if (area < config_params['min_particle_area'] or area > config_params['max_particle_area'] or
                    not (config_params['desired_aspect_ratio_min'] <= aspect_ratio <= config_params['desired_aspect_ratio_max']) or
                    extent < config_params['desired_extent_min'] or
                    eccentricity > config_params['max_eccentricity_for_particle'] or
                    not gaussian_fit_success or
                    gaussian_fit_residual_sum > config_params['max_gaussian_residual_sum'] or
                    fitted_sigma_aspect_ratio > config_params['max_gaussian_sigma_aspect_ratio'] or
                    lap_var <= config_params['laplacian_var_threshold']):
                            
                    continue 
            
            cv2.add(filtered_mask, current_particle_mask, dst=filtered_mask)

            embedding = None # Initialize embedding to None
            if siamese_model is not None:
                embedding = get_particle_embedding(
                    original_image=original_image, 
                    centroid_x=centroid_x, 
                    centroid_y=centroid_y, 
                    patch_size=CONFIG['siamese_patch_size'],
                    siamese_model=siamese_model,
                    device=device
                )

            kept_particles_data.append({
                'frame': frame_num,
                'x': centroid_x,
                'y': centroid_y,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'eccentricity': eccentricity,
                'laplacian_variance': lap_var,
                'gaussian_fit_residual_sum': gaussian_fit_residual_sum,
                'fitted_sigma_aspect_ratio': fitted_sigma_aspect_ratio,
                'Ag_gaussian_amplitude': fitted_amplitude, 
                'sigma_x_pixels': fitted_sigma_x,
                'sigma_y_pixels': fitted_sigma_y,
                'fwhm_x_pixels': fwhm_x, 
                'fwhm_y_pixels': fwhm_y,
                'fwhm_avg_pixels': fwhm_avg,
                'gaussian_offset': fitted_offset,
                'r_squared_gaussian': r_squared_gaussian, 
                'rmse_gaussian': rmse_gaussian, 
                'Ibcnt': Ibcnt, 
                'Ib_photons_emitted': Ib, 
                'Idetect_photons_camera': Idetect, 
                'Iphelect_photoelectrons': Iphelect,
                'siamese_embedding': embedding, 
            })
        return filtered_mask, kept_particles_data

    all_frames_particle_data = []

    image_paths = sorted([p for p in original_img_dir.iterdir() if p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')])

    if not image_paths:
        print(f"WARNING: No original image files found in {original_img_dir} for filtering. Skipping.")
        return False, pd.DataFrame() # Return False to signal failure

    first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_image is None:
        print(f"ERROR: Could not read first image {image_paths[0]} to determine dimensions. Plotting limits might be off.")
        img_height, img_width = 100, 100 
    else:
        img_height, img_width = first_image.shape[:2] 

    for img_path in tqdm(image_paths, desc="Filtering & Extracting Properties"):
        frame_id_stem = img_path.stem
        
        try:
            # Assuming frame number is the numeric part at the end of the filename
            frame_num = int("".join(filter(str.isdigit, Path(img_path).stem.split('_')[-1])))
        except ValueError:
            print(f"Warning: Could not extract numeric frame ID from '{frame_id_stem}'. Using 0.")
            frame_num = 0

        mask_filename = f"{frame_id_stem}_predict_mask.png"
        mask_path = unet_mask_dir / mask_filename

        try:
            original_image, unet_mask = load_image_and_mask_for_filter(img_path, mask_path)
            
            if np.sum(unet_mask) == 0:
                continue

            filtered_mask, kept_particles_data = process_frame_for_filter(original_image, unet_mask, frame_num, config)

            output_filtered_mask_path = filtered_masks_dir / f"{frame_id_stem}_filtered_mask.png"
            cv2.imwrite(str(output_filtered_mask_path), filtered_mask)

            all_frames_particle_data.extend(kept_particles_data)

        except FileNotFoundError as e:
            print(f"Error: {e}. Skipping frame {frame_id_stem}.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {img_path.name}: {e}")

    if not all_frames_particle_data:
        print("\n--- No particles were detected and kept across all frames based on current thresholds. ---")
        print("Please review image data, U-Net performance, and adjust filtering thresholds.")
        return False, pd.DataFrame()

    df_filtered_particles = pd.DataFrame(all_frames_particle_data)
    initial_results_csv_path = final_results_dir / 'filtered_particle_data_for_tracking.csv'
    df_filtered_particles.to_csv(str(initial_results_csv_path), index=False)
    print(f"\n--- Initial particle data (pre-tracking) saved to {initial_results_csv_path} ---")
    print(f"Total detections before tracking: {len(df_filtered_particles)}")
    
    print(f"DEBUG: X coordinates range: min={df_filtered_particles['x'].min():.2f}, max={df_filtered_particles['x'].max():.2f}")
    print(f"DEBUG: Y coordinates range: min={df_filtered_particles['y'].min():.2f}, max={df_filtered_particles['y'].max():.2f}")

    print("\n--- Starting nearest-neighbor particle linking ---")

    df_filtered_particles['frame'] = df_filtered_particles['frame'].astype(int)

    # Initial simple linking (frame-to-frame only)
    linked_particles = simple_nearest_neighbor_linking(
        df_filtered_particles,
        max_displacement_px=config['trackpy_search_range']
    )

    linked_particles = linked_particles.reset_index(drop=True)

    print("Custom nearest-neighbor linking complete.")
    print(f"Total raw trajectories found: {linked_particles['particle'].nunique()}")

    # --- Manual Stub Filtering ---
    min_len = config['min_trajectory_length']
    traj_lengths = linked_particles.groupby('particle').size()
    valid_ids = traj_lengths[traj_lengths >= min_len].index

    filtered_linked_particles = linked_particles[linked_particles['particle'].isin(valid_ids)].copy()

    def _ensure_frame_is_only_column(df: pd.DataFrame) -> pd.DataFrame:
        """Make sure 'frame' exists only as a column (not also an index level)."""
        out = df.copy()
        # Figure out current index names (MultiIndex-safe)
        idx_names = []
        if isinstance(out.index, pd.MultiIndex):
            idx_names = [n for n in out.index.names if n is not None]
        elif out.index.name is not None:
            idx_names = [out.index.name]

        if 'frame' in idx_names:
            # If 'frame' is already a column too, just drop the index.
            # If not present as a column, bring it out with reset_index()
            if 'frame' in out.columns:
                out = out.reset_index(drop=True)
            else:
                out = out.reset_index()  # brings index levels (incl. 'frame') into columns
        else:
            # No 'frame' index level; still drop index to avoid weirdness later
            out = out.reset_index(drop=True)

        return out
    
    filtered_linked_particles = _ensure_frame_is_only_column(filtered_linked_particles)

    print(f"Total trajectories after filtering (min length {config['min_trajectory_length']} frames): {filtered_linked_particles['particle'].nunique()}")

    # Save the raw (pre-judge) link result
    linked_raw_csv_path = final_results_dir / 'linked_particle_trajectories_raw.csv'
    filtered_linked_particles.to_csv(str(linked_raw_csv_path), index=False)
    print(f"Raw linked particle data (pre-judge) saved to {linked_raw_csv_path}")

    # --- RF JUDGE (optional) ---
    judged_df = filtered_linked_particles
    merge_log_df = pd.DataFrame()
    num_merged = 0

    if CONFIG.get('judge_enabled', True):
        print("\n--- Applying RF Judge to correct re-identifications ---")
        judged_df, merge_log_df, num_merged = _apply_rf_judge_and_merge(filtered_linked_particles, CONFIG)
        print(f"RF Judge finished: merges performed = {num_merged}")
    else:
        print("RF Judge disabled; skipping ID corrections.")

    # Save judged output + log
    linked_judged_csv_path = final_results_dir / 'linked_particle_trajectories_judged.csv'
    judged_df.to_csv(str(linked_judged_csv_path), index=False)
    print(f"Judged particle data saved to {linked_judged_csv_path}")

    if not merge_log_df.empty:
        judge_log_path = final_results_dir / 'judge_merge_log.csv'
        merge_log_df.to_csv(str(judge_log_path), index=False)
        print(f"Judge merge log saved to {judge_log_path}")

    # for downstream steps, use the judged IDs
    final_linked_for_downstream = judged_df

    # --- MSD Calculation ---
    msd_df = pd.DataFrame()
    if not final_linked_for_downstream.empty:
        msd_df = calculate_msd(final_linked_for_downstream, config['pixelsize_nm'], config['frame_rate_hz'], config['msd_max_lag_frames'])
        if not msd_df.empty:
            msd_csv_path = final_results_dir / 'msd_results.csv'
            msd_df.to_csv(str(msd_csv_path), index=False)
            print(f"MSD results saved to {msd_csv_path}")

            print("\n--- Generating MSD plot ---")
            Illustration.plot_individual_msd_curves(msd_df, final_results_dir, video_base_name, config['pixelsize_nm'])
            print("MSD plot generation complete.")

            print("\n--- Generating Detected Photons Over Time plot ---")
            Illustration.plot_detected_photons_over_time(
                final_linked_for_downstream,
                final_results_dir,
                video_base_name,
                config['frame_rate_hz']
            )
            print("Detected Photons Over Time plot generation complete.")
    else:
        print("No linked particles found after judge; skipping MSD calculation and plotting.")

    print("\n--- Generating a simple trajectory plot ---")
    try:
        if not filtered_linked_particles.empty:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot each trajectory individually so we can assign a label for the legend
            unique_particles = final_linked_for_downstream['particle'].unique()
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            for i, particle_id in enumerate(unique_particles):
                trajectory = final_linked_for_downstream[final_linked_for_downstream['particle'] == particle_id]
                ax.plot(trajectory['x'], trajectory['y'],
                        label=f'ID {particle_id}',
                        color=colors[i % len(colors)])

            ax.set_title('Particle Trajectories')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.grid(True)
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)
            ax.set_aspect('equal', adjustable='box')

            # Add the legend to the plot, placing it outside the main plot area
            ax.legend(title='Particle ID', bbox_to_anchor=(1.05, 1), loc='upper left')

            traj_plot_path = final_results_dir / 'particle_trajectories_with_legend.png'
            fig.savefig(str(traj_plot_path), bbox_inches='tight') # Use bbox_inches to ensure legend is not cut off
            print(f"Trajectory plot with legend saved to {traj_plot_path}")

            if config['show_final_trajectory_plot']:
                plt.show()

            plt.close(fig)
        else:
            print("No trajectories to plot after filtering.")
    except Exception as e:
        print(f"Error generating trajectory plot with legend: {e}")

    return True, final_linked_for_downstream

def evaluate_tracking_performance(judged_df, ground_truth_df, video_base_name, output_dir, distance_threshold=3.0):
    """
    Evaluates detection metrics (Precision/Recall/F1) using pre-linked data, 
    and tracking metrics (Fragmentation/False Linkage/Completeness/Assoc F1) using judged trajectories.
    """
    print(f"\n--- Evaluating performance for '{video_base_name}' ---")

    if ground_truth_df.empty:
        print("Skipping evaluation: ground truth dataframe is empty.")
        return

    # =========================================================
    # 1. Standard Detection Metrics (from PRE-LINKED data)
    # =========================================================
    pre_link_csv_path = output_dir / 'filtered_particle_data_for_tracking.csv'
    
    if not pre_link_csv_path.exists():
        print(f"Error: Could not find {pre_link_csv_path} for detection evaluation.")
        precision, recall, f1_score = 0, 0, 0
    else:
        detected_df = pd.read_csv(pre_link_csv_path)
        det_df = detected_df[['frame', 'x', 'y']].copy()
        det_df = det_df.rename(columns={'x': 'x_det', 'y': 'y_det'})
        det_df['det_id'] = det_df.index
        
        # Map detected points to GT points
        merged_det = pd.merge(det_df, ground_truth_df, left_on='frame', right_on='Frame', how='left')
        merged_det = merged_det.dropna(subset=['X_pix', 'Y_pix'])
        
        merged_det['distance'] = np.sqrt(
            (merged_det['x_det'] - merged_det['X_pix'])**2 +
            (merged_det['y_det'] - merged_det['Y_pix'])**2
        )

        valid_matches_det = merged_det[merged_det['distance'] <= distance_threshold].sort_values('distance')
        valid_matches_det = valid_matches_det.drop_duplicates(subset=['frame', 'ID']) 
        valid_matches_det = valid_matches_det.drop_duplicates(subset=['frame', 'det_id']) 
        
        TP = len(valid_matches_det)
        total_detected = len(det_df)
        total_gt = len(ground_truth_df)
        
        precision = TP / total_detected if total_detected > 0 else 0
        recall = TP / total_gt if total_gt > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # =========================================================
    # 2. Custom Trajectory Metrics (from POST-JUDGED data)
    # =========================================================
    if judged_df.empty:
        print("Judged dataframe is empty. Tracking metrics will be 0.")
        fragmentation_rate, false_linkage_rate, avg_completeness = 0, 0, 0
        assoc_precision, assoc_recall, assoc_f1 = 0, 0, 0
    else:
        judged_eval_df = judged_df.rename(columns={'x': 'x_tracked', 'y': 'y_tracked', 'particle': 'particle_tracked'})
        
        merged_track = pd.merge(judged_eval_df, ground_truth_df, left_on='frame', right_on='Frame', how='left')
        merged_track['distance'] = np.sqrt(
            (merged_track['x_tracked'] - merged_track['X_pix'])**2 +
            (merged_track['y_tracked'] - merged_track['Y_pix'])**2
        )

        idx = merged_track.groupby(['frame', 'particle_tracked'])['distance'].idxmin()
        matched_track_df = merged_track.loc[idx].reset_index(drop=True)
        matched_track_df.rename(columns={'ID': 'gt_particle'}, inplace=True)
        
        # --- Fragmentation Rate ---
        gt_trajectories = ground_truth_df.groupby('ID')
        gt_fragmented_count = 0
        gt_total_count = ground_truth_df['ID'].nunique()
        
        for gt_id, gt_traj in gt_trajectories:
            matching_tracked_particles = matched_track_df[matched_track_df['gt_particle'] == gt_id]['particle_tracked'].unique()
            if len(matching_tracked_particles) > 1:
                gt_fragmented_count += 1
        
        fragmentation_rate = gt_fragmented_count / gt_total_count if gt_total_count > 0 else 0
        
        # --- False Linkage Rate ---
        tracked_trajectories = judged_eval_df.groupby('particle_tracked')
        false_linkage_count = 0
        tracked_total_count = judged_eval_df['particle_tracked'].nunique()

        for tracked_id, tracked_traj in tracked_trajectories:
            matching_gt_particles = matched_track_df[matched_track_df['particle_tracked'] == tracked_id]['gt_particle'].unique()
            if len(matching_gt_particles) > 1:
                false_linkage_count += 1

        false_linkage_rate = false_linkage_count / tracked_total_count if tracked_total_count > 0 else 0

        # --- Trajectory Completeness ---
        completeness_scores = []
        for gt_id, gt_traj in gt_trajectories:
            gt_len = len(gt_traj)
            if gt_len == 0:
                continue
                
            matched_points = matched_track_df[matched_track_df['gt_particle'] == gt_id]
            total_recovered_len = matched_points['frame'].nunique()
            
            completeness = total_recovered_len / gt_len
            completeness_scores.append(completeness)

        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0

        # =========================================================
        # --- NEW: Association F1-Score ---
        # =========================================================
        # Enforce distance threshold: if distance is too far, it's not a valid GT match
        matched_track_df.loc[matched_track_df['distance'] > distance_threshold, 'gt_particle'] = np.nan
        
        # Evaluate Tracker's Links (Precision)
        pred_tracks = matched_track_df.sort_values(['particle_tracked', 'frame']).copy()
        pred_tracks['next_particle_tracked'] = pred_tracks['particle_tracked'].shift(-1)
        pred_tracks['next_gt_particle'] = pred_tracks['gt_particle'].shift(-1)
        
        # Links made by the tracker
        made_links = pred_tracks[pred_tracks['particle_tracked'] == pred_tracks['next_particle_tracked']]
        
        # True Positive Links: Both ends map to the SAME valid GT ID
        tp_mask = (made_links['gt_particle'] == made_links['next_gt_particle']) & (made_links['gt_particle'].notna())
        TP_links = tp_mask.sum()
        FP_links = (~tp_mask).sum()
        
        # Evaluate Ground Truth Links (Recall)
        gt_tracks = ground_truth_df.sort_values(['ID', 'Frame']).copy()
        # Map tracked particles back to GT points
        gt_mapped = pd.merge(gt_tracks, matched_track_df[['frame', 'gt_particle', 'particle_tracked']], 
                             left_on=['Frame', 'ID'], right_on=['frame', 'gt_particle'], how='left')
        
        gt_mapped['next_ID'] = gt_mapped['ID'].shift(-1)
        gt_mapped['next_particle_tracked'] = gt_mapped['particle_tracked'].shift(-1)
        
        # Links that exist in the Ground Truth
        gt_made_links = gt_mapped[gt_mapped['ID'] == gt_mapped['next_ID']]
        
        # Links successfully tracked: Both ends map to the SAME valid tracked ID
        gt_tp_mask = (gt_made_links['particle_tracked'] == gt_made_links['next_particle_tracked']) & (gt_made_links['particle_tracked'].notna())
        FN_links = (~gt_tp_mask).sum()
        
        assoc_precision = TP_links / (TP_links + FP_links) if (TP_links + FP_links) > 0 else 0
        assoc_recall = TP_links / (TP_links + FN_links) if (TP_links + FN_links) > 0 else 0
        assoc_f1 = (2 * assoc_precision * assoc_recall) / (assoc_precision + assoc_recall) if (assoc_precision + assoc_recall) > 0 else 0

    
    # =========================================================
    # 3. Log Results
    # =========================================================
    metrics = {
        'video': video_base_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fragmentation_rate': fragmentation_rate,
        'false_linkage_rate': false_linkage_rate,
        'avg_trajectory_completeness': avg_completeness,
        'assoc_precision': assoc_precision,
        'assoc_recall': assoc_recall,
        'assoc_f1_score': assoc_f1
    }

    metrics_df = pd.DataFrame([metrics])
    log_path = output_dir / 'evaluation_metrics.csv'
    
    if not log_path.exists():
        metrics_df.to_csv(log_path, index=False)
    else:
        metrics_df.to_csv(log_path, mode='a', header=False, index=False)
    
    print("\n--- Evaluation Results ---")
    print(f"Detection Precision (Pre-Link): {precision:.4f}")
    print(f"Detection Recall (Pre-Link):    {recall:.4f}")
    print(f"Detection F1-Score (Pre-Link):  {f1_score:.4f}")
    print("-" * 26)
    print(f"Fragmentation Rate (Post-Link): {fragmentation_rate:.4f}")
    print(f"False Linkage Rate (Post-Link): {false_linkage_rate:.4f}")
    print(f"Avg Completeness (Post-Link):   {avg_completeness:.4f}")
    print("-" * 26)
    print(f"Assoc Precision (Post-Link):    {assoc_precision:.4f}")
    print(f"Assoc Recall (Post-Link):       {assoc_recall:.4f}")
    print(f"Assoc F1-Score (Post-Link):     {assoc_f1:.4f}")
    print(f"Metrics saved to: {log_path}")
    
    return metrics


# --- Main Pipeline Execution Orchestrator for Batch Mode ---
def run_full_pipeline_batch():
    """
    Orchestrates the entire SMLM particle detection, filtering, and tracking pipeline
    for all .seq files or .tif files in the input directory.
    """
    print("\n███ Starting SMLM Particle Tracking Pipeline (Batch Mode) ███")

    setup_directories() # Ensure top-level directories exist

    # --- UPDATED LOGIC: Find video subdirectories instead of files ---
    video_dirs = sorted([
        d for d in CONFIG['input_dir'].iterdir()
        if d.is_dir() and any(f.suffix.lower() in ('.tif', '.tiff') for f in d.iterdir() if f.is_file())
    ])
    
    if not video_dirs:
        print(f"ERROR: No video subdirectories found in {CONFIG['input_dir']}. Aborting.")
        return

    print(f"\nFound {len(video_dirs)} video folders to process in {CONFIG['input_dir']}.")

    for i, video_dir in enumerate(video_dirs):
        print(f"\n--- Processing Video {i+1}/{len(video_dirs)}: '{video_dir.name}' ---")
        
        video_base_name = video_dir.name
        
        # --- DYNAMICALLY SET DIRECTORIES FOR THIS RUN ---
        temp_video_output_dir_for_this_run = video_dir
        unet_masks_output_dir_for_this_run = CONFIG['unet_masks_dir'] / video_base_name
        filtered_masks_output_dir_for_this_run = CONFIG['filtered_masks_dir'] / video_base_name
        final_results_output_dir_for_this_run = CONFIG['final_results_dir'] / video_base_name

        # --- GT Mask Cache Mode: optionally reuse existing masks and skip U-Net ---
        use_cached_masks = False
        if CONFIG.get('gt_mask_cache_enabled', False):
            use_cached_masks = gt_mask_cache_is_usable(
                temp_video_output_dir_for_this_run,
                unet_masks_output_dir_for_this_run,
                CONFIG.get('gt_mask_cache_glob', '*_predict_mask.png'),
                CONFIG.get('gt_mask_cache_require_full_match', True),
            )

        clean_directories(
            temp_video_output_dir_for_this_run,
            unet_masks_output_dir_for_this_run,
            filtered_masks_output_dir_for_this_run,
            final_results_output_dir_for_this_run,
            CONFIG['clean_previous_run_data'],
            preserve_unet_masks_dir=use_cached_masks
        )
        
        # Re-create directories as needed
        unet_masks_output_dir_for_this_run.mkdir(parents=True, exist_ok=True)
        filtered_masks_output_dir_for_this_run.mkdir(parents=True, exist_ok=True)
        final_results_output_dir_for_this_run.mkdir(parents=True, exist_ok=True)


        print(f"Processing video frames from: {temp_video_output_dir_for_this_run}")
        print(f"U-Net masks for '{video_base_name}' in: {unet_masks_output_dir_for_this_run}")
        print(f"Filtered masks for '{video_base_name}' in: {filtered_masks_output_dir_for_this_run}")
        print(f"Final results for '{video_base_name}' in: {final_results_output_dir_for_this_run}")

        print(f"\n--- Using FIXED normalization statistics for U-Net input for '{video_base_name}' ---")
        print(f"Fixed Normalization Stats: Mean={CONFIG['fixed_norm_mean']:.6f}, Std={CONFIG['fixed_norm_std']:.6f}")

        if use_cached_masks:
            print(f"--- Stage 2: Skipping U-Net prediction for '{video_base_name}' (using cached masks) ---")
            success = True
        else:
            success = run_unet_prediction(
            temp_video_output_dir_for_this_run,
            unet_masks_output_dir_for_this_run,
            CONFIG['unet_model_path'],
            CONFIG['unet_img_scale'],
            CONFIG['unet_threshold'],
            CONFIG['fixed_norm_mean'],
            CONFIG['fixed_norm_std'],
            CONFIG['save_debug_images_unet_stage']
        )
        if not success:
            print(f"Skipping rest of pipeline for '{video_dir.name}': U-Net prediction failed.")
            continue 

        success, judged_df = run_defocusing_and_extraction_and_tracking(
            temp_video_output_dir_for_this_run,
            unet_masks_output_dir_for_this_run,
            filtered_masks_output_dir_for_this_run,
            final_results_output_dir_for_this_run,
            CONFIG,
            video_base_name # Pass the video_base_name here
        )
        if not success:
            print(f"Skipping video overlay for '{video_dir.name}': Filtering/Tracking failed or no particles found.")
            continue
        
        # --- NEW: Run performance evaluation if enabled ---
        if CONFIG['evaluate_tracking_performance']:
            gt_file_path = CONFIG['ground_truth_dir'] / f"{video_base_name}{CONFIG['ground_truth_filename_pattern']}"
            if gt_file_path.exists():
                try:
                    ground_truth_df = pd.read_csv(gt_file_path)
                    evaluate_tracking_performance(judged_df, ground_truth_df, video_base_name, final_results_output_dir_for_this_run)
                except Exception as e:
                    print(f"ERROR: Failed to load or process ground truth file '{gt_file_path}': {e}")
            else:
                print(f"Ground truth file not found at '{gt_file_path}'. Skipping evaluation.")


        print(f"\n███ Finished processing '{video_dir.name}' ███")

    print("\n███ SMLM Particle Tracking Pipeline (Batch Mode) Complete! ███")


# --- Main Pipeline Execution Orchestrator for Single File ---
def run_full_pipeline_single_file():
    """
    Orchestrates the entire SMLM particle detection, filtering, and tracking pipeline
    for a single .seq file selected via GUI.
    """
    print("\n███ Starting SMLM Particle Tracking Pipeline (Single File Mode) ███")

    setup_directories()
    
    selected_input_file = select_seq_file_gui()
    if not selected_input_file:
        print("No file selected. Pipeline aborted.")
        return

    # Check if the selected file is in a video subdirectory
    if selected_input_file.parent.name.startswith('video_'):
        # If so, process the entire parent directory
        video_dir = selected_input_file.parent
        video_base_name = video_dir.name
        temp_video_output_dir_for_this_run = video_dir
    else:
        # If not, assume it's a single file and handle it with the old logic
        video_base_name = selected_input_file.stem
        temp_video_output_dir_for_this_run = CONFIG['temp_video_frames_dir'] / video_base_name
        temp_video_output_dir_for_this_run.mkdir(parents=True, exist_ok=True)
        # Handle SEQ vs TIFF for single file mode
        if selected_input_file.suffix.lower() == '.seq':
            success = run_matlab_preprocessing(
                selected_input_file,
                temp_video_output_dir_for_this_run,
                CONFIG['matlab_exe_path'],
                CONFIG['matlab_scripts_dir'],
                CONFIG['matlab_seq_to_tif_script_name']
            )
            if not success:
                print("Pipeline aborted due to MATLAB preprocessing failure.")
                return
        elif selected_input_file.suffix.lower() == '.tif':
            shutil.copy(selected_input_file, temp_video_output_dir_for_this_run / selected_input_file.name)
        else:
            print(f"Unsupported file type '{selected_input_file.suffix.lower()}'. Pipeline aborted.")
            return

    # Set up other directories based on the video name
    unet_masks_output_dir_for_this_run = CONFIG['unet_masks_dir'] / video_base_name
    filtered_masks_output_dir_for_this_run = CONFIG['filtered_masks_dir'] / video_base_name
    final_results_output_dir_for_this_run = CONFIG['final_results_dir'] / video_base_name

    # --- GT Mask Cache Mode: optionally reuse existing masks and skip U-Net ---
    use_cached_masks = False
    if CONFIG.get('gt_mask_cache_enabled', False):
        use_cached_masks = gt_mask_cache_is_usable(
            temp_video_output_dir_for_this_run,
            unet_masks_output_dir_for_this_run,
            CONFIG.get('gt_mask_cache_glob', '*_predict_mask.png'),
            CONFIG.get('gt_mask_cache_require_full_match', True),
        )

    clean_directories(
        temp_video_output_dir_for_this_run,
        unet_masks_output_dir_for_this_run,
        filtered_masks_output_dir_for_this_run,
        final_results_output_dir_for_this_run,
        CONFIG['clean_previous_run_data'],
        preserve_unet_masks_dir=use_cached_masks
    )
    
    unet_masks_output_dir_for_this_run.mkdir(parents=True, exist_ok=True)
    filtered_masks_output_dir_for_this_run.mkdir(parents=True, exist_ok=True)
    final_results_output_dir_for_this_run.mkdir(parents=True, exist_ok=True)

    print(f"Processing video frames from: {temp_video_output_dir_for_this_run}")
    print(f"U-Net masks for this run will be in: {unet_masks_output_dir_for_this_run}")
    print(f"Filtered masks for this run will be in: {filtered_masks_output_dir_for_this_run}")
    print(f"Final results for this run will be in: {final_results_output_dir_for_this_run}")

    print(f"\n--- Using FIXED normalization statistics for U-Net input ---")
    print(f"Fixed Normalization Stats: Mean={CONFIG['fixed_norm_mean']:.6f}, Std={CONFIG['fixed_norm_std']:.6f}")

    if use_cached_masks:
        print(f"--- Stage 2: Skipping U-Net prediction for '{video_base_name}' (using cached masks) ---")
        success = True
    else:
        success = run_unet_prediction(
        temp_video_output_dir_for_this_run,
        unet_masks_output_dir_for_this_run,
        CONFIG['unet_model_path'],
        CONFIG['unet_img_scale'],
        CONFIG['unet_threshold'],
        CONFIG['fixed_norm_mean'],
        CONFIG['fixed_norm_std'],
        CONFIG['save_debug_images_unet_stage']
    )
    if not success:
        print("Pipeline aborted due to U-Net prediction failure.")
        return

    success, judged_df = run_defocusing_and_extraction_and_tracking(
        temp_video_output_dir_for_this_run,
        unet_masks_output_dir_for_this_run,
        filtered_masks_output_dir_for_this_run,
        final_results_output_dir_for_this_run,
        CONFIG,
        video_base_name
    )
    if not success:
        print("Pipeline aborted due to filtering/tracking failure.")
        return
    
    # --- NEW: Run performance evaluation if enabled ---
    if CONFIG['evaluate_tracking_performance']:
        gt_file_path = CONFIG['ground_truth_dir'] / f"{video_base_name}{CONFIG['ground_truth_filename_pattern']}"
        if gt_file_path.exists():
            try:
                ground_truth_df = pd.read_csv(gt_file_path)
                evaluate_tracking_performance(judged_df, ground_truth_df, video_base_name, final_results_output_dir_for_this_run)
            except Exception as e:
                print(f"ERROR: Failed to load or process ground truth file '{gt_file_path}': {e}")
        else:
            print(f"Ground truth file not found at '{gt_file_path}'. Skipping evaluation.")


    print("\n███ SMLM Particle Tracking Pipeline Finished Successfully! ███")
    print(f"Results for '{video_base_name}' are located in: {final_results_output_dir_for_this_run}")


# --- Main Entry Point ---
if __name__ == "__main__":
    if CONFIG['batch_mode_enabled']:
        run_full_pipeline_batch()
    else:
        run_full_pipeline_single_file()