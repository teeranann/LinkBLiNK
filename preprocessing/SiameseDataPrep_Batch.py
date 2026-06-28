import pandas as pd
import numpy as np
import os
import cv2
import random
from tqdm import tqdm
import skimage.io as io
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
# Output directories (kept hardcoded as requested)
PATCHES_OUTPUT_DIR = r'E:\AI_Project\TestSample\Siamese\Output\Patches'
PAIRS_OUTPUT_DIR = r'E:\AI_Project\TestSample\Siamese\Output\Pairs'

# Desired size of the square image patches (e.g., 32x32 pixels)
PATCH_SIZE = 32

# Number of positive and negative pairs to generate per unique particle
# Adjust these based on your total particle count and available disk space/time
NUM_POSITIVE_PAIRS_PER_PARTICLE = 50
NUM_NEGATIVE_PAIRS_PER_PARTICLE = 50

# --- Helper Functions ---

def load_image_16bit(image_path):
    """Loads a 16-bit grayscale image."""
    img = io.imread(image_path)
    if img.dtype != np.uint16:
        # print(f"Warning: Image {image_path} is not uint16. Converting to uint16.") # Removed verbose warning
        img = img.astype(np.uint16)
    return img

def extract_patch(image, x, y, patch_size):
    """
    Extracts a square patch centered at (x, y) from the image.
    Handles boundaries by padding with zeros.
    """
    half_patch = patch_size // 2
    height, width = image.shape
    
    x1 = int(x - half_patch)
    y1 = int(y - half_patch)
    x2 = int(x + half_patch)
    y2 = int(y + half_patch)

    padded_patch = np.zeros((patch_size, patch_size), dtype=image.dtype)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(width, x2)
    src_y2 = min(height, y2)

    dst_x1 = max(0, -x1)
    dst_y1 = max(0, -y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        padded_patch[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    
    return padded_patch

def normalize_patch(patch):
    """Normalizes a patch to 0-1 float range."""
    patch_float = patch.astype(np.float32)
    max_val = np.max(patch_float)
    if max_val > 0:
        return patch_float / max_val
    return patch_float

# --- Main Script ---

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # --- NEW: GUI for selecting root directories ---
    RESULTS_ROOT_DIR = filedialog.askdirectory(
        title="Select the ROOT directory containing all 'linked_particle_trajectories.csv' subfolders (e.g., '.../result/')"
    )
    if not RESULTS_ROOT_DIR:
        print("No results root directory selected. Exiting.")
        exit()
    print(f"Selected Results Root Directory: {RESULTS_ROOT_DIR}")

    FRAMES_ROOT_DIR = filedialog.askdirectory(
        title="Select the ROOT directory containing all raw '.tif' image subfolders (e.g., '.../temp_video_frames/')"
    )
    if not FRAMES_ROOT_DIR:
        print("No frames root directory selected. Exiting.")
        exit()
    print(f"Selected Frames Root Directory: {FRAMES_ROOT_DIR}")
    # --- END NEW GUI ---

    os.makedirs(PATCHES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PAIRS_OUTPUT_DIR, exist_ok=True)

    # --- NEW: Master lists to accumulate data across all videos ---
    master_all_particle_patches_info = {} # Global: {global_p_id: [(frame_num, video_subfolder_name, patch_filepath), ...]}
    current_global_particle_id_counter = 0

    # Get list of all video subfolders (assuming they are common between results and frames)
    video_subfolders = sorted([d for d in os.listdir(RESULTS_ROOT_DIR) if os.path.isdir(os.path.join(RESULTS_ROOT_DIR, d))])

    if not video_subfolders:
        print(f"Error: No subfolders found in '{RESULTS_ROOT_DIR}'. Please ensure your videos are in subfolders.")
        exit()

    print(f"\n--- Starting Batch Processing of {len(video_subfolders)} Videos ---")
    
    # Loop through each video subfolder
    for video_idx, video_subfolder_name in enumerate(tqdm(video_subfolders, desc="Processing Videos")):
        current_video_linked_csv_path = os.path.join(RESULTS_ROOT_DIR, video_subfolder_name, 'linked_particle_trajectories.csv')
        current_video_frames_dir = os.path.join(FRAMES_ROOT_DIR, video_subfolder_name)

        print(f"\n--- Processing Video {video_idx+1}/{len(video_subfolders)}: '{video_subfolder_name}' ---")

        # --- Load Linked Trajectories for Current Video ---
        if not os.path.exists(current_video_linked_csv_path):
            print(f"WARNING: CSV not found for '{video_subfolder_name}' at '{current_video_linked_csv_path}'. Skipping this video.")
            continue
        
        df_linked_current_video = pd.read_csv(current_video_linked_csv_path)

        if df_linked_current_video.empty:
            print(f"WARNING: CSV is empty for '{video_subfolder_name}'. Skipping this video.")
            continue

        # --- Prepare Image Naming and Paths for Current Video ---
        first_image_file = None
        if not os.path.exists(current_video_frames_dir):
            print(f"WARNING: Image directory not found for '{video_subfolder_name}' at '{current_video_frames_dir}'. Skipping this video.")
            continue

        for f in os.listdir(current_video_frames_dir):
            if "_frame_" in f and (f.lower().endswith('.tif') or f.lower().endswith('.tiff') or f.lower().endswith('.png')):
                first_image_file = f
                break
        
        if not first_image_file:
            print(f"WARNING: No image files matching 'PREFIX_frame_XXXX.tif/png' found directly in '{current_video_frames_dir}'. Skipping this video.")
            continue

        try:
            parts = first_image_file.split('_frame_')
            video_prefix = parts[0]
            file_extension = os.path.splitext(first_image_file)[1]
            # print(f"DEBUG: Video '{video_subfolder_name}': Auto-detected prefix '{video_prefix}', extension '{file_extension}'")
        except IndexError:
            print(f"ERROR: Could not auto-detect video prefix from sample image '{first_image_file}' in '{current_video_frames_dir}'. Skipping this video.")
            continue

        frame_to_filename_current_video = {
            row['frame']: os.path.join(current_video_frames_dir, f"{video_prefix}_frame_{int(row['frame']):04d}{file_extension}")
            for idx, row in df_linked_current_video[['frame']].drop_duplicates().iterrows()
        }

        loaded_images_cache = {} # Cache for loaded images for current video

        # --- Extract Patches for Current Video ---
        # Remap local particle IDs to global particle IDs for this video
        local_to_global_particle_id_map = {}
        
        unique_particles_local = df_linked_current_video['particle'].unique()
        
        for local_p_id in unique_particles_local:
            particle_df = df_linked_current_video[df_linked_current_video['particle'] == local_p_id].copy()
            
            # --- CRITICAL: Increment global particle ID for each new unique particle ---
            global_p_id = current_global_particle_id_counter
            local_to_global_particle_id_map[local_p_id] = global_p_id
            current_global_particle_id_counter += 1

            master_all_particle_patches_info[global_p_id] = [] # Initialize list for this global particle ID

            for idx, row in particle_df.iterrows():
                frame_num = row['frame']
                x_coord = row['x']
                y_coord = row['y']
                
                image_path = frame_to_filename_current_video.get(frame_num)
                if not image_path:
                    # print(f"DEBUG_WARNING: Image path not found for frame {frame_num} (local particle {local_p_id}). Skipping.")
                    continue
                if not os.path.exists(image_path):
                    print(f"DEBUG_WARNING: Image file NOT FOUND for frame {frame_num} at '{image_path}' (local particle {local_p_id}). Skipping.")
                    continue

                if frame_num not in loaded_images_cache:
                    try:
                        loaded_images_cache[frame_num] = load_image_16bit(image_path)
                    except Exception as e:
                        print(f"DEBUG_ERROR: Could not load image '{image_path}' for frame {frame_num} (local particle {local_p_id}). Error: {e}. Skipping.")
                        continue
                current_image = loaded_images_cache[frame_num]

                try:
                    patch = extract_patch(current_image, x_coord, y_coord, PATCH_SIZE)
                    normalized_patch = normalize_patch(patch)

                    # Create global particle-specific subdirectory for patches
                    # All patches from the same global particle ID go into one folder
                    particle_patch_dir = os.path.join(PATCHES_OUTPUT_DIR, f"particle_{global_p_id}")
                    os.makedirs(particle_patch_dir, exist_ok=True)

                    # Patch filename includes video_subfolder_name and original frame_num for uniqueness
                    patch_filename = os.path.join(particle_patch_dir, f"{video_subfolder_name}_frame_{int(frame_num):04d}_patch.png")
                    cv2.imwrite(patch_filename, (normalized_patch * 255).astype(np.uint8))
                    
                    master_all_particle_patches_info[global_p_id].append((frame_num, video_subfolder_name, patch_filename)) # Store video_subfolder_name for debugging/info
                except Exception as e:
                    print(f"DEBUG_ERROR: Failed to extract/save patch for local particle {local_p_id}, frame {frame_num}. Error: {e}. Skipping detection.")
                    continue
        
        # Free up memory for images from this video after processing all its particles
        del loaded_images_cache 
        print(f"Finished processing patches for '{video_subfolder_name}'. Current global particle count: {current_global_particle_id_counter}")


    # --- FINAL Pair Generation from All Accumulated Data ---
    print(f"\n--- Finished Patch Extraction for All Videos. Total Global Unique Particles: {current_global_particle_id_counter} ---")

    # Filter particles that have at least 2 patches (detections across frames/blinks)
    final_available_global_p_ids = [
        p_id for p_id, patches_list in master_all_particle_patches_info.items() if len(patches_list) >= 2
    ]
    
    if not final_available_global_p_ids:
        print("\nERROR: No global particles found with at least 2 patches for pair generation after processing all videos.")
        print("This means all trajectories across all videos are single-frame detections. Please check Trackpy parameters (memory, search_range).")
        exit()

    print(f"\n--- Generating Training Pairs from {len(final_available_global_p_ids)} Global Particles ---")
    training_pairs = []
    
    # Use tqdm on the list of global particle IDs available for pair generation
    for global_p_id in tqdm(final_available_global_p_ids, desc="Generating pairs"):
        current_global_particle_patches = master_all_particle_patches_info[global_p_id]
        
        # --- Generate Positive Pairs ---
        num_positive_to_generate = min(NUM_POSITIVE_PAIRS_PER_PARTICLE, 
                                       len(current_global_particle_patches) * (len(current_global_particle_patches) - 1) // 2)
        
        for _ in range(num_positive_to_generate):
            patch1_info, patch2_info = random.sample(current_global_particle_patches, 2)
            training_pairs.append({
                'patch1_path': patch1_info[2], # patch_filepath is at index 2 in the tuple
                'patch2_path': patch2_info[2],
                'label': 1
            })

        # --- Generate Negative Pairs ---
        num_negative_to_generate = NUM_NEGATIVE_PAIRS_PER_PARTICLE
        
        for _ in range(num_negative_to_generate):
            other_global_p_id = global_p_id
            attempts = 0
            # Ensure we pick a *different* global particle that also has enough patches
            while (other_global_p_id == global_p_id or 
                   not master_all_particle_patches_info[other_global_p_id] or 
                   len(master_all_particle_patches_info[other_global_p_id]) < 2) and attempts < 100:
                other_global_p_id = random.choice(final_available_global_p_ids)
                attempts += 1
            
            if (other_global_p_id == global_p_id or 
                not master_all_particle_patches_info[other_global_p_id] or 
                len(master_all_particle_patches_info[other_global_p_id]) < 2):
                # This should rarely happen if final_available_global_p_ids is properly populated
                print(f"DEBUG_WARNING: Could not find valid 'other_global_p_id' for negative pair generation for particle {global_p_id}. Skipping negative pair.")
                continue

            patch1_info = random.choice(current_global_particle_patches)
            patch2_info = random.choice(master_all_particle_patches_info[other_global_p_id])
            
            training_pairs.append({
                'patch1_path': patch1_info[2],
                'patch2_path': patch2_info[2],
                'label': 0
            })

    df_pairs = pd.DataFrame(training_pairs)
    
    if df_pairs.empty:
        print("\nError: No training pairs were generated after processing all videos. The 'training_pairs' list is empty.")
        print("This is likely due to filters (e.g., no particles with >=2 patches), or all videos being skipped.")
        print("Review earlier warnings for skipped videos and ensure sufficient data quantity and quality.")
        exit()

    # --- Split Data for Training, Validation, and Testing ---
    # Split now uses final_available_global_p_ids for a clean split
    random.shuffle(final_available_global_p_ids) 
    total_split_particles = len(final_available_global_p_ids)
    train_split = int(0.7 * total_split_particles)
    val_split = int(0.15 * total_split_particles)

    train_p_ids = set(final_available_global_p_ids[:train_split])
    val_p_ids = set(final_available_global_p_ids[train_split : train_split + val_split])
    test_p_ids = set(final_available_global_p_ids[train_split + val_split :])

    print(f"\nDEBUG: Train PIDs ({len(train_p_ids)}): {list(train_p_ids)[:5]}...")
    print(f"DEBUG: Val PIDs ({len(val_p_ids)}): {list(val_p_ids)[:5]}...")
    print(f"DEBUG: Test PIDs ({len(test_p_ids)}): {list(test_p_ids)[:5]}...")

    # Filter df_pairs based on global particle IDs
    # Need to extract global particle ID from patch path: "particle_X" folder name
    train_df = df_pairs[df_pairs['patch1_path'].apply(lambda x: int(os.path.basename(os.path.dirname(x)).split('_')[1]) in train_p_ids)]
    val_df = df_pairs[df_pairs['patch1_path'].apply(lambda x: int(os.path.basename(os.path.dirname(x)).split('_')[1]) in val_p_ids)]
    test_df = df_pairs[df_pairs['patch1_path'].apply(lambda x: int(os.path.basename(os.path.dirname(x)).split('_')[1]) in test_p_ids)]

    print(f"\nGenerated {len(df_pairs)} total pairs.")
    print(f"  Training pairs: {len(train_df)}")
    print(f"  Validation pairs: {len(val_df)}")
    print(f"  Test pairs: {len(test_df)}")

    if train_df.empty or val_df.empty or test_df.empty:
        print("\nDEBUG_WARNING: One or more split dataframes are empty. This can happen with very few global particles.")
        if total_split_particles < 3: # Need at least 3 for a basic split
            print("  Consider acquiring and processing more unique particles for a meaningful split.")

    train_df.to_csv(os.path.join(PAIRS_OUTPUT_DIR, 'train_pairs.csv'), index=False)
    val_df.to_csv(os.path.join(PAIRS_OUTPUT_DIR, 'val_pairs.csv'), index=False)
    test_df.to_csv(os.path.join(PAIRS_OUTPUT_DIR, 'test_pairs.csv'), index=False)

    print(f"Pair datasets saved to {PAIRS_OUTPUT_DIR}")
    print("\nData preparation complete. You can now proceed to Siamese network definition and training.")