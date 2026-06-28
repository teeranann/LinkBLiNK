import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ==========================================
# USER CONFIGURATION
# ==========================================
# Folder containing subfolders (video_001, video_002) with pipeline results
PIPELINE_TOP_DIR = r"D:\Works\AI_Project\Process\05_Judge\Generated Particles"

# Folder containing the ground truth CSVs (video_001_ground_truth.csv)
GT_TOP_DIR = r"D:\Works\AI_Project\Process\05_Judge\Generated Particles"

# Maximum pixel distance to allow a match. 
# Anything further is considered a U-Net false positive and dropped.
MAX_MATCH_DIST = 3.0 
# ==========================================

def run_interceptor():
    print("=== Starting GT Interceptor ===")
    
    # Find all video subfolders in the pipeline directory
    subfolders = [f for f in os.listdir(PIPELINE_TOP_DIR) 
                  if os.path.isdir(os.path.join(PIPELINE_TOP_DIR, f)) and f.startswith("video_")]
    
    if not subfolders:
        print(f"No video subfolders found in {PIPELINE_TOP_DIR}")
        return

    processed_count = 0

    for video_folder in sorted(subfolders):
        print(f"\nProcessing: {video_folder}")
        
        pipeline_dir = os.path.join(PIPELINE_TOP_DIR, video_folder)
        pipeline_csv_path = os.path.join(pipeline_dir, "linked_particle_trajectories_judged.csv")
        
        gt_csv_name = f"{video_folder}_ground_truth.csv"
        gt_csv_path = os.path.join(GT_TOP_DIR, gt_csv_name)
        
        # Check if both required files exist
        if not os.path.exists(pipeline_csv_path):
            print(f"  -> SKIPPED: Missing pipeline CSV: {pipeline_csv_path}")
            continue
        if not os.path.exists(gt_csv_path):
            print(f"  -> SKIPPED: Missing GT CSV: {gt_csv_path}")
            continue

        # Load Data
        df_pipe = pd.read_csv(pipeline_csv_path)
        df_gt = pd.read_csv(gt_csv_path)

        # We will build a new array of true IDs, defaulting to -1 (unmatched)
        new_ids = np.full(len(df_pipe), -1, dtype=int)

        # Get unique frames from the pipeline data
        frames = df_pipe['frame'].unique()
        
        total_matches = 0

        for f in frames:
            # 1. Get pipeline indices and coordinates for this frame
            pipe_mask = df_pipe['frame'] == f
            pipe_idx = df_pipe.index[pipe_mask].tolist()
            if not pipe_idx: continue
            
            pipe_coords = df_pipe.loc[pipe_idx, ['x', 'y']].values

            # 2. Get GT coordinates and IDs for this frame (Only active particles, State == 1)
            gt_frame = df_gt[(df_gt['Frame'] == f) & (df_gt['State'] == 1)]
            if gt_frame.empty: continue
            
            gt_coords = gt_frame[['X_pix', 'Y_pix']].values
            gt_ids = gt_frame['ID'].values

            # 3. Calculate Euclidean distance matrix
            cost_matrix = cdist(pipe_coords, gt_coords, metric='euclidean')

            # 4. Hungarian Algorithm for optimal 1-to-1 matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 5. Apply matches that fall within the acceptable sub-pixel distance
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= MAX_MATCH_DIST:
                    # Map the GT ID to the corresponding pipeline detection
                    actual_df_index = pipe_idx[r]
                    new_ids[actual_df_index] = gt_ids[c]
                    total_matches += 1

        # Replace the flawed pipeline track IDs with the perfect GT IDs
        df_pipe['particle'] = new_ids

        # Drop any pipeline detections that couldn't be matched to a real GT particle
        # (These are U-Net false positives and will just confuse the Random Forest)
        dropped_fp = (df_pipe['particle'] == -1).sum()
        df_clean = df_pipe[df_pipe['particle'] != -1].copy()
        
        # Save a backup of the original just in case
        backup_path = os.path.join(pipeline_dir, "linked_particle_trajectories_judged.bak")
        if not os.path.exists(backup_path):
            pd.read_csv(pipeline_csv_path).to_csv(backup_path, index=False)

        # Overwrite the judged CSV so `RF gen.txt` finds it natively
        df_clean.to_csv(pipeline_csv_path, index=False)
        
        print(f"  -> Matched {total_matches} detections to GT.")
        print(f"  -> Dropped {dropped_fp} false positive detections.")
        print(f"  -> Overwrote: {pipeline_csv_path}")
        
        processed_count += 1

    print(f"\n=== FINISHED! Successfully injected GT into {processed_count} videos. ===")

if __name__ == '__main__':
    run_interceptor()