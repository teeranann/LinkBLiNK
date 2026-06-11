import os
import csv
import math
from ij import IJ
from ij.plugin import FolderOpener
from fiji.plugin.trackmate import Model, Settings, TrackMate
from fiji.plugin.trackmate.detection import MaskDetectorFactory
from fiji.plugin.trackmate.tracking.kdtree import NearestNeighborTrackerFactory
from fiji.plugin.trackmate.tracking.jaqaman import SimpleSparseLAPTrackerFactory, SparseLAPTrackerFactory

# =============================================================================
# 1. USER CONFIGURATION 
# =============================================================================

TOP_LEVEL_DIR = "D:\Works\LinkBLiNK\Playground\VideoGeneration\Scenario B2"

# Tracking mode (1 = Nearest Neighbor, 2 = Simple LAP, 3 = LAP)
MODE = 3

DISTANCE_TOLERANCE = 3.0 
MAX_LINKING_DISTANCE = 15.0
MAX_GAP_CLOSING_DISTANCE = 15.0
MAX_FRAME_GAP = 3

# =============================================================================
# 2. AUTOMATION PIPELINE
# =============================================================================

def run_batch_tracking_evaluation():
    print("=== Starting Batch Tracking Evaluation ===")
    
    mode_names = {1: "NN", 2: "SimpleLAP", 3: "LAP"}
    tracker_name = mode_names.get(MODE, "Unknown")
    output_csv_path = os.path.join(TOP_LEVEL_DIR, "batch_" + tracker_name + "_tracking_results.csv")
    
    results = []

    for item in os.listdir(TOP_LEVEL_DIR):
        video_dir = os.path.join(TOP_LEVEL_DIR, item)
        
        if os.path.isdir(video_dir) and item.startswith("video_"):
            video_name = item
            print("\nProcessing: " + video_name)
            
            mask_folder = os.path.join(TOP_LEVEL_DIR, "UNetMasks_GT", video_name)
            gt_csv_path = os.path.join(TOP_LEVEL_DIR, video_name + "_ground_truth.csv")
            
            if not os.path.exists(mask_folder) or not os.path.exists(gt_csv_path):
                print("  -> SKIPPED: Missing masks or GT CSV.")
                continue

            # --- 1. Load Ground Truth Data ---
            gt_spots = {}           # Maps frame -> list of (gt_id, x, y)
            gt_traj_lengths = {}    # Maps gt_id -> total frames it exists
            gt_sequences = {}       # Maps gt_id -> list of sorted frames [NEW FOR F1]
            
            with open(gt_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame = int(float(row['Frame']))
                    x = float(row['X_pix'])
                    y = float(row['Y_pix'])
                    gt_id = row['ID']
                    
                    if frame not in gt_spots:
                        gt_spots[frame] = []
                    gt_spots[frame].append((gt_id, x, y))
                    gt_traj_lengths[gt_id] = gt_traj_lengths.get(gt_id, 0) + 1
                    
                    if gt_id not in gt_sequences:
                        gt_sequences[gt_id] = []
                    gt_sequences[gt_id].append(frame)

            # Sort GT frame sequences
            for g_id in gt_sequences:
                gt_sequences[g_id].sort()

            # --- 2. Load Images & Run TrackMate ---
            imp = FolderOpener.open(mask_folder, " filter=.png ")
            if imp is None: continue

            cal = imp.getCalibration()
            cal.pixelWidth, cal.pixelHeight, cal.pixelDepth = 1.0, 1.0, 1.0
            cal.setUnit("pixel")
            imp.setCalibration(cal)
            imp.setDimensions(1, 1, imp.getStackSize())

            model = Model()
            settings = Settings(imp)
            settings.detectorFactory = MaskDetectorFactory()
            settings.detectorSettings = settings.detectorFactory.getDefaultSettings()

            if MODE == 1:
                settings.trackerFactory = NearestNeighborTrackerFactory()
                settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
                settings.trackerSettings['LINKING_MAX_DISTANCE'] = MAX_LINKING_DISTANCE
            elif MODE == 2:
                settings.trackerFactory = SimpleSparseLAPTrackerFactory()
                settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
                settings.trackerSettings['LINKING_MAX_DISTANCE'] = MAX_LINKING_DISTANCE
            elif MODE == 3:
                settings.trackerFactory = SparseLAPTrackerFactory()
                settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
                settings.trackerSettings['LINKING_MAX_DISTANCE'] = MAX_LINKING_DISTANCE
                settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = MAX_GAP_CLOSING_DISTANCE
                settings.trackerSettings['MAX_FRAME_GAP'] = int(MAX_FRAME_GAP)
                settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
                settings.trackerSettings['ALLOW_TRACK_MERGING'] = False

            trackmate = TrackMate(model, settings)
            if not trackmate.checkInput() or not trackmate.process():
                print("  -> ERROR: " + str(trackmate.getErrorMessage()))
                continue

            # --- 3. Extract Tracked Spots ---
            track_model = model.getTrackModel()
            track_ids = track_model.trackIDs(True)
            
            tracked_spots = {}
            total_tracked_trajectories = 0
            track_sequences = {} # Maps track_id -> list of sorted frames [NEW FOR F1]
            
            for track_id in track_ids:
                total_tracked_trajectories += 1
                spots = track_model.trackSpots(track_id)
                for spot in spots:
                    frame = int(spot.getFeature('FRAME'))
                    x = spot.getFeature('POSITION_X')
                    y = spot.getFeature('POSITION_Y')
                    
                    if frame not in tracked_spots:
                        tracked_spots[frame] = []
                    tracked_spots[frame].append((track_id, x, y))
                    
                    if track_id not in track_sequences:
                        track_sequences[track_id] = []
                    track_sequences[track_id].append(frame)

            # Sort Track frame sequences
            for t_id in track_sequences:
                track_sequences[t_id].sort()

            # --- 4. Evaluate Tracking Performance ---
            gt_to_tracks = {}         # gt_id -> set of tracked_ids
            track_to_gts = {}         # tracked_id -> set of gt_ids
            gt_recovered_frames = {}  # gt_id -> set of recovered frames
            
            spot_match = {}           # (frame, track_id) -> gt_id [NEW FOR F1]

            # Match tracked spots to GT spots
            for frame, t_spots in tracked_spots.items():
                if frame not in gt_spots: continue
                g_spots = gt_spots[frame]

                for t_id, tx, ty in t_spots:
                    best_dist = float('inf')
                    best_gt_id = None
                    
                    # Find closest GT particle
                    for g_id, gx, gy in g_spots:
                        dist = math.sqrt((tx - gx)**2 + (ty - gy)**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_gt_id = g_id
                    
                    # Log match if within threshold
                    if best_gt_id is not None and best_dist <= DISTANCE_TOLERANCE:
                        spot_match[(frame, t_id)] = best_gt_id # Save specific spot match for F1
                        
                        if best_gt_id not in gt_to_tracks: gt_to_tracks[best_gt_id] = set()
                        gt_to_tracks[best_gt_id].add(t_id)

                        if t_id not in track_to_gts: track_to_gts[t_id] = set()
                        track_to_gts[t_id].add(best_gt_id)

                        if best_gt_id not in gt_recovered_frames: gt_recovered_frames[best_gt_id] = set()
                        gt_recovered_frames[best_gt_id].add(frame)

            # Calculate Fragmentation
            gt_total_count = len(gt_traj_lengths)
            gt_fragmented_count = sum(1 for g_id, t_ids in gt_to_tracks.items() if len(t_ids) > 1)
            fragmentation_rate = float(gt_fragmented_count) / gt_total_count if gt_total_count > 0 else 0.0

            # Calculate False Linkage
            false_linkage_count = sum(1 for t_id, g_ids in track_to_gts.items() if len(g_ids) > 1)
            false_linkage_rate = float(false_linkage_count) / total_tracked_trajectories if total_tracked_trajectories > 0 else 0.0

            # Calculate Completeness
            comp_scores = []
            for g_id, length in gt_traj_lengths.items():
                if length == 0: continue
                recovered = len(gt_recovered_frames.get(g_id, set()))
                comp_scores.append(float(recovered) / length)
            
            avg_completeness = sum(comp_scores) / len(comp_scores) if len(comp_scores) > 0 else 0.0

            # --- Calculate Association F1-Score ---
            total_tracker_links = 0
            tp_links = 0
            
            # 1. Evaluate Precision (Tracker Links)
            for t_id, frames in track_sequences.items():
                for i in range(len(frames) - 1):
                    total_tracker_links += 1
                    f1 = frames[i]
                    f2 = frames[i+1]
                    
                    gt1 = spot_match.get((f1, t_id))
                    gt2 = spot_match.get((f2, t_id))
                    
                    # A link is True Positive if both ends map to the same GT ID
                    if gt1 is not None and gt2 is not None and gt1 == gt2:
                        tp_links += 1
            
            # 2. Evaluate Recall (Ground Truth Links)
            total_gt_links = 0
            for g_id, frames in gt_sequences.items():
                # A trajectory of N frames has N-1 links
                total_gt_links += max(0, len(frames) - 1)
                
            assoc_precision = float(tp_links) / total_tracker_links if total_tracker_links > 0 else 0.0
            assoc_recall = float(tp_links) / total_gt_links if total_gt_links > 0 else 0.0
            assoc_f1 = (2 * assoc_precision * assoc_recall) / (assoc_precision + assoc_recall) if (assoc_precision + assoc_recall) > 0 else 0.0

            print("  -> Frag: %.4f | False Link: %.4f | Comp: %.4f | Assoc F1: %.4f" % (fragmentation_rate, false_linkage_rate, avg_completeness, assoc_f1))
            results.append([video_name, tracker_name, fragmentation_rate, false_linkage_rate, avg_completeness, assoc_precision, assoc_recall, assoc_f1])

    # --- Export Batch CSV ---
    if len(results) > 0:
        with open(output_csv_path, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'Tracker', 'Fragmentation_Rate', 'False_Linkage_Rate', 'Completeness_Score', 'Assoc_Precision', 'Assoc_Recall', 'Assoc_F1'])
            for row in results:
                writer.writerow(row)
        print("\n=== FINISHED! Batch tracking metrics saved. ===")

run_batch_tracking_evaluation()