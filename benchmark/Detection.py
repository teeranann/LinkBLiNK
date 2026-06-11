import os
import csv
import math
from ij import IJ
from ij.plugin import FolderOpener
from fiji.plugin.trackmate import Model, Settings, TrackMate
from fiji.plugin.trackmate.detection import DogDetectorFactory, LogDetectorFactory, ThresholdDetectorFactory, HessianDetectorFactory

# =============================================================================
# 1. USER CONFIGURATION 
# =============================================================================

# Define your main directory (Use forward slashes '/' even on Windows!)
TOP_LEVEL_DIR = "D:/Works/LinkBLiNK/Playground/VideoGeneration/Scenario A"

# Choose your detection mode (1 = DoG, 2 = LoG, 3 = Threshold, 4 = Hessian)
MODE = 4 

# Distance tolerance for matching detected spots to Ground Truth (in pixels)
DISTANCE_TOLERANCE = 3.0 

# --- Detector Parameters ---
# Edit these values to fine-tune each specific algorithm.
PARAMS_DoG = {
    'RADIUS': 5.0,
    'THRESHOLD': 500.0,
    'DO_SUBPIXEL_LOCALIZATION': True,
    'DO_MEDIAN_FILTERING': False
}

PARAMS_LoG = {
    'RADIUS': 5.0,
    'THRESHOLD': 500.0,
    'DO_SUBPIXEL_LOCALIZATION': True
}

PARAMS_THRESHOLD = {
    'INTENSITY_THRESHOLD': 5000.0, 
    'SIMPLIFY_CONTOURS': True
}

PARAMS_HESSIAN = {
    'RADIUS': 5.0,
    'THRESHOLD': 500.0,
    'DO_SUBPIXEL_LOCALIZATION': True
}

# =============================================================================
# 2. AUTOMATION PIPELINE
# =============================================================================

def run_batch_evaluation():
    print("=== Starting Batch Detection Evaluation ===")
    
    # Map modes to human-readable names
    mode_names = {1: "DoG", 2: "LoG", 3: "Threshold", 4: "Mask"}
    detector_name = mode_names.get(MODE, "Unknown")
    print("Mode Selected: " + detector_name)

    # Prepare the output CSV
    output_csv_path = os.path.join(TOP_LEVEL_DIR, "batch_" + detector_name + "_results.csv")
    
    results = [] # Store row data for the CSV

    # Scan the top-level directory for video folders
    for item in os.listdir(TOP_LEVEL_DIR):
        video_dir = os.path.join(TOP_LEVEL_DIR, item)
        
        # Only process if it's a directory and starts with "video_"
        if os.path.isdir(video_dir) and item.startswith("video_"):
            video_name = item
            print("\nProcessing: " + video_name)
            
            # --- Path Resolution ---
            gt_csv_path = os.path.join(TOP_LEVEL_DIR, video_name + "_ground_truth.csv")
            
            if not os.path.exists(gt_csv_path):
                print("  -> SKIPPED: Ground truth CSV not found for " + video_name)
                continue
                
            image_folder = video_dir

            # --- Load Ground Truth ---
            gt_spots = {}
            total_gt = 0
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
                    total_gt += 1

            # --- Load Image Sequence ---
            imp = FolderOpener.open(image_folder, " filter=.tif ")
            if imp is None:
                print("  -> ERROR: Could not load image sequence.")
                continue

            # --- FORCE PIXEL CALIBRATION ---
            cal = imp.getCalibration()
            cal.pixelWidth = 1.0
            cal.pixelHeight = 1.0
            cal.pixelDepth = 1.0
            cal.setUnit("pixel")
            imp.setCalibration(cal)
            
            # --- FORCE TIME DIMENSION (THE FIX) ---
            # FolderOpener defaults to a Z-stack. We must convert it to a Time-lapse.
            stack_size = imp.getStackSize()
            imp.setDimensions(1, 1, stack_size) # (Channels, Z-slices, Frames)

            # --- Configure TrackMate ---
            model = Model()
            settings = Settings(imp)

            if MODE == 1:
                settings.detectorFactory = DogDetectorFactory()
                settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
                settings.detectorSettings['RADIUS'] = PARAMS_DoG['RADIUS']
                settings.detectorSettings['THRESHOLD'] = PARAMS_DoG['THRESHOLD']
                settings.detectorSettings['DO_SUBPIXEL_LOCALIZATION'] = PARAMS_DoG['DO_SUBPIXEL_LOCALIZATION']
            
            elif MODE == 2:
                settings.detectorFactory = LogDetectorFactory()
                settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
                settings.detectorSettings['RADIUS'] = PARAMS_LoG['RADIUS']
                settings.detectorSettings['THRESHOLD'] = PARAMS_LoG['THRESHOLD']
                settings.detectorSettings['DO_SUBPIXEL_LOCALIZATION'] = PARAMS_LoG['DO_SUBPIXEL_LOCALIZATION']
            
            elif MODE == 3:
                settings.detectorFactory = ThresholdDetectorFactory()
                settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
                settings.detectorSettings['INTENSITY_THRESHOLD'] = PARAMS_THRESHOLD['INTENSITY_THRESHOLD']
                settings.detectorSettings['SIMPLIFY_CONTOURS'] = PARAMS_THRESHOLD['SIMPLIFY_CONTOURS']
                
            
            elif MODE == 4:
                settings.detectorFactory = HessianDetectorFactory()
                settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
                settings.detectorSettings['RADIUS'] = PARAMS_HESSIAN['RADIUS']
                settings.detectorSettings['THRESHOLD'] = PARAMS_HESSIAN['THRESHOLD']
                settings.detectorSettings['DO_SUBPIXEL_LOCALIZATION'] = PARAMS_HESSIAN['DO_SUBPIXEL_LOCALIZATION']

            # --- Run Detection ---
            trackmate = TrackMate(model, settings)
            if not trackmate.execDetection():
                print("  -> TrackMate Error: " + str(trackmate.getErrorMessage()))
                continue

            # --- Extract Detected Spots ---
            spots = model.getSpots()
            det_spots = {}
            total_det = 0

            for frame in spots.keySet():
                current_frame = int(frame)
                det_spots[current_frame] = []
                for spot in spots.iterable(frame, False):
                    x = spot.getFeature('POSITION_X')
                    y = spot.getFeature('POSITION_Y')
                    det_id = spot.ID()
                    det_spots[current_frame].append((det_id, x, y))
                    total_det += 1

            # --- Evaluate Matching (TP, FP, FN) ---
            TP = 0
            for frame in det_spots.keys():
                if frame not in gt_spots:
                    continue
                
                pairs = []
                for d_id, dx, dy in det_spots[frame]:
                    for g_id, gx, gy in gt_spots[frame]:
                        dist = math.sqrt((dx - gx)**2 + (dy - gy)**2)
                        if dist <= DISTANCE_TOLERANCE:
                            pairs.append((dist, d_id, g_id))
                
                pairs.sort(key=lambda item: item[0])
                matched_det = set()
                matched_gt = set()
                
                for dist, d_id, g_id in pairs:
                    if d_id not in matched_det and g_id not in matched_gt:
                        matched_det.add(d_id)
                        matched_gt.add(g_id)
                        TP += 1

            # --- Calculate Metrics ---
            FP = total_det - TP
            FN = total_gt - TP

            precision = float(TP) / total_det if total_det > 0 else 0.0
            recall = float(TP) / total_gt if total_gt > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            print("  -> TP: %d | FP: %d | FN: %d" % (TP, FP, FN))
            print("  -> Precision: %.4f | Recall: %.4f | F1: %.4f" % (precision, recall, f1_score))

            # Store for CSV
            results.append([video_name, detector_name, total_gt, total_det, TP, FP, FN, precision, recall, f1_score])

    # --- Export to CSV ---
    if len(results) > 0:
        with open(output_csv_path, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'Detector', 'Total_GT', 'Total_Detected', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1_Score'])
            for row in results:
                writer.writerow(row)
        print("\n=== FINISHED! ===")
        print("Batch results saved to: " + output_csv_path)
    else:
        print("\n=== FINISHED! ===")
        print("No videos were successfully processed.")

# Execute the pipeline
run_batch_evaluation()