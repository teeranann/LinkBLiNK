import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tifffile import imwrite, imread
import os
import glob
import random
import json
from PIL import Image

# --- Configuration Parameters ---
# General Generation
NUM_VIDEOS_TO_GENERATE = 50 # Total videos (folders) to create
OUTPUT_ROOT_DIR = 'D:\\Works\\LinkBLiNK\\Playground\\VideoGeneration\\Scenario AX6'
BACKGROUND_DIR = 'D:\\Works\\LinkBLiNK\\Playground\\VideoGeneration\\Background' # Folder containing real background TIFFs

# Video properties
NUM_FRAMES = 500   # Total frames per video
PIXEL_SIZE_UM = 0.0359 # Micrometers per pixel (for diffusion calculation)
BIT_DEPTH = 16     # Camera bit depth (0 to 65535)
FRAME_RATE = 0.01 # Frame rate in seconds (e.g., 30 FPS)
MAX_VAL = (2**BIT_DEPTH) - 1

# Particle properties
NUM_PARTICLES = 2  # Scenario A default: focus on 2 interacting particles
# NEW: Define a range for initial particle intensity
INTENSITY_PEAK_RANGE = (8000, 30000)
# NEW: Photobleaching decay rate (0.001 means 0.1% intensity loss per frame)
BLEACHING_DECAY_RATE = 0.001
# NEW: Define a range for particle size (PSF STD)
PSF_STD_PIXELS_RANGE = (0.8, 4.0) # Standard deviation of the Gaussian PSF (in pixels)
DIFFUSION_COEFF = 0.086 # Diffusion coefficient (um^2/s) - Matches your 0.07 validation

# Realism Parameters
ELLIPTICITY_MAX = 0.15 # Max random variation for sigma_x/sigma_y (e.g., 1.0 +/- 0.15)
TAIL_FRACTION = 0.08  # Fraction of total intensity in the wide tail (3%)
TAIL_FACTOR = 4.0    # Tail sigma is this factor times the main sigma (e.g., 3x wider)

# Photophysics (Blinking)
P_ON = 0.0  # Scenario A: disable random blinking; use forced gap instead  # Probability of a particle being ON if it was OFF (reappearance)
P_OFF = 0.0  # Scenario A: disable random blinking; use forced gap instead # Probability of a particle being OFF if it was ON (disappearance)

# --- NEW FEATURES ---
# Use 999 for a chance to be a random value (0-1)
# 1. Z-axis Movement (Mid-frame disappearance)
ENABLE_Z_MOVEMENT = False  # Scenario A: keep disappearances controlled
Z_PROB_DISAPPEAR = 0.005 # Probability a particle disappears per frame due to Z-axis movement
Z_PROB_REAPPEAR = 0.8  # Probability it will eventually reappear after disappearing due to Z-axis movement
Z_REAPPEAR_RANGE = (10, 50) # Frames to wait before a Z-disappeared particle can reappear

# 2. Boundary Handling
ENABLE_BOUNDARY_HANDLING = True
# CHANGED: Removed BOUNDARY_PROB_DISAPPEAR. Now a single check determines
# if a particle reappears or disappears permanently.
BOUNDARY_PROB_REAPPEAR = 0.75 # Probability a particle reappears after leaving the boundary
BOUNDARY_REAPPEAR_RANGE = (5, 20) # Frames to wait before a particle can reappear after leaving the boundary

# 3. Particle Spawning (to prevent empty frames)
ENABLE_PARTICLE_SPAWNING = False
MIN_VISIBLE_PARTICLES = 1 # Minimum number of particles to ensure are always present

# Noise
NOISE_STD = 5.0 # Standard deviation for added Gaussian noise


# =========================
# GT MASK EXPORT (for pipeline mask-cache mode)
# =========================
GENERATE_GT_MASKS = True

# IMPORTANT: set this to the SAME root as pipeline CONFIG['unet_masks_dir']
UNET_MASK_ROOT_DIR = r'D:\Works\LinkBLiNK\Playground\VideoGeneration\Scenario AX6\UNetMasks_GT'

GT_MASK_SUFFIX = '_predict_mask.png'   # matches pipeline naming
GT_MASK_ON_VALUE = 255                 # binary mask: 0 background, 255 foreground

# radius ≈ k*sigma (in pixels). 2.0–2.5 usually looks close to U-Net masks.
GT_MASK_RADIUS_SIGMA_MULT = 2.2
GT_MASK_MIN_RADIUS_PX = 1

def _paint_particle_mask(mask_u8: np.ndarray, x: float, y: float, sigma_px: float):
    """Paint a filled disk at (x,y) with radius proportional to sigma."""
    r = int(np.ceil(GT_MASK_RADIUS_SIGMA_MULT * float(sigma_px)))
    r = max(r, GT_MASK_MIN_RADIUS_PX)

    h, w = mask_u8.shape
    xmin = max(0, int(np.floor(x - r)))
    xmax = min(w - 1, int(np.ceil(x + r)))
    ymin = max(0, int(np.floor(y - r)))
    ymax = min(h - 1, int(np.ceil(y + r)))
    if xmin > xmax or ymin > ymax:
        return

    yy, xx = np.ogrid[ymin:ymax + 1, xmin:xmax + 1]
    rr2 = (xx - x) ** 2 + (yy - y) ** 2
    sub = mask_u8[ymin:ymax + 1, xmin:xmax + 1]
    sub[rr2 <= (r * r)] = GT_MASK_ON_VALUE

def _save_mask_png(mask_u8: np.ndarray, out_path: str):
    Image.fromarray(mask_u8).save(out_path)


# --- Scenario A: close approach + forced blink gap ---
ENABLE_SCENARIO_A = True

# Which 2 particles are used for the scenario (0-indexed inside the code; ID in GT will be +1)
SCENARIO_A_IDX_A = 0  # Particle A: will disappear and then reappear near B
SCENARIO_A_IDX_B = 1  # Particle B: stays visible throughout

# This should roughly match (or be slightly smaller than) your tracker's linking radius.
SCENARIO_A_LINK_RADIUS_PIXELS = 6.0

# Minimum separation between the two tracks at the closest approach (must be < linking radius).
SCENARIO_A_CLOSE_PASS_DISTANCE_PIXELS = 20

# Forced blink gap length for particle A (frames). Pick N here.
SCENARIO_A_GAP_LENGTH_RANGE = (100, 102)

# Where the gap happens (centered around the close-approach moment).
SCENARIO_A_GAP_CENTER_FRAME = None  # None -> uses NUM_FRAMES//2 with small jitter
SCENARIO_A_GAP_CENTER_JITTER = 10   # +/- frames around mid-video

# After A comes back, keep it very close to B for a few frames to create ambiguity.
SCENARIO_A_REAPPEAR_NEAR_B_FRAMES = 5
SCENARIO_A_REAPPEAR_MIN_OFFSET = 1  # Minimum 1.2x the link radius (approx 7.2 pixels away)
SCENARIO_A_REAPPEAR_MAX_OFFSET = 2  # Maximum 2.0x the link radius (approx 12.0 pixels away)

# Deterministic path styling (keeps A & B safely in frame)
SCENARIO_A_PATH_MARGIN_PIXELS = 30
SCENARIO_A_PATH_Y_JITTER = 10
SCENARIO_A_POSITION_JITTER_STD_SCALE_RANGE = (0.15, 0.45)  # per-video random scale; multiply by step_std_pixels (diffusion-derived)
SCENARIO_A_CLOSE_FRAME_FRAC_RANGE = (0.30, 0.70)  # close-approach occurs around this fraction of the video
SCENARIO_A_CENTER_DRIFT_AMPLITUDE_RANGE = (0.0, 6.0)  # pixels; common-mode drift to add gentle curvature
SCENARIO_A_CENTER_DRIFT_PERIOD_RANGE = (200.0, 900.0) # frames; sinusoid period for curvature

# --- Core Functions ---

def get_chance_value(prob):
    """Returns the input probability or a random float if the value is 999."""
    return random.random() if prob == 999 else prob

def load_background_images(bg_dir):
    """Loads all 16-bit TIFF background images from the specified directory."""
    tiff_files = glob.glob(os.path.join(bg_dir, '*.tif'))
    if not tiff_files:
        print(f"ERROR: No TIFF files found in '{bg_dir}'. Using a constant background.")
        return None, (152, 150)
    
    sample_img = imread(tiff_files[0])
    if len(sample_img.shape) == 3:
        bg_list = [sample_img[i].astype(np.uint16) for i in range(sample_img.shape[0])]
    else:
        bg_list = [sample_img.astype(np.uint16)]

    for f in tiff_files[1:]:
        img = imread(f)
        if len(img.shape) == 3:
            bg_list.extend([img[i].astype(np.uint16) for i in range(img.shape[0])])
        else:
            bg_list.append(img.astype(np.uint16))

    print(f"Loaded {len(bg_list)} background frames for sampling.")
    if bg_list:
        return bg_list, bg_list[0].shape
    else:
        return None, (152, 150)

def generate_psf(x, y, intensity, width, height, sigma):
    """
    Generates a realistic Gaussian Point Spread Function (PSF) with
    ellipticity and a low-level broad tail component.
    """
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    ellipticity_factor = np.random.uniform(1 - ELLIPTICITY_MAX, 1 + ELLIPTICITY_MAX)
    sigma_x = sigma * ellipticity_factor
    sigma_y = sigma / ellipticity_factor
    
    main_intensity = intensity * (1 - TAIL_FRACTION)
    core_gaussian = main_intensity * np.exp(-(
        ((xx - x)**2 / (2 * sigma_x**2)) + 
        ((yy - y)**2 / (2 * sigma_y**2))
    ))
    
    tail_sigma = sigma * TAIL_FACTOR
    tail_intensity = intensity * TAIL_FRACTION
    tail_gaussian = tail_intensity * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * tail_sigma**2))
    
    psf = core_gaussian + tail_gaussian
    return psf

def run_single_simulation(video_id, bg_images, frame_dims):
    """Simulates particle movement and generates one video stack."""
    
    FRAME_HEIGHT, FRAME_WIDTH = frame_dims
    
    # Initialize Particle States and Properties
    # State: 0=OFF, 1=ON, 2=OUT_OF_FOCUS(Z), 3=OUT_OF_BOUNDS, -1=PERMANENTLY_GONE
    positions = np.random.uniform(50, FRAME_WIDTH - 50, size=(NUM_PARTICLES, 2))
    states = np.random.randint(0, 2, size=NUM_PARTICLES)
    
    # NEW: Assign a random initial intensity and PSF sigma to each particle
    initial_intensities = np.random.uniform(INTENSITY_PEAK_RANGE[0], INTENSITY_PEAK_RANGE[1], size=NUM_PARTICLES)
    particle_intensities = initial_intensities.copy()
    psf_sigmas = np.random.uniform(PSF_STD_PIXELS_RANGE[0], PSF_STD_PIXELS_RANGE[1], size=NUM_PARTICLES)
    
    # NEW: Track how long each particle has been visible for photobleaching
    # time_on should track total exposure, regardless of temporary disappearances.
    time_on = np.zeros(NUM_PARTICLES, dtype=int) 
    reappear_in_frames = np.zeros(NUM_PARTICLES, dtype=int)
    
    step_std_pixels = np.sqrt(2 * DIFFUSION_COEFF * FRAME_RATE) / PIXEL_SIZE_UM


    # --- Scenario A setup ---
    if ENABLE_SCENARIO_A:
        # Force both particles to exist and stay visible (except the forced gap on A).
        states[:] = 1

        # Give A and B different signatures (helps your Siamese + FWHM/Ibcnt judge).
        # A: slightly smaller PSF and brighter; B: larger PSF and dimmer (on average).
        psf_sigmas[SCENARIO_A_IDX_A] = np.clip(np.random.uniform(PSF_STD_PIXELS_RANGE[0], PSF_STD_PIXELS_RANGE[0] + 0.8),
                                              PSF_STD_PIXELS_RANGE[0], PSF_STD_PIXELS_RANGE[1])
        psf_sigmas[SCENARIO_A_IDX_B] = np.clip(np.random.uniform(PSF_STD_PIXELS_RANGE[1] - 0.8, PSF_STD_PIXELS_RANGE[1]),
                                              PSF_STD_PIXELS_RANGE[0], PSF_STD_PIXELS_RANGE[1])
        initial_intensities[SCENARIO_A_IDX_A] = np.random.uniform(INTENSITY_PEAK_RANGE[1] * 0.75, INTENSITY_PEAK_RANGE[1])
        initial_intensities[SCENARIO_A_IDX_B] = np.random.uniform(INTENSITY_PEAK_RANGE[0], INTENSITY_PEAK_RANGE[0] * 1.25)
        particle_intensities[:] = initial_intensities.copy()

        
        # --- Scenario A path generation (more natural) ---
        # Instead of drawing two straight-ish paths that obviously "aim" at each other,
        # we generate two smooth random walks (Brownian + velocity persistence),
        # then CUT OUT a window of length NUM_FRAMES that *happens* to contain a close approach.
        # This keeps the overall trajectories looking organic while still guaranteeing Scenario A.

        margin = int(SCENARIO_A_PATH_MARGIN_PIXELS)
        frac_lo, frac_hi = SCENARIO_A_CLOSE_FRAME_FRAC_RANGE
        close_frame = int(np.random.randint(int(frac_lo * NUM_FRAMES), max(int(frac_lo * NUM_FRAMES) + 1, int(frac_hi * NUM_FRAMES))))
        close_frame = int(np.clip(close_frame, 1, NUM_FRAMES - 2))

        # Desired closeness band at the "close_frame"
        d_target = float(SCENARIO_A_CLOSE_PASS_DISTANCE_PIXELS) * np.random.uniform(0.85, 1.15)
        d_max = min(d_target * 1.25, float(SCENARIO_A_LINK_RADIUS_PIXELS) * 0.95)
        d_min = max(1.0, d_target * 0.65)

        # Motion amplitude tuning (per video)
        jitter_scale = float(np.random.uniform(*SCENARIO_A_POSITION_JITTER_STD_SCALE_RANGE))
        walk_step_std = float(step_std_pixels) * jitter_scale

        # Make trajectories smoother than pure Brownian
        #vel_rho = 0.93
        #vel_noise_std = walk_step_std * 0.12
        #step_noise_std = walk_step_std * 0.85

        # This is Brownian
        vel_rho = 0
        vel_noise_std = 0
        step_noise_std = walk_step_std * 1

# --- NEW GUARANTEED COLLISION GENERATOR (WITH SPAWN DISTANCE CHECK) ---
        # Common stage drift (applies to BOTH particles)
        drift_amp = float(np.random.uniform(*SCENARIO_A_CENTER_DRIFT_AMPLITUDE_RANGE))
        drift_period = float(np.random.uniform(*SCENARIO_A_CENTER_DRIFT_PERIOD_RANGE))
        drift_phase = float(np.random.uniform(0, 2 * np.pi))
        drift_dir = np.random.normal(0, 1, size=2)
        drift_dir = drift_dir / (np.linalg.norm(drift_dir) + 1e-9)
        drift_t = (drift_amp * np.sin((2 * np.pi * np.arange(NUM_FRAMES) / drift_period) + drift_phase))[:, None] * drift_dir[None, :]

        def _step_particle(pos, v, t_idx_prev, t_idx_curr):
            """Calculates the next smooth position and reflects off boundaries if needed."""
            v_new = vel_rho * v + np.random.normal(0, vel_noise_std, size=2)
            step = np.random.normal(0, step_noise_std, size=2)
            drift_delta = drift_t[t_idx_curr] - drift_t[t_idx_prev]
            pos_new = pos + v_new + step + drift_delta
            
            lo_x, hi_x = margin, FRAME_WIDTH - margin
            lo_y, hi_y = margin, FRAME_HEIGHT - margin
            if pos_new[0] < lo_x:
                pos_new[0] = lo_x + (lo_x - pos_new[0])
                v_new[0] *= -0.6
            elif pos_new[0] > hi_x:
                pos_new[0] = hi_x - (pos_new[0] - hi_x)
                v_new[0] *= -0.6
            if pos_new[1] < lo_y:
                pos_new[1] = lo_y + (lo_y - pos_new[1])
                v_new[1] *= -0.6
            elif pos_new[1] > hi_y:
                pos_new[1] = hi_y - (pos_new[1] - hi_y)
                v_new[1] *= -0.6
            return pos_new, v_new

        # Define how far apart they must be at the very beginning of the video
        MIN_SPAWN_DISTANCE_PIXELS = 20.0 
        
        valid_paths = False
        while not valid_paths:
            # 1. Generate Particle B's full path normally
            posB_base_path = np.zeros((NUM_FRAMES, 2), dtype=float)
            posB_base_path[0] = np.array([np.random.uniform(margin, FRAME_WIDTH - margin),
                                          np.random.uniform(margin, FRAME_HEIGHT - margin)])
            vB = np.random.normal(0, walk_step_std * 0.20, size=2)
            for t in range(1, NUM_FRAMES):
                posB_base_path[t], vB = _step_particle(posB_base_path[t-1], vB, t-1, t)

            # 2. Force Particle A to meet Particle B exactly at 'close_frame'
            posA_base_path = np.zeros((NUM_FRAMES, 2), dtype=float)
            theta = np.random.uniform(0, 2 * np.pi)
            meet_offset = np.array([d_target * np.cos(theta), d_target * np.sin(theta)])
            posA_base_path[close_frame] = posB_base_path[close_frame] + meet_offset

            # 3. Generate Particle A's path FORWARD from the meet point
            vA_fwd = np.random.normal(0, walk_step_std * 0.20, size=2)
            for t in range(close_frame + 1, NUM_FRAMES):
                posA_base_path[t], vA_fwd = _step_particle(posA_base_path[t-1], vA_fwd, t-1, t)

            # 4. Generate Particle A's path BACKWARD from the meet point
            vA_bwd = np.random.normal(0, walk_step_std * 0.20, size=2)
            for t in range(close_frame - 1, -1, -1):
                posA_base_path[t], vA_bwd = _step_particle(posA_base_path[t+1], vA_bwd, t+1, t)

            # 5. Check if the randomly generated start positions are safely separated
            start_distance = np.linalg.norm(posA_base_path[0] - posB_base_path[0])
            if start_distance >= MIN_SPAWN_DISTANCE_PIXELS:
                valid_paths = True  # Break the loop and keep these paths!

        # Close-pass distance at the chosen frame (for metadata / sanity-checking)
        d_close = float(np.linalg.norm(posA_base_path[close_frame] - posB_base_path[close_frame]))
        meet = 0.5 * (posA_base_path[close_frame] + posB_base_path[close_frame])
        # --- END OF NEW GENERATOR ---

# Forced gap timing

        gap_len = random.randint(*SCENARIO_A_GAP_LENGTH_RANGE)
        if SCENARIO_A_GAP_CENTER_FRAME is None:
            gap_center = int(close_frame + random.randint(-SCENARIO_A_GAP_CENTER_JITTER, SCENARIO_A_GAP_CENTER_JITTER))
        else:
            gap_center = int(SCENARIO_A_GAP_CENTER_FRAME)

        gap_center = int(np.clip(gap_center, 1, NUM_FRAMES - 2))

        gap_start = max(0, gap_center - (gap_len // 2))
        gap_end = min(NUM_FRAMES, gap_start + gap_len)  # exclusive

        # We are no longer hijacking A's coordinates. 
        # A will naturally follow its pre-calculated Brownian path (posA_base_path) after the gap.
        
        # Keep these dummy variables just so your JSON metadata export doesn't crash
        reappear_offset = np.array([0.0, 0.0]) 

        # Recompute close-pass distance at the chosen frame (for metadata)
        d_close = float(np.linalg.norm(posA_base_path[close_frame] - posB_base_path[close_frame]))
        meet = 0.5 * (posA_base_path[close_frame] + posB_base_path[close_frame])

        # gap_start = max(0, gap_center - (gap_len // 2))
        # gap_end = min(NUM_FRAMES, gap_start + gap_len)  # exclusive

        # Offset for "A reappears near B" (kept constant within a video)
        
        # min_off = SCENARIO_A_LINK_RADIUS_PIXELS * float(SCENARIO_A_REAPPEAR_MIN_OFFSET)
        # max_off = SCENARIO_A_LINK_RADIUS_PIXELS * float(SCENARIO_A_REAPPEAR_MAX_OFFSET)

        # min_off = d_target * 0.90
        # max_off = d_target * 1.10

        # theta = np.random.uniform(0, 2*np.pi)
        # r = np.random.uniform(min_off, max_off)  # Now picks a safe distance between min and max!
        # reappear_offset = np.array([r*np.cos(theta), r*np.sin(theta)], dtype=float)

        # # Save a tiny metadata file inside each video folder (handy for debugging).

        # # Apply the "reappear near B" trap directly into the precomputed A path:
        # # - During [gap_start, gap_end): A will be forced OFF (in the frame loop).
        # # - For a few frames after gap_end, force A positions to be very near B.
        # # - After that, keep A's original random-walk continuation but shifted so it connects smoothly.
        # re_end = int(min(NUM_FRAMES, gap_end + SCENARIO_A_REAPPEAR_NEAR_B_FRAMES))
        # re_end = int(max(re_end, gap_end))

        # _origA = posA_base_path.copy()
        # if re_end > gap_end:
        #     for t in range(gap_end, re_end):
        #         posA_base_path[t] = posB_base_path[t] + reappear_offset + np.random.normal(0, walk_step_std * 0.10, size=2)

        #     # Shift the remaining tail of A so it continues smoothly from the forced reappearance endpoint.
        #     if re_end < NUM_FRAMES:
        #         shift = posA_base_path[re_end - 1] - _origA[re_end - 1]
        #         posA_base_path[re_end:] = _origA[re_end:] + shift
        #         posA_base_path[re_end:, 0] = np.clip(posA_base_path[re_end:, 0], margin, FRAME_WIDTH - margin)
        #         posA_base_path[re_end:, 1] = np.clip(posA_base_path[re_end:, 1], margin, FRAME_HEIGHT - margin)

        # # Recompute close-pass distance at the chosen frame (for metadata)
        # d_close = float(np.linalg.norm(posA_base_path[close_frame] - posB_base_path[close_frame]))
        # meet = 0.5 * (posA_base_path[close_frame] + posB_base_path[close_frame])

        # scenario_meta = dict(
        #             scenario="A",
        #             A_index=int(SCENARIO_A_IDX_A),
        #             B_index=int(SCENARIO_A_IDX_B),
        #             link_radius_pixels=float(SCENARIO_A_LINK_RADIUS_PIXELS),

        #             # Geometry / selection (natural random walks + window cutting)
        #             close_frame=int(close_frame),
        #             close_pass_distance_pixels=float(d_close),
        #             close_band_min_pixels=float(d_min),
        #             close_band_max_pixels=float(d_max),
        #             meet_point_pixels=[float(meet[0]), float(meet[1])],
        #             A_minus_B_at_close_pixels=[
        #                 float(posA_base_path[close_frame, 0] - posB_base_path[close_frame, 0]),
        #                 float(posA_base_path[close_frame, 1] - posB_base_path[close_frame, 1]),
        #             ],
        #             long_frames=int(NUM_FRAMES),
        #             window_start_in_long=0,
        #             jitter_scale=float(jitter_scale),
        #             drift_amp_pixels=float(drift_amp),
        #             drift_period_frames=float(drift_period),

        #             # Trap timing
        #             gap_start_frame=int(gap_start),
        #             gap_end_frame=int(gap_end),
        #             gap_length_frames=int(gap_end - gap_start),
        #             reappear_near_B_frames=int(SCENARIO_A_REAPPEAR_NEAR_B_FRAMES),
        #             reappear_offset_pixels=[float(reappear_offset[0]), float(reappear_offset[1])],
        #         )

    else:
        # Placeholders for non-scenario runs
        posA_base_path = posB_base_path = None
        close_frame = None
        d_close = None
        jitter_scale = None
        gap_start = gap_end = -1
        reappear_offset = np.zeros(2, dtype=float)
        scenario_meta = None
    
    video_dir = os.path.join(OUTPUT_ROOT_DIR, f'video_{video_id:03d}')
    os.makedirs(video_dir, exist_ok=True)
    
    # GT mask folder (mirrors pipeline unet_masks_dir/video_###)
    mask_video_dir = None
    if GENERATE_GT_MASKS:
        mask_video_dir = os.path.join(UNET_MASK_ROOT_DIR, f'video_{video_id:03d}')
        os.makedirs(mask_video_dir, exist_ok=True)
    
    ground_truth_data = []

    # Scenario A: save metadata per video folder
    if ENABLE_SCENARIO_A and 'scenario_meta' in locals() and scenario_meta is not None:
        try:
            meta_path = os.path.join(video_dir, 'scenario_meta.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(scenario_meta, f, indent=2)
        except Exception as _e:
            print(f"WARNING: could not write scenario_meta.json for video {video_id}: {_e}")

    # REVISED: Select one background image for the entire video
    if bg_images:
        bg_frame_for_video = random.choice(bg_images).astype(np.float32)
    else:
        bg_frame_for_video = np.zeros(frame_dims, dtype=np.float32)

    for frame_idx in range(NUM_FRAMES):

        

        # Check for particle count and respawn if necessary
        active_particles_count = np.sum(states >= 0)
        if ENABLE_PARTICLE_SPAWNING and active_particles_count < MIN_VISIBLE_PARTICLES:
            num_to_spawn = NUM_PARTICLES - active_particles_count
            
            permanently_gone_indices = np.where(states == -1)[0]
            
            for i in range(min(num_to_spawn, len(permanently_gone_indices))):
                particle_idx = permanently_gone_indices[i]
                
                # Reset properties for the new particle
                positions[particle_idx] = np.random.uniform(50, FRAME_WIDTH - 50, size=2)
                states[particle_idx] = 1 # Start as ON
                reappear_in_frames[particle_idx] = 0
                psf_sigmas[particle_idx] = np.random.uniform(*PSF_STD_PIXELS_RANGE)
                initial_intensities[particle_idx] = np.random.uniform(*INTENSITY_PEAK_RANGE)
                particle_intensities[particle_idx] = initial_intensities[particle_idx]
                time_on[particle_idx] = 0 # Fresh particle starts with 0 exposure

        # REVISED: Use the same background frame for every frame in this video
        frame = bg_frame_for_video.copy()
        frame += np.random.normal(0, 0.5, size=frame.shape)

        # GT mask for this frame
        mask_u8 = None
        if GENERATE_GT_MASKS:
            mask_u8 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

        # Scenario A: set A/B positions from the precomputed natural-walk paths
        if ENABLE_SCENARIO_A:
            # Small measurement jitter (optional). Motion is already in the base paths.
            meas_jitter_std = float(step_std_pixels) * float(jitter_scale) * 0.10
            posA = posA_base_path[frame_idx].copy() + np.random.normal(0, meas_jitter_std, size=2)
            posB = posB_base_path[frame_idx].copy() + np.random.normal(0, meas_jitter_std, size=2)

            positions[SCENARIO_A_IDX_A] = posA
            positions[SCENARIO_A_IDX_B] = posB

        for i in range(NUM_PARTICLES):
            laser_flicker = np.random.normal(1.0, 0.0514) 
            particle_intensities[i] = (initial_intensities[i] * np.exp(-BLEACHING_DECAY_RATE * time_on[i])) * laser_flicker

            if states[i] == -1: # Permanently gone
                continue

            # --- Scenario A: forced blink gap for A, and "reappear near B" ---
            forced_off = False
            if ENABLE_SCENARIO_A:
                if i == SCENARIO_A_IDX_A and (gap_start <= frame_idx < gap_end):
                    forced_off = True
                # Keep A close to B for a few frames right after reappearing
                if i == SCENARIO_A_IDX_A and (gap_end <= frame_idx < gap_end + SCENARIO_A_REAPPEAR_NEAR_B_FRAMES):
                    # We'll set the position after B is updated in this frame (see later).
                    pass

            # Update position
            if ENABLE_SCENARIO_A and i in (SCENARIO_A_IDX_A, SCENARIO_A_IDX_B):
                # Already updated at the start of this frame (so A can be snapped near B if needed).
                pass
            else:
                step = np.random.normal(0, step_std_pixels, size=2)
                positions[i] += step

            if states[i] == 2 or states[i] == 3: # If currently Z-disappeared or Out of Bounds
                reappear_in_frames[i] -= 1
                if reappear_in_frames[i] <= 0:
                    states[i] = 1 # Reappear (transition to ON)
                    # --- CRITICAL FIX: DO NOT RESET INTENSITY/TIME_ON HERE ---
                    # The particle intensity is calculated based on 'time_on' below,
                    # which continues to track total exposure. Resetting it here
                    # defeats the purpose of persistent photobleaching.
                    pass 
            
            # Skip rendering for non-visible particles
            if forced_off or states[i] == 0 or states[i] == 2 or states[i] == 3:
                continue
            
            # --- Photobleaching & Blinking ---
            if states[i] == 1:
                # Increment time_on counter ONLY when the particle is actively ON
                time_on[i] += 1
                
                # Apply photobleaching based on total time_on
                particle_intensities[i] = initial_intensities[i] * np.exp(-BLEACHING_DECAY_RATE * time_on[i])

                # Regular ON/OFF blinking
                if np.random.rand() < get_chance_value(P_OFF):
                    states[i] = 0
            elif states[i] == 0:
                if np.random.rand() < get_chance_value(P_ON):
                    states[i] = 1
            
            # --- Z-axis Disappearance ---
            if ENABLE_Z_MOVEMENT and states[i] == 1:
                if np.random.rand() < get_chance_value(Z_PROB_DISAPPEAR):
                    if np.random.rand() < get_chance_value(Z_PROB_REAPPEAR):
                        states[i] = 2
                        reappear_in_frames[i] = random.randint(*Z_REAPPEAR_RANGE)
                    else:
                        states[i] = -1
                        continue
            
            x, y = positions[i]
            
            # --- Boundary Check ---
            if ENABLE_BOUNDARY_HANDLING:
                if not (0 < x < FRAME_WIDTH and 0 < y < FRAME_HEIGHT):
                    if np.random.rand() < get_chance_value(BOUNDARY_PROB_REAPPEAR):
                        states[i] = 3
                        reappear_in_frames[i] = random.randint(*BOUNDARY_REAPPEAR_RANGE)
                        positions[i] = np.random.uniform(50, FRAME_WIDTH - 50, size=2)
                    else:
                        states[i] = -1
                        continue

            if states[i] == 1:
                ground_truth_data.append({
                    'VideoID': video_id,
                    'ID': i + 1,
                    'Frame': frame_idx,
                    'X_pix': x,
                    'Y_pix': y,
                    'State': states[i]
                })

                # --- Rendering: Add PSF to Frame ---
                psf = generate_psf(x, y, particle_intensities[i], FRAME_WIDTH, FRAME_HEIGHT, psf_sigmas[i])
                frame += psf


                # GT mask paint (only when particle is rendered)
                if GENERATE_GT_MASKS and (mask_u8 is not None):
                    _paint_particle_mask(mask_u8, x, y, psf_sigmas[i])
        # --- Finalize Frame ---
        # Use the gain you measured in your thesis: 0.071 e-/count
        gain = 0.071  # 

        # 1. Convert count-based frame to "photon/electron" units
        photon_frame = frame * gain

        # 2. Apply Poisson Noise (Shot Noise)
        # This makes the noise intensity-dependent, just like real physics
        noisy_photon_frame = np.random.poisson(photon_frame).astype(np.float32)

        # 3. Convert back to counts and add your NOISE_STD as the "Read Noise"
        # [cite: 531, 541]
        frame = (noisy_photon_frame / gain) + np.random.normal(0, NOISE_STD, size=frame.shape)
        
        frame_quantized = np.clip(frame, 0, MAX_VAL).astype(np.uint16)
        
        frame_filename = os.path.join(video_dir, f'frame_{frame_idx:04d}.tif')
        imwrite(frame_filename, frame_quantized)
    
        # Save GT mask PNG matching pipeline naming
        if GENERATE_GT_MASKS and (mask_video_dir is not None) and (mask_u8 is not None):
            mask_path = os.path.join(mask_video_dir, f'frame_{frame_idx:04d}{GT_MASK_SUFFIX}')
            _save_mask_png(mask_u8, mask_path)
    return pd.DataFrame(ground_truth_data)

# --- Main Execution ---
if __name__ == '__main__':
    print("--- SMLM Synthetic Data Generator ---")
    
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    
    try:
        if not os.path.isdir(BACKGROUND_DIR):
             raise FileNotFoundError(f"Directory '{BACKGROUND_DIR}' not found. Please create it and add TIFF files.")
             
        bg_list, frame_dimensions = load_background_images(BACKGROUND_DIR)
        
    except Exception as e:
        print(f"Failed to load backgrounds: {e}. Using 256x256 fallback and zero background.")
        bg_list = None
        frame_dimensions = (256, 256)

    # UPDATED: Loop through and save individual ground truth files
    for i in range(1, NUM_VIDEOS_TO_GENERATE + 1):
        print(f"Generating video {i:03d}/{NUM_VIDEOS_TO_GENERATE}...")
        gt_df = run_single_simulation(i, bg_list, frame_dimensions)
        
        # Save the ground truth for this specific video
        gt_filename = os.path.join(OUTPUT_ROOT_DIR, f'video_{i:03d}_ground_truth.csv')
        gt_df.to_csv(gt_filename, index=False)
        print(f"Ground truth for video_{i:03d} saved to: {gt_filename}")
    
    print("\nAll videos and individual ground truth files generated.")
    print("NOTE: Please copy the newly generated '_ground_truth.csv' files from the 'SyntheticVid' folder to the 'ground_truth' folder to use them with the main pipeline.")
    print("\nGeneration Complete.")