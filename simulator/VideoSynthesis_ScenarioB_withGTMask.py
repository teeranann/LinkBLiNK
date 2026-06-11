import scipy.special as sp
import numpy as np
import pandas as pd
from tifffile import imwrite, imread
import os
import glob
import random
import json

# ============================================================
#  GT U-Net Mask Export (for "mask cache" mode in MainPipeline)
#
#  This will write binary masks (0/255) with filenames like:
#     UNET_MASK_ROOT_DIR / video_### / frame_####_predict_mask.png
#
#  Make UNET_MASK_ROOT_DIR match CONFIG['unet_masks_dir'] in your pipeline.
# ============================================================

GENERATE_GT_MASKS = True
UNET_MASK_ROOT_DIR = r"D:\Works\LinkBLiNK\Playground\VideoGeneration\Scenario BX6\UNetMasks_GT"

GT_MASK_SUFFIX = "_predict_mask.png"
GT_MASK_ON_VALUE = 255
GT_MASK_RADIUS_SIGMA_MULT = 2.2   # ~ threshold at ~10% of peak for a Gaussian
GT_MASK_MIN_RADIUS_PX = 1

# Save PNG with minimal dependencies (tries imageio, then PIL, then cv2)
_save_mask_png = None
try:
    import imageio.v2 as imageio
    def _save_mask_png(path, mask_u8):
        imageio.imwrite(path, mask_u8)
except Exception:
    try:
        from PIL import Image
        def _save_mask_png(path, mask_u8):
            Image.fromarray(mask_u8).save(path)
    except Exception:
        try:
            import cv2
            def _save_mask_png(path, mask_u8):
                cv2.imwrite(path, mask_u8)
        except Exception:
            _save_mask_png = None


def _draw_filled_disk(mask_u8: np.ndarray, x: float, y: float, r: int, value: int = 255):
    """Draw a filled circle into a uint8 mask. x,y are pixel coords (float allowed)."""
    if r <= 0:
        return

    h, w = mask_u8.shape
    x = float(np.clip(x, 0, w - 1))
    y = float(np.clip(y, 0, h - 1))

    xmin = max(0, int(np.floor(x - r)))
    xmax = min(w - 1, int(np.ceil(x + r)))
    ymin = max(0, int(np.floor(y - r)))
    ymax = min(h - 1, int(np.ceil(y + r)))

    if xmin > xmax or ymin > ymax:
        return

    yy, xx = np.ogrid[ymin:ymax + 1, xmin:xmax + 1]
    rr2 = (xx - x) ** 2 + (yy - y) ** 2
    mask_u8[ymin:ymax + 1, xmin:xmax + 1][rr2 <= (r * r)] = value


def _paint_particle_mask(mask_u8: np.ndarray, x: float, y: float, sigma_pix: float):
    r = int(np.ceil(float(GT_MASK_RADIUS_SIGMA_MULT) * float(sigma_pix)))
    r = max(r, int(GT_MASK_MIN_RADIUS_PX))
    _draw_filled_disk(mask_u8, x, y, r, int(GT_MASK_ON_VALUE))
# ============================================================
#  SMLM Synthetic Data Generator — Scenario B (Decoy reappearance)
#
#  Behavior (per video):
#   - Two real particles exist at the start: A and B
#   - Particle A disappears at gap_start and never returns
#   - A *new* particle (Decoy) spawns at gap_end near A's predicted location
#   - Decoy has intentionally different signature (brightness, size, ellipticity)
#   - One random background frame is chosen ONCE per video and reused for all frames
# ============================================================

# -------------------------
# General Generation
# -------------------------
NUM_VIDEOS_TO_GENERATE = 50
OUTPUT_ROOT_DIR = r'D:\Works\LinkBLiNK\Playground\VideoGeneration\Scenario BX6'
BACKGROUND_DIR = r'D:\Works\LinkBLiNK\Playground\VideoGeneration\Background'

# Video properties
NUM_FRAMES = 500
PIXEL_SIZE_UM = 0.0359
BIT_DEPTH = 16
FRAME_RATE = 0.01
MAX_VAL = (2**BIT_DEPTH) - 1

# Particle properties
# We reserve 3 slots: A, B, Decoy.
NUM_PARTICLES = 3

# Base photophysics
INTENSITY_PEAK_RANGE = (8000, 30000)
BLEACHING_DECAY_RATE = 0.001
PSF_STD_PIXELS_RANGE = (0.8, 4.0)
DIFFUSION_COEFF = 0.086  # um^2/s

# Realism (PSF shape)
ELLIPTICITY_MAX = 0.15
ELLIPTICITY_JITTER = 0.04  # per-frame jitter around the per-particle bias
TAIL_FRACTION = 0.03
TAIL_FACTOR = 3.0

# Turn off random blinking / Z disappearance for controlled scenarios
P_ON = 0.0
P_OFF = 0.0
ENABLE_Z_MOVEMENT = False

# Noise
NOISE_STD = 5.0

# -------------------------
# Scenario B parameters
# -------------------------
ENABLE_SCENARIO_B = True

# Slot indices
SCENARIO_B_IDX_A = 0      # Particle A: disappears permanently after gap_start
SCENARIO_B_IDX_B = 1      # Particle B: stays alive throughout (background "second actor")
SCENARIO_B_IDX_DECOY = 2  # New particle, appears after the gap

# Gap timing (fraction of video)
SCENARIO_B_GAP_LENGTH_RANGE = (100,102)
SCENARIO_B_GAP_CENTER_FRAC_RANGE = (0.30, 0.70)

# Decoy placement relative to predicted A position
# "Predicted A position" is linear extrapolation from A's last 2 positions pre-gap.
SCENARIO_B_LINK_RADIUS_PIXELS = 6.0
SCENARIO_B_DECOY_OFFSET_FRAC = 0.55  # fraction of link radius for random offset magnitude
SCENARIO_B_DECOY_FOLLOW_PREDICT_FRAMES = 5  # frames to loosely follow predicted continuation

# Motion styling
PATH_MARGIN_PIXELS = 30
# (Brownian + velocity persistence)
VEL_RHO_RANGE = (0.88, 0.96)          # higher => smoother / straighter
JITTER_SCALE_RANGE = (0.18, 0.55)     # overall diffusion scaling


# -------------------------
# Helpers
# -------------------------
def load_background_images(bg_dir: str):
    # Loads all 16-bit TIFF background images from a directory; returns list-of-frames and frame shape.
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
    return (bg_list, bg_list[0].shape) if bg_list else (None, (152, 150))


def generate_psf(x, y, intensity, width, height, sigma, ellip_bias=1.0):
    """
    Realistic Gaussian PSF with per-particle ellipticity bias, 
    a broad tail component, and proper sub-pixel integration (Peak Normalized).
    """
    x_grid = np.arange(width)
    y_grid = np.arange(height)

    # Calculate final ellipticity
    ellip = float(ellip_bias) * float(np.random.uniform(1 - ELLIPTICITY_JITTER, 1 + ELLIPTICITY_JITTER))
    ellip = float(np.clip(ellip, 1 - ELLIPTICITY_MAX, 1 + ELLIPTICITY_MAX))

    sigma_x = sigma * ellip
    sigma_y = sigma / ellip

    # Sub-pixel integration using Error Function
    def pixel_integral_1d(pos, center, sig):
        left = pos - 0.5 - center
        right = pos + 0.5 - center
        return 0.5 * (sp.erf(right / (np.sqrt(2) * sig)) - sp.erf(left / (np.sqrt(2) * sig)))

    # 1. Main Core Gaussian
    x_int_core = pixel_integral_1d(x_grid, x, sigma_x)
    y_int_core = pixel_integral_1d(y_grid, y, sigma_y)
    core = np.outer(y_int_core, x_int_core)

    # 2. Wide Tail Gaussian
    tail_sigma = sigma * TAIL_FACTOR
    x_int_tail = pixel_integral_1d(x_grid, x, tail_sigma)
    y_int_tail = pixel_integral_1d(y_grid, y, tail_sigma)
    tail = np.outer(y_int_tail, x_int_tail)

    # 3. Combine base shapes
    main_intensity_ratio = 1.0 - TAIL_FRACTION
    tail_intensity_ratio = TAIL_FRACTION
    psf = (core * main_intensity_ratio) + (tail * tail_intensity_ratio)

    # 4. CRITICAL FIX: Scale the array so the *peak* matches your requested intensity
    current_max = np.max(psf)
    if current_max > 0:
        psf = (psf / current_max) * intensity
        
    return psf


def _reflect_in_bounds(pos_xy, v_xy, frame_w, frame_h, margin):
    # Reflect position/velocity at boundaries to avoid teleport/disappear.
    lo_x, hi_x = margin, frame_w - margin
    lo_y, hi_y = margin, frame_h - margin

    x, y = float(pos_xy[0]), float(pos_xy[1])
    vx, vy = float(v_xy[0]), float(v_xy[1])

    if x < lo_x:
        x = lo_x + (lo_x - x)
        vx *= -0.6
    elif x > hi_x:
        x = hi_x - (x - hi_x)
        vx *= -0.6

    if y < lo_y:
        y = lo_y + (lo_y - y)
        vy *= -0.6
    elif y > hi_y:
        y = hi_y - (y - hi_y)
        vy *= -0.6

    return np.array([x, y], dtype=float), np.array([vx, vy], dtype=float)


def gen_smooth_walk(num_frames, start_xy, step_std, frame_w, frame_h, margin, vel_rho):
    # Smooth random walk: v_t = rho*v_{t-1} + noise; x_t = x_{t-1} + v_t + Brownian_noise
    pos = np.zeros((num_frames, 2), dtype=float)
    pos[0] = start_xy.astype(float)

    v = np.random.normal(0, step_std * 0.20, size=2)
    vel_noise_std = step_std * 0.12
    step_noise_std = step_std * 0.85

    for t in range(1, num_frames):
        v = vel_rho * v + np.random.normal(0, vel_noise_std, size=2)
        step = np.random.normal(0, step_noise_std, size=2)
        pos[t] = pos[t - 1] + v + step
        pos[t], v = _reflect_in_bounds(pos[t], v, frame_w, frame_h, margin)

    return pos


def random_offset(max_r):
    # Uniform random offset inside a disk of radius max_r.
    theta = np.random.uniform(0, 2*np.pi)
    r = np.sqrt(np.random.uniform(0, 1.0)) * max_r
    return np.array([r*np.cos(theta), r*np.sin(theta)], dtype=float)


# -------------------------
# Main simulation
# -------------------------
def run_single_simulation(video_id, bg_images, frame_dims):
    FRAME_H, FRAME_W = frame_dims
    margin = int(PATH_MARGIN_PIXELS)

    # Pick ONE background frame for entire video
    if bg_images:
        bg_frame_for_video = random.choice(bg_images).astype(np.float32)
    else:
        bg_frame_for_video = np.zeros(frame_dims, dtype=np.float32)

    # Diffusion-derived step size (pixels)
    step_std_pixels = np.sqrt(2 * DIFFUSION_COEFF * FRAME_RATE) / PIXEL_SIZE_UM
    jitter_scale = float(np.random.uniform(*JITTER_SCALE_RANGE))
    walk_step_std = float(step_std_pixels) * jitter_scale
    vel_rho = float(np.random.uniform(*VEL_RHO_RANGE))

    # Build A and B trajectories (safely separated at spawn)
    MIN_SPAWN_DISTANCE_PIXELS = 40.0
    valid_spawn = False
    
    while not valid_spawn:
        startA = np.array([np.random.uniform(margin, FRAME_W - margin),
                           np.random.uniform(margin, FRAME_H - margin)], dtype=float)
        startB = np.array([np.random.uniform(margin, FRAME_W - margin),
                           np.random.uniform(margin, FRAME_H - margin)], dtype=float)
                           
        if np.linalg.norm(startA - startB) >= MIN_SPAWN_DISTANCE_PIXELS:
            valid_spawn = True

    posA_path = gen_smooth_walk(NUM_FRAMES, startA, walk_step_std, FRAME_W, FRAME_H, margin, vel_rho)
    posB_path = gen_smooth_walk(NUM_FRAMES, startB, walk_step_std, FRAME_W, FRAME_H, margin, vel_rho)

    # Scenario B timing
    gap_len = int(random.randint(*SCENARIO_B_GAP_LENGTH_RANGE))
    frac_lo, frac_hi = SCENARIO_B_GAP_CENTER_FRAC_RANGE
    gap_center = int(np.random.randint(int(frac_lo * NUM_FRAMES),
                                       max(int(frac_lo * NUM_FRAMES) + 1, int(frac_hi * NUM_FRAMES))))
    gap_center = int(np.clip(gap_center, 2, NUM_FRAMES - 2))  # keep room for v estimate

    gap_start = int(max(1, gap_center - (gap_len // 2)))
    gap_end = int(min(NUM_FRAMES, gap_start + gap_len))  # exclusive

    # Predicted A position at gap_end (linear extrapolation)
    vA = posA_path[gap_start - 1] - posA_path[gap_start - 2]
    pred_gap_end = posA_path[gap_start - 1] + vA * float(gap_end - (gap_start - 1))

    pred_gap_end[0] = float(np.clip(pred_gap_end[0], margin, FRAME_W - margin))
    pred_gap_end[1] = float(np.clip(pred_gap_end[1], margin, FRAME_H - margin))

    # Decoy spawn near predicted location, NOT exact
    max_off = float(SCENARIO_B_LINK_RADIUS_PIXELS) * float(SCENARIO_B_DECOY_OFFSET_FRAC)
    decoy_spawn = pred_gap_end + random_offset(max_off)
    decoy_spawn[0] = float(np.clip(decoy_spawn[0], margin, FRAME_W - margin))
    decoy_spawn[1] = float(np.clip(decoy_spawn[1], margin, FRAME_H - margin))

    # Decoy trajectory from gap_end to end
    decoy_path = np.full((NUM_FRAMES, 2), np.nan, dtype=float)
    if gap_end < NUM_FRAMES:
        decoy_tail = gen_smooth_walk(NUM_FRAMES - gap_end, decoy_spawn, walk_step_std, FRAME_W, FRAME_H, margin, vel_rho)
        decoy_path[gap_end:] = decoy_tail

        # Optional: for first few frames, nudge decoy toward A's predicted continuation
        follow_frames = int(min(SCENARIO_B_DECOY_FOLLOW_PREDICT_FRAMES, NUM_FRAMES - gap_end))
        if follow_frames > 0:
            for k in range(follow_frames):
                t = gap_end + k
                pred_t = posA_path[gap_start - 1] + vA * float(t - (gap_start - 1))
                pred_t[0] = float(np.clip(pred_t[0], margin, FRAME_W - margin))
                pred_t[1] = float(np.clip(pred_t[1], margin, FRAME_H - margin))
                decoy_path[t] = 0.65 * decoy_path[t] + 0.35 * pred_t

    # -------------------------
    # Particle signatures (brightness, size, shape)
    # -------------------------
    initial_intensities = np.zeros(NUM_PARTICLES, dtype=float)
    psf_sigmas = np.zeros(NUM_PARTICLES, dtype=float)
    ellip_bias = np.ones(NUM_PARTICLES, dtype=float)

    # B (normal)
    initial_intensities[SCENARIO_B_IDX_B] = np.random.uniform(INTENSITY_PEAK_RANGE[0] * 1.0, INTENSITY_PEAK_RANGE[0] * 1.6)
    psf_sigmas[SCENARIO_B_IDX_B] = np.random.uniform(PSF_STD_PIXELS_RANGE[0] + 0.8, PSF_STD_PIXELS_RANGE[1] - 0.8)
    ellip_bias[SCENARIO_B_IDX_B] = np.random.uniform(1 - ELLIPTICITY_MAX * 0.35, 1 + ELLIPTICITY_MAX * 0.35)

    # A (brighter + smaller, near-round)
    initial_intensities[SCENARIO_B_IDX_A] = np.random.uniform(INTENSITY_PEAK_RANGE[1] * 0.75, INTENSITY_PEAK_RANGE[1] * 1.00)
    psf_sigmas[SCENARIO_B_IDX_A] = np.random.uniform(PSF_STD_PIXELS_RANGE[0], PSF_STD_PIXELS_RANGE[0] + 0.9)
    ellip_bias[SCENARIO_B_IDX_A] = np.random.uniform(1 - ELLIPTICITY_MAX * 0.15, 1 + ELLIPTICITY_MAX * 0.15)


    # Decoy (dimmer + larger, more elliptical)
   #########################################################################################################################################################
    # initial_intensities[SCENARIO_B_IDX_DECOY] = np.random.uniform(INTENSITY_PEAK_RANGE[0], INTENSITY_PEAK_RANGE[0] * 1.1)
    # psf_sigmas[SCENARIO_B_IDX_DECOY] = np.random.uniform(PSF_STD_PIXELS_RANGE[1] - 0.5, PSF_STD_PIXELS_RANGE[1])
    # ellip_bias[SCENARIO_B_IDX_DECOY] = np.random.choice([0.8, 1.2])

    initial_intensities[SCENARIO_B_IDX_DECOY] = initial_intensities[SCENARIO_B_IDX_A] * 0.20
    psf_sigmas[SCENARIO_B_IDX_DECOY] = psf_sigmas[SCENARIO_B_IDX_A] * 3.00
    ellip_bias[SCENARIO_B_IDX_DECOY] = ellip_bias[SCENARIO_B_IDX_A] * 1.50


    particle_intensities = initial_intensities.copy()
    time_on = np.zeros(NUM_PARTICLES, dtype=int)

    # States: -1 gone, 0 off, 1 on
    states = np.full(NUM_PARTICLES, -1, dtype=int)
    states[SCENARIO_B_IDX_A] = 1
    states[SCENARIO_B_IDX_B] = 1
    states[SCENARIO_B_IDX_DECOY] = -1  # not spawned yet

    positions = np.zeros((NUM_PARTICLES, 2), dtype=float)

    # Metadata
    scenario_meta = dict(
        scenario="B",
        A_index=int(SCENARIO_B_IDX_A),
        B_index=int(SCENARIO_B_IDX_B),
        Decoy_index=int(SCENARIO_B_IDX_DECOY),
        link_radius_pixels=float(SCENARIO_B_LINK_RADIUS_PIXELS),
        gap_start_frame=int(gap_start),
        gap_end_frame=int(gap_end),
        gap_length_frames=int(gap_end - gap_start),
        pred_gap_end_pixels=[float(pred_gap_end[0]), float(pred_gap_end[1])],
        decoy_spawn_pixels=[float(decoy_spawn[0]), float(decoy_spawn[1])],
        decoy_offset_pixels=[float(decoy_spawn[0] - pred_gap_end[0]), float(decoy_spawn[1] - pred_gap_end[1])],
        decoy_follow_predict_frames=int(SCENARIO_B_DECOY_FOLLOW_PREDICT_FRAMES),
        signatures=dict(
            A=dict(intensity0=float(initial_intensities[SCENARIO_B_IDX_A]), sigma=float(psf_sigmas[SCENARIO_B_IDX_A]), ellip_bias=float(ellip_bias[SCENARIO_B_IDX_A])),
            B=dict(intensity0=float(initial_intensities[SCENARIO_B_IDX_B]), sigma=float(psf_sigmas[SCENARIO_B_IDX_B]), ellip_bias=float(ellip_bias[SCENARIO_B_IDX_B])),
            Decoy=dict(intensity0=float(initial_intensities[SCENARIO_B_IDX_DECOY]), sigma=float(psf_sigmas[SCENARIO_B_IDX_DECOY]), ellip_bias=float(ellip_bias[SCENARIO_B_IDX_DECOY])),
        ),
        motion=dict(
            jitter_scale=float(jitter_scale),
            vel_rho=float(vel_rho),
            step_std_pixels=float(walk_step_std),
        )
    )

    video_dir = os.path.join(OUTPUT_ROOT_DIR, f'video_{video_id:03d}')
    os.makedirs(video_dir, exist_ok=True)

    # GT mask folder (mirrors video folder naming)
    if GENERATE_GT_MASKS and _save_mask_png is not None:
        mask_video_dir = os.path.join(UNET_MASK_ROOT_DIR, os.path.basename(video_dir))
        os.makedirs(mask_video_dir, exist_ok=True)
    else:
        mask_video_dir = None


    try:
        with open(os.path.join(video_dir, 'scenario_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(scenario_meta, f, indent=2)
    except Exception as e:
        print(f"WARNING: could not write scenario_meta.json for video {video_id}: {e}")

    ground_truth_data = []

    meas_jitter_std = float(step_std_pixels) * float(jitter_scale) * 0.10

    for frame_idx in range(NUM_FRAMES):
        frame = bg_frame_for_video.copy()
        frame += np.random.normal(0, 0.5, size=frame.shape)
        
        # GT mask for this frame
        if GENERATE_GT_MASKS and mask_video_dir is not None:
            mask_u8 = np.zeros(frame.shape, dtype=np.uint8)
        else:
            mask_u8 = None

        # A & B positions from their paths
        positions[SCENARIO_B_IDX_A] = posA_path[frame_idx] + np.random.normal(0, meas_jitter_std, size=2)
        positions[SCENARIO_B_IDX_B] = posB_path[frame_idx] + np.random.normal(0, meas_jitter_std, size=2)

        # Kill A at gap_start (permanent disappearance)
        if frame_idx == gap_start:
            states[SCENARIO_B_IDX_A] = -1

        # Spawn Decoy at gap_end
        if frame_idx == gap_end and gap_end < NUM_FRAMES:
            states[SCENARIO_B_IDX_DECOY] = 1

        # Update Decoy position (after spawn)
        if states[SCENARIO_B_IDX_DECOY] == 1 and frame_idx >= gap_end:
            positions[SCENARIO_B_IDX_DECOY] = decoy_path[frame_idx] + np.random.normal(0, meas_jitter_std, size=2)

        for i in range(NUM_PARTICLES):
            if states[i] != 1:
                continue

            time_on[i] += 1
            particle_intensities[i] = initial_intensities[i] * np.exp(-BLEACHING_DECAY_RATE * time_on[i])

            x, y = positions[i]

            # Paint GT mask
            if GENERATE_GT_MASKS and (mask_u8 is not None):
                _paint_particle_mask(mask_u8, x, y, psf_sigmas[i])
            # GT row
            ground_truth_data.append({
                'VideoID': video_id,
                'ID': i + 1,          # fixed IDs: 1=A, 2=B, 3=Decoy
                'Frame': frame_idx,
                'X_pix': float(x),
                'Y_pix': float(y),
                'State': int(states[i])
            })

            # Render    
            frame += generate_psf(
                x, y,
                particle_intensities[i],
                FRAME_W, FRAME_H,
                psf_sigmas[i],
                ellip_bias=ellip_bias[i]
            )

        # --- Point 1: Physical Shot Noise (Poisson) ---
        # Based on your measured gain of 0.071 e-/count [cite: 6059]
        gain = 0.071 
        
        # Convert intensity counts to photoelectrons
        photon_frame = frame * gain
        # Ensure no negative values before Poisson sampling
        photon_frame = np.clip(photon_frame, 0, None)
        
        # Sample from Poisson distribution (Shot Noise)
        noisy_photon_frame = np.random.poisson(photon_frame).astype(np.float32)
        
        # Convert back to counts
        frame = noisy_photon_frame / gain

        # Noise + save
        frame += np.random.normal(0, NOISE_STD, size=frame.shape)
        frame_quantized = np.clip(frame, 0, MAX_VAL).astype(np.uint16)
        imwrite(os.path.join(video_dir, f'frame_{frame_idx:04d}.tif'), frame_quantized)
        # Save GT mask PNG (for cached-mask pipeline mode)
        if GENERATE_GT_MASKS and (mask_u8 is not None) and (mask_video_dir is not None):
            _save_mask_png(os.path.join(mask_video_dir, f"frame_{frame_idx:04d}_predict_mask.png"), mask_u8)

    return pd.DataFrame(ground_truth_data)


if __name__ == '__main__':
    print("--- SMLM Synthetic Data Generator — Scenario B ---")

    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    try:
        if not os.path.isdir(BACKGROUND_DIR):
            raise FileNotFoundError(f"Directory '{BACKGROUND_DIR}' not found. Please create it and add TIFF files.")
        bg_list, frame_dimensions = load_background_images(BACKGROUND_DIR)
    except Exception as e:
        print(f"Failed to load backgrounds: {e}. Using 256x256 fallback and zero background.")
        bg_list = None
        frame_dimensions = (256, 256)

    for i in range(1, NUM_VIDEOS_TO_GENERATE + 1):
        print(f"Generating video {i:03d}/{NUM_VIDEOS_TO_GENERATE}...")
        gt_df = run_single_simulation(i, bg_list, frame_dimensions)

        gt_filename = os.path.join(OUTPUT_ROOT_DIR, f'video_{i:03d}_ground_truth.csv')
        gt_df.to_csv(gt_filename, index=False)
        print(f"Ground truth saved to: {gt_filename}")

    print("\nAll videos generated.")
