import argparse
import logging
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import csv
from datetime import datetime
import glob
from PIL import Image

# --- Import Albumentations for Data Augmentation ---
try:
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2
    print("Albumentations found. Using advanced data augmentation.")
except ImportError:
    print("Albumentations not found. Please install with: pip install albumentations==1.4.0 opencv-python")
    print("Falling back to basic torchvision transforms. Data augmentation will be limited.")
    A = None
    from torchvision import transforms
    class ToTensorV2:
        def __call__(self, image, mask=None):
            image = transforms.ToTensor()(image)
            if mask is not None:
                mask = transforms.ToTensor()(mask)
                return {'image': image, 'mask': mask}
            return {'image': image}

# --- Import scipy for bandpass filtering (NOT USED FOR TRAINING DATA PREPROCESSING ANYMORE) ---
# Keeping imports here for completeness if other scripts might use these functions
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np

# Assuming these are in your project structure
from evaluate import evaluate
from unet import UNet # Your U-Net model definition
from utils.data_loading import CarvanaDataset # Your modified Dataset class


# --- Configuration: Paths and Training Parameters (Adjust these!) ---
# Dataset Paths
# IMPORTANT: For Option A, ensure TRAIN_IMG_DIR contains a COMBINED set of:
# 1. Synthetic images (generated WITHOUT bandpass filter)
# 2. Real SMLM images (raw, as they are)
# TRAIN_MASK_DIR should contain corresponding masks.
TRAIN_IMG_DIR = Path(r'E:\AI_Project\Process\01_Particle Detection\data\imgs') # Combined folder for all images (particle + background)
TRAIN_MASK_DIR = Path(r'E:\AI_Project\Process\01_Particle Detection\data\masks') # Combined folder for all masks (particle + background)

# Checkpoint directory to save trained models, history CSV, and preview images
CHECKPOINT_DIR = Path(r'E:\AI_Project\Process\01_Particle Detection\checkpoints\Architect_Test1')

# Training Parameters
TRAINING_PARAMS = {
    'epochs': 100,           # Fewer epochs for fine-tuning. Start with 20-50.
    'batch_size': 4,         # Smaller batch size often better for fine-tuning
    'learning_rate': 1e-7,   # <--- MUCH LOWER LR for fine-tuning (e.g., 1e-5, 1e-6, 1e-7)
    'validation_percent': 0.2, # Use a good validation split for real data
    'save_checkpoint': True,
    'image_scale': 1.0,
    'use_amp': False,
    'early_stopping_patience': 20, # Shorter patience for fine-tuning
    
    # pos_weight: Recalculate if real data has different particle density.
    # Otherwise, use the same value from synthetic training.
    'pos_weight_value': 1500, # Adjust based on actual combined dataset imbalance

    'plot_frequency_epochs': 1, # Plot more frequently during fine-tuning (e.g., every 5 epochs)

    # --- Placeholders for calculated normalization values ---
    # These will be calculated automatically from the RAW (0-1) images in TRAIN_IMG_DIR
    'mean_norm': 0.0,
    'std_norm': 1.0
}


# === Helper Functions for Bandpass Filtering (Kept, but not used in training pipeline) ===
# These are provided for completeness if other scripts (like predict.py) need them.
# They are NOT used in the training data loading process for Option A.
def mkffilt_py(img_shape, bp1, bp2):
    """Creates a 2D Fourier bandpass filter (Python equivalent of MATLAB mkffilt)."""
    M, N = img_shape
    cx = N // 2 + 1
    cy = M // 2 + 1
    X, Y = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    H_low = 1 / (1 + (R / bp1)**2)
    H_high = 1 / (1 + (bp2 / R)**2)
    H = H_low * H_high
    H[np.isnan(H)] = 0
    return H

def fpass_py(img, H_filter):
    """Applies a Fourier filter to an image (Python equivalent of MATLAB fpass)."""
    F = fftshift(fft2(img))
    filtered_F = F * H_filter
    filtered_img = np.real(ifft2(ifftshift(filtered_F)))
    return filtered_img

# --- NEW Helper function for calculate_dataset_stats ---
# This is a duplicate of the one in data_loading for calculate_dataset_stats's direct use
def _load_image_as_normalized_numpy_for_stats(filename):
    """
    Loads an image and converts it to a float32 NumPy array,
    preserving the original bit depth for TIFFs and normalizing to 0-1.
    """
    img = Image.open(filename)
    
    if img.mode == 'I;16' or img.mode == 'I':
        img_np = np.array(img, dtype=np.float32)
        return img_np / 65535.0 # Normalize 16-bit to 0-1
    else:
        img_np = np.array(img.convert('L'), dtype=np.float32)
        return img_np / 255.0 # Normalize 8-bit to 0-1


# === MODIFIED: calculate_dataset_stats will now correctly handle 16-bit ===
def calculate_dataset_stats(image_folder):
    """
    Calculates the mean and standard deviation of pixel values across
    all images in a folder from their 0-1 normalized state, correctly
    handling 8-bit and 16-bit images.
    """
    all_pixels = []
    
    image_paths_str = []
    for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
        image_paths_str.extend(glob.glob(os.path.join(str(image_folder), '*' + ext)))

    if not image_paths_str:
        raise ValueError(f"No image files found in {image_folder}. Please check the path and extensions.")

    logging.info(f"Calculating mean and std of normalized (0-1) data from {len(image_paths_str)} images...")

    # Optional: Debugging snippet for the first image
    if len(image_paths_str) > 0:
        # Convert to Path object here to use .suffix for robust file type check
        first_image_path = Path(image_paths_str[0]) 
        logging.info(f"DEBUG: Inspecting first image: {first_image_path}")
        try:
            img_np_raw_debug = _load_image_as_normalized_numpy_for_stats(str(first_image_path))
            
            logging.info(f"DEBUG: Normalized (0-1) min/max/mean (for first image): {img_np_raw_debug.min():.4f}, {img_np_raw_debug.max():.4f}, {img_np_raw_debug.mean():.4f}")

        except Exception as e:
            logging.error(f"DEBUG ERROR: Failed to inspect first image {first_image_path}: {e}")


    for img_path_str in tqdm(image_paths_str, desc="Processing images for stats"):
        # Use the new helper to load and normalize to 0-1
        img_np_float_0_1 = _load_image_as_normalized_numpy_for_stats(img_path_str)
        
        all_pixels.extend(img_np_float_0_1.flatten())

    all_pixels_np = np.array(all_pixels)
    
    calculated_mean = np.mean(all_pixels_np)
    calculated_std = np.std(all_pixels_np)
    
    # Add a check for extremely small std to prevent division by zero in A.Normalize
    if calculated_std < 1e-7: # Use a very small epsilon
        logging.warning(f"Calculated standard deviation is extremely small ({calculated_std:.2e}). Setting to 1.0 to prevent division by zero in normalization. This might indicate very flat images.")
        calculated_std = 1.0 # Set to 1.0 to avoid NaNs during normalization
    
    return calculated_mean, calculated_std


# === Combined Weighted Loss Functions ===
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target, pos_weight_tensor):
    bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight_tensor)
    d_loss = dice_loss(pred, target)
    return 0.5 * bce + 1.0 * d_loss # Keep this weighting for now


# === Training Function ===
def train_model(
    model,
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_percent: float,
    save_checkpoint: bool,
    img_scale: float,
    amp: bool,
    early_stopping_patience: int,
    pos_weight_value: float,
    plot_frequency_epochs: int,
    mean_norm_val: float,
    std_norm_val: float
):
    # --- Data Loading ---
    if A:
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.Normalize(mean=(mean_norm_val,), std=(std_norm_val,)), # Use calculated values from raw images
            ToTensorV2()
        ])

        val_transform = A.Compose([
            A.Normalize(mean=(mean_norm_val,), std=(std_norm_val,)), # Use calculated values from raw images
            ToTensorV2()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean_norm_val], std=[std_norm_val])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean_norm_val], std=[std_norm_val])
        ])

    # CarvanaDataset now handles 16-bit properly if data_loading.py is updated
    full_dataset = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, img_scale, transform=None)
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train_indices, val_indices = random_split(range(len(full_dataset)), [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count() if os.cpu_count() else 0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting Fine-Tuning:
        Epochs:            {epochs}
        Batch size:        {batch_size}
        Learning rate:     {learning_rate}
        Training size:     {n_train}
        Validation size:   {n_val}
        Device:            {device.type}
        Pos Weight:        {pos_weight_value}
        Early Stopping:    {early_stopping_patience} epochs
        Plot Frequency:    Every {plot_frequency_epochs} epochs
        Norm Mean:         {mean_norm_val:.6f}
        Norm Std:          {std_norm_val:.6f}
    ''')

    # --- Optimizer and Scheduler ---
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.999, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # --- Loss Function ---
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)

    # --- CSV Logging Setup ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history_csv_path = CHECKPOINT_DIR / 'training_history.csv'
    with open(history_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'train_loss', 'val_dice_score', 'norm_mean', 'norm_std'])
        csv_writer.writerow(['-', '-', '-', f"{mean_norm_val:.6f}", f"{std_norm_val:.6f}"])


    # --- Get Training Start Timestamp for Image Naming ---
    training_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.info(f"Training started at: {training_start_time}")

    # --- Training Loop ---
    best_val_score = -float('inf')
    epochs_no_improve = 0

    train_losses = []
    val_dice_scores = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                images, true_masks = batch['image'], batch['mask']
                
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
                true_masks = true_masks.to(device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = combined_loss(masks_pred.squeeze(1), true_masks.squeeze(1), pos_weight_tensor)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)

            val_score = evaluate(model, val_loader, device, amp)
            val_dice_scores.append(val_score)
            scheduler.step(val_score)
            logging.info(f'Validation Dice score: {val_score:.4f}')

            with open(history_csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([epoch, f"{avg_epoch_loss:.4f}", f"{val_score:.4f}", '-', '-'])


            if epoch % plot_frequency_epochs == 0 or epoch == epochs:
                logging.info(f'Saving prediction preview for Epoch {epoch}...')
                model.eval()
                with torch.no_grad():
                    try:
                        sample = next(iter(val_loader))
                    except StopIteration:
                        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
                        sample = next(iter(val_loader))

                    images_sample = sample['image'].to(device, dtype=torch.float32)
                    true_masks_sample = sample['mask'].to(device, dtype=torch.float32)
                    
                    masks_pred_sample = model(images_sample)
                    pred_mask_sample = (torch.sigmoid(masks_pred_sample.squeeze(1)) > 0.5).float()

                    # Need to inverse normalize image for display if plotting the actual image
                    # A.Normalize does (x - mean) / std. To inverse, x*std + mean
                    # The image shown is the output of ToTensorV2, which comes AFTER A.Normalize.
                    # This means it's already normalized for the model.
                    # If you want to view it in its "raw" (0-1, but not model-normalized) state,
                    # you'd need to inverse the normalization here.
                    # For simplicity, we assume `img_np` below is the original `images_sample[0]`
                    # after denormalization.
                    
                    # Original `img_np = images_sample[0].squeeze(0).cpu().numpy()` is *normalized*.
                    # To get a visually intuitive 0-1 range for plotting, denormalize:
                    img_np = images_sample[0].squeeze(0).cpu().numpy()
                    img_np = (img_np * std_norm_val) + mean_norm_val # Denormalize for display
                    img_np = np.clip(img_np, 0, 1) # Clip to 0-1 range for proper display

                    mask_true_np = true_masks_sample[0].squeeze(0).cpu().numpy()
                    mask_pred_np = pred_mask_sample[0].squeeze(0).cpu().numpy()

                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(img_np, cmap='gray')
                    plt.title('Image (Denormalized)') # Updated title
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(mask_true_np, cmap='gray')
                    plt.title('True Mask')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(mask_pred_np, cmap='gray')
                    plt.title('Predicted Mask')
                    plt.axis('off')
                    plt.tight_layout()
                    
                    preview_filename = CHECKPOINT_DIR / f'preview_{training_start_time}_epoch{epoch:03d}.png'
                    plt.savefig(preview_filename, bbox_inches='tight', dpi=150)
                    plt.close()
                    logging.info(f'Saved preview image to: {preview_filename}')


            # --- Optimized Checkpoint Saving ---
            if val_score > best_val_score:
                best_val_score = val_score
                epochs_no_improve = 0
                if save_checkpoint:
                    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                    # Modified: Save with epoch number for unique checkpoints
                    torch.save(model.state_dict(), str(CHECKPOINT_DIR / f'checkpoint_epoch{epoch:03d}_dice{val_score:.4f}.pth')) 
                    logging.info(f'Checkpoint epoch {epoch} saved! (Validation Dice: {best_val_score:.4f})')
            else:
                epochs_no_improve += 1
                logging.info(f'Validation Dice score did not improve for {epochs_no_improve} epochs.')
                if epochs_no_improve >= early_stopping_patience:
                    logging.info(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')
                    break

    logging.info('Training finished.')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_dice_scores) + 1), val_dice_scores, label='Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('U-Net Training History')
    plt.legend()
    plt.grid(True)
    plt.show() # This final plot will still show up


# === Script Entry Point ===
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Fine-tune U-Net on real particle images')
    parser.add_argument('--epochs', type=int, default=TRAINING_PARAMS['epochs'], help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=TRAINING_PARAMS['batch_size'], help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAINING_PARAMS['learning_rate'], help='Learning rate')
    parser.add_argument('--load', type=str, default=False, help='Load pre-trained .pth checkpoint')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # --- Calculate Normalization Statistics Automatically ---
    try:
        calculated_mean, calculated_std = calculate_dataset_stats(
            TRAIN_IMG_DIR,
        )
        TRAINING_PARAMS['mean_norm'] = calculated_mean
        TRAINING_PARAMS['std_norm'] = calculated_std
        logging.info(f"Automatically calculated normalization: Mean={calculated_mean:.6f}, Std={calculated_std:.6f}")
        if calculated_std < 1e-7:
            logging.warning("Calculated standard deviation is extremely small. This might indicate very flat images or an issue with data. Setting std to 1.0 to prevent division by zero in normalization.")
            TRAINING_PARAMS['std_norm'] = 1.0
    except Exception as e:
        logging.error(f"Failed to calculate normalization statistics: {e}. Please check TRAIN_IMG_DIR and ensure images are accessible.")
        exit()


    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)

    if args.load:
        logging.info(f'Loading model from {args.load}')
        state_dict = torch.load(args.load, map_location=device)
        state_dict.pop('mask_values', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded successfully from {args.load}')
    else:
        logging.info('No pre-trained model loaded. Starting training from scratch.')

    try:
        train_model(
            model,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=TRAINING_PARAMS['validation_percent'],
            save_checkpoint=TRAINING_PARAMS['save_checkpoint'],
            img_scale=TRAINING_PARAMS['image_scale'],
            amp=args.amp,
            early_stopping_patience=TRAINING_PARAMS['early_stopping_patience'],
            pos_weight_value=TRAINING_PARAMS['pos_weight_value'],
            plot_frequency_epochs=TRAINING_PARAMS['plot_frequency_epochs'],
            mean_norm_val=TRAINING_PARAMS['mean_norm'],
            std_norm_val=TRAINING_PARAMS['std_norm']
        )
    except KeyboardInterrupt:
        logging.info('Training interrupted by user. Saving last checkpoint...')
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(CHECKPOINT_DIR / 'INTERRUPTED_checkpoint.pth'))
        logging.info('Last checkpoint saved!')