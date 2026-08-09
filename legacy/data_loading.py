import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

# --- Original load_image function (used by BasicDataset) ---
def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

# --- Original unique_mask_values function (used by BasicDataset) ---
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

# --- BasicDataset class (remains unchanged as per your original structure) ---
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir)
                    if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            if img.ndim == 3:
                img = img[..., 0]  # drop RGB channels if present
            # Binary mask: 0 = background, 1 = particle
            mask = (img > 127).astype(np.float32)
            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]  # grayscale → [1, H, W]
            else:
                img = img.transpose((2, 0, 1))  # RGB → [C, H, W]

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for ID {name}: {mask_file}'
        mask = load_image(mask_file[0]) # Uses the top-level load_image
        img = load_image(img_file[0])   # Uses the top-level load_image

        assert img.size == mask.size, f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }

# --- Helper function for CarvanaDataset specific loading (for 16-bit handling) ---
# This loads images as NumPy arrays and normalizes them to 0-1 based on bit depth
def _load_image_as_normalized_numpy(filename):
    img = Image.open(filename)
    
    # Check for 16-bit integer modes
    if img.mode == 'I;16' or img.mode == 'I': # 'I' is for 32-bit signed int, but often used for 16-bit
        img_np = np.array(img, dtype=np.float32)
        # Normalize 16-bit images by their max possible value (65535 for unsigned)
        max_val = 65535.0
        img_np = img_np / max_val
        return img_np
    else:
        # For 8-bit images (PNG, JPG, etc.), convert to grayscale and normalize to 0-1
        img_np = np.array(img.convert('L'), dtype=np.float32)
        img_np = img_np / 255.0
        return img_np


# --- CarvanaDataset class (Modified to use the new helper for 16-bit handling) ---
class CarvanaDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float, transform=None):
        """
        Initializes the dataset.
        Args:
            images_dir (str): Directory containing image files.
            masks_dir (str): Directory containing mask files.
            scale (float): Scaling factor for images (e.g., 0.5 for half size).
            transform (callable, optional): Optional transform to be applied on a sample.
                                            This will typically be an Albumentations Compose object.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.transform = transform # Store the transform

        self.ids = [stem.stem for stem in self.images_dir.iterdir() if stem.is_file() and not str(stem.name).startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        
        mask_file_paths = list(self.masks_dir.glob(name + '_mask.*'))
        img_file_paths = list(self.images_dir.glob(name + '.*'))

        if not img_file_paths:
            raise FileNotFoundError(f'Image {name} not found in {self.images_dir}')
        if not mask_file_paths:
            raise FileNotFoundError(f'Mask for {name} not found in {self.masks_dir}. Expected mask name like {name}_mask.*')

        # Use the new helper function to load images as normalized NumPy arrays (0-1)
        img_np = _load_image_as_normalized_numpy(img_file_paths[0])
        mask_np = _load_image_as_normalized_numpy(mask_file_paths[0]) # Apply to masks too, then binarize

        # Resize if scale is not 1.0
        if self.scale != 1.0:
            h, w = img_np.shape # Assuming grayscale 2D array
            newH, newW = int(h * self.scale), int(w * self.scale)
            
            # Use PIL for resizing for simplicity, convert back to numpy
            # Need to scale to 0-255 temporarily for PIL.Image.fromarray(uint8)
            img_pil_temp = Image.fromarray((img_np * 255).astype(np.uint8)) 
            mask_pil_temp = Image.fromarray((mask_np * 255).astype(np.uint8)) 

            img_np = np.array(img_pil_temp.resize((newW, newH), resample=Image.BICUBIC), dtype=np.float32) / 255.0 # Scale back to 0-1
            mask_np = np.array(mask_pil_temp.resize((newW, newH), resample=Image.NEAREST), dtype=np.float32) / 255.0 # Scale back to 0-1


        # Apply transform if provided (Albumentations expects NumPy arrays, 0-1 range is ideal input for Normalize)
        if self.transform:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_tensor = augmented['image'] # Albumentations ToTensorV2 converts to tensor
            mask_tensor = augmented['mask'] # Albumentations ToTensorV2 converts to tensor
        else:
            # If no transform, ensure it's a PyTorch tensor with channel dimension
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        # Ensure mask is binary (0 or 1) - crucial for BCEWithLogitsLoss
        mask_tensor = (mask_tensor > 0.5).float() # Final threshold to ensure binary mask

        return {'image': img_tensor, 'mask': mask_tensor}