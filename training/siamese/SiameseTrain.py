import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image # NEW: Import PIL for torchvision transforms
import os
import pandas as pd
import skimage.io as io
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np

# --- Configuration ---
# Paths to your generated datasets
PAIRS_OUTPUT_DIR = r'E:\AI_Project\TestSample\Siamese\Output\Pairs'
PATCHES_OUTPUT_DIR = r'E:\AI_Project\TestSample\Siamese\Output\Patches'

# Training Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MARGIN = 1.0
CLASSIFICATION_THRESHOLD = 0.5

# Model saving and metrics logging
MODEL_SAVE_DIR = r'E:\AI_Project\TestSample\Siamese\Output\Models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_siamese_model.pth')
METRICS_LOG_PATH = os.path.join(MODEL_SAVE_DIR, 'training_metrics.csv')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Define PATCH_SIZE
PATCH_SIZE = 32

# --- Data Augmentation Transforms ---
# The fill value for grayscale images (0 for black)
AUG_FILL_VALUE = 0

train_transform = transforms.Compose([
    # Convert NumPy array (from CV2) to PIL Image for torchvision transforms
    # Then it will be converted to a Tensor by transforms.ToTensor()
    # No need for explicit ToPILImage() as it's handled internally if fromarray is used

    # Randomly rotate by a small degree.
    # We pass the PIL Image to this, which handles the rotation.
    transforms.RandomRotation(degrees=15, fill=AUG_FILL_VALUE), # Increased degrees slightly
    
    # Randomly translate the image. Max 10% translation in each direction.
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=AUG_FILL_VALUE),

    # Randomly scale the image (zoom in/out slightly)
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), fill=AUG_FILL_VALUE),

    # Randomly shear the image (slight distortion)
    transforms.RandomAffine(degrees=0, shear=5, fill=AUG_FILL_VALUE),

    # Randomly flip horizontally (p=0.5 means 50% chance)
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Randomly flip vertically (p=0.5 means 50% chance)
    transforms.RandomVerticalFlip(p=0.5),

    # Convert PIL Image to PyTorch Tensor.
    # This automatically scales pixel values from [0, 255] (or [0, 65535] for 16-bit PIL if correctly handled)
    # to [0.0, 1.0] and rearranges dimensions from (H, W, C) to (C, H, W).
    # For grayscale, it will be (1, H, W).
    transforms.ToTensor(),
    
    # Optional: Further normalize with mean and std if desired, but [0,1] is often sufficient.
    # transforms.Normalize(mean=[0.5], std=[0.5]) # Example for standardization
])

# For validation and test sets, we typically only apply deterministic preprocessing
# (like converting to Tensor and normalization) without random augmentations.
val_test_transform = transforms.Compose([
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor [0,1] and (C,H,W)
    # transforms.Normalize(mean=[0.5], std=[0.5]) # Example for standardization
])


# --- 1. Custom Dataset Class ---
class SiameseDataset(Dataset):
    def __init__(self, pairs_csv_path, patches_root_dir, transform=None, patch_size=32):
        self.pairs_df = pd.read_csv(pairs_csv_path)
        self.new_patches_root_dir = os.path.normpath(patches_root_dir)
        self.transform = transform
        self.patch_size = patch_size

        self.old_root_prefix = os.path.normpath(r'E:\AI_Project\TestSample\Siamese\Output\Patches') 

        self.pairs_df['patch1_path'] = self.pairs_df['patch1_path'].apply(
            lambda p: p.replace(self.old_root_prefix, self.new_patches_root_dir)
        )
        self.pairs_df['patch2_path'] = self.pairs_df['patch2_path'].apply(
            lambda p: p.replace(self.old_root_prefix, self.new_patches_root_dir)
        )

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        img1_path = row['patch1_path']
        img2_path = row['patch2_path']
        label = row['label']

        # Use cv2.IMREAD_UNCHANGED to load 16-bit images correctly
        img1_np = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
        img2_np = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

        if img1_np is None:
            raise FileNotFoundError(f"Image not found or unable to load at: {img1_path}. "
                                    f"Original path from CSV: {self.pairs_df.loc[idx, 'patch1_path']}. "
                                    f"New root: {self.new_patches_root_dir}. Old root replaced: {self.old_root_prefix}.")
        if img2_np is None:
            raise FileNotFoundError(f"Image not found or unable to load at: {img2_path}. "
                                    f"Original path from CSV: {self.pairs_df.loc[idx, 'patch2_path']}. "
                                    f"New root: {self.new_patches_root_dir}. Old root replaced: {self.old_root_prefix}.")

        # Convert 16-bit numpy array to PIL Image.
        # PIL handles the interpretation of 16-bit images for transforms.
        # Ensure the image is 'L' (8-bit grayscale) or 'I;16' (16-bit grayscale).
        # We need to explicitly convert 16-bit NumPy array to PIL 'I;16' mode
        # and then scale to [0,1] for the network input.
        
        # Scaling 16-bit image to 8-bit for PIL before common transforms can lose precision.
        # It's better to pass 16-bit data through, and let ToTensor() handle the scaling to [0,1].
        # PIL's 'I;16' mode supports 16-bit.
        # cv2.imread(IMREAD_UNCHANGED) loads it as numpy.uint16
        
        # Convert to PIL Image:
        # For a 16-bit numpy array, PIL's fromarray will interpret it as 'I;16' mode.
        img1_pil = Image.fromarray(img1_np)
        img2_pil = Image.fromarray(img2_np)

        if self.transform:
            img1 = self.transform(img1_pil)
            img2 = self.transform(img2_pil)
        else:
            # If no transform, manually convert to float32 and normalize [0,1]
            # then add channel dimension and convert to tensor.
            img1 = torch.from_numpy(img1_np.astype(np.float32) / 65535.0).unsqueeze(0)
            img2 = torch.from_numpy(img2_np.astype(np.float32) / 65535.0).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label

# --- 2. Siamese Network Architecture ---
class SiameseNet(nn.Module):
    def __init__(self, patch_size=32):
        super(SiameseNet, self).__init__()
        self.patch_size = patch_size

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        dummy_input_size = self.patch_size // (2**3)
        self.fc1_input_features = 256 * dummy_input_size * dummy_input_size

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc1_input_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# --- 3. Loss Function (Contrastive Loss) ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# --- 4. Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path, metrics_log_path, start_epoch=0):
    best_val_loss = float('inf')
    
    train_losses_history = []
    val_losses_history = []
    val_accuracies_history = []
    epoch_metrics = []

    if start_epoch > 0:
        checkpoint_to_load = os.path.join(MODEL_SAVE_DIR, f'checkpoint_epoch_{start_epoch:03d}.pth')
        if os.path.exists(checkpoint_to_load):
            print(f"Resuming training from checkpoint: {checkpoint_to_load}")
            checkpoint = torch.load(checkpoint_to_load)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['best_val_loss']
            train_losses_history = checkpoint['train_loss_history']
            val_losses_history = checkpoint['val_loss_history']
            val_accuracies_history = checkpoint['val_acc_history']
            
            random.setstate(checkpoint['rng_states']['python_random'])
            np.random.set_state(checkpoint['rng_states']['numpy_random'])
            torch.set_rng_state(checkpoint['rng_states']['torch_random'])
            if device.type == 'cuda' and checkpoint['rng_states']['torch_cuda_random'] is not None:
                torch.cuda.set_rng_state(checkpoint['rng_states']['torch_cuda_random'])
            
            print(f"Resumed from Epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint '{checkpoint_to_load}' not found. Starting training from scratch (Epoch 1).")
            start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        
        for batch_idx, (img1, img2, label) in enumerate(train_loader_tqdm):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
            train_loader_tqdm.set_postfix(loss=running_loss / ((batch_idx + 1) * img1.size(0)))

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses_history.append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
            for img1, img2, label in val_loader_tqdm:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_running_loss += loss.item() * img1.size(0)

                euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                
                predicted_labels = (euclidean_distance < CLASSIFICATION_THRESHOLD).float()
                correct_predictions += (predicted_labels == label).sum().item()
                total_samples += label.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses_history.append(epoch_val_loss)
        val_acc = correct_predictions / total_samples
        val_accuracies_history.append(val_acc)

        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'val_accuracy': val_acc
        })

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss_history': train_losses_history,
            'val_loss_history': val_losses_history,
            'val_acc_history': val_accuracies_history,
            'rng_states': {
                'python_random': random.getstate(),
                'numpy_random': np.random.get_state(),
                'torch_random': torch.get_rng_state(),
                'torch_cuda_random': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            },
            'hyperparameters': {
                'BATCH_SIZE': BATCH_SIZE,
                'NUM_EPOCHS': NUM_EPOCHS,
                'LEARNING_RATE': LEARNING_RATE,
                'MARGIN': MARGIN,
                'PATCH_SIZE': PATCH_SIZE,
                'CLASSIFICATION_THRESHOLD': CLASSIFICATION_THRESHOLD,
            }
        }
        epoch_checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch+1:03d}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Checkpoint saved for Epoch {epoch+1} to {epoch_checkpoint_path}")


        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model updated and saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    print("Training complete!")
    
    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_df.to_csv(metrics_log_path, index=False)
    print(f"Training metrics saved to {metrics_log_path}")

    return train_losses_history, val_losses_history, val_accuracies_history

# --- Main Execution ---
if __name__ == "__main__":
    RESUME_FROM_EPOCH = 0

    # 1. Data Loading - Pass the defined transforms
    train_dataset = SiameseDataset(os.path.join(PAIRS_OUTPUT_DIR, 'train_pairs.csv'), PATCHES_OUTPUT_DIR, transform=train_transform, patch_size=PATCH_SIZE)
    val_dataset = SiameseDataset(os.path.join(PAIRS_OUTPUT_DIR, 'val_pairs.csv'), PATCHES_OUTPUT_DIR, transform=val_test_transform, patch_size=PATCH_SIZE)
    test_dataset = SiameseDataset(os.path.join(PAIRS_OUTPUT_DIR, 'test_pairs.csv'), PATCHES_OUTPUT_DIR, transform=val_test_transform, patch_size=PATCH_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # 2. Model, Loss, Optimizer
    model = SiameseNet(patch_size=PATCH_SIZE).to(DEVICE)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Train the model
    print("\n--- Starting Siamese Network Training ---")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, BEST_MODEL_PATH, METRICS_LOG_PATH,
        start_epoch=RESUME_FROM_EPOCH
    )

    # --- Plotting Training Progress ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Siamese Network Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_loss_plot.png'))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Siamese Network Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'validation_accuracy_plot.png'))

    # --- Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    test_running_loss = 0.0
    test_accuracy = 0.0
    test_total_samples = 0
    test_correct_predictions = 0

    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Test Evaluation")
        for img1, img2, label in test_loader_tqdm:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            test_running_loss += loss.item() * img1.size(0)

            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            
            predicted_labels = (euclidean_distance < CLASSIFICATION_THRESHOLD).float()
            test_correct_predictions += (predicted_labels == label).sum().item()
            test_total_samples += label.size(0)

    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = test_correct_predictions / test_total_samples
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("Siamese Network training and evaluation complete!")