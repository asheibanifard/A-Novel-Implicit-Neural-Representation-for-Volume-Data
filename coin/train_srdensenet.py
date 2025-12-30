"""
Train SRDenseNet on MRI dataset with patch-based training.

Creates LR-HR pairs using Lanczos downsampling and trains
super-resolution models for scales 2, 4, and 8.

Usage:
    python train_srdensenet.py --scale 4 --epochs 100 --patch_size 48
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import zoom
import random
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from srdensenet import SRDenseNet, SRDenseNetLight


class MRIPatchDataset(Dataset):
    """
    Dataset for SR training with LR-HR patch pairs.
    
    Creates patches from MRI slices and generates LR versions
    using Lanczos/bicubic downsampling.
    """
    
    def __init__(
        self,
        data_path,
        modalities=['T1', 'PD', 'T2'],
        split='train',
        scale_factor=4,
        patch_size=48,
        patches_per_image=16,
        augment=True
    ):
        """
        Args:
            data_path: Path to preprocessed .npz file
            modalities: List of modalities to use
            split: 'train', 'val', or 'test'
            scale_factor: Downsampling factor (2, 4, or 8)
            patch_size: Size of HR patches (LR will be patch_size // scale_factor)
            patches_per_image: Number of random patches per image
            augment: Whether to apply data augmentation
        """
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // scale_factor
        self.patches_per_image = patches_per_image
        self.augment = augment
        
        # Load data
        data = np.load(data_path)
        
        # Get split indices
        if split == 'train':
            indices = data['train_idx']
        elif split == 'val':
            indices = data['val_idx']
        else:
            indices = data['test_idx']
        
        # Collect all slices
        self.images = []
        for mod in modalities:
            volume = data[mod]
            for idx in indices:
                if idx < len(volume):
                    img = volume[idx]
                    # Normalize to [0, 1]
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    self.images.append(img)
        
        print(f"Loaded {len(self.images)} images for {split} split")
        
        # Pre-extract patches for faster training
        self.patches = self._extract_patches()
    
    def _extract_patches(self):
        """Extract random patches from all images."""
        patches = []
        
        for img in self.images:
            h, w = img.shape
            
            for _ in range(self.patches_per_image):
                # Random position
                max_y = h - self.patch_size
                max_x = w - self.patch_size
                
                if max_y <= 0 or max_x <= 0:
                    continue
                
                y = random.randint(0, max_y)
                x = random.randint(0, max_x)
                
                # Extract HR patch
                hr_patch = img[y:y+self.patch_size, x:x+self.patch_size]
                
                # Create LR patch using bicubic downsampling
                lr_patch = zoom(hr_patch, 1/self.scale_factor, order=3)
                
                patches.append((lr_patch, hr_patch))
        
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        lr_patch, hr_patch = self.patches[idx]
        
        # Data augmentation
        if self.augment and random.random() > 0.5:
            # Random horizontal flip
            if random.random() > 0.5:
                lr_patch = np.fliplr(lr_patch).copy()
                hr_patch = np.fliplr(hr_patch).copy()
            # Random vertical flip
            if random.random() > 0.5:
                lr_patch = np.flipud(lr_patch).copy()
                hr_patch = np.flipud(hr_patch).copy()
            # Random 90 degree rotation
            k = random.randint(0, 3)
            lr_patch = np.rot90(lr_patch, k).copy()
            hr_patch = np.rot90(hr_patch, k).copy()
        
        # Convert to tensors
        lr_tensor = torch.from_numpy(lr_patch).float().unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_patch).float().unsqueeze(0)
        
        return lr_tensor, hr_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for lr_batch, hr_batch in dataloader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        sr_batch = model(lr_batch)
        
        # Ensure same shape
        if sr_batch.ndim == 3:
            sr_batch = sr_batch.unsqueeze(1)
        
        loss = criterion(sr_batch, hr_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    with torch.no_grad():
        for lr_batch, hr_batch in dataloader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            sr_batch = model(lr_batch)
            
            if sr_batch.ndim == 3:
                sr_batch = sr_batch.unsqueeze(1)
            
            loss = criterion(sr_batch, hr_batch)
            total_loss += loss.item()
            
            # Calculate PSNR and SSIM for each image
            sr_np = sr_batch.cpu().numpy()
            hr_np = hr_batch.cpu().numpy()
            
            for i in range(sr_np.shape[0]):
                sr_img = np.clip(sr_np[i, 0], 0, 1)
                hr_img = hr_np[i, 0]
                
                total_psnr += psnr(hr_img, sr_img, data_range=1.0)
                total_ssim += ssim(hr_img, sr_img, data_range=1.0)
                count += 1
    
    return {
        'loss': total_loss / len(dataloader),
        'psnr': total_psnr / count,
        'ssim': total_ssim / count
    }


def visualize_results(model, dataloader, device, output_path, scale_factor):
    """Visualize some SR results."""
    model.eval()
    
    # Get a batch
    lr_batch, hr_batch = next(iter(dataloader))
    lr_batch = lr_batch.to(device)
    
    with torch.no_grad():
        sr_batch = model(lr_batch)
    
    if sr_batch.ndim == 3:
        sr_batch = sr_batch.unsqueeze(1)
    
    # Plot first 4 examples
    n_show = min(4, lr_batch.shape[0])
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
    
    for i in range(n_show):
        lr_img = lr_batch[i, 0].cpu().numpy()
        hr_img = hr_batch[i, 0].numpy()
        sr_img = np.clip(sr_batch[i, 0].cpu().numpy(), 0, 1)
        
        # Bicubic upscale for comparison
        bicubic_img = zoom(lr_img, scale_factor, order=3)
        bicubic_img = np.clip(bicubic_img, 0, 1)
        
        # Crop to same size if needed
        min_h = min(hr_img.shape[0], sr_img.shape[0], bicubic_img.shape[0])
        min_w = min(hr_img.shape[1], sr_img.shape[1], bicubic_img.shape[1])
        hr_img = hr_img[:min_h, :min_w]
        sr_img = sr_img[:min_h, :min_w]
        bicubic_img = bicubic_img[:min_h, :min_w]
        
        # Calculate metrics
        sr_psnr = psnr(hr_img, sr_img, data_range=1.0)
        bicubic_psnr = psnr(hr_img, bicubic_img, data_range=1.0)
        
        axes[i, 0].imshow(lr_img, cmap='gray')
        axes[i, 0].set_title(f'LR ({lr_img.shape[0]}x{lr_img.shape[1]})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(bicubic_img, cmap='gray')
        axes[i, 1].set_title(f'Bicubic\nPSNR: {bicubic_psnr:.2f}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(sr_img, cmap='gray')
        axes[i, 2].set_title(f'SRDenseNet\nPSNR: {sr_psnr:.2f}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(hr_img, cmap='gray')
        axes[i, 3].set_title(f'HR Ground Truth')
        axes[i, 3].axis('off')
    
    plt.suptitle(f'Super Resolution Results (x{scale_factor})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train SRDenseNet on MRI data')
    parser.add_argument('--data_path', type=str, default='../dataset/preprocessed/mri_preprocessed.npz')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4, 8])
    parser.add_argument('--patch_size', type=int, default=48, help='HR patch size')
    parser.add_argument('--patches_per_image', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--light', action='store_true', help='Use lightweight model')
    parser.add_argument('--output_dir', type=str, default='sr_models')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create datasets
    print(f"\nCreating datasets for scale x{args.scale}...")
    train_dataset = MRIPatchDataset(
        args.data_path,
        split='train',
        scale_factor=args.scale,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        augment=True
    )
    
    val_dataset = MRIPatchDataset(
        args.data_path,
        split='val',
        scale_factor=args.scale,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image // 2,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train patches: {len(train_dataset)}, Val patches: {len(val_dataset)}")
    
    # Create model
    if args.light:
        model = SRDenseNetLight(
            scale_factor=args.scale,
            in_channels=1,
            out_channels=1,
            num_features=32,
            num_blocks=4
        ).to(device)
        model_name = f'srdensenet_light_x{args.scale}'
    else:
        model = SRDenseNet(
            scale_factor=args.scale,
            in_channels=1,
            out_channels=1,
            num_features=args.num_features,
            num_blocks=args.num_blocks
        ).to(device)
        model_name = f'srdensenet_x{args.scale}'
    
    print(f"\nModel: {model_name}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # L1 loss often works better for SR
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
    best_psnr = 0
    
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_psnr'].append(val_metrics['psnr'])
        history['val_ssim'].append(val_metrics['ssim'])
        
        # Update scheduler
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_metrics['loss']:.4f} - "
              f"Val PSNR: {val_metrics['psnr']:.2f} - "
              f"Val SSIM: {val_metrics['ssim']:.4f}")
        
        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': val_metrics['ssim'],
                'scale_factor': args.scale,
                'config': {
                    'num_features': args.num_features if not args.light else 32,
                    'num_blocks': args.num_blocks if not args.light else 4,
                    'light': args.light
                }
            }, output_dir / f'{model_name}_best.pth')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'psnr': val_metrics['psnr'],
        'ssim': val_metrics['ssim'],
        'scale_factor': args.scale,
    }, output_dir / f'{model_name}_final.pth')
    
    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['val_psnr'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title(f'Validation PSNR (Best: {best_psnr:.2f})')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['val_ssim'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].set_title('Validation SSIM')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'SRDenseNet x{args.scale} Training History', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_history.png', dpi=150)
    plt.close()
    
    # Visualize results
    visualize_results(model, val_loader, device, output_dir / f'{model_name}_results.png', args.scale)
    
    # Final summary
    print("\n" + "="*60)
    print(f"Training Complete - SRDenseNet x{args.scale}")
    print("="*60)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best SSIM: {max(history['val_ssim']):.4f}")
    print(f"Model saved to: {output_dir / f'{model_name}_best.pth'}")
    
    return history


def train_all_scales(data_path, epochs=100, light=False):
    """Train models for all scales (2, 4, 8)."""
    
    results = {}
    
    for scale in [2, 4, 8]:
        print("\n" + "="*70)
        print(f"TRAINING SRDenseNet x{scale}")
        print("="*70)
        
        # Adjust patch size based on scale
        if scale == 2:
            patch_size = 64
        elif scale == 4:
            patch_size = 48
        else:  # scale == 8
            patch_size = 64  # Need larger patches for 8x
        
        # Create datasets
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output_dir = Path('sr_models')
        output_dir.mkdir(exist_ok=True)
        
        train_dataset = MRIPatchDataset(
            data_path, split='train', scale_factor=scale,
            patch_size=patch_size, patches_per_image=32, augment=True
        )
        val_dataset = MRIPatchDataset(
            data_path, split='val', scale_factor=scale,
            patch_size=patch_size, patches_per_image=16, augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # Create model
        if light:
            model = SRDenseNetLight(scale_factor=scale, num_features=32, num_blocks=4).to(device)
            model_name = f'srdensenet_light_x{scale}'
        else:
            model = SRDenseNet(scale_factor=scale, num_features=64, num_blocks=8).to(device)
            model_name = f'srdensenet_x{scale}'
        
        print(f"Model parameters: {model.count_parameters():,}")
        
        # Training
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        best_psnr = 0
        
        for epoch in tqdm(range(epochs), desc=f"Training x{scale}"):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()
            
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'psnr': best_psnr,
                    'ssim': val_metrics['ssim'],
                    'scale_factor': scale,
                }, output_dir / f'{model_name}_best.pth')
        
        results[scale] = {
            'psnr': best_psnr,
            'ssim': val_metrics['ssim'],
            'params': model.count_parameters()
        }
        
        print(f"x{scale} - Best PSNR: {best_psnr:.2f} dB, SSIM: {val_metrics['ssim']:.4f}")
        
        # Visualize
        visualize_results(model, val_loader, device, output_dir / f'{model_name}_results.png', scale)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for scale, res in results.items():
        print(f"x{scale}: PSNR={res['psnr']:.2f}dB, SSIM={res['ssim']:.4f}, Params={res['params']:,}")
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Train all scales
        train_all_scales('../dataset/preprocessed/mri_preprocessed.npz', epochs=100, light=False)
    else:
        main()
