"""
Train SRDenseNet on MRI dataset - Simplified robust version.

Trains SR models for scales 2, 4, and 8 using smaller patches.
"""

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
import math


# ============================================================================
# Simple but effective SR model
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class SRNet(nn.Module):
    """Simple but effective SR network for medical images."""
    
    def __init__(self, scale_factor=4, num_features=64, num_blocks=8):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Input
        self.head = nn.Sequential(
            nn.Conv2d(1, num_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.body = nn.Sequential(*[ResBlock(num_features) for _ in range(num_blocks)])
        
        # Upsampling
        up_layers = []
        for _ in range(int(math.log2(scale_factor))):
            up_layers.extend([
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
        self.upsample = nn.Sequential(*up_layers)
        
        # Output
        self.tail = nn.Conv2d(num_features, 1, 3, padding=1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1) if x.shape[0] > x.shape[-1] else x.unsqueeze(0)
        
        # Bicubic upscale for global residual
        bicubic = torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False
        )
        
        feat = self.head(x)
        feat = self.body(feat) + feat  # Global skip
        feat = self.upsample(feat)
        out = self.tail(feat)
        
        # Global residual learning
        out = out + bicubic
        
        return out.squeeze()
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Dataset
# ============================================================================

class SRPatchDataset(Dataset):
    """SR dataset with LR-HR patch pairs."""
    
    def __init__(self, data_path, split='train', scale=4, patch_size=48, 
                 patches_per_img=16, augment=True):
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')
        
        # Load data
        data = np.load(data_path)
        indices = data[f'{split}_idx']
        
        # Collect images from all modalities
        self.images = []
        for mod in ['T1', 'PD', 'T2']:
            vol = data[mod]
            for idx in indices:
                if idx < len(vol):
                    img = vol[idx].astype(np.float32)
                    # Normalize to [0, 1]
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    self.images.append(img)
        
        # Extract patches
        self.lr_patches = []
        self.hr_patches = []
        
        for img in self.images:
            h, w = img.shape
            for _ in range(patches_per_img):
                y = random.randint(0, max(0, h - patch_size))
                x = random.randint(0, max(0, w - patch_size))
                
                hr_patch = img[y:y+patch_size, x:x+patch_size]
                if hr_patch.shape != (patch_size, patch_size):
                    continue
                
                # Downsample to create LR
                lr_patch = zoom(hr_patch, 1.0/scale, order=1)  # Bilinear
                
                self.lr_patches.append(lr_patch)
                self.hr_patches.append(hr_patch)
        
        print(f"[{split}] Loaded {len(self.lr_patches)} patches")
    
    def __len__(self):
        return len(self.lr_patches)
    
    def __getitem__(self, idx):
        lr = self.lr_patches[idx].copy()
        hr = self.hr_patches[idx].copy()
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                lr = np.fliplr(lr).copy()
                hr = np.fliplr(hr).copy()
            if random.random() > 0.5:
                lr = np.flipud(lr).copy()
                hr = np.flipud(hr).copy()
            k = random.randint(0, 3)
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()
        
        lr = torch.from_numpy(lr).float().unsqueeze(0)
        hr = torch.from_numpy(hr).float().unsqueeze(0)
        
        return lr, hr


# ============================================================================
# Training
# ============================================================================

def train_sr_model(scale, data_path, output_dir, epochs=100, batch_size=16, 
                   patch_size=48, num_features=64, num_blocks=8, lr=1e-4):
    """Train SR model for a specific scale."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Training SRNet x{scale}")
    print(f"{'='*60}")
    
    # Datasets
    train_ds = SRPatchDataset(data_path, 'train', scale, patch_size, patches_per_img=32)
    val_ds = SRPatchDataset(data_path, 'val', scale, patch_size, patches_per_img=16, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = SRNet(scale_factor=scale, num_features=num_features, num_blocks=num_blocks).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Training setup
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_psnr = 0
    history = {'train_loss': [], 'val_psnr': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            
            optimizer.zero_grad()
            sr_batch = model(lr_batch)
            if sr_batch.dim() == 3:
                sr_batch = sr_batch.unsqueeze(1)
            
            loss = criterion(sr_batch, hr_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validate
        model.eval()
        val_psnr = 0
        val_count = 0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                sr_batch = model(lr_batch)
                if sr_batch.dim() == 3:
                    sr_batch = sr_batch.unsqueeze(1)
                
                # PSNR per image
                for i in range(sr_batch.shape[0]):
                    sr = torch.clamp(sr_batch[i], 0, 1)
                    hr = hr_batch[i]
                    mse = torch.mean((sr - hr) ** 2).item()
                    if mse > 0:
                        psnr = 10 * np.log10(1.0 / mse)
                        val_psnr += psnr
                        val_count += 1
        
        val_psnr /= max(val_count, 1)
        history['val_psnr'].append(val_psnr)
        scheduler.step()
        
        # Progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} dB")
        
        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'model_state_dict': model.state_dict(),
                'scale': scale,
                'psnr': best_psnr,
                'num_features': num_features,
                'num_blocks': num_blocks
            }, output_dir / f'srnet_x{scale}_best.pth')
    
    print(f"Best PSNR: {best_psnr:.2f} dB")
    
    # Visualize
    visualize_sr(model, val_loader, device, output_dir / f'srnet_x{scale}_results.png', scale)
    
    return best_psnr, history


def visualize_sr(model, dataloader, device, save_path, scale):
    """Visualize SR results."""
    model.eval()
    lr_batch, hr_batch = next(iter(dataloader))
    lr_batch = lr_batch.to(device)
    
    with torch.no_grad():
        sr_batch = model(lr_batch)
    if sr_batch.dim() == 3:
        sr_batch = sr_batch.unsqueeze(1)
    
    n = min(4, lr_batch.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5*n))
    
    for i in range(n):
        lr = lr_batch[i, 0].cpu().numpy()
        hr = hr_batch[i, 0].numpy()
        sr = torch.clamp(sr_batch[i, 0], 0, 1).cpu().numpy()
        
        # Bicubic baseline
        bicubic = zoom(lr, scale, order=3)
        bicubic = np.clip(bicubic, 0, 1)
        
        # Ensure same size
        h, w = min(hr.shape[0], sr.shape[0], bicubic.shape[0]), min(hr.shape[1], sr.shape[1], bicubic.shape[1])
        hr, sr, bicubic = hr[:h, :w], sr[:h, :w], bicubic[:h, :w]
        
        # PSNR
        mse_sr = np.mean((sr - hr) ** 2)
        mse_bi = np.mean((bicubic - hr) ** 2)
        psnr_sr = 10 * np.log10(1.0 / (mse_sr + 1e-10))
        psnr_bi = 10 * np.log10(1.0 / (mse_bi + 1e-10))
        
        axes[i, 0].imshow(lr, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'LR ({lr.shape[0]}x{lr.shape[1]})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(bicubic, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Bicubic\nPSNR: {psnr_bi:.1f}dB')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(sr, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'SRNet\nPSNR: {psnr_sr:.1f}dB')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(hr, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title('HR (GT)')
        axes[i, 3].axis('off')
    
    plt.suptitle(f'Super Resolution x{scale}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Train SR models for all scales."""
    data_path = Path('../dataset/preprocessed/mri_preprocessed.npz')
    output_dir = Path('sr_models')
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for scale in [2, 4, 8]:
        # Adjust patch size - need larger for higher scales
        patch_size = {2: 64, 4: 48, 8: 64}[scale]
        
        psnr, hist = train_sr_model(
            scale=scale,
            data_path=data_path,
            output_dir=output_dir,
            epochs=100,
            batch_size=16,
            patch_size=patch_size,
            num_features=64,
            num_blocks=8,
            lr=1e-4
        )
        results[scale] = psnr
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for scale, psnr in results.items():
        print(f"SRNet x{scale}: Best PSNR = {psnr:.2f} dB")
    
    # Save summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    scales = list(results.keys())
    psnrs = list(results.values())
    bars = ax.bar([f'x{s}' for s in scales], psnrs, color=['#2ecc71', '#3498db', '#9b59b6'])
    ax.set_ylabel('PSNR (dB)')
    ax.set_xlabel('Scale Factor')
    ax.set_title('SRNet Performance on MRI Dataset')
    for bar, p in zip(bars, psnrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{p:.1f}', ha='center', fontsize=12)
    ax.set_ylim(0, max(psnrs) + 5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'sr_summary.png', dpi=150)
    plt.close()
    
    return results


if __name__ == '__main__':
    main()
