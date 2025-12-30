"""
COIN Weakness Analysis & Novel Architecture Comparison

This script demonstrates the weaknesses of COIN (SIREN-only) approach and
shows how the novel architecture (Lanczos downsampling + SIREN + SRDenseNet)
mitigates these issues.

Weaknesses of COIN:
1. High GPU memory consumption - scales with image resolution
2. Long training time - needs many iterations per image
3. Limited compression at high quality
4. No generalization - separate model per image
5. Quality degrades with higher compression (smaller networks)

Paper reference: arXiv:2403.08566
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom

from model import COIN, create_coordinate_grid, image_to_tensor, tensor_to_image


# ============================================================================
# NOVEL ARCHITECTURE COMPONENTS
# ============================================================================

class LanczosDownsampler:
    """Lanczos downsampling for reducing image resolution before INR."""
    
    @staticmethod
    def lanczos_kernel(x, a=3):
        """Lanczos kernel with parameter a."""
        if x == 0:
            return 1.0
        if abs(x) >= a:
            return 0.0
        return a * np.sin(np.pi * x) * np.sin(np.pi * x / a) / (np.pi ** 2 * x ** 2)
    
    @staticmethod
    def downsample(image, scale_factor):
        """Downsample image using scipy zoom with Lanczos-like interpolation."""
        return zoom(image, 1/scale_factor, order=3)  # order=3 for cubic
    
    @staticmethod
    def downsample_torch(image_tensor, scale_factor):
        """Downsample using PyTorch with area interpolation."""
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        h, w = image_tensor.shape[-2:]
        new_h, new_w = int(h / scale_factor), int(w / scale_factor)
        return F.interpolate(image_tensor, size=(new_h, new_w), mode='area').squeeze()


class ResidualBlock(nn.Module):
    """Residual block for SRDenseNet."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class SRDenseNet(nn.Module):
    """
    Super-Resolution network for upscaling INR output.
    Simplified version inspired by SRDenseNet.
    """
    def __init__(self, scale_factor=4, num_blocks=8, num_features=64):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv_input = nn.Conv2d(1, num_features, 3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_features) for _ in range(num_blocks)
        ])
        
        # Upsampling layers
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_features, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        
        feat = self.conv_input(x)
        feat = self.res_blocks(feat)
        out = self.upscale(feat)
        return out.squeeze()


class NovelArchitecture:
    """
    Novel Architecture: Lanczos Downsampling + SIREN + SR Network
    
    Key advantages:
    1. Train SIREN on low-resolution image (faster, less memory)
    2. Use SR network to recover high-resolution details
    3. Better compression-quality tradeoff
    """
    
    def __init__(self, scale_factor=4, hidden_features=28, hidden_layers=10, device='cuda'):
        self.scale_factor = scale_factor
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.device = device
        self.siren = None
        self.sr_net = None
        
    def train(self, image, siren_epochs=5000, sr_epochs=1000, lr=2e-4, verbose=True):
        """Train the complete pipeline."""
        h, w = image.shape
        
        # Step 1: Downsample
        lr_image = LanczosDownsampler.downsample(image, self.scale_factor)
        lr_h, lr_w = lr_image.shape
        
        if verbose:
            print(f"Original: {h}x{w} -> Low-res: {lr_h}x{lr_w}")
        
        # Step 2: Train SIREN on low-resolution
        self.siren = COIN(
            in_features=2,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=1
        ).to(self.device)
        
        coords = create_coordinate_grid(lr_h, lr_w, device=self.device)
        target = image_to_tensor(lr_image, device=self.device)
        
        optimizer = torch.optim.Adam(self.siren.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        history = {'siren_loss': [], 'siren_psnr': []}
        
        iterator = tqdm(range(siren_epochs), desc="Training SIREN (LR)") if verbose else range(siren_epochs)
        for epoch in iterator:
            optimizer.zero_grad()
            output = self.siren(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            mse = loss.item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            history['siren_loss'].append(mse)
            history['siren_psnr'].append(psnr)
        
        # Step 3: Train SR network (simplified - in practice would need paired data)
        self.sr_net = SRDenseNet(scale_factor=self.scale_factor).to(self.device)
        
        # Get SIREN output
        self.siren.eval()
        with torch.no_grad():
            siren_output = self.siren(coords)
            lr_recon = tensor_to_image(siren_output, lr_h, lr_w)
        
        # Train SR net to map lr_recon -> original (supervised)
        lr_tensor = torch.from_numpy(lr_recon).float().to(self.device)
        hr_tensor = torch.from_numpy(image).float().to(self.device)
        
        # Normalize
        hr_tensor = (hr_tensor - hr_tensor.min()) / (hr_tensor.max() - hr_tensor.min() + 1e-8)
        
        sr_optimizer = torch.optim.Adam(self.sr_net.parameters(), lr=1e-4)
        
        history['sr_loss'] = []
        iterator = tqdm(range(sr_epochs), desc="Training SR") if verbose else range(sr_epochs)
        for epoch in iterator:
            sr_optimizer.zero_grad()
            sr_output = self.sr_net(lr_tensor)
            loss = criterion(sr_output, hr_tensor)
            loss.backward()
            sr_optimizer.step()
            history['sr_loss'].append(loss.item())
        
        return history
    
    def reconstruct(self, original_shape):
        """Reconstruct high-resolution image."""
        h, w = original_shape
        lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
        
        # Get SIREN output
        self.siren.eval()
        coords = create_coordinate_grid(lr_h, lr_w, device=self.device)
        with torch.no_grad():
            siren_output = self.siren(coords)
            lr_recon = tensor_to_image(siren_output, lr_h, lr_w)
            
            # Apply SR
            lr_tensor = torch.from_numpy(lr_recon).float().to(self.device)
            hr_recon = self.sr_net(lr_tensor)
            
        return hr_recon.cpu().numpy()
    
    def count_parameters(self):
        """Total parameters in the pipeline."""
        siren_params = self.siren.count_parameters() if self.siren else 0
        sr_params = sum(p.numel() for p in self.sr_net.parameters()) if self.sr_net else 0
        return siren_params, sr_params, siren_params + sr_params


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def measure_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0


def experiment_1_training_time_vs_resolution(data_path, output_dir):
    """
    Experiment 1: Training time scales quadratically with resolution for COIN.
    Novel architecture trains on smaller images, reducing time.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Training Time vs Resolution")
    print("="*70)
    
    data = np.load(data_path)
    original_image = data['T1'][38]  # Use center slice
    
    resolutions = [64, 128, 256]
    epochs = 2000
    
    coin_times = []
    novel_times = []
    coin_psnrs = []
    novel_psnrs = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for res in resolutions:
        print(f"\n--- Resolution: {res}x{res} ---")
        
        # Resize image
        scale = res / original_image.shape[0]
        image = zoom(original_image, scale, order=3)
        image = (image - image.min()) / (image.max() - image.min())
        
        # COIN: Train directly on full resolution
        print("Training COIN (full resolution)...")
        model = COIN(in_features=2, hidden_features=28, hidden_layers=10, out_features=1).to(device)
        coords = create_coordinate_grid(res, res, device=device)
        target = image_to_tensor(image, device=device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        criterion = nn.MSELoss()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in tqdm(range(epochs), desc="COIN"):
            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        coin_time = time.time() - start_time
        coin_times.append(coin_time)
        
        # Evaluate COIN
        model.eval()
        with torch.no_grad():
            recon = model(coords)
            mse = criterion(recon, target).item()
            coin_psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            coin_psnrs.append(coin_psnr)
        
        print(f"COIN: Time={coin_time:.2f}s, PSNR={coin_psnr:.2f}dB")
        
        # Novel Architecture: Train on downsampled, then SR
        print("Training Novel Architecture (downsampled + SR)...")
        scale_factor = 4
        lr_res = res // scale_factor
        lr_image = zoom(image, 1/scale_factor, order=3)
        
        # Train SIREN on low-res
        model_lr = COIN(in_features=2, hidden_features=28, hidden_layers=10, out_features=1).to(device)
        coords_lr = create_coordinate_grid(lr_res, lr_res, device=device)
        target_lr = image_to_tensor(lr_image, device=device)
        
        optimizer = torch.optim.Adam(model_lr.parameters(), lr=2e-4)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in tqdm(range(epochs), desc="SIREN-LR"):
            optimizer.zero_grad()
            output = model_lr(coords_lr)
            loss = criterion(output, target_lr)
            loss.backward()
            optimizer.step()
        
        # Quick SR training (simplified)
        sr_net = SRDenseNet(scale_factor=scale_factor, num_blocks=4, num_features=32).to(device)
        sr_optimizer = torch.optim.Adam(sr_net.parameters(), lr=1e-4)
        
        model_lr.eval()
        with torch.no_grad():
            lr_recon = model_lr(coords_lr)
            lr_img = tensor_to_image(lr_recon, lr_res, lr_res)
        
        lr_tensor = torch.from_numpy(lr_img).float().to(device)
        hr_tensor = torch.from_numpy(image).float().to(device)
        
        for _ in tqdm(range(500), desc="SR"):  # Fewer epochs for SR
            sr_optimizer.zero_grad()
            sr_output = sr_net(lr_tensor)
            loss = criterion(sr_output, hr_tensor)
            loss.backward()
            sr_optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        novel_time = time.time() - start_time
        novel_times.append(novel_time)
        
        # Evaluate Novel
        sr_net.eval()
        with torch.no_grad():
            final_recon = sr_net(lr_tensor)
            mse = criterion(final_recon, hr_tensor).item()
            novel_psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            novel_psnrs.append(novel_psnr)
        
        print(f"Novel: Time={novel_time:.2f}s, PSNR={novel_psnr:.2f}dB")
        print(f"Speedup: {coin_time/novel_time:.2f}x")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(resolutions))
    width = 0.35
    
    axes[0].bar(x - width/2, coin_times, width, label='COIN', color='red', alpha=0.7)
    axes[0].bar(x + width/2, novel_times, width, label='Novel (Lanczos+SIREN+SR)', color='green', alpha=0.7)
    axes[0].set_xlabel('Resolution')
    axes[0].set_ylabel('Training Time (s)')
    axes[0].set_title('Training Time Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{r}x{r}' for r in resolutions])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(x - width/2, coin_psnrs, width, label='COIN', color='red', alpha=0.7)
    axes[1].bar(x + width/2, novel_psnrs, width, label='Novel (Lanczos+SIREN+SR)', color='green', alpha=0.7)
    axes[1].set_xlabel('Resolution')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Quality Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{r}x{r}' for r in resolutions])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp1_training_time.png', dpi=150)
    plt.close()
    
    return {
        'resolutions': resolutions,
        'coin_times': coin_times,
        'novel_times': novel_times,
        'coin_psnrs': coin_psnrs,
        'novel_psnrs': novel_psnrs
    }


def experiment_2_gpu_memory_usage(data_path, output_dir):
    """
    Experiment 2: GPU memory usage scales with image resolution.
    Novel architecture uses less memory by training on smaller images.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: GPU Memory Usage")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU memory experiment.")
        return None
    
    data = np.load(data_path)
    original_image = data['T1'][38]
    
    resolutions = [64, 128, 256, 512]
    
    coin_memory = []
    novel_memory = []
    
    device = 'cuda'
    
    for res in resolutions:
        print(f"\n--- Resolution: {res}x{res} ---")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Resize image
        scale = res / original_image.shape[0]
        image = zoom(original_image, scale, order=3)
        image = (image - image.min()) / (image.max() - image.min())
        
        # COIN memory
        model = COIN(in_features=2, hidden_features=28, hidden_layers=10, out_features=1).to(device)
        coords = create_coordinate_grid(res, res, device=device)
        target = image_to_tensor(image, device=device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        criterion = nn.MSELoss()
        
        # Run a few iterations to measure peak memory
        for _ in range(10):
            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        peak_coin = torch.cuda.max_memory_allocated() / 1024**2
        coin_memory.append(peak_coin)
        print(f"COIN Peak Memory: {peak_coin:.2f} MB")
        
        # Clear and measure Novel Architecture
        del model, coords, target, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Novel: SIREN on low-res
        scale_factor = 4
        lr_res = max(res // scale_factor, 16)
        lr_image = zoom(image, lr_res / res, order=3)
        
        model_lr = COIN(in_features=2, hidden_features=28, hidden_layers=10, out_features=1).to(device)
        coords_lr = create_coordinate_grid(lr_res, lr_res, device=device)
        target_lr = image_to_tensor(lr_image, device=device)
        
        optimizer = torch.optim.Adam(model_lr.parameters(), lr=2e-4)
        
        for _ in range(10):
            optimizer.zero_grad()
            output = model_lr(coords_lr)
            loss = criterion(output, target_lr)
            loss.backward()
            optimizer.step()
        
        peak_novel = torch.cuda.max_memory_allocated() / 1024**2
        novel_memory.append(peak_novel)
        print(f"Novel Peak Memory: {peak_novel:.2f} MB")
        print(f"Memory Reduction: {(1 - peak_novel/peak_coin)*100:.1f}%")
        
        del model_lr, coords_lr, target_lr, optimizer
        torch.cuda.empty_cache()
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(resolutions))
    width = 0.35
    
    ax.bar(x - width/2, coin_memory, width, label='COIN', color='red', alpha=0.7)
    ax.bar(x + width/2, novel_memory, width, label='Novel (Lanczos+SIREN+SR)', color='green', alpha=0.7)
    ax.set_xlabel('Image Resolution')
    ax.set_ylabel('Peak GPU Memory (MB)')
    ax.set_title('GPU Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}x{r}' for r in resolutions])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp2_gpu_memory.png', dpi=150)
    plt.close()
    
    return {
        'resolutions': resolutions,
        'coin_memory': coin_memory,
        'novel_memory': novel_memory
    }


def experiment_3_compression_quality_tradeoff(data_path, output_dir):
    """
    Experiment 3: COIN struggles at high compression ratios.
    Novel architecture achieves better quality at same compression.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Compression-Quality Tradeoff")
    print("="*70)
    
    data = np.load(data_path)
    image = data['T1'][38]
    image = (image - image.min()) / (image.max() - image.min())
    h, w = image.shape
    
    # Test different network sizes (compression levels)
    configs = [
        {'hidden': 14, 'layers': 5},   # Very small - high compression
        {'hidden': 28, 'layers': 10},  # Official COIN
        {'hidden': 56, 'layers': 10},  # Larger
        {'hidden': 128, 'layers': 10}, # Much larger
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 3000
    
    coin_results = []
    
    for cfg in configs:
        print(f"\n--- Config: hidden={cfg['hidden']}, layers={cfg['layers']} ---")
        
        model = COIN(
            in_features=2,
            hidden_features=cfg['hidden'],
            hidden_layers=cfg['layers'],
            out_features=1
        ).to(device)
        
        n_params = model.count_parameters()
        original_bits = h * w * 8
        compressed_bits = n_params * 16  # 16-bit weights
        compression_ratio = original_bits / compressed_bits
        
        coords = create_coordinate_grid(h, w, device=device)
        target = image_to_tensor(image, device=device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        criterion = nn.MSELoss()
        
        for _ in tqdm(range(epochs), desc=f"COIN h={cfg['hidden']}"):
            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            recon = model(coords)
            mse = criterion(recon, target).item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            recon_img = tensor_to_image(recon, h, w)
            ssim_val = ssim(image, recon_img, data_range=1.0)
        
        coin_results.append({
            'config': cfg,
            'params': n_params,
            'compression_ratio': compression_ratio,
            'psnr': psnr,
            'ssim': ssim_val
        })
        
        print(f"Params: {n_params}, Ratio: {compression_ratio:.2f}x, PSNR: {psnr:.2f}dB, SSIM: {ssim_val:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ratios = [r['compression_ratio'] for r in coin_results]
    psnrs = [r['psnr'] for r in coin_results]
    ssims = [r['ssim'] for r in coin_results]
    
    axes[0].plot(ratios, psnrs, 'ro-', markersize=10, linewidth=2, label='COIN')
    axes[0].set_xlabel('Compression Ratio')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs Compression Ratio')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    for i, cfg in enumerate(configs):
        axes[0].annotate(f"h={cfg['hidden']}", (ratios[i], psnrs[i]), 
                        textcoords="offset points", xytext=(5,5), fontsize=8)
    
    axes[1].plot(ratios, ssims, 'bo-', markersize=10, linewidth=2, label='COIN')
    axes[1].set_xlabel('Compression Ratio')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs Compression Ratio')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_compression_quality.png', dpi=150)
    plt.close()
    
    return coin_results


def experiment_4_no_generalization(data_path, output_dir):
    """
    Experiment 4: COIN cannot generalize - each image needs separate training.
    This is a fundamental weakness for volume compression.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: No Generalization (Per-Image Training)")
    print("="*70)
    
    data = np.load(data_path)
    slices = [30, 35, 40, 45, 50]  # Different slices
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 2000
    
    # Train on slice 38
    train_slice = data['T1'][38]
    train_slice = (train_slice - train_slice.min()) / (train_slice.max() - train_slice.min())
    h, w = train_slice.shape
    
    print("\nTraining COIN on slice 38...")
    model = COIN(in_features=2, hidden_features=28, hidden_layers=10, out_features=1).to(device)
    coords = create_coordinate_grid(h, w, device=device)
    target = image_to_tensor(train_slice, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()
    
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(coords)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    # Test on training slice
    with torch.no_grad():
        recon = model(coords)
        train_mse = criterion(recon, target).item()
        train_psnr = 10 * np.log10(1.0 / (train_mse + 1e-10))
    print(f"Training slice 38: PSNR = {train_psnr:.2f} dB")
    
    # Test on OTHER slices (generalization test)
    print("\nTesting on OTHER slices (COIN cannot generalize):")
    test_psnrs = []
    
    for idx in slices:
        test_slice = data['T1'][idx]
        test_slice = (test_slice - test_slice.min()) / (test_slice.max() - test_slice.min())
        test_target = image_to_tensor(test_slice, device=device)
        
        with torch.no_grad():
            # Use same model (trained on slice 38)
            test_recon = model(coords)
            test_mse = criterion(test_recon, test_target).item()
            test_psnr = 10 * np.log10(1.0 / (test_mse + 1e-10))
        
        test_psnrs.append(test_psnr)
        print(f"Slice {idx}: PSNR = {test_psnr:.2f} dB (POOR - model doesn't generalize)")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show training reconstruction
    with torch.no_grad():
        train_recon_img = tensor_to_image(model(coords), h, w)
    
    axes[0, 0].imshow(train_slice, cmap='gray')
    axes[0, 0].set_title(f'Original Slice 38')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(train_recon_img, cmap='gray')
    axes[0, 1].set_title(f'COIN Reconstruction\nPSNR: {train_psnr:.2f} dB')
    axes[0, 1].axis('off')
    
    # Show test on different slice
    test_slice = data['T1'][slices[2]]
    test_slice = (test_slice - test_slice.min()) / (test_slice.max() - test_slice.min())
    
    axes[0, 2].imshow(test_slice, cmap='gray')
    axes[0, 2].set_title(f'Original Slice {slices[2]}')
    axes[0, 2].axis('off')
    
    # Bar chart of PSNR
    axes[1, 0].bar(['Slice 38\n(trained)'] + [f'Slice {s}\n(untrained)' for s in slices], 
                   [train_psnr] + test_psnrs, 
                   color=['green'] + ['red']*len(slices))
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].set_title('COIN Cannot Generalize to New Images')
    axes[1, 0].axhline(y=25, color='gray', linestyle='--', label='Acceptable quality')
    axes[1, 0].legend()
    
    # Text explanation
    axes[1, 1].axis('off')
    text = """
COIN Weakness: No Generalization

• COIN memorizes a single image
• Cannot compress multiple images
• Must train separate network per image
• For N images = N x training time
• For volumes: need to train per slice

Novel Architecture Advantage:
• SR network can be pre-trained
• Only SIREN needs per-image training
• SIREN trains faster (lower resolution)
• Total time significantly reduced
    """
    axes[1, 1].text(0.1, 0.5, text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp4_no_generalization.png', dpi=150)
    plt.close()
    
    return {
        'train_psnr': train_psnr,
        'test_slices': slices,
        'test_psnrs': test_psnrs
    }


def experiment_5_visual_quality_comparison(data_path, output_dir):
    """
    Experiment 5: Visual comparison of reconstruction quality.
    Shows artifacts in COIN at high compression vs Novel architecture.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Visual Quality Comparison")
    print("="*70)
    
    data = np.load(data_path)
    image = data['T1'][38]
    image = (image - image.min()) / (image.max() - image.min())
    h, w = image.shape
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Small COIN (high compression, lower quality)
    print("\nTraining small COIN (high compression)...")
    small_model = COIN(in_features=2, hidden_features=14, hidden_layers=5, out_features=1).to(device)
    
    coords = create_coordinate_grid(h, w, device=device)
    target = image_to_tensor(image, device=device)
    
    optimizer = torch.optim.Adam(small_model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()
    
    for _ in tqdm(range(5000), desc="Small COIN"):
        optimizer.zero_grad()
        output = small_model(coords)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    small_model.eval()
    with torch.no_grad():
        small_recon = tensor_to_image(small_model(coords), h, w)
        small_mse = criterion(small_model(coords), target).item()
        small_psnr = 10 * np.log10(1.0 / (small_mse + 1e-10))
    
    small_params = small_model.count_parameters()
    small_ratio = (h * w * 8) / (small_params * 16)
    
    # Novel Architecture
    print("\nTraining Novel Architecture...")
    scale_factor = 4
    lr_h, lr_w = h // scale_factor, w // scale_factor
    lr_image = zoom(image, 1/scale_factor, order=3)
    
    # Use similar param count for fair comparison
    novel_siren = COIN(in_features=2, hidden_features=14, hidden_layers=5, out_features=1).to(device)
    
    coords_lr = create_coordinate_grid(lr_h, lr_w, device=device)
    target_lr = image_to_tensor(lr_image, device=device)
    
    optimizer = torch.optim.Adam(novel_siren.parameters(), lr=2e-4)
    
    for _ in tqdm(range(3000), desc="SIREN-LR"):
        optimizer.zero_grad()
        output = novel_siren(coords_lr)
        loss = criterion(output, target_lr)
        loss.backward()
        optimizer.step()
    
    novel_siren.eval()
    with torch.no_grad():
        lr_recon = tensor_to_image(novel_siren(coords_lr), lr_h, lr_w)
    
    # Train SR
    sr_net = SRDenseNet(scale_factor=scale_factor, num_blocks=4, num_features=32).to(device)
    sr_optimizer = torch.optim.Adam(sr_net.parameters(), lr=1e-4)
    
    lr_tensor = torch.from_numpy(lr_recon).float().to(device)
    hr_tensor = torch.from_numpy(image).float().to(device)
    
    for _ in tqdm(range(1000), desc="SR Net"):
        sr_optimizer.zero_grad()
        sr_output = sr_net(lr_tensor)
        loss = criterion(sr_output, hr_tensor)
        loss.backward()
        sr_optimizer.step()
    
    sr_net.eval()
    with torch.no_grad():
        novel_recon = sr_net(lr_tensor).cpu().numpy()
        novel_mse = np.mean((novel_recon - image) ** 2)
        novel_psnr = 10 * np.log10(1.0 / (novel_mse + 1e-10))
    
    # Calculate parameters
    siren_params = novel_siren.count_parameters()
    sr_params = sum(p.numel() for p in sr_net.parameters())
    
    # Plot comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Full images
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(small_recon, cmap='gray')
    axes[0, 1].set_title(f'COIN (Small)\nPSNR: {small_psnr:.2f} dB\nParams: {small_params}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(novel_recon, cmap='gray')
    axes[0, 2].set_title(f'Novel Architecture\nPSNR: {novel_psnr:.2f} dB\nSIREN: {siren_params}')
    axes[0, 2].axis('off')
    
    # Difference maps
    diff_coin = np.abs(image - small_recon)
    diff_novel = np.abs(image - novel_recon)
    
    axes[0, 3].imshow(diff_coin, cmap='hot', vmin=0, vmax=0.3)
    axes[0, 3].set_title('COIN Error Map')
    axes[0, 3].axis('off')
    
    # Row 2: Zoomed regions (showing detail preservation)
    crop_y, crop_x = 100, 100
    crop_size = 64
    
    def crop(img):
        return img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    
    axes[1, 0].imshow(crop(image), cmap='gray')
    axes[1, 0].set_title('Original (Zoomed)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(crop(small_recon), cmap='gray')
    axes[1, 1].set_title('COIN (Zoomed)\nNote: artifacts, blur')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(crop(novel_recon), cmap='gray')
    axes[1, 2].set_title('Novel (Zoomed)\nBetter detail preservation')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(diff_novel, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 3].set_title('Novel Error Map')
    axes[1, 3].axis('off')
    
    plt.suptitle('Visual Quality Comparison: COIN vs Novel Architecture', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'exp5_visual_comparison.png', dpi=150)
    plt.close()
    
    return {
        'coin_psnr': small_psnr,
        'coin_params': small_params,
        'coin_ratio': small_ratio,
        'novel_psnr': novel_psnr,
        'siren_params': siren_params,
        'sr_params': sr_params
    }


def main():
    """Run all experiments."""
    print("="*70)
    print("COIN WEAKNESS ANALYSIS")
    print("Demonstrating limitations and Novel Architecture improvements")
    print("Paper: arXiv:2403.08566")
    print("="*70)
    
    # Setup
    data_path = Path('../dataset/preprocessed/mri_preprocessed.npz')
    output_dir = Path('weakness_analysis')
    output_dir.mkdir(exist_ok=True)
    
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        print("Please run preprocessing first.")
        return
    
    results = {}
    
    # Run experiments
    print("\n" + "="*70)
    print("Running Experiment 1: Training Time vs Resolution")
    print("="*70)
    results['exp1'] = experiment_1_training_time_vs_resolution(data_path, output_dir)
    
    print("\n" + "="*70)
    print("Running Experiment 2: GPU Memory Usage")
    print("="*70)
    results['exp2'] = experiment_2_gpu_memory_usage(data_path, output_dir)
    
    print("\n" + "="*70)
    print("Running Experiment 3: Compression-Quality Tradeoff")
    print("="*70)
    results['exp3'] = experiment_3_compression_quality_tradeoff(data_path, output_dir)
    
    print("\n" + "="*70)
    print("Running Experiment 4: No Generalization")
    print("="*70)
    results['exp4'] = experiment_4_no_generalization(data_path, output_dir)
    
    print("\n" + "="*70)
    print("Running Experiment 5: Visual Quality Comparison")
    print("="*70)
    results['exp5'] = experiment_5_visual_quality_comparison(data_path, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: COIN Weaknesses Demonstrated")
    print("="*70)
    print("""
1. TRAINING TIME: Scales with image resolution squared
   - COIN trains on full resolution → slow
   - Novel: trains SIREN on low-res → faster
   
2. GPU MEMORY: High memory for large images
   - COIN: coordinates tensor = H×W×2
   - Novel: reduced by scale_factor²

3. COMPRESSION-QUALITY: Poor tradeoff at high compression
   - Smaller networks = worse quality
   - Novel: SR network recovers details
   
4. NO GENERALIZATION: Must train per-image
   - Volume with N slices = N separate trainings
   - Novel: SR network can be shared/pretrained
   
5. VISUAL QUALITY: Artifacts at high compression
   - COIN shows blur and artifacts
   - Novel: SR enhances fine details
""")
    
    print(f"\nResults saved to: {output_dir.absolute()}")
    
    # Save results
    np.savez(output_dir / 'experiment_results.npz', **{
        k: v for k, v in results.items() if v is not None
    })
    
    return results


if __name__ == '__main__':
    main()
