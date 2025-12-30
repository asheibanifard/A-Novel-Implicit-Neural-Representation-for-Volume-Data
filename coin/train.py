"""
Training script for COIN model on MRI slices.

Usage:
    python train.py --slice_idx 38 --epochs 2000 --hidden_features 256
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import COIN, create_coordinate_grid, image_to_tensor, tensor_to_image


def train_coin(
    image: np.ndarray,
    hidden_features: int = 256,
    hidden_layers: int = 3,
    epochs: int = 2000,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
):
    """
    Train COIN model to fit a single image.
    
    Args:
        image: (H, W) grayscale image
        hidden_features: Width of hidden layers
        hidden_layers: Number of hidden layers
        epochs: Training epochs
        lr: Learning rate
        device: Device to train on
        verbose: Print progress
        
    Returns:
        model: Trained COIN model
        history: Training history dict
    """
    height, width = image.shape
    
    # Create model
    model = COIN(
        in_features=2,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=1
    ).to(device)
    
    # Create coordinate grid and target
    coords = create_coordinate_grid(height, width, device=device)
    target = image_to_tensor(image, device=device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    history = {'loss': [], 'psnr': []}
    
    iterator = tqdm(range(epochs), desc="Training COIN") if verbose else range(epochs)
    
    for epoch in iterator:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(coords)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate PSNR
        mse = loss.item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
        history['loss'].append(mse)
        history['psnr'].append(psnr)
        
        if verbose and (epoch + 1) % 500 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Loss: {mse:.6f} - PSNR: {psnr:.2f} dB")
    
    return model, history


def reconstruct_image(model, height, width, device='cpu'):
    """Reconstruct image from trained model."""
    model.eval()
    coords = create_coordinate_grid(height, width, device=device)
    with torch.no_grad():
        output = model(coords)
    return tensor_to_image(output, height, width)


def compute_compression_ratio(model, height, width, bits_per_weight=32):
    """
    Compute compression ratio.
    
    Original size: H * W * bits_per_pixel (assuming 8-bit grayscale = 8)
    Compressed size: num_params * bits_per_weight
    """
    original_bits = height * width * 8  # 8-bit grayscale
    compressed_bits = model.count_parameters() * bits_per_weight
    ratio = original_bits / compressed_bits
    bpp = compressed_bits / (height * width)  # bits per pixel
    return ratio, original_bits / 8 / 1024, compressed_bits / 8 / 1024, bpp  # KB


def main():
    parser = argparse.ArgumentParser(description='Train COIN on MRI slice')
    parser.add_argument('--data_path', type=str, default='../dataset/preprocessed/mri_preprocessed.npz')
    parser.add_argument('--modality', type=str, default='T1', choices=['PD', 'T1', 'T2'])
    parser.add_argument('--slice_idx', type=int, default=38)
    # Official COIN config: layer_size=28, num_layers=10
    parser.add_argument('--hidden_features', type=int, default=28)
    parser.add_argument('--hidden_layers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--w0', type=float, default=30.0)
    parser.add_argument('--half_precision', action='store_true', help='Use half precision (16-bit) for compression')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    data = np.load(args.data_path)
    image = data[args.modality][args.slice_idx]
    print(f"Loaded {args.modality} slice {args.slice_idx}: shape={image.shape}")
    
    # Train model
    model, history = train_coin(
        image,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # Reconstruct
    height, width = image.shape
    reconstructed = reconstruct_image(model, height, width, device=device)
    
    # Compute metrics - full precision (32-bit)
    mse = np.mean((image - reconstructed) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    ratio_fp, orig_kb, comp_kb_fp, bpp_fp = compute_compression_ratio(model, height, width, bits_per_weight=32)
    
    # Half precision (16-bit) metrics
    ratio_hp, _, comp_kb_hp, bpp_hp = compute_compression_ratio(model, height, width, bits_per_weight=16)
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Original size: {orig_kb:.2f} KB")
    print(f"\nFull Precision (32-bit):")
    print(f"  Compressed size: {comp_kb_fp:.2f} KB")
    print(f"  BPP: {bpp_fp:.2f}")
    print(f"  Compression ratio: {ratio_fp:.2f}x")
    print(f"\nHalf Precision (16-bit):")
    print(f"  Compressed size: {comp_kb_hp:.2f} KB")
    print(f"  BPP: {bpp_hp:.2f}")
    print(f"  Compression ratio: {ratio_hp:.2f}x")
    print(f"\nFinal PSNR: {psnr:.2f} dB")
    print(f"Final MSE: {mse:.6f}")
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'psnr': psnr,
        'mse': mse
    }, output_dir / f'coin_{args.modality}_slice{args.slice_idx}.pt')
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title(f'Original ({args.modality} slice {args.slice_idx})')
    axes[0, 0].axis('off')
    
    # Reconstructed
    axes[0, 1].imshow(reconstructed, cmap='gray')
    axes[0, 1].set_title(f'Reconstructed (PSNR: {psnr:.2f} dB)')
    axes[0, 1].axis('off')
    
    # Difference
    diff = np.abs(image - reconstructed)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title(f'Absolute Difference (MSE: {mse:.6f})')
    axes[0, 2].axis('off')
    
    # Training curves
    axes[1, 0].plot(history['loss'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['psnr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('PSNR over Training')
    axes[1, 1].grid(True)
    
    # Info text
    axes[1, 2].axis('off')
    info_text = f"""
    COIN Model Summary
    ==================
    Hidden features: {args.hidden_features}
    Hidden layers: {args.hidden_layers}
    Total parameters: {model.count_parameters():,}
    
    Compression (16-bit)
    ====================
    Original: {orig_kb:.2f} KB
    Compressed: {comp_kb_hp:.2f} KB
    BPP: {bpp_hp:.2f}
    Ratio: {ratio_hp:.2f}x
    
    Quality
    =======
    PSNR: {psnr:.2f} dB
    MSE: {mse:.6f}
    """
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'coin_{args.modality}_slice{args.slice_idx}_results.png', dpi=150)
    plt.show()
    
    print(f"\n✓ Saved model and results to {output_dir}/")


if __name__ == "__main__":
    main()
