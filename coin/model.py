"""
COIN: COmpression with Implicit Neural representations
Paper: https://arxiv.org/abs/2103.03123

Implementation of the SIREN-based MLP for image compression using
Implicit Neural Representations (INR).
"""

import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    """
    Sine activation layer with special initialization as described in SIREN paper.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class COIN(nn.Module):
    """
    COIN model: MLP with SIREN activations for image compression.
    
    Maps (x, y) coordinates -> pixel intensity values.
    
    Args:
        in_features: Input dimension (2 for 2D coordinates)
        hidden_features: Hidden layer width
        hidden_layers: Number of hidden layers
        out_features: Output dimension (1 for grayscale, 3 for RGB)
        omega_0: Frequency parameter for sine activations
    """
    def __init__(
        self,
        in_features=2,
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
        omega_0=30
    ):
        super().__init__()
        
        self.net = []
        
        # First layer
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        
        # Hidden layers
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        
        # Final layer (linear, no sine activation)
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_features) / omega_0,
                np.sqrt(6 / hidden_features) / omega_0
            )
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        """
        Args:
            coords: (N, 2) tensor of (x, y) coordinates in [-1, 1]
        Returns:
            (N, out_features) tensor of pixel values
        """
        return self.net(coords)
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_coordinate_grid(height, width, device='cpu'):
    """
    Create a grid of (x, y) coordinates normalized to [-1, 1].
    
    Returns:
        coords: (H*W, 2) tensor of coordinates
    """
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    y, x = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([x.flatten(), y.flatten()], dim=-1)
    return coords


def image_to_tensor(image, device='cpu'):
    """
    Convert numpy image to normalized tensor.
    
    Args:
        image: (H, W) or (H, W, C) numpy array
    Returns:
        tensor: (H*W, C) tensor normalized to [0, 1]
    """
    if image.ndim == 2:
        image = image[..., np.newaxis]
    
    # Normalize to [0, 1]
    img_min, img_max = image.min(), image.max()
    image = (image - img_min) / (img_max - img_min + 1e-8)
    
    tensor = torch.from_numpy(image).float().to(device)
    tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor


def tensor_to_image(tensor, height, width):
    """
    Convert tensor back to numpy image.
    
    Args:
        tensor: (H*W, C) tensor
        height, width: Image dimensions
    Returns:
        image: (H, W) or (H, W, C) numpy array
    """
    tensor = tensor.detach().cpu().numpy()
    tensor = np.clip(tensor, 0, 1)
    image = tensor.reshape(height, width, -1)
    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    return image


if __name__ == "__main__":
    # Quick test
    model = COIN(in_features=2, hidden_features=256, hidden_layers=3, out_features=1)
    print(f"COIN model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    coords = create_coordinate_grid(256, 256)
    output = model(coords)
    print(f"Input shape: {coords.shape}")
    print(f"Output shape: {output.shape}")
