"""
SRDenseNet: Image Super Resolution Using Dense Skip Connections

Based on the paper and used in the novel architecture for INR compression.
This implementation uses dense blocks with growth rate and local/global feature fusion.

Reference: https://arxiv.org/abs/1802.08797 (RDN - similar architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DenseLayer(nn.Module):
    """Single dense layer with growth rate output channels."""
    
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.conv(x))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """Dense block with multiple dense layers."""
    
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate)
            )
        # Local feature fusion (1x1 conv to reduce channels)
        self.lff = nn.Conv2d(
            in_channels + num_layers * growth_rate, 
            in_channels, 
            kernel_size=1
        )
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        # Local feature fusion + local residual
        return self.lff(out) + x


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) from RDN paper."""
    
    def __init__(self, num_features, growth_rate=32, num_layers=8):
        super().__init__()
        self.dense_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.dense_layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features + i * growth_rate, growth_rate, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Local Feature Fusion
        self.lff = nn.Conv2d(num_features + num_layers * growth_rate, num_features, 1)
    
    def forward(self, x):
        features = [x]
        for layer in self.dense_layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        # Concatenate all features and apply LFF
        return self.lff(torch.cat(features, dim=1)) + x


class Upsampler(nn.Module):
    """Upsampling module using PixelShuffle."""
    
    def __init__(self, scale_factor, num_features):
        super().__init__()
        
        layers = []
        if scale_factor == 2 or scale_factor == 4 or scale_factor == 8:
            for _ in range(int(math.log2(scale_factor))):
                layers.append(nn.Conv2d(num_features, num_features * 4, 3, padding=1))
                layers.append(nn.PixelShuffle(2))
                layers.append(nn.ReLU(inplace=True))
        elif scale_factor == 3:
            layers.append(nn.Conv2d(num_features, num_features * 9, 3, padding=1))
            layers.append(nn.PixelShuffle(3))
            layers.append(nn.ReLU(inplace=True))
        
        self.upsampler = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.upsampler(x)


class SRDenseNet(nn.Module):
    """
    SRDenseNet for Single Image Super Resolution.
    
    Architecture:
    1. Shallow feature extraction (SFE)
    2. Residual Dense Blocks (RDBs) 
    3. Global Feature Fusion (GFF)
    4. Upsampling via PixelShuffle
    5. Final reconstruction
    
    Args:
        scale_factor: Upscaling factor (2, 4, or 8)
        in_channels: Input channels (1 for grayscale, 3 for RGB)
        out_channels: Output channels
        num_features: Number of feature maps (default: 64)
        growth_rate: Growth rate for dense layers (default: 32)
        num_blocks: Number of RDB blocks (default: 8)
        num_layers: Number of dense layers per RDB (default: 8)
    """
    
    def __init__(
        self,
        scale_factor=4,
        in_channels=1,
        out_channels=1,
        num_features=64,
        growth_rate=32,
        num_blocks=8,
        num_layers=8
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.num_features = num_features
        self.num_blocks = num_blocks
        
        # Shallow Feature Extraction (SFE) - two conv layers
        self.sfe1 = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Residual Dense Blocks (RDBs)
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(num_features, growth_rate, num_layers)
            for _ in range(num_blocks)
        ])
        
        # Global Feature Fusion (GFF)
        self.gff = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, 1),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )
        
        # Upsampling
        self.upsampler = Upsampler(scale_factor, num_features)
        
        # Final reconstruction
        self.reconstruction = nn.Conv2d(num_features, out_channels, 3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Handle different input formats
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        
        # Shallow feature extraction
        f_1 = self.sfe1(x)
        f_0 = self.sfe2(f_1)
        
        # Residual Dense Blocks
        rdb_outputs = []
        f = f_0
        for rdb in self.rdbs:
            f = rdb(f)
            rdb_outputs.append(f)
        
        # Global Feature Fusion
        gff_out = self.gff(torch.cat(rdb_outputs, dim=1))
        
        # Global residual learning
        f_gf = gff_out + f_1
        
        # Upsampling
        f_up = self.upsampler(f_gf)
        
        # Reconstruction
        out = self.reconstruction(f_up)
        
        return out.squeeze()
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SRDenseNetLight(nn.Module):
    """
    Lightweight SRDenseNet for faster training and inference.
    Suitable for medical imaging with smaller patch sizes.
    
    Args:
        scale_factor: Upscaling factor (2, 4, or 8)
        in_channels: Input channels (1 for grayscale)
        num_features: Number of feature maps (default: 32)
        growth_rate: Growth rate (default: 16)
        num_blocks: Number of RDB blocks (default: 4)
        num_layers: Dense layers per block (default: 4)
    """
    
    def __init__(
        self,
        scale_factor=4,
        in_channels=1,
        out_channels=1,
        num_features=32,
        growth_rate=16,
        num_blocks=4,
        num_layers=4
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(in_channels, num_features, 3, padding=1)
        
        # Dense blocks with residual connections
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i in range(num_blocks):
            self.dense_blocks.append(
                self._make_dense_block(num_features, growth_rate, num_layers)
            )
            # Transition layer to maintain channel count
            self.transitions.append(
                nn.Conv2d(num_features + num_layers * growth_rate, num_features, 1)
            )
        
        # Global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )
        
        # Upsampling
        self.upscale = self._make_upscale(scale_factor, num_features)
        
        # Output
        self.conv_output = nn.Conv2d(num_features, out_channels, 3, padding=1)
        
        self._initialize_weights()
    
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                nn.ReLU(inplace=True)
            ))
        return nn.ModuleList(layers)
    
    def _make_upscale(self, scale_factor, num_features):
        layers = []
        for _ in range(int(math.log2(scale_factor))):
            layers.extend([
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        
        # Initial features
        f0 = self.conv_input(x)
        
        # Dense blocks with global skip connections
        block_outputs = []
        f = f0
        
        for dense_block, transition in zip(self.dense_blocks, self.transitions):
            features = [f]
            for layer in dense_block:
                out = layer(torch.cat(features, dim=1))
                features.append(out)
            f = transition(torch.cat(features, dim=1)) + f  # Local residual
            block_outputs.append(f)
        
        # Global feature fusion
        gff = self.gff(torch.cat(block_outputs, dim=1))
        
        # Global residual
        f = gff + f0
        
        # Upscale and output
        f = self.upscale(f)
        out = self.conv_output(f)
        
        return out.squeeze()
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing SRDenseNet models...")
    
    for scale in [2, 4, 8]:
        # Test standard model
        model = SRDenseNet(scale_factor=scale, num_features=64, num_blocks=8).to(device)
        x = torch.randn(1, 1, 32, 32).to(device)
        y = model(x)
        print(f"SRDenseNet x{scale}: Input {x.shape} -> Output {y.shape}, Params: {model.count_parameters():,}")
        
        # Test lightweight model
        model_light = SRDenseNetLight(scale_factor=scale, num_features=32, num_blocks=4).to(device)
        y_light = model_light(x)
        print(f"SRDenseNet-Light x{scale}: Input {x.shape} -> Output {y_light.shape}, Params: {model_light.count_parameters():,}")
        
    print("\nAll tests passed!")
