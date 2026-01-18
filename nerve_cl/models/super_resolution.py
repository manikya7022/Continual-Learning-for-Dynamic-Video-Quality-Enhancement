"""
Super-Resolution Module for Video Enhancement.

Implements a lightweight temporal super-resolution model optimized
for mobile inference, exploiting inter-frame redundancy for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from nerve_cl.models.layers import (
    DepthwiseSeparableConv,
    ResidualBlock,
    PixelShuffleUpsampler,
    CBAM,
    LiteFlowNetCorrelation,
)


class FeatureExtractor(nn.Module):
    """
    Lightweight feature extractor for super-resolution.
    
    Uses depthwise separable convolutions for efficiency.
    
    Args:
        in_channels: Input image channels
        num_features: Number of feature channels
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        self.body = nn.Sequential(
            DepthwiseSeparableConv(num_features, num_features),
            DepthwiseSeparableConv(num_features, num_features),
            DepthwiseSeparableConv(num_features, num_features),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.body(feat) + feat  # Residual
        return feat


class MotionEstimator(nn.Module):
    """
    Lightweight optical flow estimator (LiteFlowNet-style).
    
    Estimates motion between frames for temporal alignment.
    
    Args:
        in_channels: Feature channels
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        self.correlation = LiteFlowNetCorrelation(max_displacement=4)
        
        # Flow prediction
        corr_channels = (2 * 4 + 1) ** 2  # 81
        self.flow_net = nn.Sequential(
            nn.Conv2d(corr_channels, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1),  # 2 channels for (dx, dy)
        )
    
    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate optical flow from feat1 to feat2.
        
        Args:
            feat1: Source features (B, C, H, W)
            feat2: Target features (B, C, H, W)
        
        Returns:
            Optical flow (B, 2, H, W)
        """
        corr = self.correlation(feat1, feat2)
        flow = self.flow_net(corr)
        return flow


def warp_features(
    features: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """
    Warp features using optical flow.
    
    Args:
        features: Features to warp (B, C, H, W)
        flow: Optical flow (B, 2, H, W)
    
    Returns:
        Warped features (B, C, H, W)
    """
    B, C, H, W = features.shape
    
    # Create grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=features.device, dtype=features.dtype),
        torch.arange(W, device=features.device, dtype=features.dtype),
        indexing='ij',
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Add flow to grid
    grid = grid + flow
    
    # Normalize to [-1, 1]
    grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
    
    # Reshape for grid_sample
    grid = grid.permute(0, 2, 3, 1)  # (B, H, W, 2)
    
    # Warp
    warped = F.grid_sample(
        features, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    
    return warped


class TemporalAggregator(nn.Module):
    """
    Temporal feature aggregation with attention.
    
    Aggregates aligned features from multiple frames using
    learned attention weights.
    
    Args:
        num_features: Feature channels
        num_frames: Number of input frames
    """
    
    def __init__(
        self,
        num_features: int = 64,
        num_frames: int = 3,
    ):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Conv2d(num_features * num_frames, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_frames, 3, 1, 1),
            nn.Softmax(dim=1),
        )
        
        # Feature refinement
        self.refine = CBAM(num_features)
    
    def forward(
        self,
        aligned_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Aggregate aligned features from multiple frames.
        
        Args:
            aligned_features: List of (B, C, H, W) tensors
        
        Returns:
            Aggregated features (B, C, H, W)
        """
        # Stack and concatenate
        stacked = torch.stack(aligned_features, dim=1)  # (B, T, C, H, W)
        B, T, C, H, W = stacked.shape
        
        concat = stacked.view(B, T * C, H, W)
        
        # Compute attention weights
        attn = self.attention(concat)  # (B, T, H, W)
        
        # Weighted sum
        attn = attn.unsqueeze(2)  # (B, T, 1, H, W)
        weighted = (stacked * attn).sum(dim=1)  # (B, C, H, W)
        
        # Refine
        output = self.refine(weighted)
        
        return output


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block for high-frequency detail extraction.
    
    Uses dense connections for feature reuse.
    
    Args:
        num_features: Number of feature channels
        growth_rate: Growth rate for dense connections
        num_layers: Number of conv layers
    """
    
    def __init__(
        self,
        num_features: int = 64,
        growth_rate: int = 32,
        num_layers: int = 5,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_channels = num_features
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, growth_rate, 3, 1, 1),
                nn.ReLU(inplace=True),
            ))
            in_channels += growth_rate
        
        # Local feature fusion
        self.lff = nn.Conv2d(in_channels, num_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        out = self.lff(torch.cat(features, dim=1))
        return out * 0.2 + x  # Residual scaling


class SuperResolutionNet(nn.Module):
    """
    Lightweight Temporal Super-Resolution Network.
    
    Features:
        - Motion-compensated feature extraction
        - Temporal aggregation with attention
        - Residual learning for high-frequency details
        - Mobile-optimized architecture
    
    Args:
        in_channels: Input image channels
        scale_factor: Upscaling factor (2, 3, or 4)
        num_features: Number of feature channels
        num_residual_blocks: Number of residual blocks
        temporal_window: Number of reference frames
    
    Example:
        >>> model = SuperResolutionNet(scale_factor=2)
        >>> lr_frames = torch.randn(1, 3, 3, 128, 128)  # 3 LR frames
        >>> hr_frame = model(lr_frames)  # Upscale center frame
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        scale_factor: int = 2,
        num_features: int = 64,
        num_residual_blocks: int = 8,
        temporal_window: int = 1,
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.temporal_window = temporal_window
        self.num_frames = 2 * temporal_window + 1
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(in_channels, num_features)
        
        # Motion estimation
        self.motion_estimator = MotionEstimator(num_features)
        
        # Temporal aggregation
        self.temporal_aggregator = TemporalAggregator(num_features, self.num_frames)
        
        # Residual blocks for high-frequency details
        self.residual_blocks = nn.Sequential(*[
            ResidualDenseBlock(num_features) for _ in range(num_residual_blocks)
        ])
        
        # Global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling
        self.upsampler = PixelShuffleUpsampler(
            in_channels=num_features,
            scale_factor=scale_factor,
            out_channels=in_channels,
        )
        
        # Bicubic upsampling for residual learning
        self.bicubic_upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bicubic',
            align_corners=False,
        )
    
    def forward(
        self,
        lr_frames: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Upscale the center frame using temporal context.
        
        Args:
            lr_frames: Low-resolution frames (B, T, C, H, W)
            return_intermediate: If True, return intermediate outputs
        
        Returns:
            Super-resolved center frame (B, C, H*scale, W*scale)
        """
        B, T, C, H, W = lr_frames.shape
        center_idx = T // 2
        
        # Extract features from all frames
        features = []
        for t in range(T):
            feat = self.feature_extractor(lr_frames[:, t])
            features.append(feat)
        
        # Align all frames to center frame
        center_feat = features[center_idx]
        aligned_features = []
        
        for t in range(T):
            if t == center_idx:
                aligned_features.append(center_feat)
            else:
                # Estimate flow from frame t to center
                flow = self.motion_estimator(features[t], center_feat)
                # Warp frame t features to center
                warped = warp_features(features[t], flow)
                aligned_features.append(warped)
        
        # Temporal aggregation
        aggregated = self.temporal_aggregator(aligned_features)
        
        # Residual blocks
        residual = self.residual_blocks(aggregated)
        
        # Global feature fusion with residual
        fused = self.gff(residual) + center_feat
        
        # Upsample
        hr_residual = self.upsampler(fused)
        
        # Add to bicubic upsampled input (residual learning)
        bicubic = self.bicubic_upsample(lr_frames[:, center_idx])
        output = bicubic + hr_residual
        
        # Clamp to valid range
        output = torch.clamp(output, 0, 1)
        
        if return_intermediate:
            return output, {
                'features': features,
                'aligned': aligned_features,
                'aggregated': aggregated,
            }
        
        return output
    
    def forward_single(self, lr_frame: torch.Tensor) -> torch.Tensor:
        """
        Upscale a single frame (no temporal information).
        
        Args:
            lr_frame: Low-resolution frame (B, C, H, W)
        
        Returns:
            Super-resolved frame (B, C, H*scale, W*scale)
        """
        # Add temporal dimension
        lr_frames = lr_frame.unsqueeze(1).expand(-1, self.num_frames, -1, -1, -1)
        return self.forward(lr_frames)
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops(self, input_size: Tuple[int, int] = (128, 128)) -> int:
        """Estimate FLOPs for given input size."""
        # Simplified estimation
        H, W = input_size
        C = 3
        F = 64  # num_features
        
        # Feature extraction
        flops = H * W * C * F * 9  # 3x3 conv
        
        # Motion estimation (correlation)
        flops += H * W * F * 81 * (self.num_frames - 1)
        
        # Residual blocks
        flops += H * W * F * F * 9 * 8  # 8 blocks
        
        # Upsampling
        scale = self.scale_factor
        flops += H * W * F * (C * scale * scale) * 9
        
        return flops


class LightweightSuperResolution(nn.Module):
    """
    Ultra-lightweight SR for mobile inference.
    
    Optimized for real-time performance on mobile GPUs.
    Under 1M parameters.
    
    Args:
        scale_factor: Upscaling factor
    """
    
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 32),
            DepthwiseSeparableConv(32, 32),
            DepthwiseSeparableConv(32, 32),
            DepthwiseSeparableConv(32, 32),
            nn.Conv2d(32, 3 * scale_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(scale_factor),
        )
        
        self.bicubic = nn.Upsample(
            scale_factor=scale_factor,
            mode='bicubic',
            align_corners=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        base = self.bicubic(x)
        return torch.clamp(base + residual, 0, 1)


if __name__ == "__main__":
    # Test the models
    print("Testing SuperResolutionNet...")
    model = SuperResolutionNet(scale_factor=2)
    print(f"Parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    lr_frames = torch.randn(2, 3, 3, 128, 128)  # B=2, T=3, C=3, H=128, W=128
    
    with torch.no_grad():
        hr = model(lr_frames)
    
    print(f"Input shape: {lr_frames.shape}")
    print(f"Output shape: {hr.shape}")
    
    # Test lightweight model
    print("\nTesting LightweightSuperResolution...")
    light_model = LightweightSuperResolution(scale_factor=2)
    print(f"Parameters: {sum(p.numel() for p in light_model.parameters()):,}")
    
    with torch.no_grad():
        hr_light = light_model(lr_frames[:, 1])  # Single frame
    print(f"Output shape: {hr_light.shape}")
