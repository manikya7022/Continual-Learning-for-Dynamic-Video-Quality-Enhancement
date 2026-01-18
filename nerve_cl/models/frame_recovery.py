"""
Video Frame Recovery Module.

Recovers corrupted or missing video frames using spatio-temporal context
from neighboring frames. Uses inpainting techniques with temporal 
consistency constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from nerve_cl.models.layers import (
    DepthwiseSeparableConv,
    ResidualBlock,
    CBAM,
    TemporalConv3D,
    PixelShuffleUpsampler,
)


class SpatialEncoder(nn.Module):
    """
    Spatial feature encoder using ResNet-style architecture.
    
    Extracts spatial features from individual frames for recovery.
    
    Args:
        in_channels: Input image channels (3 for RGB)
        base_channels: Base number of feature channels
        num_blocks: Number of residual blocks per stage
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 2,
    ):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        
        # Encoder stages
        self.stage1 = self._make_stage(base_channels, base_channels, num_blocks)
        self.stage2 = self._make_stage(base_channels, base_channels * 2, num_blocks, stride=2)
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, num_blocks, stride=2)
        
        # Attention
        self.attention = CBAM(base_channels * 4)
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a stage with residual blocks."""
        layers = []
        
        # Downsample if needed
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ))
            in_channels = out_channels
        
        # Add residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract spatial features.
        
        Args:
            x: Input frame (B, C, H, W)
        
        Returns:
            features: Encoded features (B, C', H', W')
            skip_connections: List of intermediate features for decoder
        """
        skip_connections = []
        
        x = self.stem(x)
        skip_connections.append(x)
        
        x = self.stage1(x)
        skip_connections.append(x)
        
        x = self.stage2(x)
        skip_connections.append(x)
        
        x = self.stage3(x)
        x = self.attention(x)
        
        return x, skip_connections


class TemporalEncoder(nn.Module):
    """
    Temporal feature encoder using 3D convolutions.
    
    Extracts temporal features from neighboring frames to understand
    motion and temporal context.
    
    Args:
        in_channels: Input channels per frame
        out_channels: Output feature channels
        temporal_window: Number of frames to consider
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        temporal_window: int = 3,
    ):
        super().__init__()
        
        self.temporal_window = temporal_window
        
        # 3D conv layers with (2+1)D factorization
        self.conv1 = TemporalConv3D(in_channels, 64, temporal_kernel=3)
        self.conv2 = TemporalConv3D(64, 128, temporal_kernel=3)
        self.conv3 = TemporalConv3D(128, out_channels, temporal_kernel=3)
        
        # Pooling to reduce temporal dimension
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features from frame sequence.
        
        Args:
            frames: Input frames (B, T, C, H, W)
        
        Returns:
            Temporal features (B, C', H', W')
        """
        # Reshape: (B, T, C, H, W) -> (B, C, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4)
        
        x = self.conv1(x)
        x = F.max_pool3d(x, (1, 2, 2))
        
        x = self.conv2(x)
        x = F.max_pool3d(x, (1, 2, 2))
        
        x = self.conv3(x)
        
        # Pool temporal dimension
        x = self.temporal_pool(x)
        x = x.squeeze(2)  # (B, C, H, W)
        
        return x


class FusionModule(nn.Module):
    """
    Feature fusion module combining spatial and temporal features.
    
    Uses attention-based fusion to combine information from
    corrupted frame (spatial) and reference frames (temporal).
    
    Args:
        spatial_channels: Spatial feature channels
        temporal_channels: Temporal feature channels
        out_channels: Output channels
    """
    
    def __init__(
        self,
        spatial_channels: int = 256,
        temporal_channels: int = 256,
        out_channels: int = 256,
    ):
        super().__init__()
        
        total_channels = spatial_channels + temporal_channels
        
        # Feature alignment
        self.align = nn.Conv2d(total_channels, out_channels, 1)
        
        # Attention for weighted fusion
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 2, 1),
            nn.Softmax(dim=1),
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            CBAM(out_channels),
        )
    
    def forward(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse spatial and temporal features.
        
        Args:
            spatial_feat: Spatial features (B, C1, H, W)
            temporal_feat: Temporal features (B, C2, H, W)
        
        Returns:
            Fused features (B, C_out, H, W)
        """
        # Align spatial dimensions if needed
        if spatial_feat.shape[2:] != temporal_feat.shape[2:]:
            temporal_feat = F.interpolate(
                temporal_feat,
                size=spatial_feat.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        
        # Concatenate
        concat = torch.cat([spatial_feat, temporal_feat], dim=1)
        aligned = self.align(concat)
        
        # Compute attention weights
        attn = self.attention(aligned)
        
        # Weighted combination
        spatial_proj = F.conv2d(
            spatial_feat,
            torch.ones(aligned.size(1), spatial_feat.size(1), 1, 1, device=spatial_feat.device) / spatial_feat.size(1),
        )
        temporal_proj = F.conv2d(
            temporal_feat,
            torch.ones(aligned.size(1), temporal_feat.size(1), 1, 1, device=temporal_feat.device) / temporal_feat.size(1),
        )
        
        fused = attn[:, 0:1] * spatial_proj + attn[:, 1:2] * temporal_proj
        
        # Refine
        out = self.refine(aligned + fused)
        
        return out


class Decoder(nn.Module):
    """
    Feature decoder with skip connections.
    
    Reconstructs the recovered frame from fused features.
    
    Args:
        in_channels: Input feature channels
        out_channels: Output image channels (3 for RGB)
        base_channels: Base decoder channels
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()
        
        # Decoder stages with upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, out_channels, 3, 1, 1),
            nn.Tanh(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Decode features to output frame.
        
        Args:
            x: Fused features (B, C, H, W)
            skip_connections: Optional skip connections from encoder
        
        Returns:
            Recovered frame (B, 3, H, W) in range [-1, 1]
        """
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        
        return x


class FrameRecoveryNet(nn.Module):
    """
    Complete Frame Recovery Network.
    
    Recovers corrupted/missing video frames using spatial-temporal
    context from neighboring frames.
    
    Architecture:
        1. Spatial Encoder: Extract features from corrupted frame
        2. Temporal Encoder: Extract motion features from neighbors
        3. Fusion Module: Combine spatial and temporal features
        4. Decoder: Reconstruct recovered frame
    
    Args:
        in_channels: Input image channels
        base_channels: Base feature channels
        temporal_window: Number of reference frames (before + after)
    
    Example:
        >>> model = FrameRecoveryNet()
        >>> corrupted = torch.randn(1, 3, 256, 256)
        >>> refs = torch.randn(1, 2, 3, 256, 256)  # 2 reference frames
        >>> mask = torch.zeros(1, 1, 256, 256)  # Corruption mask
        >>> recovered = model(corrupted, refs, mask)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        temporal_window: int = 2,
    ):
        super().__init__()
        
        self.temporal_window = temporal_window
        
        # Encoders
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels + 1,  # +1 for corruption mask
            base_channels=base_channels,
        )
        
        self.temporal_encoder = TemporalEncoder(
            in_channels=in_channels,
            out_channels=base_channels * 4,
            temporal_window=temporal_window,
        )
        
        # Fusion
        self.fusion = FusionModule(
            spatial_channels=base_channels * 4,
            temporal_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )
        
        # Decoder
        self.decoder = Decoder(
            in_channels=base_channels * 4,
            out_channels=in_channels,
            base_channels=base_channels,
        )
    
    def forward(
        self,
        corrupted_frame: torch.Tensor,
        reference_frames: torch.Tensor,
        corruption_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Recover a corrupted frame.
        
        Args:
            corrupted_frame: Corrupted input frame (B, C, H, W)
            reference_frames: Reference frames (B, T, C, H, W)
            corruption_mask: Binary mask (1=corrupted) (B, 1, H, W)
        
        Returns:
            Recovered frame (B, C, H, W)
        """
        B, C, H, W = corrupted_frame.shape
        
        # Create mask if not provided
        if corruption_mask is None:
            corruption_mask = torch.zeros(B, 1, H, W, device=corrupted_frame.device)
        
        # Concatenate frame with mask for spatial encoding
        spatial_input = torch.cat([corrupted_frame, corruption_mask], dim=1)
        
        # Encode
        spatial_feat, skip_connections = self.spatial_encoder(spatial_input)
        temporal_feat = self.temporal_encoder(reference_frames)
        
        # Fuse
        fused = self.fusion(spatial_feat, temporal_feat)
        
        # Decode
        recovered = self.decoder(fused, skip_connections)
        
        # Resize to match input
        if recovered.shape[2:] != (H, W):
            recovered = F.interpolate(
                recovered, size=(H, W), mode='bilinear', align_corners=False
            )
        
        # Blend: keep uncorrupted regions, use recovered for corrupted
        output = corrupted_frame * (1 - corruption_mask) + recovered * corruption_mask
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = FrameRecoveryNet()
    print(f"Frame Recovery Network: {model.get_num_parameters():,} parameters")
    
    # Test forward pass
    corrupted = torch.randn(2, 3, 256, 256)
    refs = torch.randn(2, 2, 3, 256, 256)
    mask = torch.zeros(2, 1, 256, 256)
    mask[:, :, 100:150, 100:150] = 1  # Simulate corruption
    
    with torch.no_grad():
        output = model(corrupted, refs, mask)
    
    print(f"Input shape: {corrupted.shape}")
    print(f"Output shape: {output.shape}")
