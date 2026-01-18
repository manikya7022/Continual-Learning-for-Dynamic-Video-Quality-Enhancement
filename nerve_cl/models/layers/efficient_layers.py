"""Efficient neural network layers optimized for mobile deployment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution for efficient mobile inference.
    
    Reduces computation by 3-9x compared to standard convolution:
    - Depthwise: Apply filter per channel
    - Pointwise: 1x1 conv to mix channels
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to input
        bias: If True, adds a learnable bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PixelShuffleUpsampler(nn.Module):
    """
    Efficient upsampling using sub-pixel convolution (PixelShuffle).
    
    More efficient than deconvolution/transposed convolution for
    super-resolution tasks.
    
    Args:
        in_channels: Number of input channels
        scale_factor: Upscaling factor (2, 3, or 4)
        out_channels: Number of output channels (default: 3 for RGB)
    """
    
    def __init__(
        self,
        in_channels: int,
        scale_factor: int = 2,
        out_channels: int = 3,
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        hidden_channels = out_channels * (scale_factor ** 2)
        
        self.conv = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with efficient depthwise separable convolutions.
    
    Args:
        channels: Number of channels
        use_efficient: If True, use depthwise separable convs
    """
    
    def __init__(
        self,
        channels: int,
        use_efficient: bool = True,
    ):
        super().__init__()
        
        if use_efficient:
            self.conv1 = DepthwiseSeparableConv(channels, channels)
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    """
    Channel attention module (SE-Net style).
    
    Learns to emphasize informative channels and suppress less useful ones.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial attention module.
    
    Learns to emphasize important spatial regions.
    
    Args:
        kernel_size: Convolution kernel size
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention for comprehensive feature refinement.
    
    Args:
        channels: Number of input channels
        reduction: Channel attention reduction ratio
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class TemporalConv3D(nn.Module):
    """
    Efficient 3D convolution for temporal feature extraction.
    
    Uses (2+1)D factorization: spatial conv followed by temporal conv.
    More efficient than full 3D conv while maintaining quality.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        temporal_kernel: Temporal kernel size
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
    ):
        super().__init__()
        
        # Intermediate channels
        mid_channels = (in_channels * out_channels * 3 * 3 * temporal_kernel) // (
            in_channels * 3 * 3 + out_channels * temporal_kernel
        )
        mid_channels = max(mid_channels, out_channels // 2)
        
        # Spatial conv (1x3x3)
        self.spatial = nn.Sequential(
            nn.Conv3d(
                in_channels, mid_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # Temporal conv (Tx1x1)
        self.temporal = nn.Sequential(
            nn.Conv3d(
                mid_channels, out_channels,
                kernel_size=(temporal_kernel, 1, 1),
                stride=1,
                padding=(temporal_kernel // 2, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Output tensor of shape (B, C', T, H, W)
        """
        x = self.spatial(x)
        x = self.temporal(x)
        return x


class LiteFlowNetCorrelation(nn.Module):
    """
    Lightweight correlation layer for optical flow estimation.
    
    Computes correlation between feature maps for motion estimation.
    Optimized for mobile inference.
    
    Args:
        max_displacement: Maximum displacement for correlation
    """
    
    def __init__(self, max_displacement: int = 4):
        super().__init__()
        self.max_displacement = max_displacement
        self.pad = max_displacement
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute correlation between two feature maps.
        
        Args:
            x1: First feature map (B, C, H, W)
            x2: Second feature map (B, C, H, W)
        
        Returns:
            Correlation volume (B, (2*d+1)^2, H, W)
        """
        b, c, h, w = x1.shape
        d = self.max_displacement
        
        # Pad x2
        x2_padded = F.pad(x2, [d, d, d, d])
        
        # Compute correlation
        out = []
        for i in range(2 * d + 1):
            for j in range(2 * d + 1):
                x2_slice = x2_padded[:, :, i:i+h, j:j+w]
                corr = (x1 * x2_slice).sum(dim=1, keepdim=True)
                out.append(corr)
        
        out = torch.cat(out, dim=1)
        return out / c  # Normalize by channel count
