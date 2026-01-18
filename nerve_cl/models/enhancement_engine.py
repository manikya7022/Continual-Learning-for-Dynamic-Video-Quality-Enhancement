"""
Enhancement Engine - Combined Video Enhancement Pipeline.

Orchestrates frame recovery and super-resolution for complete
video enhancement workflow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from nerve_cl.models.frame_recovery import FrameRecoveryNet
from nerve_cl.models.super_resolution import SuperResolutionNet, LightweightSuperResolution


@dataclass
class EnhancementConfig:
    """Configuration for enhancement engine."""
    
    # Frame recovery
    frame_recovery_enabled: bool = True
    recovery_base_channels: int = 64
    recovery_temporal_window: int = 2
    
    # Super resolution
    super_resolution_enabled: bool = True
    scale_factor: int = 2
    sr_num_features: int = 64
    sr_num_residual_blocks: int = 8
    sr_temporal_window: int = 1
    
    # Mode
    use_lightweight_sr: bool = False
    enhancement_mode: str = "sequential"  # sequential, parallel
    upscale_first: bool = False


class EnhancementEngine(nn.Module):
    """
    Complete Video Enhancement Engine.
    
    Combines frame recovery and super-resolution in an end-to-end
    pipeline for video quality enhancement.
    
    Features:
        - Modular design (can use recovery, SR, or both)
        - Sequential or parallel enhancement modes
        - Adaptive enhancement strength
        - Mobile-optimized inference
    
    Args:
        config: EnhancementConfig object or None for defaults
    
    Example:
        >>> engine = EnhancementEngine()
        >>> frames = torch.randn(1, 5, 3, 256, 256)
        >>> enhanced = engine(frames, center_idx=2)
    """
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        super().__init__()
        
        self.config = config or EnhancementConfig()
        
        # Initialize frame recovery
        if self.config.frame_recovery_enabled:
            self.frame_recovery = FrameRecoveryNet(
                base_channels=self.config.recovery_base_channels,
                temporal_window=self.config.recovery_temporal_window,
            )
        else:
            self.frame_recovery = None
        
        # Initialize super resolution
        if self.config.super_resolution_enabled:
            if self.config.use_lightweight_sr:
                self.super_resolution = LightweightSuperResolution(
                    scale_factor=self.config.scale_factor,
                )
            else:
                self.super_resolution = SuperResolutionNet(
                    scale_factor=self.config.scale_factor,
                    num_features=self.config.sr_num_features,
                    num_residual_blocks=self.config.sr_num_residual_blocks,
                    temporal_window=self.config.sr_temporal_window,
                )
        else:
            self.super_resolution = None
        
        # Enhancement strength (learnable or fixed)
        self.enhancement_strength = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        frames: torch.Tensor,
        center_idx: Optional[int] = None,
        corruption_mask: Optional[torch.Tensor] = None,
        enhancement_strength: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Enhance video frames.
        
        Args:
            frames: Input frames (B, T, C, H, W)
            center_idx: Index of center frame to enhance (default: T//2)
            corruption_mask: Corruption mask for frame recovery (B, 1, H, W)
            enhancement_strength: Override enhancement strength (0-1)
        
        Returns:
            Dictionary with:
                - 'enhanced': Final enhanced frame (B, C, H', W')
                - 'recovered': Recovered frame if recovery enabled
                - 'super_resolved': SR frame if SR enabled
        """
        B, T, C, H, W = frames.shape
        
        if center_idx is None:
            center_idx = T // 2
        
        results = {}
        current_frame = frames[:, center_idx]
        
        # Get reference frames
        ref_indices = [i for i in range(T) if i != center_idx]
        reference_frames = frames[:, ref_indices] if ref_indices else None
        
        # Step 1: Frame Recovery (if enabled and corruption present)
        if self.frame_recovery is not None and corruption_mask is not None:
            if corruption_mask.sum() > 0:
                recovered = self.frame_recovery(
                    corrupted_frame=current_frame,
                    reference_frames=reference_frames,
                    corruption_mask=corruption_mask,
                )
                results['recovered'] = recovered
                current_frame = recovered
        
        # Step 2: Super Resolution (if enabled)
        if self.super_resolution is not None:
            # Prepare frames for SR
            sr_window = self.config.sr_temporal_window
            
            # Get temporal context for SR
            start_idx = max(0, center_idx - sr_window)
            end_idx = min(T, center_idx + sr_window + 1)
            sr_frames = frames[:, start_idx:end_idx]
            
            # Ensure correct number of frames
            expected_frames = 2 * sr_window + 1
            if sr_frames.shape[1] < expected_frames:
                # Pad with repeated frames
                pad_needed = expected_frames - sr_frames.shape[1]
                sr_frames = torch.cat([
                    sr_frames,
                    sr_frames[:, -1:].expand(-1, pad_needed, -1, -1, -1)
                ], dim=1)
            
            # Apply SR
            if isinstance(self.super_resolution, LightweightSuperResolution):
                super_resolved = self.super_resolution(current_frame)
            else:
                super_resolved = self.super_resolution(sr_frames)
            
            results['super_resolved'] = super_resolved
            current_frame = super_resolved
        
        # Apply enhancement strength blending
        strength = enhancement_strength if enhancement_strength is not None else self.enhancement_strength.item()
        
        if strength < 1.0 and 'super_resolved' in results:
            # Blend with bicubic upsampled original
            bicubic = F.interpolate(
                frames[:, center_idx],
                size=current_frame.shape[2:],
                mode='bicubic',
                align_corners=False,
            )
            current_frame = strength * current_frame + (1 - strength) * bicubic
        
        results['enhanced'] = current_frame
        
        return results
    
    def enhance_video(
        self,
        video: torch.Tensor,
        corruption_masks: Optional[torch.Tensor] = None,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """
        Enhance an entire video sequence.
        
        Args:
            video: Input video (T, C, H, W) or (B, T, C, H, W)
            corruption_masks: Optional masks (T, 1, H, W)
            batch_size: Frames to process in parallel
        
        Returns:
            Enhanced video (T, C, H', W') or (B, T, C, H', W')
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)  # Add batch dimension
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        B, T, C, H, W = video.shape
        
        # Determine output size
        scale = self.config.scale_factor if self.super_resolution is not None else 1
        enhanced_frames = []
        
        # Process in sliding window
        window_size = 2 * max(
            self.config.recovery_temporal_window,
            self.config.sr_temporal_window
        ) + 1
        
        for t in range(T):
            # Get window of frames centered at t
            start = max(0, t - window_size // 2)
            end = min(T, t + window_size // 2 + 1)
            frames_window = video[:, start:end]
            
            # Adjust center index
            center_idx = t - start
            
            # Get corruption mask if available
            mask = corruption_masks[t:t+1] if corruption_masks is not None else None
            
            # Enhance
            result = self.forward(
                frames=frames_window,
                center_idx=center_idx,
                corruption_mask=mask,
            )
            
            enhanced_frames.append(result['enhanced'])
        
        # Stack frames
        enhanced_video = torch.stack(enhanced_frames, dim=1)
        
        if squeeze_batch:
            enhanced_video = enhanced_video.squeeze(0)
        
        return enhanced_video
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'config': {
                'frame_recovery_enabled': self.config.frame_recovery_enabled,
                'super_resolution_enabled': self.config.super_resolution_enabled,
                'scale_factor': self.config.scale_factor,
                'use_lightweight_sr': self.config.use_lightweight_sr,
            },
            'parameters': {
                'total': sum(p.numel() for p in self.parameters()),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            },
        }
        
        if self.frame_recovery is not None:
            info['parameters']['frame_recovery'] = self.frame_recovery.get_num_parameters()
        
        if self.super_resolution is not None:
            info['parameters']['super_resolution'] = self.super_resolution.get_num_parameters()
        
        return info
    
    def set_enhancement_mode(self, mode: str) -> None:
        """
        Set enhancement mode.
        
        Args:
            mode: One of 'full', 'recovery_only', 'sr_only', 'lightweight'
        """
        if mode == 'full':
            self.config.frame_recovery_enabled = True
            self.config.super_resolution_enabled = True
        elif mode == 'recovery_only':
            self.config.frame_recovery_enabled = True
            self.config.super_resolution_enabled = False
        elif mode == 'sr_only':
            self.config.frame_recovery_enabled = False
            self.config.super_resolution_enabled = True
        elif mode == 'lightweight':
            self.config.frame_recovery_enabled = False
            self.config.super_resolution_enabled = True
            self.config.use_lightweight_sr = True


class AdaptiveEnhancementEngine(EnhancementEngine):
    """
    Enhancement engine with adaptive quality-compute tradeoff.
    
    Dynamically adjusts enhancement based on:
        - Content complexity
        - Available resources (battery, compute)
        - User preferences
    """
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        super().__init__(config)
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def estimate_complexity(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Estimate content complexity (0=simple, 1=complex).
        
        Args:
            frame: Input frame (B, C, H, W)
        
        Returns:
            Complexity score (B, 1)
        """
        return self.complexity_estimator(frame)
    
    def adaptive_forward(
        self,
        frames: torch.Tensor,
        resource_budget: float = 1.0,
        user_quality_preference: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Adaptive enhancement based on resources and preferences.
        
        Args:
            frames: Input frames (B, T, C, H, W)
            resource_budget: Available resources (0-1, 1=full)
            user_quality_preference: User preference (0=battery, 1=quality)
        
        Returns:
            Enhanced frame with metadata
        """
        B, T, C, H, W = frames.shape
        center_frame = frames[:, T // 2]
        
        # Estimate complexity
        complexity = self.estimate_complexity(center_frame)
        
        # Determine enhancement strength
        enhancement_strength = (
            0.3 * resource_budget +
            0.3 * user_quality_preference +
            0.4 * complexity.mean().item()
        )
        enhancement_strength = min(1.0, max(0.3, enhancement_strength))
        
        # Choose model based on budget
        if resource_budget < 0.3:
            # Ultra-light mode
            self.set_enhancement_mode('lightweight')
        elif resource_budget < 0.6:
            # SR only
            self.set_enhancement_mode('sr_only')
        else:
            # Full enhancement
            self.set_enhancement_mode('full')
        
        # Forward pass
        results = self.forward(
            frames=frames,
            enhancement_strength=enhancement_strength,
        )
        
        results['complexity'] = complexity
        results['enhancement_strength'] = enhancement_strength
        
        return results


if __name__ == "__main__":
    # Test enhancement engine
    print("Testing EnhancementEngine...")
    
    config = EnhancementConfig(
        frame_recovery_enabled=True,
        super_resolution_enabled=True,
        scale_factor=2,
    )
    
    engine = EnhancementEngine(config)
    info = engine.get_model_info()
    print(f"Total parameters: {info['parameters']['total']:,}")
    
    # Test forward pass
    frames = torch.randn(2, 5, 3, 128, 128)
    mask = torch.zeros(2, 1, 128, 128)
    mask[:, :, 50:80, 50:80] = 1
    
    with torch.no_grad():
        results = engine(frames, corruption_mask=mask)
    
    print(f"Input shape: {frames.shape}")
    print(f"Enhanced shape: {results['enhanced'].shape}")
    
    # Test adaptive engine
    print("\nTesting AdaptiveEnhancementEngine...")
    adaptive_engine = AdaptiveEnhancementEngine(config)
    
    with torch.no_grad():
        results = adaptive_engine.adaptive_forward(
            frames,
            resource_budget=0.7,
            user_quality_preference=0.8,
        )
    
    print(f"Enhancement strength: {results['enhancement_strength']:.2f}")
    print(f"Complexity: {results['complexity'].mean().item():.2f}")
