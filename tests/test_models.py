"""Tests for neural network models."""

import pytest
import torch
from nerve_cl.models import (
    FrameRecoveryNet,
    SuperResolutionNet,
    LightweightSuperResolution,
    EnhancementEngine,
    EnhancementConfig,
)
from nerve_cl.models.layers import (
    DepthwiseSeparableConv,
    PixelShuffleUpsampler,
    ResidualBlock,
)


class TestLayers:
    """Test custom layers."""
    
    def test_depthwise_separable_conv(self):
        layer = DepthwiseSeparableConv(32, 64)
        x = torch.randn(2, 32, 16, 16)
        y = layer(x)
        assert y.shape == (2, 64, 16, 16)
    
    def test_pixel_shuffle_upsampler(self):
        layer = PixelShuffleUpsampler(64, scale_factor=2)
        x = torch.randn(2, 64, 16, 16)
        y = layer(x)
        assert y.shape == (2, 3, 32, 32)
    
    def test_residual_block(self):
        layer = ResidualBlock(64)
        x = torch.randn(2, 64, 16, 16)
        y = layer(x)
        assert y.shape == x.shape


class TestFrameRecovery:
    """Test frame recovery module."""
    
    def test_forward(self):
        model = FrameRecoveryNet()
        corrupted = torch.randn(2, 3, 128, 128)
        refs = torch.randn(2, 2, 3, 128, 128)
        mask = torch.zeros(2, 1, 128, 128)
        
        output = model(corrupted, refs, mask)
        assert output.shape == corrupted.shape
    
    def test_parameter_count(self):
        model = FrameRecoveryNet()
        assert model.get_num_parameters() > 0


class TestSuperResolution:
    """Test super-resolution module."""
    
    def test_forward(self):
        model = SuperResolutionNet(scale_factor=2)
        lr_frames = torch.randn(2, 3, 3, 64, 64)
        
        output = model(lr_frames)
        assert output.shape == (2, 3, 128, 128)
    
    def test_scale_factors(self):
        for scale in [2, 3, 4]:
            model = SuperResolutionNet(scale_factor=scale)
            lr_frames = torch.randn(1, 3, 3, 32, 32)
            output = model(lr_frames)
            assert output.shape == (1, 3, 32 * scale, 32 * scale)
    
    def test_lightweight(self):
        model = LightweightSuperResolution(scale_factor=2)
        x = torch.randn(2, 3, 64, 64)
        y = model(x)
        assert y.shape == (2, 3, 128, 128)


class TestEnhancementEngine:
    """Test combined enhancement engine."""
    
    def test_sr_only(self):
        config = EnhancementConfig(
            frame_recovery_enabled=False,
            super_resolution_enabled=True,
            scale_factor=2,
        )
        engine = EnhancementEngine(config)
        frames = torch.randn(1, 3, 3, 64, 64)
        
        results = engine(frames)
        assert 'enhanced' in results
        assert results['enhanced'].shape == (1, 3, 128, 128)
    
    def test_full_pipeline(self):
        config = EnhancementConfig(
            frame_recovery_enabled=True,
            super_resolution_enabled=True,
        )
        engine = EnhancementEngine(config)
        frames = torch.randn(1, 5, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        
        results = engine(frames, corruption_mask=mask)
        assert 'enhanced' in results
    
    def test_model_info(self):
        engine = EnhancementEngine()
        info = engine.get_model_info()
        
        assert 'config' in info
        assert 'parameters' in info
        assert info['parameters']['total'] > 0
