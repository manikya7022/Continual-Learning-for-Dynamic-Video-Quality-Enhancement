# Continual Learning for Dynamic Video Quality Enhancement

A production-grade framework for real-time video enhancement using continual learning, federated learning, and adaptive bitrate control. This system is designed to improve video streaming quality on mobile devices while preventing catastrophic forgetting when learning new content types.

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Test Cases](#test-cases)
9. [Performance and Efficiency](#performance-and-efficiency)
10. [Configuration](#configuration)

---

## Project Overview

This project addresses the challenge of video quality enhancement in mobile streaming environments. Traditional deep learning models suffer from catastrophic forgetting when adapting to new content types. Our framework combines:

- Neural video enhancement (frame recovery and super-resolution)
- Continual learning to prevent forgetting (EWC, MAML, experience replay)
- Federated learning for privacy-preserving personalization
- Reinforcement learning for adaptive bitrate control

The system targets mobile deployment with models optimized for real-time inference under resource constraints.

### Key Features

- Lightweight super-resolution with temporal consistency
- Frame recovery for corrupted video segments
- Content-adaptive enhancement with user personalization
- Battery-aware quality optimization
- Differential privacy for federated model updates

---

## System Architecture

```
Input Video Stream
        |
        v
+------------------+     +-------------------+
| Frame Recovery   | --> | Super-Resolution  |
| (Inpainting)     |     | (2x/4x upscale)   |
+------------------+     +-------------------+
        |                         |
        v                         v
+------------------+     +-------------------+
| Continual        |     | Federated         |
| Learning         |     | Learning          |
| (EWC/MAML)       |     | (Flower + DP)     |
+------------------+     +-------------------+
        |                         |
        v                         v
+------------------------------------------------+
|          Adaptive Bitrate Controller           |
|              (PPO Agent)                       |
+------------------------------------------------+
        |
        v
   Enhanced Output
```

---

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/manikya7022/Continual-Learning-for-Dynamic-Video-Quality-Enhancement.git
cd Continual-Learning-for-Dynamic-Video-Quality-Enhancement
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

### Quick Verification

```bash
python -c "from nerve_cl.models import SuperResolutionNet; print('Installation successful')"
```

---

## Project Structure

```
nerve_cl/
    __init__.py                 # Package initialization
    config/
        default.yaml            # Default configuration
    models/
        __init__.py
        frame_recovery.py       # Frame inpainting network
        super_resolution.py     # Temporal super-resolution
        enhancement_engine.py   # Combined enhancement pipeline
        layers/
            efficient_layers.py # Mobile-optimized layers
    continual/
        __init__.py
        memory.py               # Episodic memory buffer
        ewc.py                  # Elastic Weight Consolidation
        maml.py                 # Model-Agnostic Meta-Learning
        distillation.py         # Knowledge distillation
    federated/
        __init__.py
        client.py               # Flower FL client
        server.py               # Flower FL server
        privacy.py              # Differential privacy
        clustering.py           # User clustering
    abr/
        __init__.py
        environment.py          # Streaming simulation
        agent.py                # PPO agent

experiments/
    train_baseline.py           # Baseline training script
    train_continual.py          # Continual learning training
    train_federated.py          # Federated learning training
    train_abr.py                # ABR agent training

tests/
    test_models.py              # Model unit tests
    test_continual.py           # Continual learning tests
    test_abr.py                 # ABR component tests

mlops/
    drift/
        detector.py             # Data drift detection
    ab_testing/
        manager.py              # A/B testing framework

docker/
    Dockerfile                  # Container definition
    docker-compose.yml          # Multi-service setup
```

---

## Core Components

### 1. Super-Resolution Network

The SuperResolutionNet performs temporal super-resolution with motion compensation:

```python
from nerve_cl.models import SuperResolutionNet

model = SuperResolutionNet(
    scale_factor=2,          # 2x or 4x upscaling
    num_features=64,         # Feature channels
    num_residual_blocks=8,   # Depth of network
    temporal_window=3,       # Frames for temporal fusion
)

# Input: (B, T, C, H, W) - batch of frame sequences
# Output: (B, C, H*scale, W*scale) - upscaled center frame
output = model(lr_frames)
```

Key features:
- Motion estimation between frames
- Temporal feature aggregation
- Residual learning for detail preservation
- Pixel shuffle for efficient upsampling

### 2. Frame Recovery Network

Recovers corrupted or missing frames using spatial and temporal context:

```python
from nerve_cl.models import FrameRecoveryNet

model = FrameRecoveryNet()

# Inputs: corrupted frame, reference frames, corruption mask
recovered = model(corrupted_frame, reference_frames, mask)
```

### 3. Enhancement Engine

Combines frame recovery and super-resolution:

```python
from nerve_cl.models import EnhancementEngine, EnhancementConfig

config = EnhancementConfig(
    frame_recovery_enabled=True,
    super_resolution_enabled=True,
    scale_factor=2,
)
engine = EnhancementEngine(config)

results = engine(input_frames, corruption_mask=mask)
enhanced = results['enhanced']
```

### 4. Continual Learning Components

Elastic Weight Consolidation (EWC):
```python
from nerve_cl.continual import EWC

ewc = EWC(model, ewc_lambda=5000)
ewc.register_task(task_id=0, dataloader=train_loader)

# Training with EWC penalty
loss = task_loss + ewc.penalty()
```

Episodic Memory:
```python
from nerve_cl.continual import EpisodicMemory

memory = EpisodicMemory(capacity=1000, strategy='stratified')
memory.store(lr_frame, hr_frame, metadata={'content_type': 'sports'})
replay_batch = memory.sample(batch_size=32)
```

### 5. Federated Learning

```python
from nerve_cl.federated import VideoEnhancementClient, start_server

# Start server
start_server(model, num_rounds=100, min_clients=2)

# Client-side
client = VideoEnhancementClient(
    model=model,
    train_loader=train_loader,
    dp_enabled=True,
    dp_epsilon=8.0,
)
```

### 6. Adaptive Bitrate Agent

```python
from nerve_cl.abr import StreamingEnv, PPOAgent

env = StreamingEnv(max_steps=100)
agent = PPOAgent(obs_dim=7, num_actions=(5, 5))

obs, _ = env.reset()
action = agent.select_action(obs)
next_obs, reward, done, _, info = env.step(action)
```

---

## Training

### Baseline Training

```bash
python experiments/train_baseline.py \
    --data-dir data \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3
```

### Continual Learning Training

```bash
# With Elastic Weight Consolidation
python experiments/train_continual.py --strategy ewc --ewc-lambda 5000

# With Experience Replay
python experiments/train_continual.py --strategy replay --memory-size 500
```

### Federated Learning Training

```bash
# Simulation mode
python experiments/train_federated.py \
    --mode simulation \
    --num-clients 10 \
    --clients-per-round 5 \
    --num-rounds 50

# Distributed mode (start server first, then clients)
python experiments/train_federated.py --mode server
python experiments/train_federated.py --mode client --client-id 0
```

### ABR Agent Training

```bash
python experiments/train_abr.py \
    --num-steps 100000 \
    --learning-rate 3e-4
```

---

## Evaluation

### Running Evaluation

```python
import torch
from nerve_cl.models import SuperResolutionNet

# Load model
model = SuperResolutionNet(scale_factor=2, num_features=32, num_residual_blocks=4)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Compute metrics
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def compute_ssim(pred, target):
    C1, C2 = 0.01**2, 0.03**2
    mu_x, mu_y = pred.mean(), target.mean()
    sigma_x, sigma_y = pred.std(), target.std()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    return ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
```

### Evaluation Results

Results on synthetic test set (100 samples, 64x64 to 128x128 upscaling):

| Metric | Mean | Standard Deviation |
|--------|------|--------------------|
| PSNR   | 25.56 dB | 0.03 |
| SSIM   | 0.9608 | 0.0004 |
| MAE    | 0.0420 | 0.0002 |
| MSE    | 0.0028 | 0.0000 |

Comparison with baseline:

| Method | PSNR |
|--------|------|
| Bicubic Interpolation | 20.90 dB |
| SuperResolutionNet | 25.56 dB |
| Improvement | +4.66 dB |

---

## Test Cases

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v
pytest tests/test_continual.py -v
pytest tests/test_abr.py -v
```

### Test Coverage

The test suite contains 25 test cases covering:

**Model Tests (11 tests):**
- DepthwiseSeparableConv layer functionality
- PixelShuffleUpsampler output dimensions
- ResidualBlock skip connections
- FrameRecoveryNet forward pass and parameter count
- SuperResolutionNet at different scale factors (2x, 3x, 4x)
- LightweightSuperResolution variant
- EnhancementEngine SR-only and full pipeline modes

**Continual Learning Tests (8 tests):**
- EpisodicMemory store, sample, and capacity enforcement
- Stratified sampling across content types
- EWC Fisher matrix computation and penalty calculation
- MAML adaptation to new tasks
- Knowledge distillation loss computation

**ABR Tests (6 tests):**
- StreamingEnv reset and step functions
- Episode completion and termination
- PPOAgent action selection
- Training step with buffer updates
- Model save and load functionality

### Test Output

```
tests/test_abr.py::TestStreamingEnv::test_reset PASSED
tests/test_abr.py::TestStreamingEnv::test_step PASSED
tests/test_abr.py::TestStreamingEnv::test_episode PASSED
tests/test_abr.py::TestPPOAgent::test_select_action PASSED
tests/test_abr.py::TestPPOAgent::test_training_step PASSED
tests/test_abr.py::TestPPOAgent::test_save_load PASSED
tests/test_continual.py::TestEpisodicMemory::test_store_and_sample PASSED
tests/test_continual.py::TestEpisodicMemory::test_capacity_limit PASSED
tests/test_continual.py::TestEpisodicMemory::test_stratified_sampling PASSED
tests/test_continual.py::TestEWC::test_register_task PASSED
tests/test_continual.py::TestEWC::test_penalty_increases PASSED
tests/test_continual.py::TestMAML::test_adapt PASSED
tests/test_continual.py::TestDistillation::test_compute_loss PASSED
tests/test_continual.py::TestDistillation::test_register_task PASSED
tests/test_models.py::TestLayers::test_depthwise_separable_conv PASSED
tests/test_models.py::TestLayers::test_pixel_shuffle_upsampler PASSED
tests/test_models.py::TestLayers::test_residual_block PASSED
tests/test_models.py::TestFrameRecovery::test_forward PASSED
tests/test_models.py::TestFrameRecovery::test_parameter_count PASSED
tests/test_models.py::TestSuperResolution::test_forward PASSED
tests/test_models.py::TestSuperResolution::test_scale_factors PASSED
tests/test_models.py::TestSuperResolution::test_lightweight PASSED
tests/test_models.py::TestEnhancementEngine::test_sr_only PASSED
tests/test_models.py::TestEnhancementEngine::test_full_pipeline PASSED
tests/test_models.py::TestEnhancementEngine::test_model_info PASSED

============================= 25 passed in 11.94s ==============================
```

---

## Performance and Efficiency

### Model Parameters

| Model | Parameters | Size (FP32) |
|-------|------------|-------------|
| SuperResolutionNet (default) | 820,339 | 3.1 MB |
| LightweightSuperResolution | ~50,000 | 0.2 MB |
| FrameRecoveryNet | ~2.5M | 9.5 MB |
| EnhancementEngine (full) | ~3.3M | 12.6 MB |

### Computational Efficiency

Design choices for mobile deployment:

1. **Depthwise Separable Convolutions**: Reduce multiply-accumulate operations by 8-9x compared to standard convolutions.

2. **Pixel Shuffle Upsampling**: More efficient than transposed convolutions, avoids checkerboard artifacts.

3. **Temporal Window**: Using 3 frames balances quality and memory usage.

4. **Residual Learning**: Network only learns the difference, easier optimization.

### Training Efficiency

Training on synthetic dataset (500 train, 100 val samples):

| Configuration | Time per Epoch | Total Time (10 epochs) |
|---------------|----------------|------------------------|
| CPU (M1) | ~15s | ~150s |
| MPS (Apple Silicon) | ~14s | ~142s |
| CUDA (estimated) | ~5s | ~50s |

### Memory Efficiency

Episodic memory with stratified sampling:
- 1000 sample capacity
- Content-balanced storage
- O(1) insertion, O(n) sampling
- Supports streaming with recency bias

---

## Configuration

Default configuration is in `nerve_cl/config/default.yaml`. Key settings:

```yaml
# Model configuration
model:
  super_resolution:
    enabled: true
    scale_factor: 2
    num_features: 64
    num_residual_blocks: 8

# Training configuration
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100

# Continual learning
continual_learning:
  strategy: ewc
  ewc_lambda: 5000
  memory_capacity: 1000

# Federated learning
federated:
  num_clients: 10
  clients_per_round: 5
  local_epochs: 5
  dp_epsilon: 8.0

# ABR configuration
abr:
  algorithm: ppo
  gamma: 0.99
  learning_rate: 0.0003
```

---

## References

- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (EWC)
- Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation" (MAML)
- McMahan et al., "Communication-Efficient Learning of Deep Networks" (FedAvg)
- Schulman et al., "Proximal Policy Optimization Algorithms" (PPO)
