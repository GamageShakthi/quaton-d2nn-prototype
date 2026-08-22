<p align="center">
  <h1 align="center">🔬 QuATON-D²NN</h1>
  <p align="center">
    <strong>Quantization & Noise-Robust Diffractive Optical Neural Networks</strong>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#physics">Physics</a> •
    <a href="#installation">Installation</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#results">Results</a> •
    <a href="#citation">Citation</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## Overview

**QuATON-D²NN** is a PyTorch framework for simulating, training, and benchmarking multi-layer **Diffractive Deep Neural Networks (D²NN)** under real-world manufacturing and hardware constraints:

- **Phase-level quantization** (1-bit to 8-bit SLM phase modulation) via Straight-Through Estimator (STE)
- **Alignment jitter** (lateral misalignment between diffractive layers)
- **Fabrication noise** (phase surface roughness, dead pixels)
- **Detector noise** (photon shot noise, electronic readout noise)

This directly mirrors the core mechanism of **QuATON: Quantization-Aware Training of Optical Neurons**, enabling fair comparison between:

| Training Strategy | Description |
|---|---|
| **Continuous (FP32)** | Full-precision phase optimization |
| **Post-Training Quantization (PTQ)** | Quantize a trained FP32 model |
| **Quantization-Aware Training (QAT)** | Train with quantization in the loop via STE |
| **Noise-Robust Training** | Train with noise injection for hardware robustness |

## Physics

### Rayleigh-Sommerfeld Diffraction (Angular Spectrum Method)

Coherent scalar wavefield propagation through free space over distance *z*:

```
U(x,y;z) = F⁻¹{ F{U(x,y;0)} · H(f_x, f_y; z) }
```

where the bandlimited transfer function is:

```
H(f_x, f_y; z) = exp(j·2π/λ · z · √(1 - (λf_x)² - (λf_y)²))
```

for propagating waves (f_x² + f_y² ≤ 1/λ²), with evanescent wave suppression.

### D²NN Architecture

```
Input Image → Optical Encoder → [Phase Mask → Free-Space Propagation] × N → Detector → Logits
```

Each diffractive layer applies a learnable phase modulation:

```
U_out(x,y) = U_in(x,y) · exp(j·φ(x,y))
```

### Quantization-Aware Training (STE)

For a *b*-bit SLM with phase step Δφ = 2π/2ᵇ:

- **Forward pass**: φ_q = round(φ/Δφ) · Δφ
- **Backward pass**: ∂L/∂φ = ∂L/∂φ_q (straight-through)

## Project Structure

```
quaton-d2nn/
├── d2nn/                          # Core Python package
│   ├── physics/                   # ASM diffraction, optical grids
│   ├── quantization/              # STE, phase quantizers
│   ├── noise/                     # Alignment, fabrication, detector noise
│   ├── layers/                    # Encoding, diffractive, propagation, detector layers
│   ├── models/                    # D²NN classifier
│   ├── training/                  # Dataset, trainer, evaluator
│   └── visualization/             # Publication-quality plots
├── notebooks/
│   └── quaton_d2nn_workflow.ipynb  # Interactive workflow notebook
├── scripts/
│   ├── train.py                   # CLI training script
│   ├── benchmark_robustness.py    # Automated benchmarking
│   └── visualize_wavefronts.py    # Wavefront visualization
├── tests/                         # Comprehensive test suite
├── results/                       # Benchmark outputs
└── checkpoints/                   # Saved models
```

## Installation

```bash
# Clone the repository
git clone https://github.com/GamageShakthi/quaton-d2nn.git
cd quaton-d2nn

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- numpy, scipy, matplotlib, tqdm, seaborn

## Quickstart

### 1. Train a Continuous (FP32) Model

```bash
python scripts/train.py --dataset mnist --n_layers 5 --epochs 20
```

### 2. Train with 4-bit Quantization-Aware Training

```bash
python scripts/train.py --dataset mnist --n_bits 4 --n_layers 5 --epochs 30
```

### 3. Train with Noise Injection for Robustness

```bash
python scripts/train.py --dataset fashion_mnist --n_bits 4 \
    --phase_sigma 0.1 --jitter_sigma 0.5 --epochs 30
```

### 4. Benchmark Robustness

```bash
python scripts/benchmark_robustness.py \
    --checkpoint checkpoints/mnist_fp32_5layers/best_model.pt
```

### 5. Visualize Wavefronts

```bash
python scripts/visualize_wavefronts.py \
    --checkpoint checkpoints/mnist_fp32_5layers/best_model.pt
```

### 6. Python API

```python
import torch
from d2nn.models import D2NNClassifier
from d2nn.training import get_dataloader, Trainer

# Create model
model = D2NNClassifier(
    n_pixels=28, n_layers=5, n_classes=10,
    pixel_pitch=0.4e-3, wavelength=0.75e-3, z=25e-3,
    n_bits=4,  # 4-bit QAT
    phase_sigma=0.1,  # fabrication noise
    jitter_sigma=0.5,  # alignment jitter
)

# Load data
train_loader = get_dataloader("mnist", train=True, batch_size=64)
test_loader = get_dataloader("mnist", train=False, batch_size=128)

# Train
trainer = Trainer(model, lr=5e-3)
history = trainer.fit(train_loader, test_loader, n_epochs=20)
```

## Interactive Notebook

The Jupyter notebook at [`notebooks/quaton_d2nn_workflow.ipynb`](notebooks/quaton_d2nn_workflow.ipynb) provides a comprehensive, illustrated walkthrough covering:

1. **Wave Optics Fundamentals** — ASM derivation, transfer function visualization
2. **D²NN Architecture** — Layer-by-layer construction, detector design
3. **Quantization Theory** — STE mechanics, bit-width analysis
4. **Training & Evaluation** — Full training pipeline with live plots
5. **Robustness Benchmarking** — PTQ vs QAT comparison, noise sweeps
6. **Optical Field Visualization** — Wavefront intensity/phase at each layer

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_diffraction.py -v      # Physics tests
pytest tests/test_quantization.py -v     # Quantization tests
pytest tests/test_noise.py -v            # Noise model tests
pytest tests/test_model_forward.py -v    # End-to-end model tests
```

## Key References

1. **Lin et al.**, "All-optical machine learning using diffractive deep neural networks," *Science* 361(6406), 2018.
2. **Matsushima & Shimobaba**, "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation," *Optics Express* 17(22), 2009.
3. **Bengio et al.**, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation," *arXiv:1308.3432*, 2013.
4. **QuATON**, Quantization-Aware Training of Optical Neurons for hardware-deployable D²NN.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Shakthi Gamage** — [@GamageShakthi](https://github.com/GamageShakthi)
