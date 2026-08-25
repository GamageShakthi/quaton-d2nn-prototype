# QuATON-D²NN

**Quantization-Aware and Noise-Robust Diffractive Neural Networks**

A PyTorch framework for training and evaluating **Diffractive Deep Neural Networks (D²NNs)** under practical optical hardware constraints.

The project studies the effects of:

* Phase quantization
* Fabrication noise
* Layer misalignment
* Detector noise

and compares **FP32, PTQ, QAT, and noise-aware training**.

## Architecture

```text
Input
  ↓
Phase Mask → Free-Space Propagation
  ↓
Phase Mask → Free-Space Propagation
  ↓
        ...
  ↓
Detector
  ↓
Classification
```

Wave propagation is simulated using the **Angular Spectrum Method**, with phase quantization implemented using a **Straight-Through Estimator (STE)**.

## Quick Start

```bash
git clone https://github.com/GamageShakthi/quaton-d2nn.git
cd quaton-d2nn
pip install -r requirements.txt
pip install -e .
```

Train a 4-bit D²NN:

```bash
python scripts/train.py --dataset mnist --n_layers 5 --n_bits 4 --epochs 30
```

Run robustness evaluation:

```bash
python scripts/benchmark_robustness.py --checkpoint checkpoints/best_model.pt
```

## Project

```text
d2nn/
├── physics/
├── quantization/
├── noise/
├── layers/
├── models/
├── training/
└── visualization/
```

See `notebooks/quaton_d2nn_workflow.ipynb` for the full walkthrough.

## References

* Lin et al., *All-optical machine learning using diffractive deep neural networks*, Science, 2018.
* Matsushima & Shimobaba, *Band-Limited Angular Spectrum Method*, Optics Express, 2009.
* Bengio et al., *Estimating or Propagating Gradients Through Stochastic Neurons*, 2013.

## Author

**Shakthi Gamage**
[GitHub](https://github.com/GamageShakthi)

## License

MIT
