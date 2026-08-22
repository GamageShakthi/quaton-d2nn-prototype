"""
Automated robustness benchmarking script.

Runs comprehensive sweeps over:
    1. Quantization bit-widths (FP32, 8-bit, 4-bit, 3-bit, 2-bit, 1-bit).
    2. Alignment jitter levels.
    3. Fabrication phase noise levels.

Generates comparison plots and saves metrics as JSON.

Usage:
    python scripts/benchmark_robustness.py --checkpoint checkpoints/best_model.pt
"""

import sys
import os
import argparse
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from d2nn.models import D2NNClassifier
from d2nn.training import get_dataloader
from d2nn.training.evaluate import evaluate_ptq_sweep, evaluate_noise_robustness
from d2nn.visualization import plot_quantization_comparison, plot_noise_robustness


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark D²NN robustness to quantization and noise.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset for evaluation.")
    parser.add_argument("--n_pixels", type=int, default=28)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--pixel_pitch", type=float, default=0.4e-3)
    parser.add_argument("--wavelength", type=float, default=0.75e-3)
    parser.add_argument("--z", type=float, default=25e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Monte-Carlo trials for noise evaluation.")
    parser.add_argument("--results_dir", type=str, default="./results/benchmark",
                        help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = D2NNClassifier(
        n_pixels=args.n_pixels,
        n_layers=args.n_layers,
        n_classes=args.n_classes,
        pixel_pitch=args.pixel_pitch,
        wavelength=args.wavelength,
        z=args.z,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, "
          f"val_acc {checkpoint.get('val_acc', '?'):.4f})")

    # Data
    test_loader = get_dataloader(
        dataset_name=args.dataset, train=False,
        batch_size=args.batch_size, n_pixels=args.n_pixels,
    )

    # 1. Quantization sweep
    print("\n=== PTQ Sweep ===")
    ptq_results = evaluate_ptq_sweep(
        model, test_loader, bit_widths=[1, 2, 3, 4, 8], device=device
    )
    for k, v in ptq_results.items():
        print(f"  {k:>6s}: {v:.4f}")

    with open(os.path.join(args.results_dir, "ptq_sweep.json"), "w") as f:
        json.dump(ptq_results, f, indent=2)

    plot_quantization_comparison(
        ptq_results,
        save_path=os.path.join(args.results_dir, "accuracy_vs_bitwidth.png"),
        show=False,
    )

    # 2. Noise robustness
    print("\n=== Noise Robustness ===")
    noise_results = evaluate_noise_robustness(
        model, test_loader,
        jitter_levels=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
        phase_noise_levels=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
        n_trials=args.n_trials,
        device=device,
    )

    print("\n  Jitter robustness:")
    for k, v in noise_results["jitter"].items():
        print(f"    σ={k}: {v['mean']:.4f} ± {v['std']:.4f}")

    print("\n  Phase noise robustness:")
    for k, v in noise_results["phase_noise"].items():
        print(f"    σ={k}: {v['mean']:.4f} ± {v['std']:.4f}")

    with open(os.path.join(args.results_dir, "noise_robustness.json"), "w") as f:
        json.dump(noise_results, f, indent=2)

    plot_noise_robustness(
        noise_results,
        save_path=os.path.join(args.results_dir, "accuracy_vs_jitter.png"),
        show=False,
    )

    print(f"\nResults saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
