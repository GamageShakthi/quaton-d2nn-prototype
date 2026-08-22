"""
CLI training script for D²NN optical classifiers.

Usage examples:

    # Train continuous (FP32) model on MNIST
    python scripts/train.py --dataset mnist --n_layers 5 --epochs 20

    # Train 4-bit QAT model on Fashion-MNIST
    python scripts/train.py --dataset fashion_mnist --n_bits 4 --epochs 30

    # Train with noise injection for robustness
    python scripts/train.py --dataset mnist --n_bits 4 --phase_sigma 0.1 \\
        --jitter_sigma 0.5 --epochs 30
"""

import sys
import os
import argparse
import json
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from d2nn.models import D2NNClassifier
from d2nn.training import get_dataloader, Trainer
from d2nn.training.evaluate import evaluate_model
from d2nn.visualization import plot_training_history, plot_phase_masks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a D²NN optical classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size.")

    # Model architecture
    parser.add_argument("--n_pixels", type=int, default=28,
                        help="Spatial resolution of the optical grid.")
    parser.add_argument("--n_layers", type=int, default=5,
                        help="Number of diffractive layers.")
    parser.add_argument("--n_classes", type=int, default=10,
                        help="Number of output classes.")

    # Optical physics
    parser.add_argument("--pixel_pitch", type=float, default=0.4e-3,
                        help="Pixel spacing [m].")
    parser.add_argument("--wavelength", type=float, default=0.75e-3,
                        help="Optical wavelength [m].")
    parser.add_argument("--z", type=float, default=25e-3,
                        help="Inter-layer propagation distance [m].")

    # Quantization
    parser.add_argument("--n_bits", type=int, default=None,
                        help="Phase quantization bit-width (None = continuous).")

    # Noise
    parser.add_argument("--phase_sigma", type=float, default=0.0,
                        help="Fabrication phase noise std [rad].")
    parser.add_argument("--jitter_sigma", type=float, default=0.0,
                        help="Alignment jitter std [pixels].")
    parser.add_argument("--dead_pixel_prob", type=float, default=0.0,
                        help="Dead pixel probability.")

    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay.")

    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints.")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory for saving results.")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (for naming outputs).")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Experiment name
    if args.exp_name is None:
        bits_str = f"{args.n_bits}bit" if args.n_bits else "fp32"
        args.exp_name = f"{args.dataset}_{bits_str}_{args.n_layers}layers"

    # Create output directories
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    results_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config: {json.dumps(config, indent=2)}")

    # Data
    print("Loading data...")
    train_loader = get_dataloader(
        dataset_name=args.dataset, train=True,
        batch_size=args.batch_size, n_pixels=args.n_pixels,
    )
    test_loader = get_dataloader(
        dataset_name=args.dataset, train=False,
        batch_size=args.batch_size, n_pixels=args.n_pixels,
    )
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Test: {len(test_loader.dataset)} samples")

    # Model
    model = D2NNClassifier(
        n_pixels=args.n_pixels,
        n_layers=args.n_layers,
        n_classes=args.n_classes,
        pixel_pitch=args.pixel_pitch,
        wavelength=args.wavelength,
        z=args.z,
        n_bits=args.n_bits,
        phase_sigma=args.phase_sigma,
        dead_pixel_prob=args.dead_pixel_prob,
        jitter_sigma=args.jitter_sigma,
    )
    n_params = model.count_optical_parameters()
    print(f"Model: {args.n_layers} layers, {n_params:,} optical parameters")

    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.fit(train_loader, test_loader, n_epochs=args.epochs)

    # Save history
    trainer.save_history(os.path.join(results_dir, "history.json"))

    # Final evaluation
    print("\nFinal evaluation...")
    result = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {result['overall_accuracy']:.4f}")

    with open(os.path.join(results_dir, "eval_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Plots
    print("Generating plots...")
    plot_training_history(
        history,
        save_path=os.path.join(results_dir, "training_curves.png"),
        show=False,
    )
    plot_phase_masks(
        model.get_phase_masks(),
        save_path=os.path.join(results_dir, "phase_masks.png"),
        show=False,
    )

    print(f"\nDone! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
