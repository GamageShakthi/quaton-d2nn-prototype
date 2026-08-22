"""
Wavefront visualization CLI script.

Generates wavefield propagation diagrams, phase mask profiles, and
detector energy maps for a trained D²NN model.

Usage:
    python scripts/visualize_wavefronts.py --checkpoint checkpoints/best_model.pt
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from d2nn.models import D2NNClassifier
from d2nn.training import get_dataloader
from d2nn.visualization import (
    plot_phase_masks,
    plot_wavefield_propagation,
    plot_detector_energy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize D²NN wavefront propagation and optical fields."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--n_pixels", type=int, default=28)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--pixel_pitch", type=float, default=0.4e-3)
    parser.add_argument("--wavelength", type=float, default=0.75e-3)
    parser.add_argument("--z", type=float, default=25e-3)
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of test sample to visualize.")
    parser.add_argument("--output_dir", type=str, default="./results/visualizations")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
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

    # Get sample image
    test_loader = get_dataloader(
        dataset_name=args.dataset, train=False,
        batch_size=1, n_pixels=args.n_pixels, shuffle=False,
    )
    dataset = test_loader.dataset
    image, label = dataset[args.sample_idx]
    image = image.unsqueeze(0)  # (1, 1, H, W)

    print(f"Sample index: {args.sample_idx}, Label: {label}")

    # 1. Phase masks
    print("Plotting phase masks...")
    plot_phase_masks(
        model.get_phase_masks(),
        save_path=os.path.join(args.output_dir, "phase_masks.png"),
        show=False,
    )

    # 2. Wavefield propagation
    print("Plotting wavefield propagation...")
    plot_wavefield_propagation(
        model, image, device=device,
        save_path=os.path.join(args.output_dir, "wavefield_propagation.png"),
        show=False,
    )

    # 3. Detector energy
    print("Plotting detector energy...")
    plot_detector_energy(
        model, image, label=label, device=device,
        save_path=os.path.join(args.output_dir, "detector_energy.png"),
        show=False,
    )

    print(f"Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
