"""
Publication-quality visualization utilities for D²NN optical systems.

Provides plotting functions for:
    - Learned phase mask profiles (2D colormaps).
    - Wavefield intensity and phase during propagation.
    - Detector region energy distributions.
    - Training loss/accuracy curves.
    - Quantization comparison bar charts.
    - Noise robustness degradation curves.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import math
from typing import List, Dict, Optional, Tuple
import os


# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_phase_masks(
    phase_masks: List[torch.Tensor],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize learned phase masks for all diffractive layers.

    Args:
        phase_masks: List of phase tensors, each shape (H, W).
        save_path: If provided, save figure to this path.
        show: If True, display the figure.

    Returns:
        Matplotlib Figure object.
    """
    n_layers = len(phase_masks)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for i, (ax, mask) in enumerate(zip(axes, phase_masks)):
        phase_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        im = ax.imshow(
            phase_np,
            cmap="twilight",
            vmin=0,
            vmax=2 * np.pi,
            interpolation="nearest",
        )
        ax.set_title(f"Layer {i + 1}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Phase [rad]", fraction=0.046)

    fig.suptitle("Learned Diffractive Phase Masks", fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def plot_wavefield_propagation(
    model,
    sample_image: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize wavefield intensity and phase at each layer during propagation.

    Args:
        model: D²NN model.
        sample_image: Single image tensor, shape (1, 1, H, W) or (1, H, W).
        device: Torch device.
        save_path: Optional save path.
        show: Show figure.

    Returns:
        Matplotlib Figure.
    """
    model.eval()
    model.to(device)
    sample_image = sample_image.to(device)

    # Collect intermediate wavefields
    wavefields = []
    with torch.no_grad():
        u = model.encoder(sample_image)
        wavefields.append(("Input", u.squeeze(0).cpu()))

        for i, layer in enumerate(model.optical_layers):
            u = layer(u)
            layer_type = layer.__class__.__name__
            wavefields.append((f"{layer_type} {i // 2 + 1}", u.squeeze(0).cpu()))

    n = len(wavefields)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))

    for j, (name, field) in enumerate(wavefields):
        intensity = torch.abs(field).numpy() ** 2
        phase = torch.angle(field).numpy()

        # Intensity
        axes[0, j].imshow(intensity, cmap="inferno", interpolation="bilinear")
        axes[0, j].set_title(name, fontsize=9)
        axes[0, j].axis("off")

        # Phase
        axes[1, j].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                          interpolation="bilinear")
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel("Intensity |U|²", fontsize=11)
    axes[1, 0].set_ylabel("Phase ∠U", fontsize=11)
    fig.suptitle("Wavefield Propagation Through D²NN", fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def plot_detector_energy(
    model,
    sample_image: torch.Tensor,
    label: int,
    class_names: Optional[List[str]] = None,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize detector region energies for a single input.

    Args:
        model: D²NN model.
        sample_image: Input image tensor.
        label: Ground-truth label.
        class_names: Optional list of class name strings.
        device: Torch device.
        save_path: Optional save path.
        show: Show figure.

    Returns:
        Matplotlib Figure.
    """
    model.eval()
    model.to(device)
    sample_image = sample_image.to(device)

    with torch.no_grad():
        logits = model(sample_image)
        energies = logits.squeeze(0).cpu().numpy()
        pred = logits.argmax(dim=1).item()

    n_classes = len(energies)
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of detector energies
    colors = ["#2ecc71" if i == label else "#e74c3c" if i == pred and pred != label
              else "#3498db" for i in range(n_classes)]
    bars = ax1.bar(class_names, energies, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Integrated Intensity")
    ax1.set_title(f"Detector Energy  (True: {label}, Pred: {pred})")
    ax1.axhline(y=0, color="gray", linewidth=0.5)

    # Detector region overlay on output intensity
    with torch.no_grad():
        u = model.encoder(sample_image)
        for layer in model.optical_layers:
            u = layer(u)
        intensity = (torch.abs(u.squeeze(0)) ** 2).cpu().numpy()

    ax2.imshow(intensity, cmap="hot", interpolation="bilinear")
    det_mask = model.detector.get_detector_mask().cpu().numpy()
    # Overlay detector regions as colored rectangles
    cmap = plt.cm.Set3
    for c in range(n_classes):
        regions = model.detector.detector_regions[c].cpu().numpy()
        r0, r1, c0, c1 = regions
        rect = Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=1.5,
            edgecolor=cmap(c / n_classes),
            facecolor="none",
            linestyle="--",
        )
        ax2.add_patch(rect)
        ax2.text(
            (c0 + c1) / 2, (r0 + r1) / 2, str(c),
            ha="center", va="center", fontsize=7,
            color="white", fontweight="bold",
        )
    ax2.set_title("Output Intensity + Detector Regions")
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training loss and accuracy curves.

    Args:
        history: Dictionary with keys 'train_loss', 'train_acc',
            'val_loss', 'val_acc'.
        save_path: Optional save path.
        show: Show figure.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", label="Train", markersize=3,
             color="#3498db")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val", markersize=3,
             color="#e74c3c")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "o-", label="Train", markersize=3,
             color="#3498db")
    ax2.plot(epochs, history["val_acc"], "s-", label="Val", markersize=3,
             color="#e74c3c")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def plot_quantization_comparison(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart comparing accuracy across bit-widths (FP32, 8-bit, ..., 1-bit).

    Args:
        results: Dictionary mapping bit-width labels to accuracy values.
        save_path: Optional save path.
        show: Show figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = list(results.keys())
    values = [results[k] * 100 for k in labels]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("Accuracy vs. Phase Quantization Bit-Width")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def plot_noise_robustness(
    results: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot noise robustness curves with error bands.

    Args:
        results: Dictionary from ``evaluate_noise_robustness``.
        save_path: Optional save path.
        show: Show figure.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Jitter robustness
    if "jitter" in results and results["jitter"]:
        jitter_sigmas = []
        jitter_means = []
        jitter_stds = []
        for k, v in results["jitter"].items():
            jitter_sigmas.append(float(k))
            jitter_means.append(v["mean"] * 100)
            jitter_stds.append(v["std"] * 100)

        ax1.errorbar(
            jitter_sigmas, jitter_means, yerr=jitter_stds,
            fmt="o-", color="#2ecc71", capsize=4, linewidth=2, markersize=6,
            label="Accuracy",
        )
        ax1.fill_between(
            jitter_sigmas,
            np.array(jitter_means) - np.array(jitter_stds),
            np.array(jitter_means) + np.array(jitter_stds),
            alpha=0.2, color="#2ecc71",
        )
        ax1.set_xlabel("Alignment Jitter σ [pixels]")
        ax1.set_ylabel("Accuracy [%]")
        ax1.set_title("Robustness to Alignment Jitter")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Phase noise robustness
    if "phase_noise" in results and results["phase_noise"]:
        pn_sigmas = []
        pn_means = []
        pn_stds = []
        for k, v in results["phase_noise"].items():
            pn_sigmas.append(float(k))
            pn_means.append(v["mean"] * 100)
            pn_stds.append(v["std"] * 100)

        ax2.errorbar(
            pn_sigmas, pn_means, yerr=pn_stds,
            fmt="s-", color="#e74c3c", capsize=4, linewidth=2, markersize=6,
            label="Accuracy",
        )
        ax2.fill_between(
            pn_sigmas,
            np.array(pn_means) - np.array(pn_stds),
            np.array(pn_means) + np.array(pn_stds),
            alpha=0.2, color="#e74c3c",
        )
        ax2.set_xlabel("Fabrication Phase Noise σ [rad]")
        ax2.set_ylabel("Accuracy [%]")
        ax2.set_title("Robustness to Phase Fabrication Errors")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig
