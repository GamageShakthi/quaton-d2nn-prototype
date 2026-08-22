"""
Evaluation and benchmarking utilities for D²NN models.

Provides functions for:
    - Single-model evaluation (accuracy, per-class metrics).
    - Quantization sweep evaluation (FP32 → PTQ at various bit-widths).
    - Noise robustness evaluation (jitter sweep, phase noise sweep).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import copy


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    n_classes: int = 10,
) -> Dict:
    """
    Comprehensive evaluation of a D²NN model.

    Args:
        model: Trained D²NN model.
        test_loader: Test data loader.
        device: Torch device.
        n_classes: Number of classes.

    Returns:
        Dictionary with overall accuracy, per-class accuracy,
        confusion matrix, and total samples.
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    class_correct = np.zeros(n_classes)
    class_total = np.zeros(n_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1

    overall_acc = correct / total
    per_class_acc = np.where(
        class_total > 0, class_correct / class_total, 0.0
    ).tolist()

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion.tolist(),
        "total_samples": total,
        "correct": correct,
    }


@torch.no_grad()
def evaluate_ptq_sweep(
    model: nn.Module,
    test_loader: DataLoader,
    bit_widths: List[int] = [1, 2, 3, 4, 8],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluate a continuously-trained model under Post-Training Quantization
    at various bit-widths.

    Creates a deep copy of the model for each bit-width to avoid modifying
    the original.

    Args:
        model: Trained continuous (FP32) D²NN model.
        test_loader: Test data loader.
        bit_widths: List of bit-widths to evaluate.
        device: Torch device.

    Returns:
        Dictionary mapping bit-width (as string) to accuracy.
    """
    results = {}

    # First evaluate continuous (FP32) model
    fp32_result = evaluate_model(model, test_loader, device)
    results["fp32"] = fp32_result["overall_accuracy"]

    for bits in bit_widths:
        model_copy = copy.deepcopy(model)
        model_copy.apply_ptq(n_bits=bits)
        model_copy.to(device)

        result = evaluate_model(model_copy, test_loader, device)
        results[f"{bits}bit"] = result["overall_accuracy"]

        del model_copy

    return results


@torch.no_grad()
def evaluate_noise_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    jitter_levels: List[float] = [0.0, 0.25, 0.5, 1.0, 2.0],
    phase_noise_levels: List[float] = [0.0, 0.05, 0.1, 0.2, 0.5],
    n_trials: int = 5,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Evaluate model robustness under varying noise conditions via Monte-Carlo.

    For each noise level, runs ``n_trials`` forward passes and reports mean
    and std accuracy.

    Args:
        model: Trained D²NN model.
        test_loader: Test data loader.
        jitter_levels: List of alignment jitter σ values [pixels].
        phase_noise_levels: List of fabrication phase noise σ values [rad].
        n_trials: Number of Monte-Carlo trials per noise level.
        device: Torch device.

    Returns:
        Dictionary with jitter and phase noise robustness curves.
    """
    from ..layers.diffractive_layer import DiffractiveLayer

    results = {"jitter": {}, "phase_noise": {}}

    # Jitter robustness
    for sigma in jitter_levels:
        trial_accs = []
        for _ in range(n_trials):
            model_copy = copy.deepcopy(model)
            model_copy.to(device)

            # Set jitter for all layers and enable eval noise
            for module in model_copy.modules():
                if isinstance(module, DiffractiveLayer):
                    if module.jitter is not None:
                        module.jitter.sigma = sigma
                        module.jitter.eval_jitter = True
                    elif sigma > 0:
                        from ..noise.alignment import AlignmentJitter
                        module.jitter = AlignmentJitter(
                            sigma=sigma, eval_jitter=True
                        )

            result = evaluate_model(model_copy, test_loader, device)
            trial_accs.append(result["overall_accuracy"])
            del model_copy

        results["jitter"][f"{sigma:.2f}"] = {
            "mean": float(np.mean(trial_accs)),
            "std": float(np.std(trial_accs)),
        }

    # Phase noise robustness
    for sigma in phase_noise_levels:
        trial_accs = []
        for _ in range(n_trials):
            model_copy = copy.deepcopy(model)
            model_copy.to(device)

            for module in model_copy.modules():
                if isinstance(module, DiffractiveLayer):
                    module.fab_noise.phase_sigma = sigma
                    module.fab_noise.eval_noise = True

            result = evaluate_model(model_copy, test_loader, device)
            trial_accs.append(result["overall_accuracy"])
            del model_copy

        results["phase_noise"][f"{sigma:.2f}"] = {
            "mean": float(np.mean(trial_accs)),
            "std": float(np.std(trial_accs)),
        }

    return results
