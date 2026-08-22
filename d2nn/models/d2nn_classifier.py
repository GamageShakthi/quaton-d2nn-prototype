"""
Multi-layer Diffractive Deep Neural Network (D²NN) classifier.

Assembles a full optical computing pipeline:

    Input Image  →  Optical Encoder  →  [Diffract → Propagate] × N_layers
    →  Final Propagation  →  Detector  →  Class Logits

Each diffractive layer contains a learnable phase mask that is optionally
quantized (QAT / PTQ) and corrupted by physical noise during training
for robustness.

Reference
---------
Lin et al., "All-optical machine learning using diffractive deep neural
networks," Science 361(6406), 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as TF
from typing import Optional, List, Dict, Any

from ..layers.encoding import AmplitudeEncoder, PhaseEncoder
from ..layers.diffractive_layer import DiffractiveLayer
from ..layers.propagation_layer import PropagationLayer
from ..layers.detector_layer import DetectorLayer


class D2NNClassifier(nn.Module):
    """
    End-to-end D²NN image classifier.

    Args:
        n_pixels: Spatial resolution of the optical grid.
        n_layers: Number of diffractive layers (phase masks).
        n_classes: Number of output classification classes.
        pixel_pitch: Physical pixel spacing [m].
        wavelength: Optical wavelength [m].
        z: Inter-layer propagation distance [m].
        n_bits: Phase quantization bit-width. ``None`` → continuous.
        encoding: Input encoding method — ``"amplitude"`` or ``"phase"``.
        phase_sigma: Fabrication phase noise std [rad].
        dead_pixel_prob: Dead pixel probability per layer.
        jitter_sigma: Alignment jitter std [pixels].
        detector_size: Side length of each detector region [pixels].
        shot_noise_level: Shot noise for detector.
        readout_sigma: Readout noise std for detector.
        padding: Propagation padding pixels.

    Example::

        >>> model = D2NNClassifier(n_pixels=28, n_layers=5, n_classes=10,
        ...     pixel_pitch=0.4e-3, wavelength=0.75e-3, z=25e-3, n_bits=4)
        >>> x = torch.rand(4, 1, 28, 28)  # Batch of MNIST images
        >>> logits = model(x)
        >>> logits.shape  # (4, 10)
    """

    def __init__(
        self,
        n_pixels: int = 28,
        n_layers: int = 5,
        n_classes: int = 10,
        pixel_pitch: float = 0.4e-3,
        wavelength: float = 0.75e-3,
        z: float = 25e-3,
        n_bits: Optional[int] = None,
        encoding: str = "amplitude",
        phase_sigma: float = 0.0,
        dead_pixel_prob: float = 0.0,
        jitter_sigma: float = 0.0,
        detector_size: Optional[int] = None,
        shot_noise_level: float = 0.0,
        readout_sigma: float = 0.0,
        padding: int = 0,
    ):
        super().__init__()

        self.n_pixels = n_pixels
        self.n_layers = n_layers
        self.n_classes = n_classes

        # Input encoder
        if encoding == "amplitude":
            self.encoder = AmplitudeEncoder()
        elif encoding == "phase":
            self.encoder = PhaseEncoder()
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # Build alternating [DiffractiveLayer, PropagationLayer] stack
        layers = nn.ModuleList()
        for i in range(n_layers):
            layers.append(
                DiffractiveLayer(
                    n_pixels=n_pixels,
                    n_bits=n_bits,
                    phase_sigma=phase_sigma,
                    dead_pixel_prob=dead_pixel_prob,
                    jitter_sigma=jitter_sigma,
                    init_mode="uniform",
                )
            )
            layers.append(
                PropagationLayer(
                    n_pixels=n_pixels,
                    pixel_pitch=pixel_pitch,
                    wavelength=wavelength,
                    z=z,
                    padding=padding,
                )
            )
        self.optical_layers = layers

        # Output detector
        self.detector = DetectorLayer(
            n_pixels=n_pixels,
            n_classes=n_classes,
            detector_size=detector_size,
            shot_noise_level=shot_noise_level,
            readout_sigma=readout_sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full D²NN pipeline.

        Args:
            x: Input images, shape (B, 1, H, W) or (B, H, W).

        Returns:
            Classification logits, shape (B, n_classes).
        """
        # Encode input as complex wavefield
        u = self.encoder(x)

        # Pass through diffractive layers and propagation
        for layer in self.optical_layers:
            u = layer(u)

        # Detect (integrate intensity in class regions)
        logits = self.detector(u)

        return logits

    def get_phase_masks(self) -> List[torch.Tensor]:
        """
        Extract effective phase masks from all diffractive layers.

        Returns:
            List of phase tensors (one per diffractive layer), each
            shape (n_pixels, n_pixels).
        """
        masks = []
        for layer in self.optical_layers:
            if isinstance(layer, DiffractiveLayer):
                masks.append(layer.get_effective_phase().detach())
        return masks

    def get_config(self) -> Dict[str, Any]:
        """Return a dictionary of the model configuration."""
        return {
            "n_pixels": self.n_pixels,
            "n_layers": self.n_layers,
            "n_classes": self.n_classes,
            "encoding": self.encoder.__class__.__name__,
        }

    def count_optical_parameters(self) -> int:
        """Count total trainable phase parameters across all layers."""
        total = 0
        for layer in self.optical_layers:
            if isinstance(layer, DiffractiveLayer):
                total += layer.phase.numel()
        return total

    def apply_ptq(self, n_bits: int) -> None:
        """
        Apply Post-Training Quantization (PTQ) to all diffractive layers.

        Replaces each layer's quantizer with a new PhaseQuantizer at the
        specified bit-width.  Use this after training a continuous model
        to evaluate quantization degradation.

        Args:
            n_bits: Target bit-width for PTQ.
        """
        from ..quantization.quantizers import PhaseQuantizer

        for layer in self.optical_layers:
            if isinstance(layer, DiffractiveLayer):
                layer.quantizer = PhaseQuantizer(n_bits=n_bits)
