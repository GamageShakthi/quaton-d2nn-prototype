"""
Diffractive phase-mask layer — the "neuron" of a Diffractive Optical NN.

Each D²NN layer is a thin transmissive phase mask whose pixel-wise phase
modulation values are the trainable parameters.  The incoming complex
wavefield is multiplied element-wise by exp(jϕ), where ϕ is a learnable
2D phase profile.

This module integrates:
    - Learnable phase parameters (randomly initialized).
    - Optional phase quantization via PhaseQuantizer (QAT / PTQ).
    - Optional fabrication noise injection via FabricationNoise.
    - Optional alignment jitter via AlignmentJitter.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from ..quantization.quantizers import PhaseQuantizer
from ..noise.fabrication import FabricationNoise
from ..noise.alignment import AlignmentJitter


class DiffractiveLayer(nn.Module):
    """
    A single diffractive phase-mask layer (optical neuron array).

    The layer applies the transformation:

        U_out(x,y) = U_in(x,y) · exp(j · ϕ(x,y))

    where ϕ is optionally quantized and corrupted by fabrication noise.

    Args:
        n_pixels: Spatial resolution of the mask (n_pixels × n_pixels).
        n_bits: Phase quantization bit-width.  ``None`` → continuous.
        phase_sigma: Fabrication phase noise std [rad].
        dead_pixel_prob: Probability of dead pixels in the mask.
        jitter_sigma: Alignment jitter std [pixels].
        init_mode: Phase initialization — ``"uniform"`` (random [0, 2π)),
            ``"zeros"`` (all zero), or ``"xavier"`` (scaled normal).

    Attributes:
        phase (nn.Parameter): Learnable phase profile, shape (n_pixels, n_pixels).
    """

    def __init__(
        self,
        n_pixels: int = 200,
        n_bits: Optional[int] = None,
        phase_sigma: float = 0.0,
        dead_pixel_prob: float = 0.0,
        jitter_sigma: float = 0.0,
        init_mode: str = "uniform",
    ):
        super().__init__()
        self.n_pixels = n_pixels

        # Learnable phase parameters
        if init_mode == "uniform":
            init_phase = torch.rand(n_pixels, n_pixels) * 2 * math.pi
        elif init_mode == "zeros":
            init_phase = torch.zeros(n_pixels, n_pixels)
        elif init_mode == "xavier":
            init_phase = torch.randn(n_pixels, n_pixels) * math.sqrt(2.0 / n_pixels)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        self.phase = nn.Parameter(init_phase)

        # Quantization
        self.quantizer = PhaseQuantizer(n_bits=n_bits)

        # Fabrication noise
        self.fab_noise = FabricationNoise(
            phase_sigma=phase_sigma,
            dead_pixel_prob=dead_pixel_prob,
        )

        # Alignment jitter
        self.jitter = AlignmentJitter(sigma=jitter_sigma) if jitter_sigma > 0 else None

    def get_effective_phase(self) -> torch.Tensor:
        """
        Return the current effective (possibly quantized) phase profile.

        Returns:
            Phase tensor of shape (n_pixels, n_pixels).
        """
        return self.quantizer(self.phase)

    def forward(self, u_in: torch.Tensor) -> torch.Tensor:
        """
        Modulate incoming wavefield by the phase mask.

        Args:
            u_in: Complex input wavefield, shape (B, H, W).

        Returns:
            Complex output wavefield, shape (B, H, W).
        """
        # 1. Quantize phase (QAT or continuous)
        phi = self.quantizer(self.phase)

        # 2. Inject fabrication noise
        phi = self.fab_noise(phi)

        # 3. Compute complex transmission
        transmission = torch.exp(1j * phi)

        # 4. Modulate wavefield
        u_out = u_in * transmission.unsqueeze(0)  # Broadcast over batch

        # 5. Apply alignment jitter
        if self.jitter is not None:
            u_out = self.jitter(u_out)

        return u_out

    def extra_repr(self) -> str:
        return (
            f"n_pixels={self.n_pixels}, "
            f"quantizer={self.quantizer}, "
            f"jitter={'enabled' if self.jitter else 'disabled'}"
        )
