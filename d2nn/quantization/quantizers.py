"""
Phase quantization utilities for Spatial Light Modulators (SLMs).

Implements uniform phase quantization to simulate the finite number of
addressable phase levels available on real SLM / lithographic hardware.
Supports both post-training quantization (PTQ) and quantization-aware
training (QAT) via the Straight-Through Estimator.

For a b-bit SLM the discrete phase step is:

    Δϕ = 2π / 2^b

and quantization maps each continuous phase ϕ ∈ [0, 2π) to:

    ϕ_q = round(ϕ / Δϕ) · Δϕ
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .ste import ste_quantize


def uniform_phase_quantize(
    phase: torch.Tensor,
    n_bits: int,
    wrap: bool = True,
) -> torch.Tensor:
    """
    Quantize continuous phase values to uniform discrete levels.

    Args:
        phase: Continuous phase tensor (radians, arbitrary range).
        n_bits: Quantization bit-width. Number of discrete levels = 2^n_bits.
        wrap: If True, wrap phase into [0, 2π) before quantization.

    Returns:
        Quantized phase tensor with values on the discrete grid.
    """
    n_levels = 2 ** n_bits
    delta = 2.0 * math.pi / n_levels

    if wrap:
        phase_wrapped = torch.remainder(phase, 2.0 * math.pi)
    else:
        phase_wrapped = phase

    quantized = torch.round(phase_wrapped / delta) * delta

    # Re-wrap to [0, 2π) to handle rounding to exactly 2π
    if wrap:
        quantized = torch.remainder(quantized, 2.0 * math.pi)

    return quantized


class PhaseQuantizer(nn.Module):
    """
    Differentiable phase quantizer module with STE for QAT.

    During training (or when ``enabled=True``), quantizes phase values to
    discrete levels using the STE to allow gradient flow.  In eval mode
    (or when ``enabled=False``), applies hard quantization without STE.

    Args:
        n_bits: Bit-width for phase quantization (e.g. 1, 2, 3, 4, 8).
            Set to ``None`` or 0 to disable quantization entirely
            (continuous / full-precision mode).
        wrap: Whether to wrap phase into [0, 2π) before quantizing.

    Attributes:
        n_levels (int): Number of discrete phase levels (2^n_bits).
        delta (float): Phase step size (2π / n_levels).

    Example::

        >>> q = PhaseQuantizer(n_bits=2)
        >>> phi = torch.tensor([0.0, 1.0, 3.14, 5.5], requires_grad=True)
        >>> phi_q = q(phi)
        >>> phi_q.sum().backward()
        >>> phi.grad  # non-zero thanks to STE
    """

    def __init__(self, n_bits: Optional[int] = None, wrap: bool = True):
        super().__init__()
        self.n_bits = n_bits
        self.wrap = wrap

        if n_bits is not None and n_bits > 0:
            self.n_levels = 2 ** n_bits
            self.delta = 2.0 * math.pi / self.n_levels
            self.enabled = True
        else:
            self.n_levels = None
            self.delta = None
            self.enabled = False

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Apply phase quantization.

        Args:
            phase: Continuous phase tensor.

        Returns:
            Quantized phase (with STE in training mode), or continuous
            phase if quantization is disabled.
        """
        if not self.enabled:
            return phase

        phase_q = uniform_phase_quantize(phase, self.n_bits, wrap=self.wrap)

        if self.training:
            # QAT: use STE for differentiable quantization
            return ste_quantize(phase, phase_q.detach())
        else:
            # PTQ / inference: hard quantization
            return phase_q

    def extra_repr(self) -> str:
        if self.enabled:
            return f"n_bits={self.n_bits}, levels={self.n_levels}, delta={self.delta:.4f}"
        return "disabled (continuous)"
