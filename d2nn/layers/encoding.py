"""
Input encoding layers for converting images to optical wavefields.

Provides two standard encoding schemes used in D²NN literature:

1. **Amplitude encoding**: The image pixel intensity directly modulates
   the amplitude of a uniform coherent wavefront.  Phase is zero.

2. **Phase encoding**: The image pixel intensity is mapped to a phase
   value (0 → 2π), giving a phase-only object with uniform amplitude.
   This is common for SLM-based optical processors.
"""

import torch
import torch.nn as nn
import math


class AmplitudeEncoder(nn.Module):
    """
    Encode a real-valued image as the amplitude of a complex wavefield.

    The output field is:  U = I · exp(j·0) = I + 0j

    where I ∈ [0, 1] is the normalised image intensity.

    Args:
        normalize: If True, normalize input to [0, 1] range.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to complex wavefield.

        Args:
            x: Real image tensor, shape (B, 1, H, W) or (B, H, W).

        Returns:
            Complex tensor of shape (B, H, W).
        """
        if x.dim() == 4:
            x = x.squeeze(1)

        if self.normalize:
            # Per-sample min-max normalization
            B = x.shape[0]
            x_flat = x.view(B, -1)
            x_min = x_flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
            x_max = x_flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
            x = (x - x_min) / (x_max - x_min + 1e-8)

        return torch.complex(x.float(), torch.zeros_like(x.float()))


class PhaseEncoder(nn.Module):
    """
    Encode a real-valued image as the phase of a complex wavefield.

    The output field is:  U = exp(j · I · 2π)

    where I ∈ [0, 1] is the normalised image intensity.

    Args:
        phase_range: Maximum phase modulation [radians].  Default 2π.
        normalize: If True, normalize input to [0, 1] range.
    """

    def __init__(self, phase_range: float = 2.0 * math.pi, normalize: bool = True):
        super().__init__()
        self.phase_range = phase_range
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to complex wavefield via phase modulation.

        Args:
            x: Real image tensor, shape (B, 1, H, W) or (B, H, W).

        Returns:
            Complex tensor of shape (B, H, W) with unit amplitude.
        """
        if x.dim() == 4:
            x = x.squeeze(1)

        if self.normalize:
            B = x.shape[0]
            x_flat = x.view(B, -1)
            x_min = x_flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
            x_max = x_flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
            x = (x - x_min) / (x_max - x_min + 1e-8)

        phase = x.float() * self.phase_range
        return torch.exp(1j * phase)
