"""
Free-space propagation layer wrapping the ASM diffraction kernel.

This module encapsulates the Angular Spectrum Method propagation as a
torch.nn.Module so it can be stacked with DiffractiveLayer and DetectorLayer
in a Sequential-like architecture.  The transfer function is pre-computed
once and cached for efficiency.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..physics.diffraction import angular_spectrum_propagate, _build_asm_transfer_function


class PropagationLayer(nn.Module):
    """
    Free-space optical propagation between diffractive layers.

    Pre-computes the ASM transfer function at construction time and
    reuses it for all forward calls (geometry is fixed).

    Args:
        n_pixels: Spatial grid size.
        pixel_pitch: Physical pixel spacing [m].
        wavelength: Optical wavelength [m].
        z: Propagation distance [m].
        bandlimit: Apply Matsushima bandlimited window.
        padding: Zero-padding pixels on each side.

    Attributes:
        H (torch.Tensor): Cached transfer function (registered as buffer).
    """

    def __init__(
        self,
        n_pixels: int = 200,
        pixel_pitch: float = 0.4e-3,
        wavelength: float = 0.75e-3,
        z: float = 25e-3,
        bandlimit: bool = True,
        padding: int = 0,
    ):
        super().__init__()
        self.n_pixels = n_pixels
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.z = z
        self.bandlimit = bandlimit
        self.padding = padding

        # Pre-compute and register transfer function as a buffer (non-learnable)
        n_padded = n_pixels + 2 * padding if padding > 0 else n_pixels
        H = _build_asm_transfer_function(
            n_pixels=n_padded,
            pixel_pitch=pixel_pitch,
            wavelength=wavelength,
            z=z,
            bandlimit=bandlimit,
        )
        self.register_buffer("H", H)

    def forward(self, u_in: torch.Tensor) -> torch.Tensor:
        """
        Propagate complex wavefield through free space.

        Args:
            u_in: Complex wavefield, shape (B, H, W).

        Returns:
            Propagated wavefield, shape (B, H, W).
        """
        return angular_spectrum_propagate(
            u_in,
            pixel_pitch=self.pixel_pitch,
            wavelength=self.wavelength,
            z=self.z,
            bandlimit=self.bandlimit,
            padding=self.padding,
            precomputed_H=self.H,
        )

    def extra_repr(self) -> str:
        return (
            f"n_pixels={self.n_pixels}, pixel_pitch={self.pixel_pitch:.2e}, "
            f"wavelength={self.wavelength:.2e}, z={self.z:.2e}"
        )
