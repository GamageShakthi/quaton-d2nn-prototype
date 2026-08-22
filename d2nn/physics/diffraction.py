"""
Angular Spectrum Method (ASM) for Rayleigh-Sommerfeld scalar wave propagation.

Implements the exact solution to the scalar Helmholtz equation for free-space
propagation of coherent monochromatic light between parallel planes, with
bandlimited anti-aliasing to suppress evanescent-wave artifacts.

Theory
------
Given an input complex wavefield U(x, y; z=0), the propagated field at axial
distance z is:

    U(x,y;z) = F^{-1}{ F{U(x,y;0)} · H(f_x, f_y; z) }

where the transfer function is:

    H(f_x, f_y; z) = exp(j·2π/λ · z · sqrt(1 - (λ·f_x)² - (λ·f_y)²))

for propagating waves (f_x² + f_y² ≤ 1/λ²), and H = 0 for evanescent waves.

Reference
---------
Matsushima & Shimobaba, "Band-Limited Angular Spectrum Method for Numerical
Simulation of Free-Space Propagation in Far and Near Fields,"
Optics Express 17(22), 2009.
"""

import math

import torch
import torch.nn.functional as F
from typing import Optional

from .utils import create_frequency_grid


def _build_asm_transfer_function(
    n_pixels: int,
    pixel_pitch: float,
    wavelength: float,
    z: float,
    device: torch.device = torch.device("cpu"),
    bandlimit: bool = True,
) -> torch.Tensor:
    """
    Construct the bandlimited ASM transfer function H(f_x, f_y; z).

    Args:
        n_pixels: Grid size along each axis.
        pixel_pitch: Physical pixel spacing [m].
        wavelength: Optical wavelength [m].
        z: Axial propagation distance [m].
        device: Torch device.
        bandlimit: If True, zero out evanescent (non-propagating) components.

    Returns:
        Complex tensor of shape (n_pixels, n_pixels).
    """
    FX, FY = create_frequency_grid(n_pixels, pixel_pitch, device=device)

    # Squared normalized spatial frequencies
    arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2

    # Propagating-wave mask (positive radicand)
    propagating_mask = arg >= 0.0

    # Safe sqrt — clamp negative values to zero before sqrt
    sqrt_arg = torch.sqrt(torch.clamp(arg, min=0.0))

    # Phase accumulation through free space
    phase = (2.0 * torch.pi / wavelength) * z * sqrt_arg

    # Complex transfer function
    H = torch.exp(1j * phase)

    if bandlimit:
        # Bandlimit: suppress evanescent waves
        H = H * propagating_mask

        # Additional Matsushima anti-aliasing window:
        # Limit max spatial frequency to avoid transfer-function aliasing
        delta_u = 1.0 / (n_pixels * pixel_pitch)
        f_limit_x = 1.0 / (
            wavelength * math.sqrt(
                (2.0 * delta_u * z) ** 2 + 1.0
            )
        )
        f_limit_y = f_limit_x  # Square grid ⇒ symmetric limits
        window = (torch.abs(FX) <= f_limit_x) & (torch.abs(FY) <= f_limit_y)
        H = H * window

    return H


def angular_spectrum_propagate(
    u_in: torch.Tensor,
    pixel_pitch: float,
    wavelength: float,
    z: float,
    bandlimit: bool = True,
    padding: int = 0,
    precomputed_H: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Propagate a complex wavefield through free space using the Angular
    Spectrum Method (ASM).

    Supports batched input and optional zero-padding for aperture extension.

    Args:
        u_in: Input complex wavefield. Shape (H, W) or (B, H, W) or (B, 1, H, W).
        pixel_pitch: Physical pixel spacing [m].
        wavelength: Optical wavelength [m].
        z: Axial propagation distance [m].  Positive = forward.
        bandlimit: Apply Matsushima bandlimited window (recommended).
        padding: Number of pixels to zero-pad on each side before propagation.
            Cropped after propagation to maintain original spatial extent.
        precomputed_H: Pre-built transfer function tensor.  If provided,
            skip internal construction (faster for repeated calls with the
            same geometry).

    Returns:
        Propagated complex wavefield, same shape as ``u_in``.
    """
    # ---- Normalize to (B, H, W) -------------------------------------------
    squeeze_batch = False
    if u_in.dim() == 2:
        u_in = u_in.unsqueeze(0)
        squeeze_batch = True
    elif u_in.dim() == 4:
        # (B, 1, H, W) → (B, H, W)
        u_in = u_in.squeeze(1)

    B, H, W = u_in.shape
    assert H == W, "Square grids required for current ASM implementation."

    # ---- Optional zero-padding --------------------------------------------
    if padding > 0:
        u_in = F.pad(u_in, [padding] * 4, mode="constant", value=0)
        n = H + 2 * padding
    else:
        n = H

    # ---- Transfer function ------------------------------------------------
    if precomputed_H is not None:
        H_tf = precomputed_H
    else:
        H_tf = _build_asm_transfer_function(
            n_pixels=n,
            pixel_pitch=pixel_pitch,
            wavelength=wavelength,
            z=z,
            device=u_in.device,
            bandlimit=bandlimit,
        )

    # ---- Propagate in frequency domain ------------------------------------
    U = torch.fft.fft2(u_in)
    U_prop = U * H_tf.unsqueeze(0)  # Broadcast over batch
    u_out = torch.fft.ifft2(U_prop)

    # ---- Crop padding -----------------------------------------------------
    if padding > 0:
        u_out = u_out[:, padding : padding + H, padding : padding + W]

    # ---- Restore original dimensionality ----------------------------------
    if squeeze_batch:
        u_out = u_out.squeeze(0)

    return u_out
