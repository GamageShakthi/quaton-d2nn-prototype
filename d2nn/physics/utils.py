"""
Optical utility functions for spatial and frequency domain grids.

Provides helper routines for constructing coordinate grids, frequency grids,
and validating the Nyquist sampling criterion for the Angular Spectrum Method.
"""

import torch
import numpy as np
from typing import Tuple


def create_spatial_grid(
    n_pixels: int,
    pixel_pitch: float,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a 2D spatial coordinate grid centered at origin.

    Args:
        n_pixels: Number of pixels along each spatial axis.
        pixel_pitch: Physical spacing between adjacent pixels [meters].
        device: Torch device.

    Returns:
        (X, Y): Meshgrid tensors of shape (n_pixels, n_pixels) in meters.
    """
    extent = n_pixels * pixel_pitch
    coords = torch.linspace(
        -extent / 2, extent / 2 - pixel_pitch, n_pixels, device=device
    )
    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    return X, Y


def create_frequency_grid(
    n_pixels: int,
    pixel_pitch: float,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a 2D spatial-frequency grid for the FFT domain.

    The grid values are spatial frequencies in cycles/meter.

    Args:
        n_pixels: Number of pixels along each axis.
        pixel_pitch: Physical pixel spacing [meters].
        device: Torch device.

    Returns:
        (FX, FY): Meshgrid tensors of shape (n_pixels, n_pixels) in [1/m].
    """
    freq = torch.fft.fftfreq(n_pixels, d=pixel_pitch, device=device)
    FY, FX = torch.meshgrid(freq, freq, indexing="ij")
    return FX, FY


def validate_sampling(
    n_pixels: int,
    pixel_pitch: float,
    wavelength: float,
    propagation_distance: float,
) -> dict:
    """
    Validate Nyquist-Shannon sampling criteria for ASM propagation.

    Checks whether the spatial sampling is sufficient to avoid aliasing
    in the transfer function and computes key optical parameters.

    Args:
        n_pixels: Number of pixels along each axis.
        pixel_pitch: Physical pixel spacing [meters].
        wavelength: Optical wavelength [meters].
        propagation_distance: Axial propagation distance [meters].

    Returns:
        Dictionary with Fresnel number, max spatial frequency, critical
        sampling pitch, and a boolean indicating if sampling is adequate.
    """
    aperture = n_pixels * pixel_pitch
    max_freq = 1.0 / (2.0 * pixel_pitch)  # Nyquist frequency
    critical_pitch = wavelength / 2.0  # Minimum pitch to capture all propagating waves

    # Fresnel number: N_F = a^2 / (lambda * z)
    fresnel_number = (aperture / 2.0) ** 2 / (wavelength * propagation_distance)

    # Maximum spatial frequency supported by the propagation kernel
    max_propagating_freq = 1.0 / wavelength

    is_valid = pixel_pitch <= critical_pitch or max_freq >= max_propagating_freq

    return {
        "fresnel_number": fresnel_number,
        "max_nyquist_freq": max_freq,
        "max_propagating_freq": max_propagating_freq,
        "critical_pitch": critical_pitch,
        "aperture": aperture,
        "sampling_valid": is_valid,
    }
