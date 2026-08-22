"""
Detector noise model for optical neural network readout.

Models the photon shot noise and electronic readout noise inherent in
photodetector arrays (CCDs, CMOS sensors) at the output plane of a D²NN.

Physical models:

    Shot noise (Poisson):   I_shot = Poisson(I · n_photons) / n_photons
    Readout noise:          I_meas = I_shot + N(0, σ_read²)
    Dynamic range clipping: I_clipped = clamp(I_meas, 0, I_max)
"""

import torch
import torch.nn as nn


class DetectorNoise(nn.Module):
    """
    Detector noise model: shot noise + readout noise + clipping.

    This module operates on *intensity* values (|U|²), not on the
    complex field directly.

    Args:
        shot_noise_level: Mean photon count scaling factor.  Higher values
            → lower relative shot noise.  Set to 0 to disable.
        readout_sigma: Std of additive Gaussian readout noise.
        clip_max: Maximum detector saturation level. ``None`` = no clipping.
        eval_noise: If True, also inject noise during eval mode.

    Example::

        >>> det = DetectorNoise(shot_noise_level=1000, readout_sigma=0.01)
        >>> intensity = torch.rand(4, 10)  # 4 samples, 10 detector regions
        >>> noisy = det(intensity)
    """

    def __init__(
        self,
        shot_noise_level: float = 0.0,
        readout_sigma: float = 0.0,
        clip_max: float = None,
        eval_noise: bool = False,
    ):
        super().__init__()
        self.shot_noise_level = shot_noise_level
        self.readout_sigma = readout_sigma
        self.clip_max = clip_max
        self.eval_noise = eval_noise

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        """
        Apply detector noise to intensity values.

        Note: shot noise uses a Gaussian approximation of Poisson for
        differentiability (reparameterization trick).

        Args:
            intensity: Non-negative intensity tensor, any shape.

        Returns:
            Noisy intensity tensor, same shape.
        """
        if not self.training and not self.eval_noise:
            return intensity

        out = intensity.clone()

        # Shot noise — Gaussian approximation: sqrt(I/N) * N(0,1)
        if self.shot_noise_level > 0:
            # Variance of Poisson ≈ I*N, so std(I_measured) ≈ sqrt(I/N)
            shot_std = torch.sqrt(
                torch.clamp(out, min=1e-10) / self.shot_noise_level
            )
            out = out + shot_std * torch.randn_like(out.float())

        # Readout noise — additive Gaussian
        if self.readout_sigma > 0:
            out = out + self.readout_sigma * torch.randn_like(out.float())

        # Dynamic range clipping
        out = torch.clamp(out, min=0.0)
        if self.clip_max is not None:
            out = torch.clamp(out, max=self.clip_max)

        return out

    def extra_repr(self) -> str:
        return (
            f"shot_noise_level={self.shot_noise_level}, "
            f"readout_sigma={self.readout_sigma}, "
            f"clip_max={self.clip_max}"
        )
