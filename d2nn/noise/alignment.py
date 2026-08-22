"""
Alignment jitter noise model for diffractive optical layers.

Models the lateral (translational) misalignment that occurs during physical
assembly of multi-layer D²NN systems.  Each diffractive layer may be
shifted by a small random displacement in x and y, simulated via a
differentiable affine grid sampler so that gradients can flow through
the noise injection during robust training.

Physical model:

    r' = r + δ_trans,    δ_trans ~ N(0, σ_jitter²)

where σ_jitter is expressed as a fraction of the pixel pitch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentJitter(nn.Module):
    """
    Differentiable lateral alignment jitter via affine grid sampling.

    During training, applies a random sub-pixel translational shift to
    the 2D complex wavefield.  During evaluation, no jitter is applied
    (deterministic forward pass) unless ``eval_jitter=True``.

    Args:
        sigma: Standard deviation of jitter in *pixels* (e.g. 0.5 means
            the jitter std is half a pixel pitch).
        eval_jitter: If True, also apply jitter during eval mode
            (useful for Monte-Carlo robustness evaluation).

    Example::

        >>> jitter = AlignmentJitter(sigma=1.0)
        >>> u = torch.randn(4, 1, 64, 64) + 1j * torch.randn(4, 1, 64, 64)
        >>> u_shifted = jitter(u)
        >>> u_shifted.shape  # (4, 1, 64, 64)
    """

    def __init__(self, sigma: float = 0.5, eval_jitter: bool = False):
        super().__init__()
        self.sigma = sigma
        self.eval_jitter = eval_jitter

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Apply alignment jitter to a complex wavefield.

        Args:
            u: Complex tensor of shape (B, H, W) or (B, 1, H, W).

        Returns:
            Shifted complex wavefield, same shape as input.
        """
        if not self.training and not self.eval_jitter:
            return u

        # Ensure 4D for grid_sample: (B, C, H, W)
        needs_squeeze = False
        if u.dim() == 3:
            u = u.unsqueeze(1)
            needs_squeeze = True

        B, C, H, W = u.shape

        # Random translational shift in normalised grid coordinates
        # grid_sample uses [-1, 1] coordinates, so 1 pixel = 2/N
        pixel_to_norm = 2.0 / H
        shift = torch.randn(B, 2, device=u.device, dtype=torch.float32) * self.sigma * pixel_to_norm

        # Build affine matrix: identity + translation
        theta = torch.eye(2, 3, device=u.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).clone()
        theta[:, 0, 2] = shift[:, 0]  # x-shift
        theta[:, 1, 2] = shift[:, 1]  # y-shift

        grid = F.affine_grid(theta, [B, C, H, W], align_corners=False)

        # grid_sample requires real input — process real & imag separately
        u_real = F.grid_sample(u.real.float(), grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)
        u_imag = F.grid_sample(u.imag.float(), grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)
        u_out = torch.complex(u_real, u_imag)

        if needs_squeeze:
            u_out = u_out.squeeze(1)

        return u_out

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, eval_jitter={self.eval_jitter}"
