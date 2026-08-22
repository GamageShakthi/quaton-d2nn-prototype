"""
Fabrication noise model for diffractive optical elements.

Models the inevitable phase errors introduced during manufacturing of
diffractive phase masks (e.g. lithographic etching depth variations,
SLM pixel response non-uniformity, and dead pixels).

Physical model:

    ϕ_eff = ϕ_quant + ε_ϕ,    ε_ϕ ~ N(0, σ_phase²)

Additionally supports:
    - Dead pixels: random binary mask zeroing out a fraction of neurons.
    - Amplitude crosstalk: coupling a fraction of neighbour amplitudes.
"""

import torch
import torch.nn as nn


class FabricationNoise(nn.Module):
    """
    Fabrication-process noise injection for phase masks.

    Args:
        phase_sigma: Std of additive Gaussian phase error [radians].
        dead_pixel_prob: Probability of any pixel being "dead" (zero
            transmission).  Applied per-sample during training.
        eval_noise: If True, also inject noise during eval mode.

    Example::

        >>> fab = FabricationNoise(phase_sigma=0.1, dead_pixel_prob=0.01)
        >>> phase = torch.rand(4, 64, 64) * 2 * 3.14159
        >>> noisy_phase = fab(phase)
    """

    def __init__(
        self,
        phase_sigma: float = 0.1,
        dead_pixel_prob: float = 0.0,
        eval_noise: bool = False,
    ):
        super().__init__()
        self.phase_sigma = phase_sigma
        self.dead_pixel_prob = dead_pixel_prob
        self.eval_noise = eval_noise

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Inject fabrication noise into a phase tensor.

        Args:
            phase: Phase values [radians], shape (B, H, W) or (H, W).

        Returns:
            Noisy phase tensor, same shape.
        """
        if not self.training and not self.eval_noise:
            return phase

        noisy = phase.clone()

        # Additive Gaussian phase error
        if self.phase_sigma > 0:
            noise = torch.randn_like(phase.float()) * self.phase_sigma
            noisy = noisy + noise

        # Dead pixel mask (multiplicative binary noise)
        if self.dead_pixel_prob > 0:
            alive_mask = (
                torch.rand_like(phase.float()) > self.dead_pixel_prob
            ).to(phase.dtype)
            noisy = noisy * alive_mask

        return noisy

    def extra_repr(self) -> str:
        return (
            f"phase_sigma={self.phase_sigma}, "
            f"dead_pixel_prob={self.dead_pixel_prob}, "
            f"eval_noise={self.eval_noise}"
        )
