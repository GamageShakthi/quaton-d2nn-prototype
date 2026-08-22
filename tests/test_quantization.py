"""
Unit tests for phase quantization and Straight-Through Estimator (STE).

Tests:
    1. STE allows gradient flow through quantization.
    2. Quantized values are on the correct discrete grid.
    3. PhaseQuantizer produces correct number of levels.
    4. QAT mode uses STE; eval mode uses hard quantization.
"""

import sys
import os
import pytest
import torch
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from d2nn.quantization.ste import StraightThroughEstimator, ste_quantize
from d2nn.quantization.quantizers import uniform_phase_quantize, PhaseQuantizer


class TestSTE:
    """Tests for the Straight-Through Estimator."""

    def test_gradient_flow(self):
        """Gradients must flow through STE (non-zero gradient for continuous input)."""
        x = torch.tensor([0.3, 1.7, 2.5, 4.1], requires_grad=True)
        x_q = torch.round(x)

        y = ste_quantize(x, x_q.detach())
        loss = y.sum()
        loss.backward()

        # Gradient should be all ones (STE identity backward)
        np.testing.assert_array_equal(
            x.grad.numpy(), np.ones(4),
            err_msg="STE gradient should be identity (all ones)"
        )

    def test_forward_returns_quantized(self):
        """Forward pass should return the quantized values, not continuous."""
        x = torch.tensor([0.3, 1.7, 2.5])
        x_q = torch.round(x)

        y = ste_quantize(x, x_q.detach())

        np.testing.assert_array_equal(
            y.detach().numpy(), x_q.numpy(),
            err_msg="STE forward should return quantized values"
        )

    def test_ste_module(self):
        """StraightThroughEstimator module should work with custom quantize_fn."""
        quant_fn = lambda x: torch.round(x)
        ste = StraightThroughEstimator(quant_fn)

        x = torch.tensor([0.3, 1.7, 2.5], requires_grad=True)
        y = ste(x)
        y.sum().backward()

        assert x.grad is not None, "Gradient should exist"
        np.testing.assert_array_equal(x.grad.numpy(), np.ones(3))


class TestPhaseQuantization:
    """Tests for uniform phase quantization."""

    @pytest.mark.parametrize("n_bits", [1, 2, 3, 4, 8])
    def test_quantized_levels(self, n_bits):
        """Quantized phases should be on the discrete grid for each bit-width."""
        phase = torch.linspace(0, 2 * math.pi - 0.01, 100)
        q_phase = uniform_phase_quantize(phase, n_bits)

        n_levels = 2 ** n_bits
        delta = 2 * math.pi / n_levels

        # All quantized values should be multiples of delta (mod 2π)
        residual = torch.remainder(q_phase / delta, 1.0)
        # residual should be very close to 0 or 1
        near_grid = (residual < 0.01) | (residual > 0.99)
        assert near_grid.all(), \
            f"Not all values on {n_bits}-bit grid. Residuals: {residual}"

    @pytest.mark.parametrize("n_bits", [1, 2, 3, 4])
    def test_unique_levels_count(self, n_bits):
        """Number of unique quantized values should equal 2^n_bits."""
        phase = torch.linspace(0, 2 * math.pi - 0.001, 1000)
        q_phase = uniform_phase_quantize(phase, n_bits)

        n_unique = len(torch.unique(q_phase))
        expected = 2 ** n_bits

        assert n_unique == expected, \
            f"Expected {expected} unique levels for {n_bits}-bit, got {n_unique}"

    def test_wrap_to_2pi(self):
        """Quantized phases should be in [0, 2π) after wrapping."""
        phase = torch.tensor([-1.0, 7.0, 13.0, -5.0])
        q_phase = uniform_phase_quantize(phase, n_bits=4, wrap=True)

        assert (q_phase >= 0).all() and (q_phase < 2 * math.pi + 0.01).all(), \
            f"Phases not wrapped to [0, 2π): {q_phase}"


class TestPhaseQuantizerModule:
    """Tests for the PhaseQuantizer nn.Module."""

    def test_qat_gradient_flow(self):
        """In training mode, PhaseQuantizer should allow gradient flow via STE."""
        q = PhaseQuantizer(n_bits=4)
        q.train()

        phi = torch.tensor([0.5, 1.5, 3.0, 5.0], requires_grad=True)
        phi_q = q(phi)
        phi_q.sum().backward()

        assert phi.grad is not None, "Gradient should flow in QAT mode"
        assert (phi.grad != 0).any(), "Gradient should be non-zero"

    def test_eval_hard_quantization(self):
        """In eval mode, PhaseQuantizer should apply hard quantization."""
        q = PhaseQuantizer(n_bits=2)
        q.eval()

        phi = torch.tensor([0.5, 2.0, 4.0, 5.5])
        phi_q = q(phi)

        # Should be on 2-bit grid (4 levels, delta = π/2)
        delta = 2 * math.pi / 4
        residual = torch.remainder(phi_q / delta, 1.0)
        near_grid = (residual < 0.01) | (residual > 0.99)
        assert near_grid.all()

    def test_disabled_passthrough(self):
        """When n_bits=None, PhaseQuantizer should pass through unchanged."""
        q = PhaseQuantizer(n_bits=None)
        phi = torch.tensor([0.5, 1.5, 3.0])
        phi_out = q(phi)

        np.testing.assert_array_equal(phi.numpy(), phi_out.numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
