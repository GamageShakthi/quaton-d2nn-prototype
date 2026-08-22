"""
Unit tests for noise injection modules (alignment, fabrication, detector).

Tests:
    1. AlignmentJitter modifies the wavefield during training.
    2. FabricationNoise adds phase errors and dead pixels.
    3. DetectorNoise injects shot and readout noise.
    4. All noise modules are disabled in eval mode (unless eval_noise=True).
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from d2nn.noise.alignment import AlignmentJitter
from d2nn.noise.fabrication import FabricationNoise
from d2nn.noise.detector import DetectorNoise


class TestAlignmentJitter:
    """Tests for alignment jitter noise model."""

    def test_train_mode_modifies_field(self):
        """In training mode, jitter should modify the wavefield."""
        torch.manual_seed(42)
        jitter = AlignmentJitter(sigma=2.0)
        jitter.train()

        u = torch.ones(2, 1, 32, 32, dtype=torch.cfloat)
        u_shifted = jitter(u)

        # The field should differ from the original due to shifts
        # (at edges especially due to zero-padding)
        diff = torch.abs(u.squeeze(1) - u_shifted).sum().item()
        assert diff > 0, "Jitter should modify the wavefield in train mode"

    def test_eval_mode_passthrough(self):
        """In eval mode (default), jitter should not modify the field."""
        jitter = AlignmentJitter(sigma=2.0, eval_jitter=False)
        jitter.eval()

        u = torch.randn(2, 32, 32) + 1j * torch.randn(2, 32, 32)
        u_out = jitter(u)

        np.testing.assert_array_equal(
            u.numpy(), u_out.numpy(),
            err_msg="Jitter should be identity in eval mode"
        )

    def test_eval_jitter_enabled(self):
        """eval_jitter=True should apply noise even in eval mode."""
        torch.manual_seed(42)
        jitter = AlignmentJitter(sigma=2.0, eval_jitter=True)
        jitter.eval()

        u = torch.ones(2, 1, 32, 32, dtype=torch.cfloat)
        u_shifted = jitter(u)

        diff = torch.abs(u.squeeze(1) - u_shifted).sum().item()
        assert diff > 0, "eval_jitter=True should apply noise in eval mode"

    def test_output_shape(self):
        """Output shape must match input shape."""
        jitter = AlignmentJitter(sigma=1.0)
        jitter.train()

        u3d = torch.randn(4, 32, 32) + 1j * torch.randn(4, 32, 32)
        assert jitter(u3d).shape == (4, 32, 32)


class TestFabricationNoise:
    """Tests for fabrication noise model."""

    def test_phase_noise_injection(self):
        """Phase noise should modify the phase values in training mode."""
        torch.manual_seed(42)
        fab = FabricationNoise(phase_sigma=0.5)
        fab.train()

        phase = torch.ones(4, 32, 32) * 1.0
        noisy = fab(phase)

        diff = (noisy - phase).abs().sum().item()
        assert diff > 0, "Fabrication noise should modify phase in train mode"

    def test_eval_passthrough(self):
        """Fabrication noise should be disabled in eval mode (default)."""
        fab = FabricationNoise(phase_sigma=0.5)
        fab.eval()

        phase = torch.ones(4, 32, 32)
        noisy = fab(phase)

        np.testing.assert_array_equal(phase.numpy(), noisy.numpy())

    def test_dead_pixels(self):
        """Dead pixel mask should zero out some phase values."""
        torch.manual_seed(42)
        fab = FabricationNoise(phase_sigma=0.0, dead_pixel_prob=0.5)
        fab.train()

        phase = torch.ones(1, 64, 64)
        noisy = fab(phase)

        # About 50% of pixels should be zeroed
        n_dead = (noisy == 0).sum().item()
        total = phase.numel()
        ratio = n_dead / total

        assert 0.3 < ratio < 0.7, \
            f"Dead pixel ratio {ratio:.2f} outside expected range [0.3, 0.7]"


class TestDetectorNoise:
    """Tests for detector noise model."""

    def test_shot_noise(self):
        """Shot noise should modify intensity values."""
        torch.manual_seed(42)
        det = DetectorNoise(shot_noise_level=100)
        det.train()

        intensity = torch.ones(4, 10) * 0.5
        noisy = det(intensity)

        diff = (noisy - intensity).abs().sum().item()
        assert diff > 0, "Shot noise should modify intensity"

    def test_readout_noise(self):
        """Readout noise should be additive Gaussian."""
        torch.manual_seed(42)
        det = DetectorNoise(readout_sigma=0.1)
        det.train()

        intensity = torch.ones(4, 10)
        noisy = det(intensity)

        diff = (noisy - intensity).abs().sum().item()
        assert diff > 0, "Readout noise should modify intensity"

    def test_non_negative(self):
        """Output intensity should be non-negative after clamping."""
        torch.manual_seed(42)
        det = DetectorNoise(readout_sigma=10.0)  # Very high noise
        det.train()

        intensity = torch.ones(4, 10) * 0.01
        noisy = det(intensity)

        assert (noisy >= 0).all(), "Intensity should be non-negative"

    def test_eval_passthrough(self):
        """Detector noise disabled in eval mode by default."""
        det = DetectorNoise(shot_noise_level=1000, readout_sigma=0.5)
        det.eval()

        intensity = torch.ones(4, 10)
        noisy = det(intensity)

        np.testing.assert_array_equal(intensity.numpy(), noisy.numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
