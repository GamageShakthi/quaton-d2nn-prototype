"""
End-to-end tests for the D²NN model forward and backward passes.

Tests:
    1. Forward pass produces correct output shape (B, n_classes).
    2. Backward pass computes non-zero gradients for phase parameters.
    3. Model works with different quantization bit-widths.
    4. PTQ application changes quantizer settings.
    5. Parameter counting is correct.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from d2nn.models import D2NNClassifier


class TestD2NNForward:
    """Tests for D²NN model forward pass."""

    @pytest.fixture
    def model(self):
        """Create a small D²NN model for testing."""
        return D2NNClassifier(
            n_pixels=28,
            n_layers=3,
            n_classes=10,
            pixel_pitch=0.4e-3,
            wavelength=0.75e-3,
            z=25e-3,
        )

    @pytest.fixture
    def batch(self):
        """Create a small batch of fake MNIST images."""
        return torch.rand(4, 1, 28, 28)

    def test_output_shape(self, model, batch):
        """Forward pass should produce (B, n_classes) logits."""
        logits = model(batch)
        assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"

    def test_output_real(self, model, batch):
        """Logits should be real-valued (not complex)."""
        logits = model(batch)
        assert not logits.is_complex(), "Logits should be real, not complex"

    def test_output_nonnegative(self, model, batch):
        """Intensity-based logits should be non-negative."""
        model.eval()  # No noise
        logits = model(batch)
        assert (logits >= 0).all(), "Intensity logits should be non-negative"


class TestD2NNBackward:
    """Tests for D²NN model backward pass (gradient flow)."""

    def test_gradient_flow_continuous(self):
        """Continuous model should have non-zero gradients for phase params."""
        model = D2NNClassifier(
            n_pixels=28, n_layers=2, n_classes=10,
            pixel_pitch=0.4e-3, wavelength=0.75e-3, z=25e-3,
        )
        batch = torch.rand(2, 1, 28, 28)
        labels = torch.tensor([3, 7])

        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        # Check that at least one diffractive layer has non-zero gradient
        has_grad = False
        from d2nn.layers.diffractive_layer import DiffractiveLayer
        for module in model.modules():
            if isinstance(module, DiffractiveLayer):
                if module.phase.grad is not None and module.phase.grad.abs().sum() > 0:
                    has_grad = True
                    break

        assert has_grad, "Phase parameters should have non-zero gradients"

    def test_gradient_flow_qat(self):
        """QAT (quantized) model should still have non-zero gradients via STE."""
        model = D2NNClassifier(
            n_pixels=28, n_layers=2, n_classes=10,
            pixel_pitch=0.4e-3, wavelength=0.75e-3, z=25e-3,
            n_bits=4,  # 4-bit QAT
        )
        model.train()
        batch = torch.rand(2, 1, 28, 28)
        labels = torch.tensor([3, 7])

        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        from d2nn.layers.diffractive_layer import DiffractiveLayer
        has_grad = False
        for module in model.modules():
            if isinstance(module, DiffractiveLayer):
                if module.phase.grad is not None and module.phase.grad.abs().sum() > 0:
                    has_grad = True
                    break

        assert has_grad, "QAT phase params should have gradients via STE"


class TestD2NNConfiguration:
    """Tests for model configuration and utilities."""

    def test_parameter_count(self):
        """Optical parameter count should equal n_layers × n_pixels²."""
        n_layers = 3
        n_pixels = 28
        model = D2NNClassifier(
            n_pixels=n_pixels, n_layers=n_layers, n_classes=10,
            pixel_pitch=0.4e-3, wavelength=0.75e-3, z=25e-3,
        )
        expected = n_layers * n_pixels * n_pixels
        actual = model.count_optical_parameters()
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_ptq_application(self):
        """apply_ptq should change quantizer settings on all layers."""
        model = D2NNClassifier(
            n_pixels=28, n_layers=3, n_classes=10,
            pixel_pitch=0.4e-3, wavelength=0.75e-3, z=25e-3,
        )

        # Initially continuous
        from d2nn.layers.diffractive_layer import DiffractiveLayer
        for module in model.modules():
            if isinstance(module, DiffractiveLayer):
                assert not module.quantizer.enabled

        # Apply PTQ
        model.apply_ptq(n_bits=4)

        for module in model.modules():
            if isinstance(module, DiffractiveLayer):
                assert module.quantizer.enabled
                assert module.quantizer.n_bits == 4

    def test_get_phase_masks(self):
        """get_phase_masks should return correct number and shape."""
        n_layers = 4
        n_pixels = 28
        model = D2NNClassifier(
            n_pixels=n_pixels, n_layers=n_layers, n_classes=10,
            pixel_pitch=0.4e-3, wavelength=0.75e-3, z=25e-3,
        )
        masks = model.get_phase_masks()
        assert len(masks) == n_layers
        for mask in masks:
            assert mask.shape == (n_pixels, n_pixels)

    @pytest.mark.parametrize("n_bits", [None, 1, 2, 4, 8])
    def test_forward_all_bitwidths(self, n_bits):
        """Model should produce valid output at all bit-widths."""
        model = D2NNClassifier(
            n_pixels=28, n_layers=2, n_classes=10,
            pixel_pitch=0.4e-3, wavelength=0.75e-3, z=25e-3,
            n_bits=n_bits,
        )
        model.eval()
        batch = torch.rand(2, 1, 28, 28)
        logits = model(batch)

        assert logits.shape == (2, 10)
        assert not torch.isnan(logits).any(), f"NaN in logits for n_bits={n_bits}"
        assert not torch.isinf(logits).any(), f"Inf in logits for n_bits={n_bits}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
