"""
Unit tests for Angular Spectrum Method (ASM) diffraction propagation.

Tests:
    1. Energy conservation: ||U_z||² ≈ ||U_0||² for free-space propagation.
    2. Phase consistency: Plane wave acquires correct axial phase shift.
    3. Output shape correctness for batched inputs.
    4. Bandlimited transfer function zeroes evanescent waves.
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from d2nn.physics.diffraction import angular_spectrum_propagate, _build_asm_transfer_function
from d2nn.physics.utils import create_spatial_grid, create_frequency_grid, validate_sampling


class TestASMPropagation:
    """Tests for the Angular Spectrum Method propagation kernel."""

    # Standard optical parameters
    N = 64
    PIXEL_PITCH = 0.4e-3  # 0.4 mm
    WAVELENGTH = 0.75e-3  # 0.75 mm (THz regime)
    Z = 25e-3  # 25 mm

    def test_energy_conservation(self):
        """Total energy (integrated intensity) must be conserved in free space."""
        # Create a random complex input field
        u_in = torch.randn(self.N, self.N) + 1j * torch.randn(self.N, self.N)
        energy_in = torch.sum(torch.abs(u_in) ** 2).item()

        # Use bandlimit=False for exact energy conservation test,
        # since the Matsushima window intentionally discards frequency content
        u_out = angular_spectrum_propagate(
            u_in,
            pixel_pitch=self.PIXEL_PITCH,
            wavelength=self.WAVELENGTH,
            z=self.Z,
            bandlimit=False,
        )
        energy_out = torch.sum(torch.abs(u_out) ** 2).item()

        # Allow small numerical error from FFT round-trip
        np.testing.assert_allclose(energy_in, energy_out, rtol=1e-4,
                                   err_msg="Energy not conserved in ASM propagation")

    def test_plane_wave_phase_shift(self):
        """A uniform plane wave should acquire phase exp(j·2π·z/λ)."""
        u_in = torch.ones(self.N, self.N, dtype=torch.cfloat)

        u_out = angular_spectrum_propagate(
            u_in,
            pixel_pitch=self.PIXEL_PITCH,
            wavelength=self.WAVELENGTH,
            z=self.Z,
            bandlimit=False,  # Disable bandlimit for clean plane wave
        )

        # Expected phase shift for on-axis plane wave
        expected_phase = 2.0 * np.pi * self.Z / self.WAVELENGTH
        expected_phase_wrapped = expected_phase % (2 * np.pi)

        # Check center pixel phase
        center = self.N // 2
        actual_phase = torch.angle(u_out[center, center]).item() % (2 * np.pi)

        np.testing.assert_allclose(
            actual_phase, expected_phase_wrapped, atol=0.1,
            err_msg="Plane wave phase shift incorrect"
        )

    def test_batch_dimension(self):
        """Output shape must match input shape for batched input."""
        B = 4
        u_in = torch.randn(B, self.N, self.N) + 1j * torch.randn(B, self.N, self.N)

        u_out = angular_spectrum_propagate(
            u_in,
            pixel_pitch=self.PIXEL_PITCH,
            wavelength=self.WAVELENGTH,
            z=self.Z,
        )

        assert u_out.shape == (B, self.N, self.N), \
            f"Expected shape {(B, self.N, self.N)}, got {u_out.shape}"

    def test_4d_input(self):
        """(B, 1, H, W) input should also work and return (B, H, W)."""
        B = 2
        u_in = torch.randn(B, 1, self.N, self.N) + 1j * torch.randn(B, 1, self.N, self.N)

        u_out = angular_spectrum_propagate(
            u_in,
            pixel_pitch=self.PIXEL_PITCH,
            wavelength=self.WAVELENGTH,
            z=self.Z,
        )

        assert u_out.shape == (B, self.N, self.N)

    def test_evanescent_suppression(self):
        """Bandlimited TF should zero out evanescent spatial frequencies."""
        H = _build_asm_transfer_function(
            n_pixels=self.N,
            pixel_pitch=self.PIXEL_PITCH,
            wavelength=self.WAVELENGTH,
            z=self.Z,
            bandlimit=True,
        )

        FX, FY = create_frequency_grid(self.N, self.PIXEL_PITCH)
        evanescent_mask = (FX ** 2 + FY ** 2) > (1.0 / self.WAVELENGTH) ** 2

        # All evanescent components should be zero
        evanescent_energy = torch.abs(H[evanescent_mask]).sum().item()
        assert evanescent_energy == 0.0, \
            f"Evanescent energy should be 0, got {evanescent_energy}"

    def test_padding(self):
        """Padded propagation should return original spatial dimensions."""
        u_in = torch.randn(self.N, self.N) + 1j * torch.randn(self.N, self.N)

        u_out = angular_spectrum_propagate(
            u_in,
            pixel_pitch=self.PIXEL_PITCH,
            wavelength=self.WAVELENGTH,
            z=self.Z,
            padding=16,
        )

        assert u_out.shape == (self.N, self.N), \
            f"Padding should not change output shape. Got {u_out.shape}"


class TestGridUtilities:
    """Tests for spatial and frequency grid construction."""

    def test_spatial_grid_centering(self):
        """Spatial grid should be centered at (0, 0)."""
        N = 32
        pitch = 1e-3
        X, Y = create_spatial_grid(N, pitch)

        # Mean should be close to 0 (off by half a pixel pitch)
        assert abs(X.mean().item()) < pitch, "X grid not centered"
        assert abs(Y.mean().item()) < pitch, "Y grid not centered"

    def test_frequency_grid_nyquist(self):
        """Maximum frequency should be ±1/(2·pitch) (Nyquist)."""
        N = 64
        pitch = 0.5e-3
        FX, FY = create_frequency_grid(N, pitch)

        expected_max = 1.0 / (2.0 * pitch)
        actual_max = FX.abs().max().item()

        # FFT frequencies go up to Nyquist
        np.testing.assert_allclose(actual_max, expected_max, rtol=0.05)

    def test_validate_sampling(self):
        """Sampling validator should return correct keys."""
        result = validate_sampling(
            n_pixels=64, pixel_pitch=0.4e-3,
            wavelength=0.75e-3, propagation_distance=25e-3,
        )
        assert "fresnel_number" in result
        assert "sampling_valid" in result
        assert isinstance(result["fresnel_number"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
