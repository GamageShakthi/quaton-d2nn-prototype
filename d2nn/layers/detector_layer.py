"""
Detector layer for the output plane of a Diffractive Optical Neural Network.

Divides the output intensity pattern into spatial regions, one per class,
and integrates the optical power in each region to produce classification
logits.  This mirrors the physical setup where photodetectors are placed
at designated locations on the output plane.

Detector region layout:
    For C classes on an N×N grid, the detector plane is partitioned into
    C non-overlapping rectangular regions.  The integrated intensity in
    each region serves as the logit for the corresponding class.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from ..noise.detector import DetectorNoise


class DetectorLayer(nn.Module):
    """
    Spatial-region intensity detector for classification.

    Partitions the output plane into class-specific regions and integrates
    optical intensity (|U|²) to produce logits.

    Args:
        n_pixels: Output plane grid size.
        n_classes: Number of classification classes (detector regions).
        detector_size: Side length of each square detector region [pixels].
            If None, automatically computed from grid size and n_classes.
        shot_noise_level: Shot noise parameter for detector.
        readout_sigma: Readout noise std.

    Example::

        >>> det = DetectorLayer(n_pixels=200, n_classes=10)
        >>> u = torch.randn(4, 200, 200, dtype=torch.cfloat)
        >>> logits = det(u)
        >>> logits.shape  # (4, 10)
    """

    def __init__(
        self,
        n_pixels: int = 200,
        n_classes: int = 10,
        detector_size: Optional[int] = None,
        shot_noise_level: float = 0.0,
        readout_sigma: float = 0.0,
    ):
        super().__init__()
        self.n_pixels = n_pixels
        self.n_classes = n_classes

        # Compute detector region size and positions
        if detector_size is None:
            # Place detectors in a row or grid arrangement
            detector_size = max(4, n_pixels // (n_classes + 2))
        self.detector_size = detector_size

        # Generate detector center positions
        # Arrange detectors evenly around the center of the output plane
        centers = self._compute_detector_centers(n_pixels, n_classes, detector_size)
        # Register as buffer: (n_classes, 4) → [row_start, row_end, col_start, col_end]
        self.register_buffer("detector_regions", centers)

        # Detector noise
        self.det_noise = DetectorNoise(
            shot_noise_level=shot_noise_level,
            readout_sigma=readout_sigma,
        )

    @staticmethod
    def _compute_detector_centers(
        n_pixels: int, n_classes: int, det_size: int
    ) -> torch.Tensor:
        """
        Compute detector region bounding boxes arranged in a circular pattern.

        Returns:
            Tensor of shape (n_classes, 4): [row_start, row_end, col_start, col_end].
        """
        regions = []
        center = n_pixels / 2.0
        # Arrange detectors in a circle around the center
        radius = n_pixels / 4.0

        for i in range(n_classes):
            angle = 2.0 * math.pi * i / n_classes
            cx = int(center + radius * math.cos(angle))
            cy = int(center + radius * math.sin(angle))

            # Bounding box
            half = det_size // 2
            r_start = max(0, cy - half)
            r_end = min(n_pixels, cy + half)
            c_start = max(0, cx - half)
            c_end = min(n_pixels, cx + half)

            regions.append([r_start, r_end, c_start, c_end])

        return torch.tensor(regions, dtype=torch.long)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Integrate intensity over detector regions to produce logits.

        Args:
            u: Complex wavefield at output plane, shape (B, H, W).

        Returns:
            Logits tensor of shape (B, n_classes).
        """
        # Compute intensity
        intensity = torch.abs(u) ** 2  # (B, H, W)

        B = intensity.shape[0]
        logits = torch.zeros(B, self.n_classes, device=u.device, dtype=torch.float32)

        for c in range(self.n_classes):
            r0, r1, c0, c1 = self.detector_regions[c]
            region_intensity = intensity[:, r0:r1, c0:c1]
            # Sum (integrate) intensity over the detector region
            logits[:, c] = region_intensity.sum(dim=(-2, -1))

        # Apply detector noise
        logits = self.det_noise(logits)

        return logits

    def get_detector_mask(self) -> torch.Tensor:
        """
        Generate a visualization mask showing detector region locations.

        Returns:
            Integer mask of shape (n_pixels, n_pixels), 0 = background,
            1..n_classes = detector index.
        """
        mask = torch.zeros(self.n_pixels, self.n_pixels, dtype=torch.long)
        for c in range(self.n_classes):
            r0, r1, c0, c1 = self.detector_regions[c]
            mask[r0:r1, c0:c1] = c + 1
        return mask

    def extra_repr(self) -> str:
        return (
            f"n_pixels={self.n_pixels}, n_classes={self.n_classes}, "
            f"detector_size={self.detector_size}"
        )
