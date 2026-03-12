"""Tests for spikeout.geometry."""

import numpy as np
import pytest
from spikeout.geometry import sinogram_rho_to_physical, radon_line_to_image


class TestSinogramRhoToPhysical:
    """Tests for sinogram row → physical ρ conversion."""

    def test_centre_is_zero(self):
        """The centre row should map to ρ = 0."""
        n_rho = 500
        centre = n_rho // 2
        assert sinogram_rho_to_physical(centre, n_rho) == 0

    def test_above_centre_is_negative(self):
        """Rows below the centre index should give negative ρ."""
        assert sinogram_rho_to_physical(100, 500) < 0

    def test_below_centre_is_positive(self):
        """Rows above the centre index should give positive ρ."""
        assert sinogram_rho_to_physical(300, 500) > 0

    def test_vectorised(self):
        """Should work on arrays."""
        indices = np.array([100, 250, 400])
        result = sinogram_rho_to_physical(indices, 500)
        assert result.shape == (3,)
        assert result[0] < 0
        assert result[1] == 0
        assert result[2] > 0


class TestRadonLineToImage:
    """Tests for (ρ,θ) → pixel endpoint conversion."""

    def test_centred_horizontal(self):
        """ρ=0, θ=0 should give a horizontal line through the centre."""
        shape = (100, 100)
        (x1, y1), (x2, y2), angle = radon_line_to_image(0, 0, shape)
        # θ=0: projection direction is along x, so the line (perpendicular
        # to projection) runs vertically in Radon coords.
        # With the y-flip, direction is (-sin0, -cos0) = (0, -1) in plot
        # coords, i.e. a vertical line.  angle = atan2(-1, 0) = 270°
        assert angle == pytest.approx(270.0, abs=1.0)
        # Both endpoints should have x ≈ centre (vertical line)
        assert x1 == pytest.approx(50.0, abs=1.0)
        assert x2 == pytest.approx(50.0, abs=1.0)

    def test_centred_theta90(self):
        """ρ=0, θ=90 should give a horizontal line through the centre."""
        shape = (100, 100)
        (x1, y1), (x2, y2), angle = radon_line_to_image(0, 90, shape)
        # θ=90: direction is (-sin90, -cos90) = (-1, 0) → angle=180°
        assert angle == pytest.approx(180.0, abs=1.0)
        # Both endpoints should have y ≈ centre (horizontal line)
        assert y1 == pytest.approx(50.0, abs=1.0)
        assert y2 == pytest.approx(50.0, abs=1.0)

    def test_rho_shifts_line(self):
        """Non-zero ρ should move the line off-centre."""
        shape = (200, 200)
        # ρ=0 through centre
        (_, y1_c), (_, y2_c), _ = radon_line_to_image(0, 90, shape)
        # ρ=+20 should shift the line
        (_, y1_s), (_, y2_s), _ = radon_line_to_image(20, 90, shape)
        # The line should be at a different y position
        assert not pytest.approx(y1_c, abs=5.0) == y1_s

    def test_endpoints_span_image(self):
        """With default padding, endpoints should extend past the image."""
        shape = (100, 100)
        (x1, y1), (x2, y2), _ = radon_line_to_image(0, 45, shape)
        diag = np.hypot(*shape)
        endpoint_span = np.hypot(x2 - x1, y2 - y1)
        assert endpoint_span > diag

    def test_angle_wraps_to_360(self):
        """Returned angle should be in [0, 360)."""
        for theta in [0, 45, 90, 135, 170]:
            _, _, angle = radon_line_to_image(0, theta, (100, 100))
            assert 0 <= angle < 360

    def test_rectangular_image(self):
        """Should work with non-square images."""
        shape = (100, 200)
        (x1, y1), (x2, y2), angle = radon_line_to_image(0, 45, shape)
        assert np.isfinite(x1) and np.isfinite(y1)
        assert np.isfinite(angle)
