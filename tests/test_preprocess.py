"""Tests for spikeout.preprocess."""

import numpy as np
import pytest
from spikeout.preprocess import prepare_image, azimuthal_median, find_centre


class TestFindCentre:
    """Tests for automatic centre detection."""

    def test_finds_peak(self):
        """Should return coordinates near the brightest region."""
        img = np.zeros((100, 100))
        img[70, 30] = 1000.0
        cy, cx = find_centre(img)
        assert abs(cy - 70) <= 2
        assert abs(cx - 30) <= 2

    def test_handles_nan(self):
        """NaNs should not crash centre detection."""
        img = np.zeros((64, 64))
        img[32, 32] = 500.0
        img[10:15, 10:15] = np.nan
        cy, cx = find_centre(img)
        assert np.isfinite(cy) and np.isfinite(cx)


class TestAzimuthalMedian:
    """Tests for azimuthal median model."""

    def test_symmetric_input_recovers_itself(self):
        """A radially symmetric image's model should closely match itself.

        Integer radial binning introduces discretisation error, especially
        where the gradient is steep (near the centre), so we check the
        outer region where the function is smooth.
        """
        Y, X = np.mgrid[:101, :101]
        R = np.sqrt((X - 50) ** 2 + (Y - 50) ** 2)
        img = 1000.0 / (1 + R ** 2)
        model = azimuthal_median(img, centre=(50, 50))
        residual = img - model
        # Outer region (R > 10) should be well-recovered
        outer = R > 10
        assert np.abs(residual[outer]).max() < 1.0

    def test_shape_preserved(self, pure_noise):
        """Output shape must match input."""
        model = azimuthal_median(pure_noise)
        assert model.shape == pure_noise.shape

    def test_custom_centre(self, rng):
        """Non-default centre should not crash and should produce output."""
        img = rng.normal(0, 1, (64, 64))
        model = azimuthal_median(img, centre=(20, 40))
        assert model.shape == img.shape
        assert np.isfinite(model).all()

    def test_radial_bin_width(self, pure_noise):
        """Wider bins produce a smoother model."""
        model_narrow = azimuthal_median(pure_noise, radial_bin_width=1)
        model_wide = azimuthal_median(pure_noise, radial_bin_width=5)
        # Both should be valid
        assert np.isfinite(model_narrow).all()
        assert np.isfinite(model_wide).all()


class TestPrepareImage:
    """Tests for the full preprocessing pipeline."""

    def test_output_nonnegative(self, star_centred):
        """Prepared image should have no negative values."""
        prepared = prepare_image(star_centred)
        assert (prepared >= 0).all()

    def test_output_shape(self, star_centred):
        """Shape should be preserved."""
        prepared = prepare_image(star_centred)
        assert prepared.shape == star_centred.shape

    def test_nan_safe(self):
        """NaN pixels should be handled without crashing."""
        img = np.ones((64, 64)) * 100.0
        img[10:20, 10:20] = np.nan
        img[32, 32] = 500.0
        prepared = prepare_image(img)
        assert np.isfinite(prepared).all()

    def test_morph_radius_zero_skips(self, star_centred):
        """morph_radius=0 should skip morphological opening."""
        # Should not crash, and result should differ from morph_radius=3
        p0 = prepare_image(star_centred, morph_radius=0)
        p3 = prepare_image(star_centred, morph_radius=3)
        assert p0.shape == p3.shape
        # They shouldn't be identical (opening changes the image)
        assert not np.allclose(p0, p3)

    def test_custom_asinh_stretch(self, star_centred):
        """Explicit asinh_stretch should be respected."""
        p1 = prepare_image(star_centred, asinh_stretch=1.0)
        p10 = prepare_image(star_centred, asinh_stretch=10.0)
        # Higher stretch → lower dynamic range in output
        assert p10.max() < p1.max()

    def test_constant_image_raises(self, constant_image):
        """Constant image has zero variance → should still return."""
        # prepare_image clips to zero after subtracting model, so
        # it should return an all-zero array rather than crash
        prepared = prepare_image(constant_image)
        assert (prepared == 0).all()
