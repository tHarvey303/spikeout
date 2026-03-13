"""Tests for spikeout.regions.halo_mask."""

import numpy as np
import pytest
from spikeout.regions import halo_mask


# ── helpers ───────────────────────────────────────────────────────────────────

def _radial_dist(shape, centre=None):
    nrows, ncols = shape
    cy, cx = (nrows / 2.0, ncols / 2.0) if centre is None else centre
    Y, X = np.mgrid[:nrows, :ncols]
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)


def _moffat_star(rng, size=256, psf_peak=5000, bg=100, bg_sigma=5, seed=0):
    """Synthetic star with a Moffat-like 1/(1+r²) halo, centred in the image."""
    rng_local = np.random.default_rng(seed)
    img = rng_local.normal(bg, bg_sigma, (size, size)).astype(float)
    cy, cx = size / 2.0, size / 2.0
    Y, X = np.mgrid[:size, :size]
    R2 = (X - cx) ** 2 + (Y - cy) ** 2
    img += psf_peak / (1.0 + R2)
    return img


# ── basic return types ────────────────────────────────────────────────────────

class TestHaloMaskReturnTypes:

    def test_returns_tuple(self, star_centred):
        result = halo_mask(star_centred)
        assert isinstance(result, tuple) and len(result) == 2

    def test_mask_is_bool_ndarray(self, star_centred):
        mask, _ = halo_mask(star_centred)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_mask_shape_matches_image(self, star_centred):
        mask, _ = halo_mask(star_centred)
        assert mask.shape == star_centred.shape

    def test_radius_is_positive_float(self, star_centred):
        _, r = halo_mask(star_centred)
        assert isinstance(r, float)
        assert r > 0.0


# ── geometric correctness ─────────────────────────────────────────────────────

class TestHaloMaskGeometry:

    def test_mask_is_circular(self, star_centred):
        """All True pixels must lie within halo_r of the centre (±1 px)."""
        mask, r = halo_mask(star_centred)
        R = _radial_dist(star_centred.shape)
        # Pixels in mask should all be within r (with 1-pixel rounding margin)
        assert np.all(R[mask] <= r + 1.0)

    def test_centre_pixel_always_masked(self, star_centred):
        nrows, ncols = star_centred.shape
        mask, _ = halo_mask(star_centred)
        assert mask[nrows // 2, ncols // 2]

    def test_mask_connected_circle(self, star_centred):
        """No pixel outside the disc should be True."""
        mask, r = halo_mask(star_centred)
        R = _radial_dist(star_centred.shape)
        # Everything outside r+1 must be False
        outside = R > r + 1.0
        assert not mask[outside].any()

    def test_custom_centre_shifts_mask(self):
        """The mask centre follows the specified star position."""
        size = 128
        img = np.ones((size, size)) * 100.0
        # Add a bright PSF at (row=40, col=90)
        Y, X = np.mgrid[:size, :size]
        img += 3000.0 / (1.0 + (X - 90) ** 2 + (Y - 40) ** 2)

        mask_default, _ = halo_mask(img)
        mask_custom, _ = halo_mask(img, centre=(40, 90))

        # Centre pixel at star position must be masked when using custom centre
        assert mask_custom[40, 90]
        # Default (image centre) and custom centres give different masks
        assert not np.array_equal(mask_default, mask_custom)


# ── threshold behaviour ───────────────────────────────────────────────────────

class TestHaloMaskThreshold:

    def test_higher_nsigma_gives_smaller_or_equal_radius(self, star_centred):
        _, r_low = halo_mask(star_centred, threshold_nsigma=2.0)
        _, r_high = halo_mask(star_centred, threshold_nsigma=8.0)
        assert r_low >= r_high

    def test_min_radius_enforced_when_threshold_very_high(self, star_centred):
        min_r = 20.0
        _, r = halo_mask(star_centred, threshold_nsigma=1e6, min_radius=min_r)
        assert r >= min_r

    def test_min_radius_enforced_for_noise_image(self, pure_noise):
        min_r = 8.0
        _, r = halo_mask(pure_noise, min_radius=min_r)
        assert r >= min_r

    def test_noise_image_gives_small_radius(self, pure_noise):
        """For a structureless noise image the detected radius should be small."""
        min_r = 5.0
        _, r = halo_mask(pure_noise, min_radius=min_r, threshold_nsigma=3.0)
        # Should not extend more than ~10 px beyond min_radius
        assert r <= min_r + 15.0

    def test_bright_star_larger_radius_than_noise(self, star_centred, pure_noise):
        _, r_star = halo_mask(star_centred)
        _, r_noise = halo_mask(pure_noise)
        assert r_star > r_noise


# ── profile radius scaling ────────────────────────────────────────────────────

class TestHaloMaskRadiusScaling:

    def test_brighter_star_larger_radius(self):
        """Higher PSF peak → wider halo → larger mask radius."""
        img_dim = _moffat_star(None, psf_peak=500, seed=1)
        img_bright = _moffat_star(None, psf_peak=10000, seed=1)
        _, r_dim = halo_mask(img_dim, threshold_nsigma=3.0)
        _, r_bright = halo_mask(img_bright, threshold_nsigma=3.0)
        assert r_bright > r_dim

    def test_radius_increases_monotonically_with_brightness(self):
        peaks = [300, 1000, 3000, 10000, 30000]
        radii = []
        for p in peaks:
            img = _moffat_star(None, psf_peak=p, seed=2)
            _, r = halo_mask(img, threshold_nsigma=3.0)
            radii.append(r)
        assert all(radii[i] <= radii[i + 1] for i in range(len(radii) - 1))


# ── robustness ────────────────────────────────────────────────────────────────

class TestHaloMaskRobustness:

    def test_nan_core_does_not_crash(self, star_centred):
        img = star_centred.copy()
        nrows, ncols = img.shape
        img[nrows // 2 - 10:nrows // 2 + 10, ncols // 2 - 10:ncols // 2 + 10] = np.nan
        mask, r = halo_mask(img)
        assert mask.dtype == bool
        assert r > 5.0

    def test_all_nan_annulus_does_not_crash(self, star_centred):
        """An image with a NaN outer ring should still return a valid mask."""
        img = star_centred.copy()
        nrows, ncols = img.shape
        R = _radial_dist(img.shape)
        img[R > 100] = np.nan
        mask, r = halo_mask(img, max_radius=90.0)
        assert mask.dtype == bool
        assert r >= 5.0

    def test_spike_lines_do_not_inflate_radius(self, star_centred):
        """Adding bright spike lines should not significantly inflate the radius.

        The azimuthal median is robust to structures covering < 50 % of an
        annulus (a pair of opposite spike lines covers ≪ 50 % of each annulus).
        """
        from skimage.draw import line as draw_line
        img_no_spike = star_centred.copy()
        img_spike = star_centred.copy()
        nrows, ncols = img_spike.shape
        cy, cx = nrows // 2, ncols // 2
        # Draw a horizontal spike
        rr, cc = draw_line(cy, 0, cy, ncols - 1)
        img_spike[rr, cc] += 5000

        _, r_no = halo_mask(img_no_spike)
        _, r_sp = halo_mask(img_spike)
        # Spike should not inflate radius by more than a few pixels
        assert abs(r_sp - r_no) <= 5.0

    def test_neighbour_does_not_dominate_radius(self, star_with_neighbour):
        """A compact bright neighbour should not drive the mask to the image edge."""
        mask, r = halo_mask(star_with_neighbour)
        max_r = min(star_with_neighbour.shape) / 2.0
        # Radius should not consume the entire image
        assert r < max_r * 0.85

    def test_offset_star_with_correct_centre(self, star_offset):
        """Providing the correct off-centre position gives a reasonable radius."""
        from spikeout.preprocess import find_centre
        cy, cx = find_centre(star_offset)
        mask, r = halo_mask(star_offset, centre=(cy, cx))
        # Centre pixel of the star should be masked
        assert mask[int(round(cy)), int(round(cx))]
        assert r > 5.0

    def test_smooth_bins_zero_does_not_crash(self, star_centred):
        mask, r = halo_mask(star_centred, smooth_bins=0)
        assert mask.dtype == bool

    def test_small_image(self):
        """Very small images (32×32) should not raise exceptions."""
        rng = np.random.default_rng(7)
        img = rng.normal(100, 5, (32, 32)).astype(float)
        Y, X = np.mgrid[:32, :32]
        img += 500.0 / (1.0 + (X - 16) ** 2 + (Y - 16) ** 2)
        mask, r = halo_mask(img)
        assert mask.shape == (32, 32)
        assert r > 0.0


# ── max_radius parameter ──────────────────────────────────────────────────────

class TestHaloMaskMaxRadius:

    def test_max_radius_caps_result(self, star_centred):
        cap = 30.0
        mask, r = halo_mask(star_centred, max_radius=cap, threshold_nsigma=1.0)
        assert r <= cap + 1.0  # 1-px rounding

    def test_default_max_radius_does_not_exceed_image_edge(self, star_centred):
        mask, r = halo_mask(star_centred)
        nrows, ncols = star_centred.shape
        max_possible = min(nrows, ncols) / 2.0
        assert r <= max_possible + 1.0
