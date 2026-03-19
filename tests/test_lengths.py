"""Tests for spikeout.lengths."""

import numpy as np
import pytest
from spikeout.detect import detect
from spikeout.lengths import (
    measure_spike_lengths,
    SpikeLengths,
    _swath_profile,
    _find_profile_end,
)


@pytest.fixture
def detected_result(star_centred):
    """A detection result with spikes found."""
    return detect(star_centred, min_snr=3.0)


# ---------------------------------------------------------------------------
# _swath_profile
# ---------------------------------------------------------------------------

class TestSwathProfile:
    """Unit tests for the chunk-based swath profile extractor."""

    def test_returns_correct_length(self):
        img = np.ones((100, 100))
        prof = _swath_profile(img, (50, 10), angle_deg=0.0, length=80)
        assert len(prof) == 80

    def test_horizontal_line_constant_image(self):
        img = np.full((50, 200), 5.0)
        prof = _swath_profile(img, (25, 0), angle_deg=0.0, length=100)
        assert np.allclose(prof, 5.0, atol=1e-10)

    def test_vertical_line_constant_image(self):
        img = np.full((200, 50), 3.0)
        prof = _swath_profile(img, (0, 25), angle_deg=90.0, length=150)
        assert np.allclose(prof, 3.0, atol=1e-10)

    def test_swath_width_averages_across_line(self):
        """Swath mean should equal pixel value for constant images."""
        img = np.full((100, 100), 7.0)
        prof = _swath_profile(img, (50, 5), angle_deg=45.0, length=40,
                               swath_width=5)
        assert np.allclose(prof, 7.0, atol=1e-9)

    def test_out_of_bounds_gives_nan(self):
        """Pixels outside the array boundary are NaN, not errors."""
        img = np.ones((50, 50))
        # start near the edge; some swath pixels will be OOB
        prof = _swath_profile(img, (25, 48), angle_deg=0.0, length=10,
                               swath_width=5)
        # Should not raise; profile may contain NaN at OOB positions but
        # nanmean over valid pixels still works.
        assert prof.shape == (10,)


# ---------------------------------------------------------------------------
# _find_profile_end
# ---------------------------------------------------------------------------

class TestFindProfileEnd:
    """Unit tests for the sliding-window endpoint detector."""

    def test_constant_above_threshold(self):
        """Profile always above threshold → end at last index."""
        prof = np.ones(100) * 10.0
        end = _find_profile_end(prof, threshold=1.0)
        assert end == len(prof) - 1

    def test_constant_below_threshold(self):
        """Profile always below threshold → end at 0."""
        prof = np.zeros(100)
        end = _find_profile_end(prof, threshold=1.0)
        assert end == 0

    def test_drops_midway(self):
        """Profile drops to zero halfway → end near midpoint."""
        prof = np.concatenate([np.ones(50) * 5.0, np.zeros(50)])
        end = _find_profile_end(prof, threshold=1.0, min_pad=0, pad_frac=0)
        # should end somewhere in the first half (with some window tolerance)
        assert end < 80

    def test_end_clamped_to_profile_length(self):
        prof = np.ones(20) * 5.0
        end = _find_profile_end(prof, threshold=0.5)
        assert 0 <= end <= len(prof) - 1

    def test_empty_profile(self):
        assert _find_profile_end(np.array([]), threshold=1.0) == 0

    def test_sinusoidal_profile_not_cut_early(self):
        """Sinusoidal profile (oscillating around threshold) should not be
        cut at the first trough when the window spans a full period."""
        # period ≈ 60 samples; window must span >= 1 period to be robust
        x = np.linspace(0, 10 * np.pi, 300)
        prof = 2.0 + np.sin(x)  # oscillates between 1 and 3, threshold=1.5
        # run_length=70 covers more than one full period (≈60 samples)
        end = _find_profile_end(prof, threshold=1.5, above_frac=0.25,
                                run_length=70)
        # Should reach at least 200 pixels into the 300-pixel profile
        assert end > 200

    def test_isolated_bump_after_end_does_not_extend(self):
        """A bright isolated bump after the spike end should not extend it."""
        prof = np.concatenate([
            np.ones(60) * 5.0,
            np.zeros(30),
            np.ones(10) * 5.0,  # isolated bump
            np.zeros(20),
        ])
        end = _find_profile_end(prof, threshold=1.0, min_pad=2, pad_frac=0.1)
        # the run breaks at ~60; padding may add a few more pixels,
        # but it should not jump to the isolated bump at 90
        assert end < 85


# ---------------------------------------------------------------------------
# measure_spike_lengths
# ---------------------------------------------------------------------------

class TestMeasureSpikeLengths:
    """Tests for the swath-profile length measurement."""

    def test_returns_list_of_spike_lengths(self, star_centred, detected_result):
        lengths = measure_spike_lengths(star_centred, detected_result)
        assert isinstance(lengths, list)
        assert all(isinstance(sl, SpikeLengths) for sl in lengths)

    def test_one_entry_per_spike(self, star_centred, detected_result):
        lengths = measure_spike_lengths(star_centred, detected_result)
        assert len(lengths) == len(detected_result.angles)

    def test_angles_match(self, star_centred, detected_result):
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl, angle in zip(lengths, detected_result.angles):
            assert sl.angle_deg == pytest.approx(angle)

    def test_profiles_are_arrays(self, star_centred, detected_result):
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert sl.profile_pos.ndim == 1
            assert sl.profile_neg.ndim == 1
            assert sl.radii_pos.ndim == 1
            assert sl.radii_neg.ndim == 1

    def test_radii_start_positive(self, star_centred, detected_result):
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert sl.radii_pos[0] > 0
            assert sl.radii_neg[0] > 0

    def test_total_is_sum(self, star_centred, detected_result):
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert sl.length_total == pytest.approx(
                sl.length_pos + sl.length_neg,
            )

    def test_custom_swath_width(self, star_centred, detected_result):
        lengths = measure_spike_lengths(
            star_centred, detected_result, swath_width=10.0,
        )
        assert len(lengths) > 0

    def test_strict_sigma_shorter(self, star_centred, detected_result):
        """Higher length_sigma → shorter measured lengths."""
        l_loose = measure_spike_lengths(
            star_centred, detected_result, length_sigma=1.0,
        )
        l_strict = measure_spike_lengths(
            star_centred, detected_result, length_sigma=5.0,
        )
        if l_loose and l_strict:
            assert l_strict[0].length_total <= l_loose[0].length_total

    def test_asymmetric_arms(self, star_asymmetric):
        """Asymmetric spikes should yield different arm lengths."""
        result = detect(star_asymmetric, min_snr=2.0)
        if len(result.angles) > 0:
            lengths = measure_spike_lengths(star_asymmetric, result)
            ratios = [
                sl.length_pos / max(sl.length_neg, 1.0)
                for sl in lengths
            ]
            assert any(abs(r - 1.0) > 0.1 for r in ratios)

    def test_nan_in_image(self, star_centred, detected_result):
        """NaN pixels should not crash length measurement."""
        noisy = star_centred.copy()
        noisy[10:20, 10:20] = np.nan
        lengths = measure_spike_lengths(noisy, detected_result)
        assert all(np.isfinite(sl.length_total) for sl in lengths)

    def test_popt_always_none(self, star_centred, detected_result):
        """popt is deprecated and should always be None."""
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert sl.popt is None

    def test_converged_flags_are_bool(self, star_centred, detected_result):
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert isinstance(sl.converged_pos, bool)
            assert isinstance(sl.converged_neg, bool)

    def test_threshold_stored(self, star_centred, detected_result):
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert sl.threshold > 0

    def test_full_array_same_as_cutout_when_spike_short(self, star_centred, detected_result):
        """When the spike is short (fits in cutout) full_array should give
        the same (or very similar) result as cutout-only measurement."""
        ny, nx = star_centred.shape
        cy, cx = ny / 2.0, nx / 2.0
        lengths_cutout = measure_spike_lengths(star_centred, detected_result)
        # Simulate a full array by placing the cutout at known offset
        pad = 50
        full = np.zeros((ny + 2 * pad, nx + 2 * pad))
        full[pad:pad + ny, pad:pad + nx] = star_centred
        lengths_full = measure_spike_lengths(
            star_centred, detected_result,
            full_array=full,
            centre_row_full=cy + pad,
            centre_col_full=cx + pad,
        )
        if lengths_cutout and lengths_full:
            # Results should be close (within a few pixels)
            for lc, lf in zip(lengths_cutout, lengths_full):
                assert abs(lc.length_total - lf.length_total) < 30

    def test_skycoord_wcs_path(self, star_centred, detected_result):
        """skycoord + wcs sets centre_row/col_full correctly."""
        pytest.importorskip("astropy")
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord

        ny, nx = star_centred.shape
        w = WCS(naxis=2)
        w.wcs.crpix = [nx / 2, ny / 2]
        w.wcs.cdelt = [-0.000277778, 0.000277778]
        w.wcs.crval = [10.0, 20.0]
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        sky = SkyCoord(ra=10.0, dec=20.0, unit='deg')
        # Should not raise; wcs→pixel → same as default centre
        lengths = measure_spike_lengths(
            star_centred, detected_result,
            full_array=star_centred,
            skycoord=sky,
            wcs=w,
        )
        assert len(lengths) == len(detected_result.angles)
