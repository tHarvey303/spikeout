"""Tests for spikeout.lengths."""

import numpy as np
import pytest
from spikeout.detect import detect
from spikeout.lengths import measure_spike_lengths, SpikeLengths


@pytest.fixture
def detected_result(star_centred):
    """A detection result with spikes found."""
    return detect(star_centred, min_snr=3.0)


class TestMeasureSpikeLengths:
    """Tests for the swath-profile length measurement."""

    def test_returns_list_of_spike_lengths(self, star_centred, detected_result):
        """Should return a list of SpikeLengths."""
        lengths = measure_spike_lengths(star_centred, detected_result)
        assert isinstance(lengths, list)
        assert all(isinstance(sl, SpikeLengths) for sl in lengths)

    def test_one_entry_per_spike(self, star_centred, detected_result):
        """Should have one SpikeLengths per detected angle."""
        lengths = measure_spike_lengths(star_centred, detected_result)
        assert len(lengths) == len(detected_result.angles)

    def test_angles_match(self, star_centred, detected_result):
        """Each SpikeLengths.angle_deg should match the detection result."""
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl, angle in zip(lengths, detected_result.angles):
            assert sl.angle_deg == pytest.approx(angle)

    def test_profiles_are_arrays(self, star_centred, detected_result):
        """Swath profiles should be 1-D arrays."""
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert sl.profile_pos.ndim == 1
            assert sl.profile_neg.ndim == 1
            assert sl.radii_pos.ndim == 1
            assert sl.radii_neg.ndim == 1

    def test_radii_start_positive(self, star_centred, detected_result):
        """Radii should start at step > 0 (not at the centre)."""
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert sl.radii_pos[0] > 0
            assert sl.radii_neg[0] > 0

    def test_total_is_sum(self, star_centred, detected_result):
        """length_total should be length_pos + length_neg."""
        lengths = measure_spike_lengths(star_centred, detected_result)
        for sl in lengths:
            assert sl.length_total == pytest.approx(
                sl.length_pos + sl.length_neg,
            )

    def test_custom_swath_width(self, star_centred, detected_result):
        """Explicit swath_width should not crash."""
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
        # At least one arm should be shorter with stricter threshold
        if l_loose and l_strict:
            assert l_strict[0].length_total <= l_loose[0].length_total

    def test_asymmetric_arms(self, star_asymmetric):
        """Asymmetric spikes should yield different arm lengths."""
        result = detect(star_asymmetric, min_snr=2.0)
        if len(result.angles) > 0:
            lengths = measure_spike_lengths(star_asymmetric, result)
            # At least one spike should have notably different arms
            ratios = [
                sl.length_pos / max(sl.length_neg, 1.0)
                for sl in lengths
            ]
            # Not all ratios should be exactly 1
            assert any(abs(r - 1.0) > 0.1 for r in ratios)

    def test_nan_in_image(self, star_centred, detected_result):
        """NaN pixels should not crash length measurement."""
        noisy = star_centred.copy()
        noisy[10:20, 10:20] = np.nan
        lengths = measure_spike_lengths(noisy, detected_result)
        assert all(np.isfinite(sl.length_total) for sl in lengths)
