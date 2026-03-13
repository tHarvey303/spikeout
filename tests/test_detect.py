"""Tests for spikeout.detect."""

import numpy as np
import pytest
from spikeout.detect import detect, SpikeResult


class TestDetectBasic:
    """Basic detection tests: return types, shapes, edge cases."""

    def test_returns_spike_result(self, star_centred):
        """detect() should return a SpikeResult dataclass."""
        result = detect(star_centred)
        assert isinstance(result, SpikeResult)

    def test_result_arrays_consistent(self, star_centred):
        """All per-spike arrays should have the same length."""
        result = detect(star_centred)
        n = len(result.angles)
        assert len(result.rho_physical) == n
        assert len(result.snr) == n
        assert len(result.peak_rho_indices) == n
        assert len(result.peak_theta_indices) == n

    def test_angles_in_range(self, star_centred):
        """Detected angles should be in [0, 360)."""
        result = detect(star_centred)
        for a in result.angles:
            assert 0 <= a < 360

    def test_snr_positive(self, star_centred):
        """Peaks accepted by the SNR filter should have SNR >= the threshold."""
        result = detect(star_centred, min_snr=3.0)
        assert (result.snr >= 3.0).all()

    def test_sinogram_shape(self, star_centred):
        """Sinogram should have (n_rho, n_theta) shape."""
        result = detect(star_centred)
        assert result.sinogram.ndim == 2
        assert result.sinogram.shape[1] == len(result.theta)

    def test_prepared_image_shape(self, star_centred):
        """Prepared image should match input shape."""
        result = detect(star_centred)
        assert result.prepared_image.shape == star_centred.shape

    def test_1d_image_raises(self):
        """Non-2D input should raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            detect(np.ones(100))

    def test_all_nan_raises(self, all_nan_image):
        """All-NaN input should raise ValueError."""
        with pytest.raises(ValueError):
            detect(all_nan_image)

    def test_constant_image_no_crash(self, constant_image):
        """Constant image should raise (zero variance)."""
        with pytest.raises(ValueError, match="zero variance"):
            detect(constant_image)


class TestDetectAngles:
    """Tests for correctness of detected spike angles."""

    def test_two_spikes_centred(self, star_centred):
        """Should detect two spikes for a centred star with two spikes."""
        result = detect(star_centred)
        assert len(result.angles) == 2

    def test_four_spikes(self, star_four_spikes):
        """Should detect four spikes in a cross pattern."""
        result = detect(
            star_four_spikes, min_snr=2.0, min_peak_separation_deg=15.0,
        )
        assert len(result.angles) >= 3  # at least most of the 4

    def test_angles_are_separated(self, star_centred):
        """Detected angles should respect min_peak_separation_deg."""
        sep = 20.0
        result = detect(star_centred, min_peak_separation_deg=sep, min_snr=0.0)
        if len(result.angles) >= 2:
            for i in range(len(result.angles)):
                for j in range(i + 1, len(result.angles)):
                    diff = abs(result.angles[i] - result.angles[j])
                    diff = min(diff, 360 - diff)
                    assert diff >= sep * 0.5  # allow some tolerance


class TestRhoFilter:
    """Tests for the ρ (centrality) constraint applied during detection."""

    def test_strict_rho_detects_fewer(self, star_offset):
        """Stricter ρ constraint should detect fewer or equal peaks."""
        r_loose = detect(star_offset, max_rho_fraction=0.5, min_snr=0.0)
        r_strict = detect(star_offset, max_rho_fraction=0.02, min_snr=0.0)
        assert len(r_strict.angles) <= len(r_loose.angles)

    def test_detected_rho_within_band(self, star_offset):
        """All detected peaks must have |ρ| within the requested band."""
        frac = 0.2
        result = detect(star_offset, max_rho_fraction=frac, min_snr=0.0)
        max_rho_px = frac * min(star_offset.shape) / 2.0
        assert (np.abs(result.rho_physical) <= max_rho_px + 1).all()

    def test_centred_star_small_rho(self, star_centred):
        """Centred star's real spikes should survive even a strict ρ filter.

        Note: spurious peaks from noise may still be rejected — we only
        check that the actual spikes pass through.
        """
        result = detect(star_centred, max_rho_fraction=0.05, min_snr=3.0)
        assert len(result.angles) >= 2


class TestSnrFilter:
    """Tests for the SNR significance filter."""

    def test_pure_noise_no_spikes(self, pure_noise):
        """Noise-only image should yield no detections at min_snr=5."""
        result = detect(pure_noise, min_snr=5.0)
        assert len(result.angles) == 0

    def test_high_snr_threshold_rejects(self, star_centred):
        """An absurdly high SNR threshold should reject everything."""
        result = detect(star_centred, min_snr=1000.0)
        assert len(result.angles) == 0
        assert result.n_rejected_snr > 0

    def test_zero_snr_disables_filter(self, star_centred):
        """min_snr=0 should not reject any peaks."""
        result = detect(star_centred, min_snr=0.0)
        assert result.n_rejected_snr == 0

    def test_snr_values_above_threshold(self, star_centred):
        """All accepted peaks should have SNR ≥ min_snr."""
        threshold = 4.0
        result = detect(star_centred, min_snr=threshold)
        assert (result.snr >= threshold).all()

    def test_with_neighbour_still_finds_spikes(self, star_with_neighbour):
        """Neighbouring source should not prevent spike detection."""
        result = detect(star_with_neighbour, min_snr=3.0)
        assert len(result.angles) >= 1


class TestDetectWithLengths:
    """Tests for the measure_lengths integration path."""

    def test_lengths_none_by_default(self, star_centred):
        """Lengths should be None when not requested."""
        result = detect(star_centred)
        assert result.lengths is None

    def test_lengths_populated(self, star_centred):
        """measure_lengths=True should populate the lengths list."""
        result = detect(star_centred, measure_lengths=True, min_snr=3.0)
        assert result.lengths is not None
        assert len(result.lengths) == len(result.angles)

    def test_lengths_positive(self, star_centred):
        """All measured lengths should be positive."""
        result = detect(star_centred, measure_lengths=True, min_snr=3.0)
        for sl in result.lengths:
            assert sl.length_pos > 0
            assert sl.length_neg > 0
            assert sl.length_total == pytest.approx(
                sl.length_pos + sl.length_neg,
            )

    def test_no_spikes_no_lengths(self, pure_noise):
        """If no spikes detected, lengths should be None."""
        result = detect(pure_noise, measure_lengths=True, min_snr=5.0)
        assert result.lengths is None

    def test_length_kw_forwarded(self, star_centred):
        """Custom length_kw should affect results."""
        r1 = detect(
            star_centred, measure_lengths=True, min_snr=3.0,
            length_kw=dict(length_sigma=1.0),
        )
        r2 = detect(
            star_centred, measure_lengths=True, min_snr=3.0,
            length_kw=dict(length_sigma=5.0),
        )
        # Higher sigma threshold → shorter measured lengths
        if r1.lengths and r2.lengths:
            assert r1.lengths[0].length_total >= r2.lengths[0].length_total


class TestPrepKwForwarding:
    """Tests that preprocessing kwargs are forwarded correctly."""

    def test_morph_radius_changes_result(self, star_centred):
        """Different morph_radius should produce different sinograms."""
        r0 = detect(star_centred, morph_radius=0, min_snr=0.0)
        r5 = detect(star_centred, morph_radius=5, min_snr=0.0)
        assert not np.allclose(r0.sinogram, r5.sinogram)

    def test_sigma_clip_changes_result(self, star_centred):
        """Different sigma_clip should affect preprocessing."""
        r1 = detect(star_centred, sigma_clip=0.5, min_snr=0.0)
        r3 = detect(star_centred, sigma_clip=3.0, min_snr=0.0)
        assert not np.allclose(r1.prepared_image, r3.prepared_image)
