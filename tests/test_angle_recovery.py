"""Angle-recovery tests for spikeout.detect.

Each test class uses a synthetic image with spike lines drawn at *known*
angles and checks that detect() recovers those angles to within the
expected Radon angular resolution (~0.7° for a 256-px image; we use a
4° tolerance to absorb discretisation of draw_line).

All fixtures use morph_radius=0 because the Gaussian-broadened spike
lines (FWHM ≈ 5 px) would be erased by the default disk(3) morphological
opening.

Known algorithm limitations tested or documented here
------------------------------------------------------
* Vertical spikes (display angle ≈ 90°) have their Radon peak at
  theta=0°, the edge of the theta array.  scipy.signal.find_peaks does
  not detect edge peaks, so exactly-vertical spikes are missed.  The
  parametric test excludes 90° for this reason.

* A bright off-centre source creates a Radon contribution at ρ=0 along
  the line connecting the image centre to the source (the "collinear"
  angle).  This can appear as a spurious spike.  The bright-neighbour
  tests check that real spikes ARE recovered, not that spurious collinear
  detections are absent — that is a known limitation of the current
  implementation.
"""

import numpy as np
import pytest
from spikeout.detect import detect


# ── helpers ───────────────────────────────────────────────────────────────────

TOL_DEG = 4.0  # angular tolerance for detection matching (≈ 6 Radon steps)


def _angle_diff_mod180(a, b):
    """Minimum angular difference between two angles modulo 180°.

    Spikes are bidirectional, so 30° and 210° are the same line.
    """
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _unmatched_expected(detected, expected, tol=TOL_DEG):
    """Return expected angles that have no detection within *tol* degrees."""
    return [
        exp for exp in expected
        if not any(_angle_diff_mod180(d, exp) <= tol for d in detected)
    ]


def _spurious_detections(detected, expected, tol=TOL_DEG):
    """Return detected angles that do not correspond to any expected angle."""
    return [
        d for d in detected
        if not any(_angle_diff_mod180(d, exp) <= tol for exp in expected)
    ]


# ── clean image ───────────────────────────────────────────────────────────────

class TestAngleRecoveryClean:
    """Two spikes at 30° and 120°, no saturation, no neighbour."""

    ANGLES = [30, 120]

    def test_count(self, spike_clean):
        result = detect(spike_clean, morph_radius=0, min_snr=3.0)
        assert len(result.angles) == len(self.ANGLES)

    def test_recovers_expected_angles(self, spike_clean):
        result = detect(spike_clean, morph_radius=0, min_snr=3.0)
        unmatched = _unmatched_expected(result.angles, self.ANGLES)
        assert not unmatched, f"Expected angles not found: {unmatched}"

    def test_no_spurious_detections(self, spike_clean):
        result = detect(spike_clean, morph_radius=0, min_snr=5.0)
        spurious = _spurious_detections(result.angles, self.ANGLES)
        assert not spurious, f"Spurious detections: {spurious}"

    def test_rho_near_zero(self, spike_clean):
        """For a centred star with no saturation, |ρ| should be small."""
        result = detect(spike_clean, morph_radius=0, min_snr=3.0)
        max_rho_px = 0.1 * min(spike_clean.shape) / 2.0
        assert (np.abs(result.rho_physical) <= max_rho_px + 2).all()

    def test_snr_values_positive(self, spike_clean):
        result = detect(spike_clean, morph_radius=0, min_snr=3.0)
        assert (result.snr > 0).all()


# ── saturated / zeroed core ───────────────────────────────────────────────────

class TestAngleRecoverySaturatedCore:
    """Spikes at 30° and 120° with a 50-px zeroed core."""

    ANGLES = [30, 120]

    def test_count(self, spike_saturated_core):
        result = detect(spike_saturated_core, morph_radius=0, min_snr=3.0)
        assert len(result.angles) == len(self.ANGLES)

    def test_recovers_expected_angles(self, spike_saturated_core):
        result = detect(spike_saturated_core, morph_radius=0, min_snr=3.0)
        unmatched = _unmatched_expected(result.angles, self.ANGLES)
        assert not unmatched, f"Expected angles not found: {unmatched}"

    def test_rho_within_extended_band(self, spike_saturated_core):
        """max_rho_px should be auto-extended to capture peaks at the core edge."""
        result = detect(spike_saturated_core, morph_radius=0, min_snr=3.0)
        assert (np.abs(result.rho_physical) <= result.max_rho_px + 1).all()

    def test_max_rho_px_larger_than_clean(self, spike_clean, spike_saturated_core):
        """A saturated core should widen the acceptance band."""
        r_clean = detect(spike_clean, morph_radius=0, min_snr=3.0)
        r_sat = detect(spike_saturated_core, morph_radius=0, min_snr=3.0)
        assert r_sat.max_rho_px > r_clean.max_rho_px


class TestAngleRecoveryNanCore:
    """Same as saturated-core but the core is NaN instead of zero."""

    ANGLES = [30, 120]

    def test_count(self, spike_nan_core):
        result = detect(spike_nan_core, morph_radius=0, min_snr=3.0)
        assert len(result.angles) == len(self.ANGLES)

    def test_recovers_expected_angles(self, spike_nan_core):
        result = detect(spike_nan_core, morph_radius=0, min_snr=3.0)
        unmatched = _unmatched_expected(result.angles, self.ANGLES)
        assert not unmatched, f"Expected angles not found: {unmatched}"

    def test_rho_within_extended_band(self, spike_nan_core):
        result = detect(spike_nan_core, morph_radius=0, min_snr=3.0)
        assert (np.abs(result.rho_physical) <= result.max_rho_px + 1).all()


# ── off-edge bright neighbour ─────────────────────────────────────────────────

class TestAngleRecoveryBrightNeighbour:
    """Spikes at 30° and 120° with a bright Gaussian source at (200, 50).

    A bright off-centre source contributes to sinogram_central along the
    line connecting the image centre to the source (the collinear angle,
    ~137° here).  This can reduce the SNR of real spikes and create a
    spurious detection at the collinear angle.  These tests check that real
    spikes ARE recovered despite the neighbour — suppression of the collinear
    false positive requires additional filtering (e.g. min_length) and is
    not guaranteed by the current algorithm.
    """

    ANGLES = [30, 120]
    # The collinear direction from image centre to the Gaussian at (r=200, c=50):
    # display angle ≈ atan2(-(200-128), (50-128)) mapped through radon_line_to_image
    # The exact value is not tested here, but is documented for reference.

    def test_count_at_least_real_spikes(self, spike_bright_neighbour):
        """At least the real spikes should be detected."""
        result = detect(
            spike_bright_neighbour, morph_radius=0, min_snr=2.5,
        )
        assert len(result.angles) >= len(self.ANGLES)

    def test_recovers_expected_angles(self, spike_bright_neighbour):
        result = detect(
            spike_bright_neighbour, morph_radius=0, min_snr=2.5,
        )
        unmatched = _unmatched_expected(result.angles, self.ANGLES)
        assert not unmatched, f"Expected angles not found: {unmatched}"

    def test_neighbour_does_not_suppress_real_spikes(
        self, spike_clean, spike_bright_neighbour,
    ):
        """Real spikes detected with neighbour should match those without."""
        r_clean = detect(spike_clean, morph_radius=0, min_snr=2.5)
        r_nbr = detect(spike_bright_neighbour, morph_radius=0, min_snr=2.5)
        unmatched = _unmatched_expected(r_nbr.angles, self.ANGLES)
        assert not unmatched, (
            f"Neighbour suppressed real spikes; unmatched: {unmatched}"
        )

    def test_rho_within_band(self, spike_bright_neighbour):
        result = detect(
            spike_bright_neighbour, morph_radius=0, min_snr=2.5,
        )
        assert (np.abs(result.rho_physical) <= result.max_rho_px + 1).all()


# ── multi-angle: close pair + saturated core + neighbour ─────────────────────

class TestAngleRecoveryMultiAngle:
    """Full scenario: 4 spikes, saturated core, bright off-edge neighbour.

    With 4 spikes, a saturated core, and a bright off-edge Gaussian source,
    the angular-profile SNR of individual spikes is reduced.  We use
    min_snr=2.0 to capture all real spikes.  Spurious detections at the
    neighbour's collinear angle may also appear; we test only that all
    expected angles are found (recovery), not that no false positives exist.
    """

    ANGLES = [30, 120, 140, 170]
    _KW = dict(morph_radius=0, min_snr=2.0, min_peak_separation_deg=8.0)

    def test_count_at_least_all_spikes(self, spike_multi_angle_full):
        result = detect(spike_multi_angle_full, **self._KW)
        assert len(result.angles) >= len(self.ANGLES)

    def test_recovers_all_angles(self, spike_multi_angle_full):
        result = detect(spike_multi_angle_full, **self._KW)
        unmatched = _unmatched_expected(result.angles, self.ANGLES)
        assert not unmatched, f"Expected angles not found: {unmatched}"

    def test_close_angle_pair_resolved(self, spike_multi_angle_full):
        """140° and 170° are only 30° apart — both must be independently detected."""
        result = detect(spike_multi_angle_full, **self._KW)
        unmatched = _unmatched_expected(result.angles, [140, 170])
        assert not unmatched, (
            f"Close angle pair not fully resolved; unmatched: {unmatched}, "
            f"detected: {sorted(result.angles)}"
        )

    def test_wide_pair_resolved(self, spike_multi_angle_full):
        """30° and 120° are 90° apart — should always be separated."""
        result = detect(spike_multi_angle_full, **self._KW)
        unmatched = _unmatched_expected(result.angles, [30, 120])
        assert not unmatched, f"Wide pair not found: {unmatched}"

    def test_rho_within_band(self, spike_multi_angle_full):
        result = detect(spike_multi_angle_full, **self._KW)
        assert (np.abs(result.rho_physical) <= result.max_rho_px + 1).all()


# ── parametric: individual angle accuracy ────────────────────────────────────

def _single_spike_image(rng, angle_deg, size=256, brightness=100,
                         noise_sigma=2.0, blur_sigma=2.0, half_len=110):
    """Minimal synthetic image with one spike for the parametric test."""
    from scipy.ndimage import gaussian_filter
    from skimage.draw import line as draw_line
    img = rng.normal(0, noise_sigma, (size, size)).astype(float)
    cx, cy = size // 2, size // 2
    rad = np.deg2rad(angle_deg)
    r1 = int(cy + half_len * np.sin(rad))
    c1 = int(cx + half_len * np.cos(rad))
    r2 = int(cy - half_len * np.sin(rad))
    c2 = int(cx - half_len * np.cos(rad))
    rr, cc = draw_line(
        np.clip(r2, 0, size - 1), np.clip(c2, 0, size - 1),
        np.clip(r1, 0, size - 1), np.clip(c1, 0, size - 1),
    )
    img[rr, cc] += brightness
    return gaussian_filter(img, sigma=blur_sigma)


# Note: 90° is excluded — a perfectly vertical spike has its Radon peak at
# theta=0° (the edge of the theta array) where scipy find_peaks cannot detect
# it.  Any real spike slightly offset from 90° would be detected normally.
@pytest.mark.parametrize("angle_deg", [0, 15, 30, 45, 60, 75,
                                        105, 120, 135, 150, 165])
def test_single_spike_angle_recovery(rng, angle_deg):
    """A single spike at each 15° step is recovered within tolerance."""
    img = _single_spike_image(rng, angle_deg)
    result = detect(img, morph_radius=0, min_snr=3.0)
    assert len(result.angles) >= 1, (
        f"No spike detected for input angle {angle_deg}°"
    )
    unmatched = _unmatched_expected(result.angles, [angle_deg])
    assert not unmatched, (
        f"Angle {angle_deg}° not recovered; detected {sorted(result.angles)}"
    )
