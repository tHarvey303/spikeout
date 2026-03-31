"""Tests for star-centre correction via Radon peak analysis.

Covers:
- calculate_star_offset  (geometry)
- detect() recenter_for_lengths / max_center_offset
- spike_box_regions / spike_mask using corrected_centre automatically
- write_ds9_regions using corrected_centre
"""

import numpy as np
import pytest

from spikeout.geometry import calculate_star_offset
from spikeout.detect import detect
from spikeout.regions import spike_box_regions, spike_mask, write_ds9_regions


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_box(s):
    """Parse 'box(x,y,len,wid,angle)' → (x, y, length, width, angle)."""
    return tuple(float(v) for v in s[4:-1].split(","))


# Loose detect kwargs for the offset fixture (star is ~18 px off centre).
_OFFSET_KW = dict(min_snr=3.0, max_rho_fraction=0.3)


# ── calculate_star_offset ─────────────────────────────────────────────────────


class TestCalculateStarOffset:

    def test_exact_two_spikes(self):
        """With 2 non-parallel spikes the system is exactly determined."""
        dx_true, dy_true = 5.0, -3.0
        peaks = [
            (dx_true * np.cos(t) + dy_true * np.sin(t), t)
            for t in (np.deg2rad(30), np.deg2rad(120))
        ]
        dx, dy = calculate_star_offset(peaks)
        assert dx == pytest.approx(dx_true, abs=1e-9)
        assert dy == pytest.approx(dy_true, abs=1e-9)

    def test_overdetermined_four_spikes(self):
        """Least-squares over 4 spikes at different angles recovers exact offset."""
        dx_true, dy_true = -8.0, 4.0
        peaks = [
            (dx_true * np.cos(t) + dy_true * np.sin(t), t)
            for t in (np.deg2rad(th) for th in [10, 60, 100, 150])
        ]
        dx, dy = calculate_star_offset(peaks)
        assert dx == pytest.approx(dx_true, abs=1e-9)
        assert dy == pytest.approx(dy_true, abs=1e-9)

    def test_centred_star_gives_zero_offset(self):
        """rho = 0 for all angles → offset = (0, 0)."""
        peaks = [(0.0, np.deg2rad(th)) for th in [30, 90, 150]]
        dx, dy = calculate_star_offset(peaks)
        assert abs(dx) < 1e-10
        assert abs(dy) < 1e-10

    def test_returns_float_tuple(self):
        peaks = [(1.0, np.deg2rad(45)), (0.5, np.deg2rad(135))]
        dx, dy = calculate_star_offset(peaks)
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_noisy_overdetermined(self):
        """With mild noise on rho the recovery should still be close."""
        rng = np.random.default_rng(0)
        dx_true, dy_true = 7.0, -2.0
        thetas = np.deg2rad([20, 50, 80, 110, 140, 170])
        rho_true = dx_true * np.cos(thetas) + dy_true * np.sin(thetas)
        rho_noisy = rho_true + rng.normal(0, 0.1, len(thetas))
        peaks = list(zip(rho_noisy, thetas))
        dx, dy = calculate_star_offset(peaks)
        assert abs(dx - dx_true) < 1.0
        assert abs(dy - dy_true) < 1.0


# ── detect recentering fields ─────────────────────────────────────────────────


class TestDetectRecenterFields:

    def test_fields_none_when_disabled(self, star_offset):
        result = detect(star_offset, recenter_for_lengths=False, **_OFFSET_KW)
        assert result.star_centre_offset is None
        assert result.corrected_centre is None

    def test_fields_none_when_no_spikes(self):
        """High SNR threshold → 0 spikes → no offset computed."""
        img = np.random.default_rng(0).normal(100, 5, (64, 64))
        result = detect(img, recenter_for_lengths=True, min_snr=1000.0)
        assert result.star_centre_offset is None
        assert result.corrected_centre is None

    def test_star_centre_offset_populated(self, star_offset):
        result = detect(star_offset, recenter_for_lengths=True, **_OFFSET_KW)
        if len(result.angles) < 2:
            pytest.skip("fewer than 2 spikes detected")
        assert result.star_centre_offset is not None
        dx, dy = result.star_centre_offset
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_corrected_centre_populated(self, star_offset):
        result = detect(star_offset, recenter_for_lengths=True, **_OFFSET_KW)
        if len(result.angles) < 2:
            pytest.skip("fewer than 2 spikes detected")
        assert result.corrected_centre is not None
        assert len(result.corrected_centre) == 2

    def test_corrected_centre_consistent_with_offset(self, star_offset):
        """corrected_centre should equal (ny/2 - dy, nx/2 + dx)."""
        ny, nx = star_offset.shape
        result = detect(star_offset, recenter_for_lengths=True, **_OFFSET_KW)
        if result.star_centre_offset is None or result.corrected_centre is None:
            pytest.skip("offset not computed")
        dx, dy = result.star_centre_offset
        expected_row = ny / 2.0 - dy
        expected_col = nx / 2.0 + dx
        corr_row, corr_col = result.corrected_centre
        assert corr_row == pytest.approx(expected_row, abs=1e-9)
        assert corr_col == pytest.approx(expected_col, abs=1e-9)

    def test_max_center_offset_accepted(self, star_offset):
        """Generous threshold → corrected_centre is set."""
        result = detect(star_offset, recenter_for_lengths=True,
                        max_center_offset=100.0, **_OFFSET_KW)
        if result.star_centre_offset is not None:
            assert result.corrected_centre is not None

    def test_max_center_offset_rejected(self, star_offset):
        """Zero threshold → corrected_centre is None but offset still recorded."""
        result = detect(star_offset, recenter_for_lengths=True,
                        max_center_offset=0.001, **_OFFSET_KW)
        if result.star_centre_offset is not None:
            assert result.corrected_centre is None

    def test_offset_distance_within_threshold(self, star_offset):
        """Computed offset must be ≤ max_center_offset for centre to be set."""
        result = detect(star_offset, recenter_for_lengths=True,
                        max_center_offset=50.0, **_OFFSET_KW)
        if result.star_centre_offset is None:
            pytest.skip("offset not computed")
        dx, dy = result.star_centre_offset
        dist = np.hypot(dx, dy)
        if result.corrected_centre is not None:
            assert dist <= 50.0

    def test_repr_includes_offset(self, star_offset):
        result = detect(star_offset, recenter_for_lengths=True, **_OFFSET_KW)
        if result.star_centre_offset is not None:
            assert "star_offset" in repr(result)


# ── accuracy: corrected_centre closer to truth than image centre ──────────────


class TestRecenterAccuracy:

    def test_corrected_centre_closer_to_true_position(self, star_offset_with_centre):
        """Radon-derived centre must land closer to the true star than image centre."""
        img, true_col, true_row = star_offset_with_centre
        ny, nx = img.shape

        result = detect(img, recenter_for_lengths=True,
                        max_center_offset=50.0, **_OFFSET_KW)
        if result.corrected_centre is None:
            pytest.skip("no corrected_centre available")

        corr_row, corr_col = result.corrected_centre
        d_corrected = np.hypot(corr_col - true_col, corr_row - true_row)
        d_image_cen = np.hypot(nx / 2 - true_col, ny / 2 - true_row)
        assert d_corrected < d_image_cen


# ── lengths measured using the corrected centre ───────────────────────────────


class TestLengthsWithCorrectedCentre:

    def test_lengths_populated(self, star_offset):
        result = detect(star_offset, measure_lengths=True,
                        recenter_for_lengths=True, **_OFFSET_KW)
        assert result.lengths is not None
        assert len(result.lengths) == len(result.angles)

    def test_lengths_finite_and_positive(self, star_offset):
        result = detect(star_offset, measure_lengths=True,
                        recenter_for_lengths=True, **_OFFSET_KW)
        for sl in result.lengths:
            assert np.isfinite(sl.length_total)
            assert sl.length_total > 0

    def test_lengths_still_valid_for_centred_star(self, star_centred):
        """Recentering on a well-centred star should still produce valid lengths."""
        result = detect(star_centred, measure_lengths=True,
                        recenter_for_lengths=True, min_snr=3.0)
        assert result.lengths is not None
        assert len(result.lengths) > 0
        for sl in result.lengths:
            assert np.isfinite(sl.length_total)
            assert sl.length_total > 0


# ── regions: spike_box_regions uses corrected_centre automatically ────────────


class TestSpikeBoxRegionsUseCorrectedCentre:

    def _detect_plain_and_recentered(self, img):
        r_plain = detect(img, measure_lengths=True, **_OFFSET_KW)
        r_recen = detect(img, measure_lengths=True,
                         recenter_for_lengths=True, **_OFFSET_KW)
        return r_plain, r_recen

    def test_box_centre_shifts(self, star_offset):
        r_plain, r_recen = self._detect_plain_and_recentered(star_offset)
        if not r_recen.corrected_centre:
            pytest.skip("no corrected centre")
        if not r_plain.lengths or not r_recen.lengths:
            pytest.skip("no lengths")

        regs_plain = spike_box_regions(r_plain, star_offset.shape)
        regs_recen = spike_box_regions(r_recen, star_offset.shape)

        shifted = any(
            abs(_parse_box(rp)[0] - _parse_box(rr)[0]) > 1e-3 or
            abs(_parse_box(rp)[1] - _parse_box(rr)[1]) > 1e-3
            for rp, rr in zip(regs_plain, regs_recen)
        )
        assert shifted, "Box centres should differ when corrected_centre is set"

    def test_explicit_centre_overrides_corrected_centre(self, star_offset):
        r_recen = detect(star_offset, measure_lengths=True,
                         recenter_for_lengths=True, **_OFFSET_KW)
        if not r_recen.corrected_centre or not r_recen.lengths:
            pytest.skip("need corrected_centre and lengths")

        explicit = (10, 10)
        regs_explicit = spike_box_regions(r_recen, star_offset.shape,
                                          centre=explicit)
        regs_auto = spike_box_regions(r_recen, star_offset.shape)

        x_exp, y_exp = _parse_box(regs_explicit[0])[:2]
        x_auto, y_auto = _parse_box(regs_auto[0])[:2]
        assert abs(x_exp - x_auto) > 1.0 or abs(y_exp - y_auto) > 1.0


# ── regions: spike_mask uses corrected_centre automatically ──────────────────


class TestSpikeMaskUsesCorrectedCentre:

    def test_mask_differs_with_corrected_centre(self, star_offset):
        r_plain = detect(star_offset, measure_lengths=True, **_OFFSET_KW)
        r_recen = detect(star_offset, measure_lengths=True,
                         recenter_for_lengths=True, **_OFFSET_KW)
        if not r_recen.corrected_centre:
            pytest.skip("no corrected centre")
        if not r_plain.lengths or not r_recen.lengths:
            pytest.skip("no lengths")

        mask_plain = spike_mask(r_plain, star_offset.shape)
        mask_recen = spike_mask(r_recen, star_offset.shape)
        assert not np.array_equal(mask_plain, mask_recen)

    def test_explicit_centre_overrides_corrected_centre(self, star_offset):
        r_recen = detect(star_offset, measure_lengths=True,
                         recenter_for_lengths=True, **_OFFSET_KW)
        if not r_recen.corrected_centre or not r_recen.lengths:
            pytest.skip("need corrected_centre and lengths")

        mask_auto = spike_mask(r_recen, star_offset.shape)
        mask_explicit = spike_mask(r_recen, star_offset.shape, centre=(10, 10))
        assert not np.array_equal(mask_auto, mask_explicit)


# ── write_ds9_regions uses corrected_centre ───────────────────────────────────


class TestWriteDs9RegionsUsesCorrectedCentre:

    def test_file_box_differs_with_corrected_centre(self, star_offset, tmp_path):
        r_plain = detect(star_offset, measure_lengths=True, **_OFFSET_KW)
        r_recen = detect(star_offset, measure_lengths=True,
                         recenter_for_lengths=True, **_OFFSET_KW)
        if not r_recen.corrected_centre:
            pytest.skip("no corrected centre")
        if not r_plain.lengths or not r_recen.lengths:
            pytest.skip("no lengths")

        p_plain = tmp_path / "plain.reg"
        p_recen = tmp_path / "recen.reg"
        write_ds9_regions(p_plain, r_plain, star_offset.shape)
        write_ds9_regions(p_recen, r_recen, star_offset.shape)

        boxes_plain = [l for l in p_plain.read_text().splitlines()
                       if l.startswith("box(")]
        boxes_recen = [l for l in p_recen.read_text().splitlines()
                       if l.startswith("box(")]

        assert len(boxes_plain) == len(boxes_recen)
        shifted = any(
            abs(_parse_box(bp)[0] - _parse_box(br)[0]) > 1e-3 or
            abs(_parse_box(bp)[1] - _parse_box(br)[1]) > 1e-3
            for bp, br in zip(boxes_plain, boxes_recen)
        )
        assert shifted, "DS9 box coordinates should shift when corrected_centre is used"

    def test_explicit_centre_overrides_in_file(self, star_offset, tmp_path):
        r_recen = detect(star_offset, measure_lengths=True,
                         recenter_for_lengths=True, **_OFFSET_KW)
        if not r_recen.corrected_centre or not r_recen.lengths:
            pytest.skip("need corrected_centre and lengths")

        p_auto = tmp_path / "auto.reg"
        p_explicit = tmp_path / "explicit.reg"
        write_ds9_regions(p_auto, r_recen, star_offset.shape)
        write_ds9_regions(p_explicit, r_recen, star_offset.shape,
                          centre=(10, 10))

        auto_box = _parse_box(
            next(l for l in p_auto.read_text().splitlines()
                 if l.startswith("box("))
        )
        exp_box = _parse_box(
            next(l for l in p_explicit.read_text().splitlines()
                 if l.startswith("box("))
        )
        assert abs(auto_box[0] - exp_box[0]) > 1.0 or abs(auto_box[1] - exp_box[1]) > 1.0
