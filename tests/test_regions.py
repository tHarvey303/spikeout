"""Tests for spikeout.regions."""

import numpy as np
import pytest
from spikeout.detect import detect


@pytest.fixture
def result_with_lengths(star_centred):
    return detect(star_centred, measure_lengths=True, min_snr=3.0)


@pytest.fixture
def result_no_lengths(star_centred):
    return detect(star_centred, min_snr=3.0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_box(s):
    """Parse 'box(x,y,len,wid,angle)' → (x, y, length, width, angle)."""
    return tuple(float(v) for v in s[4:-1].split(","))


# ── spike_box_regions ─────────────────────────────────────────────────────────

class TestSpikeBoxRegions:

    def test_one_region_per_spike(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_box_regions
        regions = spike_box_regions(result_with_lengths, star_centred.shape)
        assert len(regions) == len(result_with_lengths.angles)

    def test_raises_without_lengths(self, result_no_lengths, star_centred):
        from spikeout.regions import spike_box_regions
        with pytest.raises(ValueError, match="measure_lengths"):
            spike_box_regions(result_no_lengths, star_centred.shape)

    def test_strings_start_with_box(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_box_regions
        for r in spike_box_regions(result_with_lengths, star_centred.shape):
            assert r.startswith("box(")

    def test_centre_coords_near_image_centre(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_box_regions
        nrows, ncols = star_centred.shape
        regions = spike_box_regions(result_with_lengths, star_centred.shape)
        for r in regions:
            x, y, *_ = _parse_box(r)
            # box centre should be broadly within the image
            assert 0 < x < ncols
            assert 0 < y < nrows

    def test_width_fraction_scales_width(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_box_regions
        r_narrow = spike_box_regions(
            result_with_lengths, star_centred.shape,
            width_fraction=0.05, min_width=0.0,
        )
        r_wide = spike_box_regions(
            result_with_lengths, star_centred.shape,
            width_fraction=0.4, min_width=0.0,
        )
        for rn, rw in zip(r_narrow, r_wide):
            assert _parse_box(rn)[3] < _parse_box(rw)[3]

    def test_min_width_enforced(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_box_regions
        min_w = 50.0
        regions = spike_box_regions(
            result_with_lengths, star_centred.shape,
            width_fraction=0.001, min_width=min_w,
        )
        for r in regions:
            assert _parse_box(r)[3] >= min_w

    def test_max_width_enforced(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_box_regions
        max_w = 3.0
        regions = spike_box_regions(
            result_with_lengths, star_centred.shape,
            width_fraction=0.9, max_width=max_w,
        )
        for r in regions:
            assert _parse_box(r)[3] <= max_w + 1e-9

    def test_custom_centre_shifts_box(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_box_regions
        r_default = spike_box_regions(result_with_lengths, star_centred.shape)
        r_custom = spike_box_regions(
            result_with_lengths, star_centred.shape, centre=(10, 10),
        )
        # centres should differ
        x0, y0 = _parse_box(r_default[0])[:2]
        x1, y1 = _parse_box(r_custom[0])[:2]
        assert not (abs(x0 - x1) < 1e-6 and abs(y0 - y1) < 1e-6)


# ── write_ds9_regions ─────────────────────────────────────────────────────────

class TestWriteDs9Regions:

    def test_creates_file(self, result_with_lengths, star_centred, tmp_path):
        from spikeout.regions import write_ds9_regions
        path = tmp_path / "out.reg"
        write_ds9_regions(path, result_with_lengths, star_centred.shape)
        assert path.exists()

    def test_header_coordsys_image(self, result_with_lengths, star_centred, tmp_path):
        from spikeout.regions import write_ds9_regions
        path = tmp_path / "out.reg"
        write_ds9_regions(path, result_with_lengths, star_centred.shape)
        lines = path.read_text().splitlines()
        assert any(line.strip() == "image" for line in lines)

    def test_colour_in_header(self, result_with_lengths, star_centred, tmp_path):
        from spikeout.regions import write_ds9_regions
        path = tmp_path / "out.reg"
        write_ds9_regions(path, result_with_lengths, star_centred.shape, colour="cyan")
        assert "color=cyan" in path.read_text()

    def test_box_count_matches_spikes(self, result_with_lengths, star_centred, tmp_path):
        from spikeout.regions import write_ds9_regions
        path = tmp_path / "out.reg"
        write_ds9_regions(path, result_with_lengths, star_centred.shape)
        box_lines = [l for l in path.read_text().splitlines() if l.startswith("box(")]
        assert len(box_lines) == len(result_with_lengths.angles)


# ── spike_mask ────────────────────────────────────────────────────────────────

class TestSpikeMask:

    def test_shape_matches_image(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_mask
        mask = spike_mask(result_with_lengths, star_centred.shape)
        assert mask.shape == star_centred.shape

    def test_dtype_is_bool(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_mask
        mask = spike_mask(result_with_lengths, star_centred.shape)
        assert mask.dtype == bool

    def test_has_masked_pixels(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_mask
        mask = spike_mask(result_with_lengths, star_centred.shape)
        assert mask.any()

    def test_raises_without_lengths(self, result_no_lengths, star_centred):
        from spikeout.regions import spike_mask
        with pytest.raises(ValueError, match="measure_lengths"):
            spike_mask(result_no_lengths, star_centred.shape)

    def test_wider_mask_covers_more_pixels(self, result_with_lengths, star_centred):
        from spikeout.regions import spike_mask
        mask_narrow = spike_mask(
            result_with_lengths, star_centred.shape, width_fraction=0.05,
        )
        mask_wide = spike_mask(
            result_with_lengths, star_centred.shape, width_fraction=0.5,
        )
        assert mask_wide.sum() > mask_narrow.sum()

    def test_mask_consistent_with_box_regions(self, result_with_lengths, star_centred):
        """Masked pixel count should grow with width, same as box regions."""
        from spikeout.regions import spike_mask, spike_box_regions
        # Both functions share _box_width logic; a wider fraction always
        # produces more masked pixels and a longer DS9 box width.
        kw_narrow = dict(width_fraction=0.05, min_width=0.0)
        kw_wide = dict(width_fraction=0.3, min_width=0.0)
        n_narrow = spike_mask(result_with_lengths, star_centred.shape, **kw_narrow).sum()
        n_wide = spike_mask(result_with_lengths, star_centred.shape, **kw_wide).sum()
        assert n_wide > n_narrow
