"""Tests for spikeout.stars._proximity_filter."""

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.table import Table


def _make_table(ra, dec, mags, mag_col="phot_g_mean_mag"):
    """Build a minimal catalogue table as expected by _proximity_filter."""
    return Table({
        "ra_epoch": np.array(ra, dtype=float),
        "dec_epoch": np.array(dec, dtype=float),
        mag_col: np.array(mags, dtype=float),
    })


# ── _proximity_filter ─────────────────────────────────────────────────────────

class TestProximityFilter:

    from spikeout.stars import _proximity_filter as _pf

    def test_empty_table_unchanged(self):
        from spikeout.stars import _proximity_filter
        t = _make_table([], [], [])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert len(out) == 0
        assert n == 0

    def test_single_source_unchanged(self):
        from spikeout.stars import _proximity_filter
        t = _make_table([10.0], [20.0], [15.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert len(out) == 1
        assert n == 0

    def test_well_separated_pair_unchanged(self):
        """Two sources >5 arcsec apart should both survive."""
        from spikeout.stars import _proximity_filter
        # 0.1 deg ≈ 360 arcsec apart
        t = _make_table([10.0, 10.1], [20.0, 20.0], [14.0, 16.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert len(out) == 2
        assert n == 0

    def test_close_pair_keeps_brighter(self):
        """Two sources within threshold: fainter (higher mag) is removed."""
        from spikeout.stars import _proximity_filter
        # 0.001 deg ≈ 3.6 arcsec; threshold 5 arcsec → they are close
        t = _make_table([10.0, 10.001], [20.0, 20.0], [14.0, 16.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert n == 1
        assert len(out) == 1
        assert float(out["phot_g_mean_mag"][0]) == pytest.approx(14.0)

    def test_close_pair_keeps_brighter_reversed(self):
        """Same as above but fainter source comes first in the table."""
        from spikeout.stars import _proximity_filter
        t = _make_table([10.0, 10.001], [20.0, 20.0], [16.0, 14.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert n == 1
        assert len(out) == 1
        assert float(out["phot_g_mean_mag"][0]) == pytest.approx(14.0)

    def test_three_close_sources_keeps_brightest(self):
        """Three sources all within threshold of each other → only brightest kept."""
        from spikeout.stars import _proximity_filter
        # Place all three within ~2 arcsec of each other
        eps = 0.0005  # ~1.8 arcsec
        t = _make_table(
            [10.0, 10.0 + eps, 10.0 + 2 * eps],
            [20.0, 20.0, 20.0],
            [12.0, 14.0, 16.0],  # brightest first
        )
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=10.0)
        mags_out = sorted(float(m) for m in out["phot_g_mean_mag"])
        assert mags_out[0] == pytest.approx(12.0)

    def test_distant_sources_not_affected_by_close_pair(self):
        """A close pair plus a well-separated bright source: only faint of pair removed."""
        from spikeout.stars import _proximity_filter
        # sources 0 and 1 are close; source 2 is far away
        t = _make_table(
            [10.0, 10.001, 15.0],
            [20.0, 20.0,   20.0],
            [14.0, 16.0,   13.0],
        )
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert n == 1  # only the faint close neighbour removed
        assert len(out) == 2
        mags_out = sorted(float(m) for m in out["phot_g_mean_mag"])
        assert mags_out == pytest.approx([13.0, 14.0])

    def test_nan_mag_treated_as_faintest(self):
        """A source with NaN magnitude loses to any source with a measured mag."""
        from spikeout.stars import _proximity_filter
        t = _make_table([10.0, 10.001], [20.0, 20.0], [np.nan, 15.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert n == 1
        assert len(out) == 1
        assert np.isfinite(float(out["phot_g_mean_mag"][0]))

    def test_equal_mags_removes_second(self):
        """For equal magnitudes the second source (higher index) is removed."""
        from spikeout.stars import _proximity_filter
        t = _make_table([10.0, 10.001], [20.0, 20.0], [15.0, 15.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert n == 1
        assert len(out) == 1

    def test_threshold_boundary_just_inside(self):
        """Sources just inside the threshold are filtered."""
        from spikeout.stars import _proximity_filter
        # 4 arcsec apart; threshold 5 arcsec
        sep_deg = 4.0 / 3600.0
        t = _make_table([10.0, 10.0 + sep_deg], [20.0, 20.0], [14.0, 16.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert n == 1

    def test_threshold_boundary_just_outside(self):
        """Sources just outside the threshold are not filtered."""
        from spikeout.stars import _proximity_filter
        # 6 arcsec apart; threshold 5 arcsec
        sep_deg = 6.0 / 3600.0
        t = _make_table([10.0, 10.0 + sep_deg], [20.0, 20.0], [14.0, 16.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=5.0)
        assert n == 0
        assert len(out) == 2

    def test_returns_count_equals_removed(self):
        """Returned n_removed matches the difference in table length."""
        from spikeout.stars import _proximity_filter
        t = _make_table([10.0, 10.001, 10.002], [20.0, 20.0, 20.0],
                        [12.0, 14.0, 16.0])
        out, n = _proximity_filter(t, "phot_g_mean_mag", sep_arcsec=10.0)
        assert n == len(t) - len(out)

    def test_custom_mag_col(self):
        """Works with a non-default magnitude column name."""
        from spikeout.stars import _proximity_filter
        t = Table({
            "ra_epoch": [10.0, 10.001],
            "dec_epoch": [20.0, 20.0],
            "my_mag": [14.0, 16.0],
        })
        out, n = _proximity_filter(t, "my_mag", sep_arcsec=5.0)
        assert n == 1
        assert float(out["my_mag"][0]) == pytest.approx(14.0)
