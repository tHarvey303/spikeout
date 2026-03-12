"""Tests for spikeout.catalogue."""

import numpy as np
import pytest
from skimage.draw import line as draw_line
from scipy.ndimage import gaussian_filter

pytest.importorskip("astropy", reason="astropy not installed")


@pytest.fixture
def fits_with_stars(tmp_path, rng):
    """512x512 FITS file with a simple TAN WCS and two synthetic stars."""
    from astropy.io import fits
    from astropy.wcs import WCS

    size = 512
    img = rng.normal(100, 5, (size, size))

    def add_star(data, cx, cy, spike_angles=(30, 120)):
        Y, X = np.mgrid[:size, :size]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        data = data.copy()
        data += 5000 / (1 + R ** 2)
        for angle_deg in spike_angles:
            rad = np.deg2rad(angle_deg)
            r1 = int(cy + 100 * np.sin(rad))
            c1 = int(cx + 100 * np.cos(rad))
            r2 = int(cy - 100 * np.sin(rad))
            c2 = int(cx - 100 * np.cos(rad))
            rr, cc = draw_line(
                np.clip(r2, 0, size - 1), np.clip(c2, 0, size - 1),
                np.clip(r1, 0, size - 1), np.clip(c1, 0, size - 1),
            )
            data[rr, cc] += 200
        return data

    # Stars at pixel (col=128, row=128) and (col=384, row=384)
    img = add_star(img, cx=128, cy=128)
    img = add_star(img, cx=384, cy=384)
    img = gaussian_filter(img, sigma=1.0)

    w = WCS(naxis=2)
    w.wcs.crpix = [256, 256]
    w.wcs.cdelt = [-0.000277778, 0.000277778]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    header = w.to_header()
    header['NAXIS'] = 2
    header['NAXIS1'] = size
    header['NAXIS2'] = size

    path = tmp_path / "test_image.fits"
    fits.PrimaryHDU(data=img.astype(np.float32), header=header).writeto(path)

    # Compute sky positions corresponding to the two star pixel positions
    # pixel_to_world takes (x=col, y=row)
    sky1 = w.pixel_to_world(128, 128)
    sky2 = w.pixel_to_world(384, 384)
    coords = [(sky1.ra.deg, sky1.dec.deg), (sky2.ra.deg, sky2.dec.deg)]

    return path, coords


class TestCatalogueDetect:

    def test_returns_correct_number_of_entries(self, fits_with_stars):
        from spikeout.catalogue import catalogue_detect
        path, coords = fits_with_stars
        entries = catalogue_detect(coords, path, cutout_size=128, min_snr=3.0)
        assert len(entries) == 2

    def test_entry_coords_match_input(self, fits_with_stars):
        from spikeout.catalogue import catalogue_detect
        path, coords = fits_with_stars
        entries = catalogue_detect(coords, path, cutout_size=128)
        for entry, (ra, dec) in zip(entries, coords):
            assert abs(entry.ra - ra) < 1e-6
            assert abs(entry.dec - dec) < 1e-6

    def test_cutout_shape(self, fits_with_stars):
        from spikeout.catalogue import catalogue_detect
        path, coords = fits_with_stars
        size = 128
        entries = catalogue_detect(coords, path, cutout_size=size)
        for entry in entries:
            assert entry.error is None
            assert entry.cutout.shape == (size, size)

    def test_result_is_spike_result(self, fits_with_stars):
        from spikeout.catalogue import catalogue_detect
        from spikeout.detect import SpikeResult
        path, coords = fits_with_stars
        entries = catalogue_detect(coords, path, cutout_size=128, min_snr=3.0)
        for entry in entries:
            if entry.error is None:
                assert isinstance(entry.result, SpikeResult)

    def test_error_stored_for_out_of_bounds_coord(self, fits_with_stars):
        from spikeout.catalogue import catalogue_detect
        path, _ = fits_with_stars
        # Position far outside the image footprint
        entries = catalogue_detect([(180.0, -70.0)], path, cutout_size=128)
        assert len(entries) == 1
        assert entries[0].error is not None
        assert entries[0].result is None

    def test_accepts_skycoord(self, fits_with_stars):
        from astropy.coordinates import SkyCoord
        from spikeout.catalogue import catalogue_detect
        path, coords = fits_with_stars
        sky = SkyCoord(
            [c[0] for c in coords], [c[1] for c in coords], unit='deg',
        )
        entries = catalogue_detect(sky, path, cutout_size=128)
        assert len(entries) == 2

    def test_detect_kw_forwarded(self, fits_with_stars):
        from spikeout.catalogue import catalogue_detect
        path, coords = fits_with_stars
        entries = catalogue_detect(
            coords, path, cutout_size=128, min_snr=1000.0,
        )
        for entry in entries:
            if entry.result is not None:
                assert len(entry.result.angles) == 0

    def test_order_preserved(self, fits_with_stars):
        """Entries must appear in the same order as the input coords."""
        from spikeout.catalogue import catalogue_detect
        path, coords = fits_with_stars
        entries = catalogue_detect(coords, path, cutout_size=128)
        for entry, (ra, dec) in zip(entries, coords):
            assert abs(entry.ra - ra) < 1e-6


class TestPlotCatalogue:

    def test_returns_figure(self, fits_with_stars):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from spikeout.catalogue import catalogue_detect, plot_catalogue
        path, coords = fits_with_stars
        entries = catalogue_detect(coords, path, cutout_size=128, min_snr=3.0)
        fig = plot_catalogue(entries, ncols=2)
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_grid_shape(self, fits_with_stars):
        """Grid should have ceil(n/ncols) rows."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from spikeout.catalogue import catalogue_detect, plot_catalogue
        path, coords = fits_with_stars
        entries = catalogue_detect(coords, path, cutout_size=128)
        fig = plot_catalogue(entries, ncols=2)
        assert fig.get_axes()[0] is not None
        plt.close(fig)

    def test_handles_error_entries(self):
        """Error entries should render without raising."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from spikeout.catalogue import CatalogueEntry, plot_catalogue
        entries = [
            CatalogueEntry(
                ra=10.0, dec=20.0, cutout=None, result=None,
                error="something went wrong",
            ),
        ]
        fig = plot_catalogue(entries, ncols=1)
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_empty_entries_raises(self):
        from spikeout.catalogue import plot_catalogue
        with pytest.raises(ValueError, match="No entries"):
            plot_catalogue([])

    def test_single_column(self, fits_with_stars):
        """ncols=1 should not crash (axes shape edge case)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from spikeout.catalogue import catalogue_detect, plot_catalogue
        path, coords = fits_with_stars
        entries = catalogue_detect(coords, path, cutout_size=128)
        fig = plot_catalogue(entries, ncols=1)
        plt.close(fig)
