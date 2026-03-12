"""Shared fixtures for spikeout tests."""

import numpy as np
import pytest
from skimage.draw import line as draw_line
from scipy.ndimage import gaussian_filter


@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def pure_noise(rng):
    """128×128 Gaussian noise image with no structure."""
    return rng.normal(100, 5, (128, 128))


@pytest.fixture
def smooth_galaxy(rng):
    """128×128 image with a smooth elliptical galaxy (no spikes)."""
    img = rng.normal(100, 5, (128, 128))
    Y, X = np.mgrid[:128, :128]
    img += 500 * np.exp(-((X - 64) ** 2 / (2 * 20 ** 2)
                          + (Y - 64) ** 2 / (2 * 10 ** 2)))
    return img


def _make_star_with_spikes(
    rng, size=256, spike_angles=(30, 120), centre_offset=(5, -5),
    spike_brightness=200, psf_peak=5000, spike_half_len=100,
    asymmetric=False,
):
    """Helper to build a synthetic star + diffraction spikes."""
    cx = size // 2 + centre_offset[0]
    cy = size // 2 + centre_offset[1]

    img = rng.normal(100, 5, (size, size))

    # Moffat-like PSF core + halo
    Y, X = np.mgrid[:size, :size]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    img += psf_peak / (1 + R ** 2)

    # Diffraction spikes
    for angle_deg in spike_angles:
        rad = np.deg2rad(angle_deg)
        pos_len = spike_half_len
        neg_len = int(spike_half_len * 0.6) if asymmetric else spike_half_len

        r1 = int(cy + pos_len * np.sin(rad))
        c1 = int(cx + pos_len * np.cos(rad))
        r2 = int(cy - neg_len * np.sin(rad))
        c2 = int(cx - neg_len * np.cos(rad))

        rr, cc = draw_line(
            np.clip(r2, 0, size - 1), np.clip(c2, 0, size - 1),
            np.clip(r1, 0, size - 1), np.clip(c1, 0, size - 1),
        )
        img[rr, cc] += spike_brightness

    img = gaussian_filter(img, sigma=1.0)
    return img, cx, cy


@pytest.fixture
def star_centred(rng):
    """256×256 star at image centre with two spikes at 30° and 120°."""
    img, cx, cy = _make_star_with_spikes(rng, centre_offset=(0, 0))
    return img


@pytest.fixture
def star_offset(rng):
    """256×256 star slightly off-centre with two spikes at 30° and 120°."""
    img, cx, cy = _make_star_with_spikes(rng, centre_offset=(15, -10))
    return img


@pytest.fixture
def star_asymmetric(rng):
    """256×256 star with asymmetric spike arm lengths."""
    img, cx, cy = _make_star_with_spikes(rng, asymmetric=True)
    return img


@pytest.fixture
def star_four_spikes(rng):
    """256×256 star with four spikes (typical cross pattern)."""
    img, cx, cy = _make_star_with_spikes(
        rng, spike_angles=(0, 45, 90, 135),
    )
    return img


@pytest.fixture
def star_with_neighbour(rng):
    """256×256 star with spikes + a compact neighbouring source."""
    img, cx, cy = _make_star_with_spikes(rng)
    Y, X = np.mgrid[:256, :256]
    img += 800 * np.exp(-((X - 200) ** 2 + (Y - 50) ** 2) / (2 * 4 ** 2))
    return img


@pytest.fixture
def constant_image():
    """Constant image (zero variance)."""
    return np.full((64, 64), 100.0)


@pytest.fixture
def all_nan_image():
    """All-NaN image."""
    return np.full((64, 64), np.nan)
