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


# ── Synthetic spike-line fixtures ─────────────────────────────────────────────
#
# These images contain drawn spike lines (no PSF) and are intended for
# precise angle-recovery tests.  Morphological opening with a large
# structuring element would erase thin Gaussian-broadened lines, so tests
# that use these fixtures call detect() with morph_radius=0.


def _make_spike_image(
    rng,
    size=256,
    spike_angles=(30, 120),
    spike_brightness=100,
    noise_sigma=2.0,
    spike_half_len=110,
    blur_sigma=2.0,
    saturated_core_half=0,
    neighbour=None,
):
    """Synthetic image with drawn spike lines at known angles.

    Parameters
    ----------
    spike_angles : sequence of float
        Angles in degrees, measured CCW from +col axis (array frame).
    saturated_core_half : int
        If > 0, zero out a square of side ``2*saturated_core_half`` at
        the image centre to simulate a saturated/masked core.
    neighbour : ((row, col), peak_brightness, sigma) or None
        Adds a compact Gaussian source.  Using a Gaussian (rather than a
        flat circle) ensures the neighbour's Radon contribution at ρ ≈ 0
        is negligible even when the source is near the image edge — a flat
        filled circle can produce a large chord at ρ ≈ 0 that dominates
        the sinogram threshold and suppresses real spike detections.
    blur_sigma : float
        Gaussian blur applied after drawing; broadens spikes so they
        survive the sigma-clip preprocessing step.
    """
    img = rng.normal(0, noise_sigma, (size, size)).astype(float)
    cx, cy = size // 2, size // 2

    for angle_deg in spike_angles:
        rad = np.deg2rad(angle_deg)
        r1 = int(cy + spike_half_len * np.sin(rad))
        c1 = int(cx + spike_half_len * np.cos(rad))
        r2 = int(cy - spike_half_len * np.sin(rad))
        c2 = int(cx - spike_half_len * np.cos(rad))
        rr, cc = draw_line(
            np.clip(r2, 0, size - 1), np.clip(c2, 0, size - 1),
            np.clip(r1, 0, size - 1), np.clip(c1, 0, size - 1),
        )
        img[rr, cc] += spike_brightness

    img = gaussian_filter(img, sigma=blur_sigma)

    if saturated_core_half > 0:
        n = saturated_core_half
        img[cy - n:cy + n, cx - n:cx + n] = 0

    if neighbour is not None:
        (nr, nc), nbr_peak, nbr_sigma = neighbour
        Y, X = np.ogrid[:size, :size]
        img += nbr_peak * np.exp(
            -((X - nc) ** 2 + (Y - nr) ** 2) / (2 * nbr_sigma ** 2)
        )

    return img


@pytest.fixture
def spike_clean(rng):
    """256×256 spikes at 30° and 120°, no saturation, no neighbour."""
    return _make_spike_image(rng, spike_angles=(30, 120))


@pytest.fixture
def spike_saturated_core(rng):
    """256×256 spikes at 30° and 120° with a zeroed 50-px core."""
    return _make_spike_image(
        rng, spike_angles=(30, 120), saturated_core_half=25,
    )


@pytest.fixture
def spike_nan_core(rng):
    """256×256 spikes at 30° and 120° with a NaN core."""
    img = _make_spike_image(rng, spike_angles=(30, 120))
    cx, cy = 128, 128
    img[cy - 25:cy + 25, cx - 25:cx + 25] = np.nan
    return img


@pytest.fixture
def spike_bright_neighbour(rng):
    """256×256 spikes at 30° and 120° plus a bright Gaussian source near edge.

    The neighbour uses a compact Gaussian (not a flat circle) so that its
    Radon projection at ρ ≈ 0 (central band) is negligible.
    """
    return _make_spike_image(
        rng,
        spike_angles=(30, 120),
        neighbour=((200, 50), 2000, 5),   # (row, col), peak, sigma
    )


@pytest.fixture
def spike_multi_angle_full(rng):
    """256×256 spikes at 30°, 120°, 140°, 170° with saturated core + neighbour.

    Mirrors the user's full scenario: close angle pairs, saturated core,
    and an off-edge bright source that should not create false detections.
    """
    return _make_spike_image(
        rng,
        spike_angles=(30, 120, 140, 170),
        saturated_core_half=25,
        neighbour=((200, 50), 2000, 5),   # (row, col), peak, sigma
    )
