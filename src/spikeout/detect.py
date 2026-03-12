"""Core diffraction-spike detection via the Radon transform."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from .stats import mad_std
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks
from skimage.transform import radon

from .geometry import sinogram_rho_to_physical, radon_line_to_image
from .preprocess import prepare_image
from .lengths import SpikeLengths, measure_spike_lengths

__all__ = ["SpikeResult", "detect"]


@dataclass
class SpikeResult:
    """Container for spike-detection output.

    Attributes
    ----------
    angles : ndarray
        Image-plane angles of detected spikes (degrees, display frame).
    rho_physical : ndarray
        Signed perpendicular distance (pixels) of each line from the
        image centre.
    snr : ndarray
        Signal-to-noise ratio of each peak in the angular profile.
    sinogram : ndarray
        Full Radon sinogram.
    theta : ndarray
        Projection-angle grid (degrees, 0–180).
    peak_rho_indices : ndarray
        Sinogram row indices of accepted peaks.
    peak_theta_indices : ndarray
        Sinogram column indices of accepted peaks.
    prepared_image : ndarray
        The preprocessed image that was fed to the Radon transform.
    n_rejected_snr : int
        Number of peaks rejected by the SNR significance filter.
    lengths : list of SpikeLengths or None
        Spike-arm lengths (populated when ``measure_lengths=True``).
    """
    angles: np.ndarray
    rho_physical: np.ndarray
    snr: np.ndarray
    sinogram: np.ndarray
    theta: np.ndarray
    peak_rho_indices: np.ndarray
    peak_theta_indices: np.ndarray
    prepared_image: np.ndarray
    n_rejected_snr: int = 0
    lengths: Optional[List[SpikeLengths]] = None


def _angular_profile_snr(sinogram, peak_theta_indices):
    """Compute the SNR of each peak in the max-over-ρ angular profile.

    SNR = (peak_value − median) / σ_MAD  of the profile.

    This measures how prominent each spike is relative to the overall
    azimuthal structure.  Stars without real spikes (or extended galaxies)
    produce a flat profile where no peak is significant.
    """
    max_profile = np.max(sinogram, axis=0)
    median = np.median(max_profile)
    sigma = mad_std(max_profile)

    if sigma <= 0:
        return np.full(len(peak_theta_indices), 0.0)

    return (max_profile[peak_theta_indices] - median) / sigma


def detect(
    image,
    # ── Radon grid ──
    angular_resolution=None,
    # ── peak finding ──
    peak_prominence=0.3,
    min_peak_separation_deg=5.0,
    local_max_window=(7, 5),
    # ── quality filters ──
    max_rho_fraction=0.1,
    min_snr=5.0,
    # ── optional length measurement ──
    measure_lengths=False,
    length_kw=None,
    # ── preprocessing ──
    **prep_kw,
):
    """Detect diffraction spikes in an image.

    Parameters
    ----------
    image : 2-D array
        Input image (NaN-safe).

    angular_resolution : float or *None*
        Angular step in degrees.  Default: ``180 / max(image.shape)``.

    peak_prominence : float
        Minimum peak height as a fraction of the sinogram maximum.
        Controls the initial candidate selection.

    min_peak_separation_deg : float
        Minimum angular separation (degrees) between reported peaks.

    local_max_window : (int, int)
        ``(rho, theta)`` window for the 2-D local-maximum filter.

    max_rho_fraction : float
        Maximum ``|ρ|`` as a fraction of half the smallest image
        dimension that a line may have to be considered.  Diffraction
        spikes pass through the image centre, so the sinogram is
        restricted to this central ρ band *before* peak detection —
        this prevents bright off-axis sources from winning the
        per-angle competition and masking weaker on-axis spikes.

        Typical values::

            0.05 — strict, star must be well centred
            0.10 — default, allows slight miscentering
            0.20 — lenient, for poor centring or small cutouts
            1.00 — no restriction

    min_snr : float
        Minimum signal-to-noise ratio for a peak in the angular profile
        (max-over-ρ) to be accepted.  Measured as
        ``(peak − median) / σ_MAD`` of the profile.

        This is the primary guard against false detections in images
        where spikes are absent (faint stars, extended galaxies, noise).

        Typical values::

            3.0  — lenient, accepts marginal spikes
            5.0  — default, good balance
            8.0  — conservative, only strong spikes
            0.0  — disable significance filtering

    measure_lengths : bool
        If *True*, measure each spike arm's length via swath profiles.

    length_kw : dict or *None*
        Extra keyword arguments forwarded to
        `~spikeout.lengths.measure_spike_lengths`.

    **prep_kw
        Extra keyword arguments forwarded to
        `~spikeout.preprocess.prepare_image` (``centre`` —
        ``'center'`` | ``'auto'`` | ``(row, col)``,
        ``radial_bin_width``, ``morph_radius``, ``sigma_clip``,
        ``asinh_stretch``).

    Returns
    -------
    SpikeResult
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")

    # ── early validation ─────────────────────────────────────────────────
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Image is entirely NaN / Inf")
    if np.ptp(finite) == 0:
        raise ValueError("Image has zero variance (constant or trivial)")

    # ── preprocessing ────────────────────────────────────────────────────
    prepared = prepare_image(image, **prep_kw)

    # ── Radon transform ──────────────────────────────────────────────────
    if angular_resolution is None:
        n_angles = max(prepared.shape)
    else:
        n_angles = int(np.ceil(180.0 / angular_resolution))

    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = radon(prepared, theta=theta, circle=True)
    n_rho = sinogram.shape[0]

    # ── centrality mask (applied during detection, not post-hoc) ─────────
    # Diffraction spikes pass through the image centre, so we restrict
    # the sinogram to the central ρ band before peak detection.  This
    # prevents bright off-axis peaks from winning the per-angle competition
    # and then being discarded, which would mask weaker on-axis spikes.
    max_rho_px = max_rho_fraction * min(image.shape) / 2.0
    all_rho_phys = sinogram_rho_to_physical(np.arange(n_rho), n_rho)
    rho_central = np.abs(all_rho_phys) <= max_rho_px  # shape (n_rho,)
    sinogram_central = sinogram * rho_central[:, np.newaxis]

    # ── 2-D peak detection ───────────────────────────────────────────────
    abs_threshold = peak_prominence * np.max(sinogram_central)
    local_max = maximum_filter(sinogram, size=local_max_window)
    peak_map = (sinogram == local_max) & (sinogram > abs_threshold)

    max_along_rho = np.max(sinogram_central * peak_map, axis=0)

    min_sep_idx = max(1, int(np.round(
        min_peak_separation_deg / (180.0 / n_angles),
    )))
    peaks_1d, _ = find_peaks(
        max_along_rho,
        height=abs_threshold,
        distance=min_sep_idx,
    )

    peak_rho_idx = np.array(
        [np.argmax(sinogram_central[:, ti]) for ti in peaks_1d],
    )
    rho_phys = sinogram_rho_to_physical(peak_rho_idx, n_rho).astype(float)

    # ── filter: significance (SNR) ────────────────────────────────────────
    if len(peaks_1d) > 0 and min_snr > 0:
        snr_values = _angular_profile_snr(sinogram, peaks_1d)
        snr_mask = snr_values >= min_snr
        n_rejected_snr = int(np.sum(~snr_mask))

        peaks_1d = peaks_1d[snr_mask]
        peak_rho_idx = peak_rho_idx[snr_mask]
        rho_phys = rho_phys[snr_mask]
        snr_values = snr_values[snr_mask]
    else:
        snr_values = _angular_profile_snr(sinogram, peaks_1d) \
            if len(peaks_1d) > 0 else np.array([])
        n_rejected_snr = 0

    # ── display angles ───────────────────────────────────────────────────
    if len(peaks_1d) > 0:
        image_angles = np.array([
            radon_line_to_image(
                rho_phys[i], theta[peaks_1d[i]], image.shape,
            )[2]
            for i in range(len(peaks_1d))
        ])
    else:
        image_angles = np.array([])

    result = SpikeResult(
        angles=image_angles,
        rho_physical=rho_phys,
        snr=snr_values,
        sinogram=sinogram,
        theta=theta,
        peak_rho_indices=peak_rho_idx,
        peak_theta_indices=peaks_1d,
        prepared_image=prepared,
        n_rejected_snr=n_rejected_snr,
    )

    # ── optional length measurement ──────────────────────────────────────
    if measure_lengths and len(peaks_1d) > 0:
        kw = length_kw or {}
        result.lengths = measure_spike_lengths(image, result, **kw)

    return result
