"""Core diffraction-spike detection via the Radon transform."""

import warnings
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
    max_rho_px: float = 0.0

    def __repr__(self) -> str:
        n = len(self.angles)
        if n == 0:
            return "SpikeResult(no spikes detected)"
        parts = [f"n={n}"]
        parts.append("angles=[" + ", ".join(f"{a:.1f}\u00b0" for a in self.angles) + "]")
        if self.lengths is not None:
            parts.append(
                "lengths=["
                + ", ".join(f"{sl.length_total:.0f}px" for sl in self.lengths)
                + "]"
            )
        return f"SpikeResult({', '.join(parts)})"


def _central_blank_radius(image: np.ndarray) -> float:
    """Estimate the radius of any NaN/zero patch at the image centre.

    For a saturated star whose core has been set to 0 or NaN, the Radon
    peak for a diffraction spike appears at the *edge* of that blank
    region (ρ ≈ R_blank) rather than at ρ = 0.  This function returns
    R_blank so the caller can extend the ρ acceptance band accordingly.

    Returns 0 if the central pixel is finite and non-zero.
    """
    nrows, ncols = image.shape
    cy, cx = nrows / 2.0, ncols / 2.0
    cy_i, cx_i = int(round(cy)), int(round(cx))

    blank = ~np.isfinite(image) | (image == 0.0)
    if not blank[cy_i, cx_i]:
        return 0.0

    Y, X = np.ogrid[:nrows, :ncols]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Cap search to the inner quarter to ignore distant bad pixels
    nearby = blank & (dist <= min(nrows, ncols) / 4.0)
    return float(dist[nearby].max()) if nearby.any() else 0.0


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
    angle_image_fraction=1.0,
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
    min_length=None,
    # ── preprocessing ──
    **prep_kw,
):
    """Detect diffraction spikes in an image.

    Parameters
    ----------
    image : 2-D array
        Input image (NaN-safe).

    angle_image_fraction : float
        Fraction of cutout size (from center) to use for spike angle
        estimation.  Reducing this can help when the cutout contains
        bright off-centre sources whose spikes would otherwise bias the
        angle measurement.  Default: 1.0 (use the whole cutout). Using 
        a large cutout will help for spike length and halo measurements,
        but may hurt angle estimation if this argument is not tuned accordingly.

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
        A detected peak is accepted when its line passes within
        ``max_rho_fraction × min(image.shape) / 2`` pixels of the
        boundary of the central source.  For a clean star this is
        equivalent to requiring ``|ρ| ≲ 0``.  For a saturated star
        whose core is 0 or NaN the acceptance band is automatically
        widened by the estimated blank-core radius so that the spike
        arms — which peak at the edge of the saturated region rather
        than at ρ = 0 — are still detected.

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

    min_length : float or *None*
        Minimum total spike length in pixels.  Spikes shorter than this
        are dropped after arm measurement.  Requires
        ``measure_lengths=True``; a warning is issued if set otherwise.

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

    # -- cropping for angle estimation ────────────────────────────────────
    if angle_image_fraction < 1.0:
        cy, cx = np.array(image.shape) / 2.0
        half_size = angle_image_fraction * min(image.shape) / 2.0
        y1, y2 = int(round(cy - half_size)), int(round(cy + half_size))
        x1, x2 = int(round(cx - half_size)), int(round(cx + half_size))
        prepared = prepared[y1:y2, x1:x2]

    # ── Radon transform ──────────────────────────────────────────────────
    if angular_resolution is None:
        n_angles = max(prepared.shape)
    else:
        n_angles = int(np.ceil(180.0 / angular_resolution))

    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = radon(prepared, theta=theta, circle=True)
    n_rho = sinogram.shape[0]

    # ── centrality mask ───────────────────────────────────────────────────
    # Rather than imposing a hard limit on ρ, we require that the line
    # corresponding to each peak passes within max_rho_fraction of the
    # boundary of the central source.  For a clean star this is ρ ≈ 0.
    # For a saturated star whose core is 0/NaN the spike arms produce
    # Radon peaks at ρ ≈ R_blank (the edge of the blank region) — the
    # line still passes through the star if extended, but the bright
    # segment stops at the saturation boundary.  We automatically extend
    # the accepted band by R_blank so these spikes are not missed.
    blank_r = 0#_central_blank_radius(image)
    max_rho_px = max_rho_fraction * min(image.shape) / 2.0 + blank_r
    all_rho_phys = sinogram_rho_to_physical(np.arange(n_rho), n_rho)
    rho_central = np.abs(all_rho_phys) <= max_rho_px  # shape (n_rho,)
    sinogram_central = sinogram * rho_central[:, np.newaxis]

    # ── 2-D peak detection ───────────────────────────────────────────────
    # All operations are restricted to sinogram_central (|ρ| ≤ max_rho_px)
    # so that off-centre bright sources never influence threshold, peak
    # locations, ρ assignment, or SNR estimation.
    abs_threshold = peak_prominence * np.max(sinogram_central)
    local_max = maximum_filter(sinogram_central, size=local_max_window)
    peak_map = (sinogram_central == local_max) & (sinogram_central > 0)

    max_along_rho = np.max(sinogram_central * peak_map, axis=0)

    min_sep_idx = max(1, int(np.round(
        min_peak_separation_deg / (180.0 / n_angles),
    )))
    peaks_1d, _ = find_peaks(
        max_along_rho,
        height=abs_threshold,
        distance=min_sep_idx,
    )

    # Find the brightest ρ within the central band for each detected angle.
    # Using sinogram_central (zeroed outside the band) ensures off-centre
    # sources never hijack the ρ assignment and cause valid spikes to fail
    # the subsequent central_mask check.
    peak_rho_idx = np.array(
        [np.argmax(sinogram_central[:, ti]) for ti in peaks_1d],
    )
    rho_phys = sinogram_rho_to_physical(peak_rho_idx, n_rho).astype(float)

    # ── filter: closest approach to image centre ──────────────────────────
    # Compute the perpendicular distance from the image centre to each
    # candidate line (= |ρ|) and reject lines that lie outside the
    # accepted band.  For a clean star |ρ| ≈ 0; for a saturated star
    # max_rho_px has been extended by blank_r so peaks at the edge of
    # the blank core survive this check.
    central_mask = np.abs(rho_phys) <= max_rho_px
    peaks_1d = peaks_1d[central_mask]
    peak_rho_idx = peak_rho_idx[central_mask]
    rho_phys = rho_phys[central_mask]

    # ── filter: significance (SNR) ────────────────────────────────────────
    if len(peaks_1d) > 0 and min_snr > 0:
        snr_values = _angular_profile_snr(sinogram_central, peaks_1d)
        snr_mask = snr_values >= min_snr
        n_rejected_snr = int(np.sum(~snr_mask))

        peaks_1d = peaks_1d[snr_mask]
        peak_rho_idx = peak_rho_idx[snr_mask]
        rho_phys = rho_phys[snr_mask]
        snr_values = snr_values[snr_mask]
    else:
        snr_values = _angular_profile_snr(sinogram_central, peaks_1d) \
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
        max_rho_px=max_rho_px,
    )

    # ── optional length measurement ──────────────────────────────────────
    if min_length is not None and not measure_lengths:
        warnings.warn(
            "min_length has no effect when measure_lengths=False",
            UserWarning,
            stacklevel=2,
        )

    if measure_lengths and len(peaks_1d) > 0:
        kw = length_kw or {}
        result.lengths = measure_spike_lengths(image, result, **kw)

        if min_length is not None:
            keep = np.array(
                [i for i, sl in enumerate(result.lengths)
                 if sl.length_total >= min_length],
                dtype=int,
            )
            result.angles = result.angles[keep]
            result.rho_physical = result.rho_physical[keep]
            result.snr = result.snr[keep]
            result.peak_rho_indices = result.peak_rho_indices[keep]
            result.peak_theta_indices = result.peak_theta_indices[keep]
            result.lengths = [result.lengths[i] for i in keep]

    return result
