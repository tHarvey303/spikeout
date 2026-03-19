"""Measure the length of each diffraction-spike arm via swath profile extraction."""

import math
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple
from scipy.ndimage import median_filter

from .stats import mad_std, estimate_background
from .preprocess import find_centre

__all__ = ["SpikeLengths", "measure_spike_lengths"]


@dataclass
class SpikeLengths:
    """Length measurement for one spike (two arms).

    Attributes
    ----------
    angle_deg : float
        Display-frame angle of this spike.
    length_pos, length_neg : float
        Length (pixels) of the positive and negative arms.
    length_total : float
        Sum of both arms.
    profile_pos, profile_neg : ndarray
        Swath profile along each arm (raw-image values, smoothed).
    radii_pos, radii_neg : ndarray
        Radial distance array for each arm (pixels from star centre).
    converged_pos, converged_neg : bool
        True if the arm end was found within the extracted profile.
        False if the spike extended to the profile boundary (array edge
        or max_output_length); the endpoint is then the profile boundary.
    popt : ndarray or None
        Deprecated — always None.  Retained for backwards compatibility.
    threshold : float
        Detection threshold used for arm endpoint measurement.
    background_profile : tuple of (r_bg, p_bg) or None
        (radii, profile) of the median background profile from off-spike angles.
    swath_width : float
        Swath width used for profile extraction (pixels).
    """
    angle_deg: float
    length_pos: float
    length_neg: float
    length_total: float
    profile_pos: np.ndarray
    profile_neg: np.ndarray
    radii_pos: np.ndarray
    radii_neg: np.ndarray
    converged_pos: bool = True
    converged_neg: bool = True
    popt: Optional[np.ndarray] = None  # deprecated, always None
    threshold: float = 0.0
    background_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None
    swath_width: float = 0.0


# ---------------------------------------------------------------------------
# Swath profile extraction (chunk-based, memmap-friendly)
# ---------------------------------------------------------------------------

def _swath_profile(
    data: np.ndarray,
    start: tuple,
    angle_deg: float,
    length: int,
    swath_width: int = 1,
    chunk_length: int = 512,
    reducer: Literal["mean", "median", "sum"] = "mean",
    step: float = 1.0,
) -> np.ndarray:
    """Extract a 1-D swath profile along a line from a 2-D array.

    Chunk-based extraction: reads one rectangular bounding-box slice per
    chunk, so a single IO operation covers ``chunk_length`` samples.
    Works efficiently on NumPy memmaps (e.g. astropy FITS memmaps).

    Parameters
    ----------
    data : array-like, shape (nrow, ncol)
        The 2-D image.  Row-major (C-order) layout assumed.
    start : (row0, col0)
        Starting pixel coordinate (sub-pixel floats accepted).
    angle_deg : float
        Angle in the display frame (degrees CCW from +x / +col axis,
        with +y pointing up / +row increasing upward).
        Compatible with ``SpikeResult.angles``.
    length : int
        Number of sample steps along the line.
    swath_width : int
        Width of the swath perpendicular to the line (pixels).
        Rounded up to odd so the line is centred.
    chunk_length : int
        Samples per IO chunk.
    reducer : {"mean", "median", "sum"}
        How to collapse the swath width into a single value per sample.
    step : float
        Sampling interval (pixels).

    Returns
    -------
    profile : np.ndarray, shape (length,)
    """
    swath_width = max(1, swath_width)
    if swath_width % 2 == 0:
        swath_width += 1
    half_w = swath_width // 2

    angle_rad = math.radians(angle_deg)
    dc = math.cos(angle_rad)   # column step per unit along line
    dr = math.sin(angle_rad)   # row step    (display +y = +row)
    pc = -math.sin(angle_rad)  # perpendicular col direction
    pr = math.cos(angle_rad)   # perpendicular row direction

    nrow, ncol = data.shape
    reduce_fn = _get_reducer(reducer)
    profile = np.empty(length, dtype=np.float64)

    for chunk_start in range(0, length, chunk_length):
        chunk_end = min(chunk_start + chunk_length, length)
        n_samples = chunk_end - chunk_start

        t = chunk_start + np.arange(n_samples, dtype=np.float64) * step

        rows_c = start[0] + t * dr
        cols_c = start[1] + t * dc

        offsets = np.arange(-half_w, half_w + 1, dtype=np.float64)

        rows_all = rows_c[:, None] + offsets[None, :] * pr
        cols_all = cols_c[:, None] + offsets[None, :] * pc

        ri = np.rint(rows_all).astype(np.intp)
        ci = np.rint(cols_all).astype(np.intp)

        r_min, r_max = int(ri.min()), int(ri.max())
        c_min, c_max = int(ci.min()), int(ci.max())

        r_min_c = max(r_min, 0)
        r_max_c = min(r_max, nrow - 1)
        c_min_c = max(c_min, 0)
        c_max_c = min(c_max, ncol - 1)

        chunk_data = np.asarray(
            data[r_min_c: r_max_c + 1, c_min_c: c_max_c + 1]
        )

        li = ri - r_min_c
        lj = ci - c_min_c

        valid = (
            (ri >= 0) & (ri < nrow) &
            (ci >= 0) & (ci < ncol)
        )

        vals = np.full(ri.shape, np.nan, dtype=np.float64)
        vals[valid] = chunk_data[li[valid], lj[valid]]

        profile[chunk_start:chunk_end] = reduce_fn(vals)

    return profile


def _get_reducer(name: str):
    """Return a swath-reduction function (n_samples, swath_width) → (n_samples,)."""
    if name == "mean":
        return lambda v: np.nanmean(v, axis=1)
    if name == "median":
        return lambda v: np.nanmedian(v, axis=1)
    if name == "sum":
        return lambda v: np.nansum(v, axis=1)
    raise ValueError(f"Unknown reducer {name!r}; use 'mean', 'median', or 'sum'.")


# ---------------------------------------------------------------------------
# Profile endpoint detection
# ---------------------------------------------------------------------------

def _find_profile_end(
    profile: np.ndarray,
    threshold: float,
    run_length: Optional[int] = None,
    run_frac: float = 0.05,
    min_run: int = 16,
    above_frac: float = 0.25,
    pad_frac: float = 0.5,
    min_pad: int = 16,
) -> int:
    """Find the index where a spike profile drops below threshold.

    Uses a sliding window: a window is "in signal" if at least
    ``above_frac`` of its samples exceed ``threshold``.  The scan
    walks forward from index 0 and stops at the first window that
    fails — isolated bumps further out (neighbouring stars) cannot
    restart the run.  After finding the last passing window, padding
    is added to avoid cutting off a sinusoidal trough.

    Parameters
    ----------
    profile : 1-D array
        Extracted swath profile (linear flux units).
    threshold : float
        Noise-floor level (same units as ``profile``).
    run_length : int or None
        Explicit window size.  If None, computed as
        ``max(min_run, int(len(profile) * run_frac))``.
    run_frac : float
        Window size as a fraction of profile length.
    min_run : int
        Absolute minimum window size in pixels.
    above_frac : float
        Fraction of samples in a window that must exceed ``threshold``.
    pad_frac : float
        After finding the raw endpoint, extend by ``pad_frac × W``.
    min_pad : int
        Minimum padding in pixels.

    Returns
    -------
    end_idx : int
        Index of the estimated endpoint (inclusive), clamped to
        ``[0, len(profile) - 1]``.
    """
    n = len(profile)
    if n == 0:
        return 0

    W = run_length if run_length is not None else max(min_run, int(n * run_frac))
    W = max(1, W)

    above = (profile >= threshold)

    cum = np.zeros(n + 1, dtype=np.int64)
    cum[1:] = np.cumsum(above)

    required = max(1, int(W * above_frac))

    last_passing = -1
    for i in range(n):
        j = min(i + W, n)
        count = int(cum[j] - cum[i])
        if count >= required:
            last_passing = i
        else:
            break

    if last_passing < 0:
        return 0

    raw_end = last_passing + W - 1
    pad = max(min_pad, int(W * pad_frac))
    end_idx = min(raw_end + pad, n - 1)

    return int(end_idx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_core_radius(image, centre=None):
    """Radius of the NaN/zero patch centred on ``centre``."""
    ny, nx = image.shape
    if centre is None:
        cy, cx = ny / 2.0, nx / 2.0
    else:
        cy, cx = centre
    cy_i, cx_i = int(round(cy)), int(round(cx))
    if not (0 <= cy_i < ny and 0 <= cx_i < nx):
        return 0.0
    blank = ~np.isfinite(image) | (image == 0.0)
    if not blank[cy_i, cx_i]:
        return 0.0
    Y, X = np.ogrid[:ny, :nx]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    nearby = blank & (dist <= min(ny, nx) / 4.0)
    return float(dist[nearby].max()) if nearby.any() else 0.0


def _arm_start(cx, cy, angle_deg, offset_px):
    """Row, col start point offset along ``angle_deg`` from (cy, cx)."""
    rad = math.radians(angle_deg)
    return cy + offset_px * math.sin(rad), cx + offset_px * math.cos(rad)


def _max_length_in_array(row_start, col_start, angle_deg, nrow, ncol, step=1.0):
    """Maximum number of steps before the line leaves the array."""
    rad = math.radians(angle_deg)
    dr = math.sin(rad)
    dc = math.cos(rad)

    limits = []
    if dr > 0:
        limits.append((nrow - 1 - row_start) / dr)
    elif dr < 0:
        limits.append(-row_start / dr)
    if dc > 0:
        limits.append((ncol - 1 - col_start) / dc)
    elif dc < 0:
        limits.append(-col_start / dc)

    if not limits:
        return 0
    return max(1, int(min(limits) / step))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_spike_lengths(
    image,
    result,
    swath_width=None,
    length_sigma=2.0,
    override_threshold=None,
    centre=None,
    full_array=None,
    centre_row_full=None,
    centre_col_full=None,
    skycoord=None,
    wcs=None,
    smooth_size_cross=20,
    max_radius=None,
    median_subtract=False,
    background_profiles=True,
    max_output_length=5000,
    swath_chunk_length=512,
    reducer="mean",
    run_frac=0.05,
    min_run=16,
    above_frac=0.25,
    pad_frac=0.5,
    radial_bin_width=2,
):
    """Measure the length of each detected spike arm via swath profiles.

    Profiles are extracted along each arm from the raw image using a
    chunk-based swath extraction that is efficient on memory-mapped arrays.

    **Two-stage extraction**: the profile is first extracted within the
    initial image cutout.  If the spike reaches the cutout boundary
    without dropping below the threshold *and* a ``full_array`` is
    provided, the profile is re-extracted from the full image (memmap)
    to find the true endpoint.

    Parameters
    ----------
    image : 2-D array
        Original (un-preprocessed) image cutout.
    result : `~spikeout.detect.SpikeResult`
        Output of `~spikeout.detect.detect`.
    swath_width : float or None
        Width (pixels) of the perpendicular sampling band.
        Default: ``max(3, min(image.shape) * 0.02)``.
    length_sigma : float
        Threshold multiplier: ``background + length_sigma × σ_sky``.
        Does not apply if ``override_threshold`` is provided.
    override_threshold : float or None
        If provided, use this absolute threshold instead of the estimated
        background + length_sigma × σ_sky.
    centre : (row, col) or None
        Star centre in the cutout.  Auto-detected if None.
    full_array : 2-D array or None
        Full memory-mapped image (e.g. ``hdul[0].data`` with
        ``memmap=True``).  Used for extended arm extraction when the
        spike reaches the cutout edge.
    centre_row_full : float or None
        Star centre row in ``full_array`` pixel coordinates.
        Required when ``full_array`` is provided (unless ``skycoord``
        and ``wcs`` are given).
    centre_col_full : float or None
        Star centre column in ``full_array`` pixel coordinates.
    skycoord : `~astropy.coordinates.SkyCoord` or None
        Sky position of the star.  Converted to pixel coordinates via
        ``wcs`` when ``centre_row_full`` / ``centre_col_full`` are not
        provided.
    wcs : `~astropy.wcs.WCS` or None
        WCS for ``full_array``.  Required when ``skycoord`` is given.
    smooth_size_cross : int
        Median-filter window applied to each arm profile before
        endpoint detection.  Suppresses noise spikes in the outer tail.
    max_radius : float or None
        Maximum walk distance in the cutout (pixels).
        Defaults to the cutout diagonal.
    median_subtract : bool
        If True, subtract the azimuthal median before measuring.
    background_profiles : bool
        If True, extract background profiles from off-spike angles.
    max_output_length : int
        Maximum arm length returned (pixels).
    swath_chunk_length : int
        Chunk size for the swath extractor (tune for IO performance).
    reducer : {"mean", "median", "sum"}
        How to collapse the swath width into one value per sample.
    run_frac : float
        ``_find_profile_end`` window size as fraction of profile length.
    min_run : int
        ``_find_profile_end`` minimum window size (pixels).
    above_frac : float
        ``_find_profile_end`` fraction of window above threshold required.
    pad_frac : float
        ``_find_profile_end`` padding fraction.
    radial_bin_width : int
        Annular bin width for azimuthal-median background estimation.

    Returns
    -------
    lengths : list of `SpikeLengths`
    """
    image = np.asarray(image, dtype=float)

    # ── resolve full-array star centre from sky coord ─────────────────────
    if skycoord is not None and wcs is not None:
        if centre_row_full is None or centre_col_full is None:
            px, py = wcs.world_to_pixel(skycoord)
            centre_col_full = float(px)
            centre_row_full = float(py)

    print(f'col: {centre_col_full}, row: {centre_row_full}')

    # ── blank-core radius (saturated / NaN core) ──────────────────────────
    if centre is None:
        centre = find_centre(image)
    cy, cx = centre
    blank_r = _blank_core_radius(image, centre=centre) + 3.0
    print(f'Blank-core radius: {blank_r:.1f} pixels')
    print(f'Cutout centre: ({cx:.1f}, {cy:.1f})')
    # ── working copy (NaN → 0 for safe indexing) ──────────────────────────
    img = image.copy()
    img[~np.isfinite(img)] = 0.0

    if median_subtract:
        from .preprocess import azimuthal_median
        model = azimuthal_median(
            image, centre=centre, radial_bin_width=radial_bin_width,
        )
        img = (image - model).copy()
        img[~np.isfinite(img)] = 0.0

    if swath_width is None:
        swath_width = max(3.0, min(image.shape) * 0.02)
    swath_width_int = max(1, int(round(swath_width)))

    ny, nx = img.shape

    if max_radius is None:
        max_radius = float(np.hypot(ny, nx))

    # ── Background noise (sep-based when available, fallback otherwise) ───
    Y_g, X_g = np.ogrid[:ny, :nx]
    R_from_centre = np.sqrt((X_g - cx) ** 2 + (Y_g - cy) ** 2)

    # Inner exclusion: generous halo radius (blank_r + 20% of half-image)
    halo_inner_r = max(blank_r + 5, 0.20 * min(ny, nx))
    if override_threshold is not None:
        threshold = override_threshold
        bg_level = np.nan
        sigma_bg = np.nan
    else:
        bg_level, sigma_bg = estimate_background(img, cy, cx, halo_inner_r)
        threshold = bg_level + length_sigma * sigma_bg

    print(f'Threshold: {threshold:.4f} (bg {bg_level:.4f} + {length_sigma} × σ {sigma_bg:.4f})')

    theta = result.theta
    pk_th = result.peak_theta_indices

    # ── optional background profiles ──────────────────────────────────────
    bg_profile_result = None
    if background_profiles and len(result.angles) > 0:
        bg_profs = []
        rng = np.random.default_rng(0)
        random_angles = rng.uniform(0, 360, size=80)
        for bg_angle in random_angles:
            if np.all(
                np.abs((bg_angle - result.angles + 180) % 360 - 180) > 10
            ):
                row_s, col_s = _arm_start(cx, cy, bg_angle, blank_r + 1)
                bg_len = _max_length_in_array(row_s, col_s, bg_angle, ny, nx)
                bg_len = min(bg_len, int(max_radius))
                if bg_len < 4:
                    continue
                p_bg = _swath_profile(
                    img, (row_s, col_s), bg_angle, bg_len,
                    swath_width=swath_width_int,
                    chunk_length=swath_chunk_length,
                    reducer=reducer,
                )
                r_bg = blank_r + 1 + np.arange(bg_len, dtype=float)
                bg_profs.append((r_bg, p_bg))
            if len(bg_profs) >= 6:
                break
        if bg_profs:
            r_comb = np.concatenate([r for r, p in bg_profs])
            p_comb = np.concatenate([p for r, p in bg_profs])
            order = np.argsort(r_comb)
            r_s, p_s = r_comb[order], p_comb[order]
            sz = max(1, smooth_size_cross)
            p_smooth = median_filter(p_s, size=sz, mode='nearest')
            bg_profile_result = (r_s, p_smooth)

    # ── per-spike measurement ─────────────────────────────────────────────
    lengths = []
    for i in range(len(result.angles)):
        rho = result.rho_physical[i]
        th_rad = np.deg2rad(theta[pk_th[i]])
        # star-centre position in the cutout (column, row)
        x0 = nx / 2.0 + rho * np.cos(th_rad)
        y0 = ny / 2.0 - rho * np.sin(th_rad)
        angle = result.angles[i]

        arm_results = {}
        for label, arm_angle in [("pos", angle), ("neg", (angle + 180.0) % 360.0)]:
            # ── Step 1: extract profile from the cutout ───────────────────
            row_s, col_s = _arm_start(x0, y0, arm_angle, blank_r + 1)
            cutout_len = _max_length_in_array(row_s, col_s, arm_angle, ny, nx)
            cutout_len = min(cutout_len, int(max_radius))
            cutout_len = max(cutout_len, 1)

            profile = _swath_profile(
                img, (row_s, col_s), arm_angle, cutout_len,
                swath_width=swath_width_int,
                chunk_length=swath_chunk_length,
                reducer=reducer,
            )
            # radii from star centre
            radii = blank_r + 1 + np.arange(len(profile), dtype=float)

            # smooth before endpoint detection
            sz = max(1, int(smooth_size_cross))
            profile_smooth = (
                median_filter(profile, size=sz, mode='nearest')
                if sz > 1 else profile.copy()
            )

            end_idx = _find_profile_end(
                profile_smooth, threshold,
                run_frac=run_frac, min_run=min_run,
                above_frac=above_frac, pad_frac=pad_frac,
            )

            # Check if spike reached the cutout boundary
            at_cutout_edge = (end_idx >= len(profile) - 1)

            # ── Step 2: extend into full_array if needed ──────────────────
            if at_cutout_edge and full_array is not None:
                if centre_row_full is not None and centre_col_full is not None:
                    # Compute offset from full-array star centre
                    fa_nrow, fa_ncol = full_array.shape
                    fa_row_s, fa_col_s = _arm_start(
                        centre_col_full, centre_row_full,
                        arm_angle, blank_r + 1,
                    )
                    full_len = _max_length_in_array(
                        fa_row_s, fa_col_s, arm_angle, fa_nrow, fa_ncol,
                    )
                    full_len = min(full_len, max_output_length)
                    full_len = max(full_len, 1)

                    profile_full = _swath_profile(
                        full_array, (fa_row_s, fa_col_s), arm_angle, full_len,
                        swath_width=swath_width_int,
                        chunk_length=swath_chunk_length,
                        reducer=reducer,
                    )
                    radii_full = blank_r + 1 + np.arange(
                        len(profile_full), dtype=float
                    )

                    sz = max(1, int(smooth_size_cross))
                    profile_full_smooth = (
                        median_filter(profile_full, size=sz, mode='nearest')
                        if sz > 1 else profile_full.copy()
                    )

                    end_idx_full = _find_profile_end(
                        profile_full_smooth, threshold,
                        run_frac=run_frac, min_run=min_run,
                        above_frac=above_frac, pad_frac=pad_frac,
                    )
                    converged = (end_idx_full < len(profile_full) - 1)
                    length_px = float(radii_full[end_idx_full])
                    length_px = min(length_px, float(max_output_length))
                    arm_results[label] = (
                        length_px, radii_full, profile_full_smooth, converged,
                    )
                    continue
            elif at_cutout_edge and full_array is None:
                print(
                    f"Warning: spike arm at angle {angle:.1f}° reached cutout edge "
                    "and no full_array provided; length may be underestimated."
                )

            # ── Step 1 result ─────────────────────────────────────────────
            converged = not at_cutout_edge
            length_px = float(radii[end_idx])
            length_px = min(length_px, float(max_output_length))
            arm_results[label] = (length_px, radii, profile_smooth, converged)

        lengths.append(SpikeLengths(
            angle_deg=angle,
            length_pos=arm_results["pos"][0],
            length_neg=arm_results["neg"][0],
            length_total=arm_results["pos"][0] + arm_results["neg"][0],
            profile_pos=arm_results["pos"][2],
            profile_neg=arm_results["neg"][2],
            radii_pos=arm_results["pos"][1],
            radii_neg=arm_results["neg"][1],
            converged_pos=arm_results["pos"][3],
            converged_neg=arm_results["neg"][3],
            popt=None,
            threshold=threshold,
            background_profile=bg_profile_result,
            swath_width=swath_width,
        ))

    return lengths
