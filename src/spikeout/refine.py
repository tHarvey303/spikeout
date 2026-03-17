"""Multi-stage spike length refinement via direct pixel probing.

The fitting-based length estimate from :func:`~spikeout.measure_spike_lengths`
(Stage 1) extrapolates a power-law model to find the threshold crossing.
Because the length scales as ``r ∝ A^(1/γ)``, small errors in the amplitude
or slope propagate into significant length errors.

This module provides two additional stages:

**Stage 2 — bracketing**
    The fitted model gives the positions of oscillation *peaks* — the radii
    where the spike signal is at its local maximum for each fringe period.
    A small number of peaks just inside and outside the Stage-1 estimate are
    sampled directly from the image.  This locates a bracket
    ``[r_lo, r_hi]`` that contains the true endpoint, without relying on any
    model extrapolation.  Sampling at peaks (rather than arbitrary positions)
    gives the strongest possible detection at each radius: if even the peak
    is below threshold the whole fringe period is below threshold.

**Stage 3 — binary search**
    The bracket is narrowed iteratively.  At each step the midpoint of
    ``[r_lo, r_hi]`` is computed, the nearest oscillation peak to that
    midpoint is located, and the image is sampled there.  After
    ``n_binary_steps`` iterations the bracket half-width is
    ``(r_hi − r_lo) / 2^n`` — sub-pixel for typical fringe spacings with
    6–8 steps.

Both stages read only a narrow perpendicular swath at each probe radius
(``≈ swath_width`` pixels).  When the image is provided as a FITS path the
file is opened with ``memmap=True`` so only those pixels are paged from disk.
"""

from __future__ import annotations

import numpy as np
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from .lengths import SpikeLengths, _blank_core_radius
from .preprocess import find_centre

if TYPE_CHECKING:
    from .detect import SpikeResult

__all__ = ["refine_spike_lengths"]

ImageOrPath = Union[np.ndarray, str, Path]


# ── low-level helpers ──────────────────────────────────────────────────────────

def _open_data(
    image_or_path: ImageOrPath,
    hdu_index: int = 0,
    cutout_slice: Optional[Tuple[slice, slice]] = None,
) -> Tuple[np.ndarray, bool, object]:
    """Return *(data, opened_fits, hdul_or_None)*.

    When *image_or_path* is a path the FITS is opened with ``memmap=True``
    so reads are demand-paged.  ``cutout_slice = (row_slice, col_slice)``
    immediately slices the memmap to the region of interest; only those
    tiles will ever be paged from disk.
    """
    if isinstance(image_or_path, (str, Path)):
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError(
                "astropy is required to open FITS files in refine_spike_lengths. "
                "Install with: pip install 'spikeout[astropy]'"
            )
        hdul = fits.open(str(image_or_path), memmap=True, mode='denywrite')
        raw = hdul[hdu_index].data
        if cutout_slice is not None:
            # np.asarray forces the slice into a real array — only these
            # tiles are read from disk.
            data = np.asarray(raw[cutout_slice], dtype=float)
        else:
            data = raw  # keep as memmap; individual probes page their tiles
        return data, True, hdul
    else:
        return np.asarray(image_or_path, dtype=float), False, None


def _sample_swath_at_r(
    data: np.ndarray,
    x0: float,
    y0: float,
    angle_deg: float,
    r: float,
    swath_width: float,
    combine: callable = np.nanmedian,
) -> float:
    """Sample a single perpendicular swath at radius *r* along the spike.

    Computes the ``swath_width`` pixel positions of the perpendicular band,
    then reads a tight bounding-box slice from *data*.  When *data* is a
    memmap array this minimises disk I/O by paging only the relevant tiles.

    Returns ``NaN`` if fewer than 2 valid (in-bounds, finite) pixels exist.
    """
    rad = np.deg2rad(angle_deg)
    cx = x0 + r * np.cos(rad)
    cy = y0 + r * np.sin(rad)
    dx_perp = -np.sin(rad)
    dy_perp = np.cos(rad)

    half_w = swath_width / 2.0
    n_perp = max(3, int(np.ceil(swath_width)))
    offsets = np.linspace(-half_w, half_w, n_perp)

    px = cx + offsets * dx_perp
    py = cy + offsets * dy_perp
    ix = np.round(px).astype(int)
    iy = np.round(py).astype(int)

    ny, nx = data.shape
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if valid.sum() < 2:
        return np.nan

    iy_v, ix_v = iy[valid], ix[valid]
    y_lo = int(iy_v.min())
    y_hi = int(iy_v.max()) + 1
    x_lo = int(ix_v.min())
    x_hi = int(ix_v.max()) + 1
    # np.asarray forces the memmap read of this bounding box.
    strip = np.asarray(data[y_lo:y_hi, x_lo:x_hi], dtype=float)

    vals = strip[iy_v - y_lo, ix_v - x_lo]
    finite = vals[np.isfinite(vals)]
    return float(combine(finite)) if finite.size > 0 else np.nan


def _oscillation_peak_radii(
    popt: np.ndarray,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    """Return the radii of oscillation peaks within ``[r_min, r_max]``.

    For the model ``(A/r^γ)·(1 − C·exp(−r/r_dec)·cos(2π·f·r + φ))``,
    peaks of the cosine term occur where ``cos(2π·f·r + φ) = −1``, giving:

        r_peak_k = (k + 0.5 − φ/(2π)) / f,   k = 0, 1, 2, …

    These are exact for constant C; the exponential decay shifts the true
    maximum by O(1/(f·r_dec)) which is negligible when r_dec is large.

    Returns an empty array if *popt* has an unexpected shape or ``f ≤ 0``.
    """
    if popt is None or len(popt) != 6:
        return np.array([])
    _A, _gamma, f, _C, _r_dec, phi = popt
    if f <= 0:
        return np.array([])

    # k range such that r_peak_k ∈ [r_min, r_max]:
    #   r_min ≤ (k + 0.5 − φ/(2π)) / f   →   k ≥ f·r_min − 0.5 + φ/(2π)
    #   (k + 0.5 − φ/(2π)) / f ≤ r_max   →   k ≤ f·r_max − 0.5 + φ/(2π)
    phi_turns = phi / (2.0 * np.pi)
    k_lo = int(np.floor(f * r_min - 0.5 + phi_turns)) - 1
    k_hi = int(np.ceil(f * r_max - 0.5 + phi_turns)) + 1

    peaks = []
    for k in range(k_lo, k_hi + 1):
        r_pk = (k + 0.5 - phi_turns) / f
        if r_min <= r_pk <= r_max:
            peaks.append(r_pk)
    return np.array(sorted(peaks))


def _nearest_peak(popt: np.ndarray, r_target: float, r_lo: float, r_hi: float) -> float:
    """Return the oscillation peak nearest to *r_target* within ``[r_lo, r_hi]``.

    Falls back to *r_target* itself if no peaks are found in the window.
    """
    peaks = _oscillation_peak_radii(popt, r_lo, r_hi)
    if len(peaks) == 0:
        return r_target
    return float(peaks[int(np.argmin(np.abs(peaks - r_target)))])


def _fallback_probe_grid(r_min: float, r_max: float, n: int = 20) -> np.ndarray:
    """Geometric probe grid used when the oscillation model is unavailable."""
    return np.geomspace(max(r_min, 1.0), r_max, n)


# ── per-arm refinement ─────────────────────────────────────────────────────────

def _refine_arm(
    data: np.ndarray,
    x0: float,
    y0: float,
    angle_deg: float,
    sl: SpikeLengths,
    arm: str,
    blank_r: float = 0.0,
    n_bracket_peaks: int = 3,
    n_binary_steps: int = 6,
) -> Tuple[float, float, float, int, bool]:
    """Refine one arm endpoint via direct image probing (Stages 2 and 3).

    Parameters
    ----------
    data : 2-D array
        Image (may be a memmap).
    x0, y0 : float
        Spike line origin in pixel coordinates.
    angle_deg : float
        Walking direction for this arm (degrees CCW from +x).
    sl : SpikeLengths
        Stage-1 result for this spike.
    arm : ``'pos'`` or ``'neg'``
        Which arm to refine.
    blank_r : float
        Inner exclusion radius (blank/saturated core).
    n_bracket_peaks, n_binary_steps : int
        See :func:`refine_spike_lengths`.

    Returns
    -------
    refined : float
        Best-estimate length (midpoint of final bracket).
    r_lo : float
        Last radius directly confirmed above threshold.
    bracket_half : float
        Half-width of the final bracket (pixels).
    n_probes : int
        Total number of image reads.
    converged : bool
        ``False`` if the spike was still detected at the outermost probed
        peak (endpoint not found within the search range).
    """
    r_est = sl.length_pos if arm == 'pos' else sl.length_neg
    threshold = sl.threshold
    swath_width = max(sl.swath_width, 3.0)
    bg_profile = sl.background_profile
    bg_sub = sl.background_subtracted

    # Background interpolator — used to isolate spike signal from raw flux.
    if bg_sub and bg_profile is not None:
        r_bg_arr, p_bg_arr = bg_profile
        p_bg_safe = np.maximum(p_bg_arr, 1e-30)

        def _bg_at(r: float) -> float:
            return float(np.interp(r, r_bg_arr, p_bg_safe,
                                   left=float(p_bg_safe[0]),
                                   right=float(p_bg_safe[-1])))
    else:
        def _bg_at(r: float) -> float:  # noqa: E306
            return 0.0

    n_probes = [0]

    def _above(r: float) -> bool:
        n_probes[0] += 1
        raw = _sample_swath_at_r(data, x0, y0, angle_deg, r, swath_width)
        if np.isnan(raw):
            return False
        return bool((raw - _bg_at(r)) > threshold)

    # Upper bound on probe radius: 2.5× the Stage-1 estimate, capped at the
    # image diagonal.  This allows the refinement to extend the length if the
    # Stage-1 estimate was an underestimate.
    r_max_probe = min(r_est * 2.5, float(np.hypot(*data.shape)))
    r_min_probe = max(blank_r + 1.0, r_est * 0.1)

    has_model = (sl.popt is not None and len(sl.popt) == 6 and sl.popt[2] > 0)

    if has_model:
        all_peaks = _oscillation_peak_radii(sl.popt, r_min_probe, r_max_probe)
    else:
        all_peaks = _fallback_probe_grid(r_min_probe, r_max_probe)

    if len(all_peaks) < 2:
        # Profile too short or no oscillation: return Stage-1 estimate unchanged.
        return r_est, r_est, 0.0, 0, True

    # ── Stage 2: bracket by probing peaks around the Stage-1 estimate ─────
    # Probe n_bracket_peaks peaks on each side of r_est.
    inner_peaks = all_peaks[all_peaks <= r_est][-n_bracket_peaks:]
    outer_peaks = all_peaks[all_peaks > r_est][:n_bracket_peaks]

    r_lo = r_min_probe
    for r_pk in inner_peaks:
        if _above(r_pk):
            r_lo = r_pk

    r_hi: Optional[float] = None
    for r_pk in outer_peaks:
        if not _above(r_pk):
            r_hi = r_pk
            break
        else:
            r_lo = r_pk  # spike confirmed beyond r_est

    if r_hi is None:
        # Spike confirmed at every probed outer peak: endpoint not found.
        # Return the last confirmed radius as a lower bound.
        return r_lo, r_lo, 0.0, n_probes[0], False

    # ── Stage 3: binary search within [r_lo, r_hi] ────────────────────────
    for _ in range(n_binary_steps):
        if r_hi - r_lo < 0.5:   # sub-pixel: stop early
            break
        r_mid = (r_lo + r_hi) / 2.0
        # Always probe at the oscillation peak nearest to the midpoint so
        # we get the strongest signal and the most decisive binary answer.
        if has_model:
            r_probe = _nearest_peak(sl.popt, r_mid, r_lo, r_hi)
        else:
            r_probe = r_mid

        if _above(r_probe):
            r_lo = r_probe
        else:
            r_hi = r_probe

    refined = (r_lo + r_hi) / 2.0
    bracket_half = (r_hi - r_lo) / 2.0
    return refined, r_lo, bracket_half, n_probes[0], True


# ── public API ─────────────────────────────────────────────────────────────────

def refine_spike_lengths(
    result: SpikeResult,
    image_or_path: ImageOrPath,
    hdu_index: int = 0,
    cutout_slice: Optional[Tuple[slice, slice]] = None,
    centre: Optional[Tuple[float, float]] = None,
    n_bracket_peaks: int = 3,
    n_binary_steps: int = 6,
    swath_width: Optional[float] = None,
) -> List[SpikeLengths]:
    """Refine spike arm lengths by probing oscillation peaks directly.

    Takes a :class:`~spikeout.SpikeResult` whose ``lengths`` list has been
    populated by :func:`~spikeout.measure_spike_lengths` (Stage 1) and
    applies two further stages:

    **Stage 2 — bracketing**
        Oscillation peak positions are computed from the fitted model
        parameters.  ``n_bracket_peaks`` peaks on each side of the
        current Stage-1 estimate are sampled directly from the image to
        establish a confirmed lower bound ``r_lo`` (last detected peak) and
        a rejected upper bound ``r_hi`` (first undetected peak), bracketing
        the true endpoint within one fringe period independently of any
        power-law extrapolation.

        If no model is available (``popt`` is *None* or the fit failed), a
        geometric grid of probe radii is used as a fallback.

    **Stage 3 — binary search**
        The bracket ``[r_lo, r_hi]`` is narrowed iteratively.  At each step
        the nearest oscillation peak to the current midpoint is sampled.
        After ``n_binary_steps`` iterations the bracket half-width is
        ``(r_hi − r_lo) / 2^n``; 6 steps give 64× compression, 8 steps 256×.

    Each probe reads only ``≈ swath_width`` pixels.  When *image_or_path* is
    a FITS path the file is opened with ``memmap=True`` so only the probed
    pixels are paged from disk.

    Parameters
    ----------
    result : SpikeResult
        Must have ``result.lengths`` populated (``measure_lengths=True`` in
        :func:`~spikeout.detect`).
    image_or_path : ndarray or str or Path
        The image array *in the same pixel frame that was used for detection*,
        **or** a path to the FITS file.  When a FITS path is given use
        ``cutout_slice`` to restrict I/O to the relevant cutout region.
    hdu_index : int
        HDU index (used only when *image_or_path* is a FITS path).
    cutout_slice : ``(row_slice, col_slice)`` or *None*
        Sub-region of the FITS to read, e.g.
        ``(slice(y0, y1), slice(x0, x1))``.  Only relevant when a FITS path
        is given; pass *None* to read the entire HDU (may page a lot of data).
    centre : ``(row, col)`` or *None*
        Star centre in the image frame.  Auto-detected from the image if
        *None*.
    n_bracket_peaks : int
        Number of oscillation peaks to probe on each side of the Stage-1
        estimate during bracketing.  3 (default) gives a search window of
        ±3 fringe periods around the current estimate.
    n_binary_steps : int
        Binary-search refinement iterations.  6 (default) gives a final
        bracket half-width of ``initial_bracket / 64``.
    swath_width : float or *None*
        Perpendicular sampling width (pixels) for all probes.  Defaults to
        ``SpikeLengths.swath_width`` from the Stage-1 result.

    Returns
    -------
    list of SpikeLengths
        Updated copies of the input lengths.  The following fields are
        populated in addition to the usual Stage-1 fields:

        ``length_pos``, ``length_neg``, ``length_total``
            Refined endpoint estimates (midpoint of the final bracket).
        ``length_pos_lo``, ``length_neg_lo``
            Last radius at which the arm was directly confirmed (lower bound).
        ``length_pos_bracket``, ``length_neg_bracket``
            Bracket half-width in pixels (uncertainty estimate).
        ``n_probes_pos``, ``n_probes_neg``
            Number of image reads per arm.
        ``converged_pos``, ``converged_neg``
            *False* if the spike was still detected at the outermost probed
            peak (endpoint beyond the search range).
        ``refined``
            Always *True* for refined results.

    Notes
    -----
    The function does **not** modify *result* in-place.  Assign the return
    value back::

        result.lengths = refine_spike_lengths(result, image)
    """
    if result.lengths is None or len(result.lengths) == 0:
        return []

    data, opened, hdul = _open_data(image_or_path, hdu_index, cutout_slice)

    try:
        ny, nx = data.shape

        if centre is None:
            img_for_centre = np.where(np.isfinite(data), data, 0.0)
            centre = find_centre(img_for_centre)
        _cy, _cx = centre

        blank_r = _blank_core_radius(data, centre=centre) + 3.0

        theta = result.theta
        pk_th = result.peak_theta_indices

        updated: List[SpikeLengths] = []
        for i, sl in enumerate(result.lengths):
            rho = result.rho_physical[i]
            th_rad = np.deg2rad(float(theta[pk_th[i]]))
            x0 = nx / 2.0 + rho * np.cos(th_rad)
            y0 = ny / 2.0 - rho * np.sin(th_rad)
            angle = result.angles[i]

            # Optionally override swath_width for all probes.
            sl_probe = sl
            if swath_width is not None:
                sl_probe = copy(sl)
                sl_probe.swath_width = float(swath_width)

            # Positive arm: walk in direction `angle`
            r_pos, r_lo_pos, bk_pos, n_pos, conv_pos = _refine_arm(
                data, x0, y0, angle, sl_probe, 'pos',
                blank_r=blank_r,
                n_bracket_peaks=n_bracket_peaks,
                n_binary_steps=n_binary_steps,
            )
            # Negative arm: walk in direction `angle + 180`
            r_neg, r_lo_neg, bk_neg, n_neg, conv_neg = _refine_arm(
                data, x0, y0, (angle + 180.0) % 360.0, sl_probe, 'neg',
                blank_r=blank_r,
                n_bracket_peaks=n_bracket_peaks,
                n_binary_steps=n_binary_steps,
            )

            updated.append(SpikeLengths(
                angle_deg=sl.angle_deg,
                length_pos=r_pos,
                length_neg=r_neg,
                length_total=r_pos + r_neg,
                profile_pos=sl.profile_pos,
                profile_neg=sl.profile_neg,
                radii_pos=sl.radii_pos,
                radii_neg=sl.radii_neg,
                converged_pos=conv_pos,
                converged_neg=conv_neg,
                popt=sl.popt,
                threshold=sl.threshold,
                background_profile=sl.background_profile,
                swath_width=sl.swath_width,
                background_subtracted=sl.background_subtracted,
                length_pos_lo=r_lo_pos,
                length_neg_lo=r_lo_neg,
                length_pos_bracket=bk_pos,
                length_neg_bracket=bk_neg,
                n_probes_pos=n_pos,
                n_probes_neg=n_neg,
                refined=True,
            ))

    finally:
        if opened and hdul is not None:
            hdul.close()

    return updated
