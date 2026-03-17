"""Measure the length of each diffraction-spike arm via Fraunhofer profile fitting.

Background-subtraction approach
--------------------------------
Rather than fitting the full model ``a·sinc(b·r)² + c/r^α + d`` (5 free
parameters), we extract background profiles from random angles far from any
spike, subtract them from the spike-arm profiles, and then fit only the
sinc² spike signal with 2 free parameters ``(a, b)``.  This avoids the
dangerous coupled fitting of the stellar core/halo that can fail or produce
unphysical results.

Care is taken to prevent the background dropping to near zero at large radii
(which would cause log-log space to diverge): the interpolated background is
floored at a small positive value before subtraction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, root_scalar

from .stats import mad_std
from .preprocess import azimuthal_median, find_centre

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
        Median-smoothed swath profile along each arm used for endpoint
        detection.  When ``background_subtracted`` is *True* these are the
        background-subtracted signal profiles; otherwise they are the raw
        swath values.
    radii_pos, radii_neg : ndarray
        Radial distance array for each arm (pixels from star centre).
    converged_pos, converged_neg : bool
        True if the arm end was found as a threshold crossing within the
        measured profile.  False means the endpoint was obtained by
        extrapolating the fitted Fraunhofer envelope beyond the profile edge.
    popt : ndarray or None
        Fitted Fraunhofer parameters ``(a, b, c, alpha, d)``.
    threshold : float
        Detection threshold used for arm endpoint measurement.
    background_profile : tuple of (r_bg, p_bg)
        (radii, profile) of the median background profile extracted from random angles, for sanity check and potential future use.
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
    popt: Optional[np.ndarray] = None
    threshold: float = 0.0
    background_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None
    swath_width: float = 0.0
    background_subtracted: bool = False

# ── helpers ───────────────────────────────────────────────────────────────────

def _blank_core_radius(image, centre=None):
    """Radius of the NaN/zero patch centred on ``centre``.

    A saturated or masked star core is represented as NaN or exactly zero.
    Returns 0.0 if the centre pixel is finite and non-zero.
    """
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


def _extract_swath_profile(image, x0, y0, angle_deg, swath_width, combine_function=np.nansum,
                           step=1.0, max_radius=None):
    """Extract a median swath profile along a direction from a point.

    At each step *r* from ``(x0, y0)`` along *angle_deg*, samples a
    perpendicular band of *swath_width* pixels and takes the median.

    Parameters
    ----------
    image : 2-D array
    x0, y0 : float
        Starting point (pixel coords, ``origin='lower'``).
    angle_deg : float
        Direction to walk (degrees CCW from +x in display frame).
    swath_width : float
        Full width (pixels) of the perpendicular band.
    combine_function : callable
        Function to combine the perpendicular samples at each radius.  Default
        is `numpy.nansum`.
    step : float
        Radial step size (pixels).
    max_radius : float or *None*
        Maximum walk distance.  Defaults to the image diagonal.

    Returns
    -------
    radii : 1-D array
    profile : 1-D array
    """
    ny, nx = image.shape
    if max_radius is None:
        max_radius = np.hypot(ny, nx)

    rad = np.deg2rad(angle_deg)
    dx_along = np.cos(rad)
    dy_along = np.sin(rad)
    dx_perp = -dy_along
    dy_perp = dx_along

    half_w = swath_width / 2.0
    n_perp = max(3, int(np.ceil(swath_width)))
    perp_offsets = np.linspace(-half_w, half_w, n_perp)

    radii = np.arange(step, max_radius, step)
    profile = np.full(len(radii), np.nan)

    for i, r in enumerate(radii):
        cx = x0 + r * dx_along
        cy = y0 + r * dy_along

        px = cx + perp_offsets * dx_perp
        py = cy + perp_offsets * dy_perp

        ix = np.round(px).astype(int)
        iy = np.round(py).astype(int)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

        if valid.sum() < max(2, n_perp // 2):
            break

        vals = image[iy[valid], ix[valid]]
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size > 0:
            profile[i] = combine_function(finite_vals)

    last_valid = np.where(np.isfinite(profile))[0]
    if len(last_valid) == 0:
        return radii[:1], profile[:1]
    end = last_valid[-1] + 1
    return radii[:end], profile[:end]


# ── Fraunhofer profile model ──────────────────────────────────────────────────

def _fraunhofer_model(r, a, b, c, alpha, d):
    """Fraunhofer sinc² spike + power-law halo + constant background.

    ``a · sinc(b·r)²  +  c / r^alpha  +  d``

    ``np.sinc`` is the normalised sinc: ``sin(π x) / (π x)``.
    """
    return a * np.sinc(b * r) ** 2 + c / r ** alpha + d


def _log_fraunhofer_model(r, a, b, c, alpha, d):
    """log₁₀ of ``_fraunhofer_model``, used for log-space curve_fit."""
    val = _fraunhofer_model(r, a, b, c, alpha, d)
    return np.log10(np.maximum(val, 1e-30))


def _log_sinc2_model(r, a, b):
    """log₁₀ of ``a · sinc(b·r)²``, used after background subtraction."""
    val = a * np.sinc(b * r) ** 2
    return np.log10(np.maximum(val, 1e-30))


def _powerlaw_osc_model(r, A, gamma, f, C, phi):
    """Power-law envelope modulated by a sinusoidal oscillation.

    ``(A / r^gamma) · (1 - C · cos(2π·f·r + φ))``

    This is the physically correct model for the background-subtracted
    diffraction spike radial profile:
    - ``A / r^gamma`` is the power-law envelope (for a monochromatic spike
      the slope is γ=2; broadband averaging can change this).
    - The ``(1 - C·cos(...))`` factor captures the sinusoidal fringes.
      It is a direct generalisation of sinc²:
      ``sinc²(b·r) = (1 - cos(2π·b·r)) / (2π²b²r²)`` — i.e. γ=2, C=1,
      f=b, with A = 1/(2π²b²).
    - C is the fringe contrast (0 ≤ C < 1 keeps the model strictly
      positive in log space; C→1 recovers the sinc² zeros).
    - f is the fringe spatial frequency (cycles per pixel).
    - φ is the phase offset.

    Unlike fitting sinc² directly, this model never touches zero when C<1,
    so log-space fitting is well-conditioned.  The oscillation frequency is
    initialised from an FFT of the envelope-subtracted residuals, avoiding
    the degenerate b→0 failure mode of the raw sinc² fit.
    """
    return (A / r ** gamma) * (1.0 - C * np.cos(2.0 * np.pi * f * r + phi))


def _log_powerlaw_osc_model(r, A, gamma, f, C, phi):
    """log₁₀ of ``_powerlaw_osc_model``."""
    val = _powerlaw_osc_model(r, A, gamma, f, C, phi)
    return np.log10(np.maximum(val, 1e-30))


def _powerlaw_osc_envelope_crossing(A, gamma, C, threshold):
    """Analytic threshold-crossing radius for the power-law+oscillation model.

    The upper envelope is ``A·(1+C) / r^gamma`` (cos = -1, oscillation peak).
    The spike is considered ended when even the oscillation peaks fall below
    *threshold*:

        r_end = (A·(1+C) / threshold)^(1/γ)

    Returns *None* if the envelope amplitude never exceeds *threshold*.
    """
    if threshold <= 0:
        return None
    amp = A * (1.0 + C)
    if amp <= threshold:
        return None
    return float((amp / threshold) ** (1.0 / gamma))


def _envelope_model(r, a, b, c, alpha, d):
    """Upper envelope of the Fraunhofer spike model.

    Replaces ``sinc(b·r)² ≤ 1/(π b r)²`` by its envelope, giving a
    monotonically decreasing upper bound used for threshold crossing.
    """
    return a / (np.pi * b * r) ** 2 + c / r ** alpha + d


def _estimate_p0(r, p):
    """Estimate initial Fraunhofer model parameters from profile data.

    Strategy
    --------
    1. ``d0`` — lower percentile of the outer tail, where the spike has
       faded to sky/halo level.  The full-profile P5 is in the signal region
       and gives a grossly inflated estimate.
    2. ``(c0, alpha0)`` — log-log fit on the raw outer half of the profile,
       *without* subtracting ``d0``.  Subtracting a noisy ``d0`` from
       already-small tail values produces near-zero residuals, causing the
       power-law slope to diverge and ``alpha`` to hit its bound.  sinc²
       contributes little in the outer half, so the raw slope ≈ ``−alpha``.
    3. ``a0`` — peak of the halo-subtracted inner profile.
    4. ``b0`` — half-power radius of the halo-subtracted spike component.
       ``sinc²(b·r) = 0.5`` at ``b·r ≈ 0.45``.
    """
    # 1. Background
    n_outer = max(5, len(p) // 4)
    d0 = max(float(np.percentile(p[-n_outer:], 10)), 1e-10)

    # 2. Halo: log-log fit on raw outer half (no d0 subtraction)
    n_halo = max(5, len(p) // 2)
    r_halo = r[-n_halo:]
    p_halo = np.maximum(p[-n_halo:], 1e-30)
    try:
        slope, intercept = np.polyfit(np.log(r_halo), np.log(p_halo), 1)
        alpha0 = float(np.clip(-slope, 0.5, 3.0))
        c0 = max(float(np.exp(intercept)), d0)
    except Exception:
        alpha0 = 1.5
        c0 = max(float(np.median(p)) * float(r[len(r) // 2]) ** 1.5, d0)

    # 3. Spike amplitude: peak of inner profile minus estimated halo
    n_inner = max(3, len(p) // 3)
    r_safe = np.maximum(r[:n_inner], 0.5)
    halo_inner = c0 / r_safe ** alpha0 + d0
    a0 = max(float(np.max(np.maximum(p[:n_inner] - halo_inner, 0))), d0)

    # 4. Sinc² frequency: half-power of halo-subtracted spike
    r_safe_full = np.maximum(r, 0.5)
    halo_full = c0 / r_safe_full ** alpha0 + d0
    p_spike = np.maximum(p - halo_full, 0)
    half_target = 0.5 * p_spike.max()
    if half_target > 0:
        candidates = r[p_spike >= half_target]
        r_half = float(candidates[-1]) if len(candidates) else float(r[-1])
    else:
        r_half = float(r[-1])
    b0 = float(np.clip(0.45 / max(r_half, 1.0), 1e-5, 0.05))

    print(f"Initial parameter estimates: a={a0:.2e}, b={b0:.2e}, c={c0:.2e}, alpha={alpha0:.2f}, d={d0:.2e}")

    return [a0, b0, c0, alpha0, d0]


def _fit_spike_signal_only(
    r_pos: np.ndarray,
    p_pos: np.ndarray,
    r_neg: np.ndarray,
    p_neg: np.ndarray,
    r_bg: np.ndarray,
    p_bg: np.ndarray,
    blank_r: float = 0.0,
    smooth_size: int = 10,
    smooth_size_cross: Optional[int] = None,
    r_min: Optional[float] = None,
    r_max_fraction: float = 0.7,
    bg_floor: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Fit a power-law+oscillation model to background-subtracted spike profiles.

    Interpolates the background profile (from off-spike angles) onto each
    arm's radii and subtracts it, then fits:

        ``signal(r) = (A / r^γ) · (1 − C · cos(2π·f·r + φ))``

    This is the physically correct model: the radial spike profile has a
    power-law envelope (γ≈2 for monochromatic; broadband can differ) with
    sinusoidal fringes from diffraction.  Unlike fitting sinc² directly,
    C < 1 keeps the model strictly positive so log-space fitting is
    well-conditioned and the optimizer cannot escape into the b→0 degenerate
    solution.

    The oscillation frequency ``f`` is initialised from an FFT of the
    envelope-subtracted residuals.

    Parameters
    ----------
    r_pos, p_pos, r_neg, p_neg : ndarray
        Arm radii and raw profiles (both arms of one spike).
    r_bg, p_bg : ndarray
        Background profile radii and values (median of off-spike angles).
    blank_r : float
        Core radius to exclude from fitting.
    smooth_size, smooth_size_cross : int
        Median-filter windows for fitting data and endpoint detection.
    r_min : float or *None*
        Minimum fitting radius; defaults to ``blank_r``.
    r_max_fraction : float
        Upper fitting boundary as fraction of profile length.
    bg_floor : float or *None*
        Minimum background value before subtraction.  Prevents near-zero
        background from causing log-space explosions.  Defaults to the 5th
        percentile of positive background values.

    Returns
    -------
    popt : ndarray ``[A, gamma, f, C, phi]`` or *None*
        Fitted parameters of ``_powerlaw_osc_model``.
    p_pos_sig_s, p_neg_sig_s : ndarray
        Background-subtracted profiles smoothed for endpoint detection.
    """
    sz_cross = max(1, int(smooth_size_cross if smooth_size_cross is not None
                          else smooth_size * 2))
    sz_fit = max(1, max(3, int(smooth_size) // 2))
    fit_r_min = max(blank_r, r_min if r_min is not None else 1.0)

    # Floor the background to avoid near-zero values causing log-space
    # divergence after subtraction.
    pos_bg = p_bg[p_bg > 0]
    if bg_floor is None:
        bg_floor = float(np.percentile(pos_bg, 5)) if pos_bg.size > 0 else 1e-10
    bg_floor = max(bg_floor, 1e-10)
    p_bg_safe = np.maximum(p_bg, bg_floor)

    def _subtract_bg(r_arm, p_arm, sz):
        p_sm = median_filter(p_arm, size=sz, mode='nearest') if sz > 1 else p_arm.copy()
        p_bg_interp = np.interp(r_arm, r_bg, p_bg_safe,
                                left=p_bg_safe[0], right=p_bg_safe[-1])
        return p_sm - p_bg_interp

    # Lightly smoothed signal profiles for fitting
    p_pos_sig_fit = _subtract_bg(r_pos, p_pos, sz_fit)
    p_neg_sig_fit = _subtract_bg(r_neg, p_neg, sz_fit)

    # Heavily smoothed signal profiles for endpoint threshold crossing
    p_pos_sig_s = _subtract_bg(r_pos, p_pos, sz_cross)
    p_neg_sig_s = _subtract_bg(r_neg, p_neg, sz_cross)

    eps = 1e-30

    # Build minimum-envelope on a shared radius grid.
    r_hi_grid = min(float(r_pos[-1]), float(r_neg[-1])) * r_max_fraction
    r_comb, p_comb = np.array([]), np.array([])
    if fit_r_min < r_hi_grid:
        r_grid = np.arange(fit_r_min, r_hi_grid, 1.0)
        if len(r_grid) >= 6:
            pp = np.interp(r_grid, r_pos, p_pos_sig_fit)
            pn = np.interp(r_grid, r_neg, p_neg_sig_fit)
            both_positive = (pp > 0) & (pn > 0)
            p_min = np.minimum(np.maximum(pp, eps), np.maximum(pn, eps))
            valid = both_positive & (p_min > eps)
            r_comb = r_grid[valid]
            p_comb = p_min[valid]

    # Fallback: concatenate individually masked arm data.
    if len(r_comb) < 6:
        mask_pos = (
            (r_pos > fit_r_min)
            & (r_pos < r_max_fraction * float(r_pos[-1]))
            & (p_pos_sig_fit > 0)
        )
        mask_neg = (
            (r_neg > fit_r_min)
            & (r_neg < r_max_fraction * float(r_neg[-1]))
            & (p_neg_sig_fit > 0)
        )
        r_comb = np.concatenate([r_pos[mask_pos], r_neg[mask_neg]])
        p_comb = np.concatenate([
            np.maximum(p_pos_sig_fit[mask_pos], eps),
            np.maximum(p_neg_sig_fit[mask_neg], eps),
        ])

    if len(r_comb) < 6:
        return None, p_pos_sig_s, p_neg_sig_s

    # ── Step 1: fit power-law envelope in log-log space ───────────────────
    # Use the upper percentile at each radius to trace the envelope rather
    # than the raw minimum, since oscillation troughs pull the minimum down.
    try:
        slope, intercept = np.polyfit(np.log10(r_comb), np.log10(p_comb), 1)
        gamma0 = float(np.clip(-slope, 0.5, 4.0))
        A0 = float(10.0 ** intercept)
    except Exception:
        gamma0 = 2.0
        A0 = float(np.median(p_comb)) * float(np.median(r_comb)) ** gamma0
    A0 = max(A0, eps)

    # ── Step 2: estimate oscillation frequency from FFT of residuals ──────
    # Interpolate onto a uniform 1-px grid, divide by the fitted envelope,
    # then FFT to find the dominant fringe frequency.
    r_uni = np.arange(r_comb[0], r_comb[-1] + 1, 1.0)
    p_uni = np.interp(r_uni, r_comb, p_comb)
    envelope0 = A0 / r_uni ** gamma0
    # Normalised residual: (signal / envelope) - 1  ∈ [-1, +1]
    norm_resid = p_uni / np.maximum(envelope0, eps) - 1.0
    fft_coeffs = np.fft.rfft(norm_resid)
    freqs = np.fft.rfftfreq(len(norm_resid), d=1.0)
    # Ignore DC (index 0); find dominant positive frequency
    power = np.abs(fft_coeffs[1:]) ** 2
    dominant = int(np.argmax(power)) + 1
    f0 = float(freqs[dominant])
    f0 = max(f0, 1.0 / len(r_uni))   # at least one cycle over the profile

    # ── Step 3: initial C and phi from the dominant FFT component ─────────
    phase0 = float(np.angle(fft_coeffs[dominant]))
    C0 = float(np.clip(2.0 * np.abs(fft_coeffs[dominant]) / len(norm_resid), 0.0, 0.9))
    C0 = max(C0, 0.1)   # ensure some oscillation is allowed

    p0 = [A0, gamma0, f0, C0, phase0]
    bounds = (
        [eps,  0.5, 1.0 / len(r_uni), 0.0, -np.pi],
        [np.inf, 4.0, 0.5,             0.99,  np.pi],
    )

    try:
        popt, _ = curve_fit(
            _log_powerlaw_osc_model, r_comb, np.log10(p_comb),
            p0=p0, bounds=bounds, maxfev=10000,
        )
        return np.array(popt), p_pos_sig_s, p_neg_sig_s
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"power-law+oscillation signal fit failed: {e}")
        return None, p_pos_sig_s, p_neg_sig_s


def _fit_shared_halo(
    arm_pairs_list,
    blank_r: float = 0.0,
    smooth_size: int = 10,
    r_min: Optional[float] = None,
    r_max_fraction: float = 0.7,
    halo_region_start: float = 0.4,
) -> Optional[Tuple[float, float, float]]:
    """Fit the shared PSF halo + background from all spike arms jointly.

    The halo ``c / r^alpha + d`` is azimuthally symmetric, so all spike
    arms from the same star share the same ``(c, alpha, d)``.  Fitting
    these three parameters from all arms together gives a much more robust
    estimate than trying to separate them per arm.

    Uses the outer ``[halo_region_start, r_max_fraction]`` fraction of each
    arm's profile, where the sinc² oscillations have largely decayed and the
    signal is halo-dominated.

    For each spike pair the two arms are interpolated onto a shared radius
    grid and the element-wise minimum is taken before contributing to the
    joint fit.  This suppresses contamination from neighbouring sources: a
    neighbour inflating one arm at a given radius is discarded in favour of
    the lower, uncontaminated arm.

    Parameters
    ----------
    arm_pairs_list : list of (r_pos, p_pos, r_neg, p_neg) tuples
        Both arms for each detected spike.
    blank_r, smooth_size, r_min, r_max_fraction
        Same meaning as in `_fit_spike_profile`.
    halo_region_start : float
        Inner boundary of the halo-fitting window, as a fraction of each
        arm's profile length.  Default 0.4 avoids the bright inner peak
        where sinc² dominates.

    Returns
    -------
    (c, alpha, d) : tuple of float, or *None* on failure
    """
    sz = max(1, int(smooth_size))
    fit_r_min = max(blank_r, r_min if r_min is not None else 1.0)

    r_all, p_all = [], []
    for r_pos, p_pos, r_neg, p_neg in arm_pairs_list:
        p_pos_s = median_filter(p_pos, size=sz, mode='nearest') if sz > 1 else p_pos.copy()
        p_neg_s = median_filter(p_neg, size=sz, mode='nearest') if sz > 1 else p_neg.copy()

        # For each arm, attempt to form a minimum-envelope with the opposite arm
        # over the halo window, falling back to the arm alone when the other is
        # too short to cover the relevant radius range.
        for r_arm, p_arm_s, r_other, p_other_s in [
            (r_pos, p_pos_s, r_neg, p_neg_s),
            (r_neg, p_neg_s, r_pos, p_pos_s),
        ]:
            if len(r_arm) < 4:
                continue
            r_max_arm = float(r_arm[-1])
            r_lo = max(fit_r_min, halo_region_start * r_max_arm)
            r_hi = r_max_fraction * r_max_arm
            if r_lo >= r_hi:
                continue
            r_grid = np.arange(r_lo, r_hi, 1.0)
            if len(r_grid) < 3:
                continue
            p_arm_interp = np.interp(r_grid, r_arm, p_arm_s)
            if len(r_other) >= 4 and float(r_other[-1]) >= r_lo:
                p_other_interp = np.interp(r_grid, r_other, p_other_s,
                                           left=np.nan, right=np.nan)
                both_valid = np.isfinite(p_other_interp) & (p_arm_interp > 0) & (p_other_interp > 0)
                if both_valid.sum() >= 3:
                    r_all.append(r_grid[both_valid])
                    p_all.append(np.minimum(p_arm_interp[both_valid], p_other_interp[both_valid]))
                    continue
            # Fallback: use this arm alone
            mask = p_arm_interp > 0
            if mask.sum() >= 3:
                r_all.append(r_grid[mask])
                p_all.append(p_arm_interp[mask])

    if not r_all:
        return None

    r_comb = np.concatenate(r_all)
    p_comb = np.concatenate(p_all)

    if len(r_comb) < 5:
        return None

    d0 = max(float(np.percentile(p_comb, 10)), 1e-10)
    p_above = np.maximum(p_comb - d0, 1e-10)
    try:
        slope, intercept = np.polyfit(np.log(r_comb), np.log(p_above), 1)
        alpha0 = float(np.clip(-slope, 0.1, 3.0))
        c0 = max(float(np.exp(intercept)), d0)
    except Exception:
        alpha0, c0 = 1.5, float(np.nanmedian(p_comb))

    def _log_halo(r, c, alpha, d):
        return np.log10(np.maximum(c / r ** alpha + d, 1e-30))

    try:
        popt, _ = curve_fit(
            _log_halo, r_comb, np.log10(p_comb),
            p0=[c0, alpha0, d0],
            bounds=([0.0, 0.1, 0.0], [np.inf, 3.0, np.inf]),
            maxfev=5000,
        )
        return tuple(float(v) for v in popt)  # (c, alpha, d)
    except (RuntimeError, ValueError, TypeError):
        return None


def _fit_spike_profile(
    r_pos: np.ndarray,
    p_pos: np.ndarray,
    r_neg: np.ndarray,
    p_neg: np.ndarray,
    blank_r: float = 0.0,
    smooth_size: int = 10,
    smooth_size_cross: Optional[int] = None,
    r_min: Optional[float] = None,
    r_max_fraction: float = 0.7,
    shared_halo: Optional[Tuple[float, float, float]] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Smooth both arm profiles and fit a combined Fraunhofer+halo model.

    Both arms are fit jointly in log₁₀ space.  When ``shared_halo`` is
    provided the halo parameters ``(c, alpha, d)`` are fixed, and only the
    sinc² spike parameters ``(a, b)`` are free.  This gives a much better-
    constrained fit for short profiles and when multiple spikes are present.
    Falls back to the full 5-parameter fit if the constrained fit fails.

    The two arms are interpolated onto a shared 1-px radius grid and the
    element-wise minimum is used as the fitting profile.  This suppresses
    contamination from neighbouring sources.  Falls back to concatenation if
    the arms do not share enough radial overlap.

    Parameters
    ----------
    r_pos, p_pos : ndarray
        Radii and raw-image swath-profile values for the positive arm.
    r_neg, p_neg : ndarray
        Same for the negative arm.
    blank_r : float
        Radius of any NaN/zero core to exclude from fitting.
    smooth_size : int
        Median-filter window used for the fitting data (lighter: half of
        ``smooth_size_cross``).  The sinc² fringe spacing (~1/b ≈ 200–1000 px)
        greatly exceeds this window, so no fringe information is lost.
    smooth_size_cross : int or *None*
        Median-filter window applied to the profiles returned for endpoint
        detection.  Defaults to ``2 * smooth_size``.  A larger window
        suppresses noise spikes in the faint outer tail, preventing false
        early termination of the threshold crossing.
    r_min : float or *None*
        Minimum radius included in the fit.  Defaults to ``blank_r``.
    r_max_fraction : float
        Upper fitting boundary as a fraction of each arm's profile length.
    shared_halo : (c, alpha, d) or *None*
        Pre-fitted halo parameters shared across all spikes on this star.
        When provided the fit has only 2 free parameters ``(a, b)``.

    Returns
    -------
    popt : ndarray or *None*
        Fitted parameters ``(a, b, c, alpha, d)``, or *None* on failure.
    p_pos_smooth, p_neg_smooth : ndarray
        Profiles smoothed with ``smooth_size_cross`` for endpoint detection.
    """
    sz_cross = max(1, int(smooth_size_cross if smooth_size_cross is not None
                          else smooth_size * 2))
    sz_fit = max(1, max(3, int(smooth_size) // 2))

    # Lightly smoothed profiles used only for building the fitting data.
    p_pos_fit = median_filter(p_pos, size=sz_fit, mode='nearest') if sz_fit > 1 else p_pos.copy()
    p_neg_fit = median_filter(p_neg, size=sz_fit, mode='nearest') if sz_fit > 1 else p_neg.copy()

    # More heavily smoothed profiles returned for threshold crossing.
    p_pos_s = median_filter(p_pos, size=sz_cross, mode='nearest') if sz_cross > 1 else p_pos.copy()
    p_neg_s = median_filter(p_neg, size=sz_cross, mode='nearest') if sz_cross > 1 else p_neg.copy()

    fit_r_min = max(blank_r, r_min if r_min is not None else 1.0)

    # Build minimum-envelope on a shared radius grid.  At each radius the
    # lower of the two (lightly smoothed) arm values is used; a neighbour
    # source inflating one arm is discarded in favour of the other.
    r_hi_grid = min(float(r_pos[-1]), float(r_neg[-1])) * r_max_fraction
    r_comb, p_comb = np.array([]), np.array([])
    if fit_r_min < r_hi_grid:
        r_grid = np.arange(fit_r_min, r_hi_grid, 1.0)
        if len(r_grid) >= 6:
            p_pos_interp = np.interp(r_grid, r_pos, p_pos_fit)
            p_neg_interp = np.interp(r_grid, r_neg, p_neg_fit)
            p_min = np.minimum(p_pos_interp, p_neg_interp)
            valid = p_min > 0
            r_comb = r_grid[valid]
            p_comb = p_min[valid]

    # Fallback: concatenate individually masked arm data.
    if len(r_comb) < 6:
        mask_pos = (
            (r_pos > fit_r_min)
            & (r_pos < r_max_fraction * float(r_pos[-1]))
            & (p_pos_fit > 0)
        )
        mask_neg = (
            (r_neg > fit_r_min)
            & (r_neg < r_max_fraction * float(r_neg[-1]))
            & (p_neg_fit > 0)
        )
        r_comb = np.concatenate([r_pos[mask_pos], r_neg[mask_neg]])
        p_comb = np.concatenate([p_pos_fit[mask_pos], p_neg_fit[mask_neg]])

    if len(r_comb) < 6:
        return None, p_pos_s, p_neg_s

    # ── constrained fit: (a, b) only, halo fixed ──────────────────────────
    if shared_halo is not None:
        c_s, alpha_s, d_s = shared_halo

        def _log_model_ab(r, a, b):
            val = a * np.sinc(b * r) ** 2 + c_s / r ** alpha_s + d_s
            return np.log10(np.maximum(val, 1e-30))

        # Initial (a, b): subtract known halo, estimate from residual spike
        p_spike = np.maximum(p_comb - c_s / r_comb ** alpha_s - d_s, 1e-10)
        a0 = max(float(np.percentile(p_spike, 90)), 1e-10)
        p_half = 0.5 * p_spike.max()
        cands = r_comb[p_spike >= p_half]
        r_half = float(cands[-1]) if len(cands) else float(r_comb[len(r_comb) // 4])
        b0 = float(np.clip(0.6 / max(r_half, 1.0), 1e-5, 0.05))

        try:
            (a_fit, b_fit), _ = curve_fit(
                _log_model_ab, r_comb, np.log10(p_comb),
                p0=[a0, b0],
                bounds=([0.0, 1e-5], [np.inf, 0.1]),
                maxfev=5000,
            )
            return np.array([a_fit, b_fit, c_s, alpha_s, d_s]), p_pos_s, p_neg_s
        except (RuntimeError, ValueError, TypeError):
            pass  # fall through to unconstrained fit

    # ── unconstrained fit: all 5 parameters ───────────────────────────────
    p0 = _estimate_p0(r_comb, p_comb)

    try:
        popt, _ = curve_fit(
            _log_fraunhofer_model,
            r_comb,
            np.log10(p_comb),
            p0=p0,
            maxfev=10000,
        )
        return popt, p_pos_s, p_neg_s
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"Fraunhofer fit failed: {e}")
        return None, p_pos_s, p_neg_s


def _find_envelope_crossing(popt, threshold, r_start):
    """Find the radius where the Fraunhofer envelope crosses *threshold*.

    Searches from *r_start* outward (for unconverged arms the profile ends
    at *r_start*, so the crossing must be extrapolated beyond it).  Falls
    back to a search inside ``[1 px, r_start]`` if the envelope is already
    below threshold there.

    Returns the crossing radius, or *None* if the envelope never drops
    below *threshold* (e.g. ``threshold <= d``).
    """
    a, b, c, alpha, d = popt
    if threshold <= d:
        return None  # envelope asymptotes to d, never drops below threshold

    def f(r):
        return _envelope_model(r, a, b, c, alpha, d) - threshold

    # Case 1: envelope already below threshold at r_start — find crossing inside
    if f(r_start) <= 0:
        r_lo = max(1.0, 1.0 / (np.pi * b))
        if f(r_lo) <= 0:
            return float(r_lo)
        try:
            res = root_scalar(f, bracket=[r_lo, r_start], method='brentq')
            return float(res.root) if res.converged else None
        except ValueError:
            return None

    # Case 2: envelope still above threshold at r_start — extrapolate outward
    r_search = r_start
    for _ in range(20):
        r_search *= 2.0
        if f(r_search) <= 0:
            try:
                res = root_scalar(f, bracket=[r_start, r_search], method='brentq')
                return float(res.root) if res.converged else None
            except ValueError:
                return None
    return None  # envelope never drops below threshold


def _find_profile_crossing(
    radii,
    profile_smooth,
    threshold,
    blank_r: float = 0.0,
):
    """Return the first radius after the last above-threshold point.

    Skips the blank-core region (``r ≤ blank_r``) and finds the last radius
    where the smoothed profile is at or above *threshold*, then returns the
    next radius as the endpoint.

    This is inherently robust to sinc² oscillation troughs: a temporary dip
    below threshold that later recovers simply moves ``last_above`` forward,
    so the trough is never mistaken for the arm end.

    Returns ``(radius, True)`` when the profile drops and stays below
    *threshold* within the measured extent.  Returns ``(None, False)`` if the
    profile is still above *threshold* at the outermost measured radius
    (spike reaches the profile edge without converging).
    """
    skip_before = int(np.searchsorted(radii, blank_r))
    above = np.where(profile_smooth[skip_before:] >= threshold)[0]
    if len(above) == 0:
        # Profile never rises above threshold beyond blank_r
        return float(radii[skip_before]), True
    last_above = above[-1] + skip_before
    if last_above + 1 < len(radii):
        return float(radii[last_above + 1]), True
    return None, False


# ── public API ────────────────────────────────────────────────────────────────

def measure_spike_lengths(
    image,
    result,
    swath_width=None,
    length_sigma=2.0,
    centre=None,
    radial_bin_width=2,
    smooth_size=10,
    smooth_size_cross=None,
    r_min=None,
    r_max_fraction=0.7,
    max_radius=None,
    median_subtract=False,
    background_profiles=True,
    max_output_length=None,
    background_subtract=True,
):
    """Measure the length of each detected spike arm.

    Background-subtraction mode (``background_subtract=True``, default)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Profiles from random angles far from any spike are extracted and
    median-combined to form a background profile that captures the stellar
    halo + sky at each radius.  This is subtracted from each spike arm
    profile, isolating the diffraction-spike signal.  Only the sinc²
    parameters ``(a, b)`` are then fitted (2 free parameters), giving a
    robust fit that does not require the halo/core to be modelled jointly
    with the spike.

    The interpolated background is floored before subtraction to prevent
    log-space divergence where the background profile approaches zero at
    large radii.

    The arm endpoint threshold becomes ``length_sigma × σ_sky`` (not
    ``background + length_sigma × σ_sky``) because the background has
    already been subtracted.

    Full-model mode (``background_subtract=False``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Both arms of each spike are jointly fitted in log₁₀ space to the model

        ``f(r) = a · sinc(b r)²  +  c / r^α  +  d``

    where the halo parameters ``(c, α, d)`` are first estimated from all
    arms combined (``_fit_shared_halo``), then held fixed while ``(a, b)``
    are refined per spike.  Falls back to the unconstrained 5-parameter fit
    if the shared-halo estimate is unavailable.

    Common behaviour
    ~~~~~~~~~~~~~~~~
    The arm endpoint is where the smoothed profile first drops below the
    threshold **and** the model envelope also predicts the spike has ended,
    preventing sinc² oscillation troughs from being mistaken for the end.
    If the spike extends to the profile edge the envelope is extrapolated;
    ``SpikeLengths.converged_pos/neg`` is *False* in that case.

    Parameters
    ----------
    image : 2-D array
        Original (un-preprocessed) image.
    result : `~spikeout.detect.SpikeResult`
        Output of `~spikeout.detect.detect`.
    swath_width : float or *None*
        Width (pixels) of the perpendicular sampling band.  Default:
        ``max(3, min(image.shape) * 0.02)``.
    length_sigma : float
        Arm endpoint threshold multiplier applied to the sky noise ``σ_sky``.
        In background-subtract mode the threshold is ``length_sigma × σ_sky``;
        in full-model mode it is ``background + length_sigma × σ_sky``.
    centre : (row, col) or *None*
        Star centre.  Auto-detected if *None*.
    radial_bin_width : int
        Annular bin width for background estimation.
    smooth_size : int
        Median-filter window for the fitting data (half this value is used
        internally as ``sz_fit``).  Default 10.
    smooth_size_cross : int or *None*
        Median-filter window applied to each arm profile for threshold
        crossing / endpoint detection.  Defaults to ``2 * smooth_size``.
    r_min : float or *None*
        Minimum radius included in the fit.  Defaults to the blank-core
        radius (so saturated/NaN cores are excluded).
    r_max_fraction : float
        Upper boundary of the fitting region as a fraction of the profile
        length.  Default 0.7.
    max_radius : float or *None*
        Maximum swath-walk radius (pixels).  Defaults to the image diagonal.
    median_subtract : bool
        If *True*, subtract the azimuthal median from the image before
        measuring lengths.
    background_profiles : bool
        If *True* (or when ``background_subtract=True``), extract background
        profiles from random off-spike angles.  Always forced *True* when
        ``background_subtract=True``.
    max_output_length : int or *None*
        Maximum length of each arm allowed (pixels).
    background_subtract : bool
        If *True* (default), subtract the median background profile from each
        spike arm before fitting, then fit only the sinc² signal.  If *False*,
        use the full 5-parameter Fraunhofer fit.

    Returns
    -------
    lengths : list of `SpikeLengths`
    """
    image = np.asarray(image, dtype=float)

    # Blank-core radius — needed to skip corrupted inner profile segment
    blank_r = _blank_core_radius(image, centre=centre) + 3.0

    # Working copy with NaNs filled so swath sampling always has data
    img = image.copy()
    img[~np.isfinite(img)] = 0.0

    if centre is None:
        centre = find_centre(image)

    if median_subtract:
        model = azimuthal_median(
            image, centre=centre, radial_bin_width=radial_bin_width,
        )
        residual = image - model
        bg_level = 0.0
    else:
        residual = image
        bg_level = None  # computed below after cy/cx are set

    if swath_width is None:
        swath_width = max(3.0, min(image.shape) * 0.02)

    cy, cx = centre
    ny, nx = img.shape

    if max_radius is None:
        max_radius = float(np.hypot(ny, nx))

    # ── Background noise via pixel-to-pixel differences ───────────────────
    # Differencing adjacent pixels cancels the smooth PSF halo gradient,
    # leaving only sky noise.
    Y_g, X_g = np.ogrid[:ny, :nx]
    R_from_centre = np.sqrt((X_g - cx) ** 2 + (Y_g - cy) ** 2)
    r_noise_annulus = 0.75 * min(ny, nx) / 2.0

    dx_diff = np.diff(img, axis=1)
    dy_diff = np.diff(img, axis=0)
    noise_dx = dx_diff[R_from_centre[:, :-1] > r_noise_annulus]
    noise_dy = dy_diff[R_from_centre[:-1, :] > r_noise_annulus]
    noise_all = np.concatenate([noise_dx, noise_dy])
    noise_all = noise_all[np.isfinite(noise_all)]

    if noise_all.size >= 20:
        sigma_bg = float(mad_std(noise_all)) / np.sqrt(2)
    else:
        sigma_bg = float(mad_std(img[np.isfinite(img)]))

    if bg_level is None:
        outer_px = img[R_from_centre > r_noise_annulus]
        outer_px = outer_px[np.isfinite(outer_px)]
        bg_level = float(np.median(outer_px)) if outer_px.size >= 10 \
            else float(np.median(img[np.isfinite(img)]))

    theta = result.theta
    pk_th = result.peak_theta_indices

    # ── Pass 1: extract all arm profiles ─────────────────────────────────
    arm_profiles = []
    for i in range(len(result.angles)):
        rho = result.rho_physical[i]
        th_rad = np.deg2rad(theta[pk_th[i]])
        x0 = nx / 2.0 + rho * np.cos(th_rad)
        y0 = ny / 2.0 - rho * np.sin(th_rad)
        angle = result.angles[i]

        r_pos, p_pos = _extract_swath_profile(
            img, x0, y0, angle, swath_width, max_radius=max_radius,
        )
        r_neg, p_neg = _extract_swath_profile(
            img, x0, y0, (angle + 180.0) % 360.0, swath_width,
            max_radius=max_radius,
        )
        arm_profiles.append((r_pos, p_pos, r_neg, p_neg, x0, y0, angle))

    # ── Background profile extraction ────────────────────────────────────
    # Always extract when using background subtraction; optionally otherwise.
    bg_profile_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if background_subtract or background_profiles:
        bg_samples = []
        random_angles = np.random.uniform(0, 360, size=40)
        for bg_angle in random_angles:
            # Require > 10° separation from every spike arm direction.
            if np.all(np.abs((bg_angle - result.angles + 180) % 360 - 180) > 10):
                r_bg_s, p_bg_s = _extract_swath_profile(
                    img, cx, cy, bg_angle, swath_width, max_radius=max_radius,
                )
                bg_samples.append((r_bg_s, p_bg_s))
            if len(bg_samples) >= 6:
                break

        if bg_samples:
            # Combine all samples onto a sorted radius grid and median-smooth.
            r_bg_comb = np.concatenate([r for r, p in bg_samples])
            p_bg_comb = np.concatenate([p for r, p in bg_samples])
            sort_idx = np.argsort(r_bg_comb)
            r_bg_sorted = r_bg_comb[sort_idx]
            p_bg_sorted = p_bg_comb[sort_idx]
            sz = max(1, int(smooth_size))
            p_bg_smooth = (
                median_filter(p_bg_sorted, size=sz, mode='nearest')
                if sz > 1 else p_bg_sorted.copy()
            )
            bg_profile_data = (r_bg_sorted, p_bg_smooth)

    # ── Threshold ─────────────────────────────────────────────────────────
    if background_subtract and bg_profile_data is not None:
        # After subtracting the background the signal should be zero in
        # sky-only regions; threshold is purely noise-based.
        threshold = length_sigma * max(sigma_bg, 1e-10)
    else:
        threshold = bg_level + length_sigma * max(sigma_bg, 1e-10)

    # ── Pass 2: per-spike fit and model-aware endpoint detection ──────────
    # Choose fitting strategy: background-subtract (preferred) or full model.
    use_bg_sub = background_subtract and bg_profile_data is not None

    if not use_bg_sub:
        # Pre-compute shared halo for the full-model path.
        arm_pairs = [
            (r_pos, p_pos, r_neg, p_neg)
            for (r_pos, p_pos, r_neg, p_neg, *_) in arm_profiles
        ]
        shared_halo = _fit_shared_halo(
            arm_pairs,
            blank_r=blank_r,
            smooth_size=smooth_size,
            r_min=r_min,
            r_max_fraction=r_max_fraction,
        )
    else:
        shared_halo = None

    lengths = []
    for r_pos, p_pos, r_neg, p_neg, x0, y0, angle in arm_profiles:

        if use_bg_sub:
            r_bg_fit, p_bg_fit = bg_profile_data
            popt, p_pos_s, p_neg_s = _fit_spike_signal_only(
                r_pos, p_pos, r_neg, p_neg,
                r_bg_fit, p_bg_fit,
                blank_r=blank_r,
                smooth_size=smooth_size,
                smooth_size_cross=smooth_size_cross,
                r_min=r_min,
                r_max_fraction=r_max_fraction,
            )
        else:
            popt, p_pos_s, p_neg_s = _fit_spike_profile(
                r_pos, p_pos, r_neg, p_neg,
                blank_r=blank_r,
                smooth_size=smooth_size,
                smooth_size_cross=smooth_size_cross,
                r_min=r_min,
                r_max_fraction=r_max_fraction,
                shared_halo=shared_halo,
            )

        arm_results = {}
        for label, radii_arm, p_arm_s in [
            ("pos", r_pos, p_pos_s),
            ("neg", r_neg, p_neg_s),
        ]:
            r_cross, converged = _find_profile_crossing(
                radii_arm, p_arm_s, threshold, blank_r=blank_r,
            )
            if converged:
                arm_results[label] = (r_cross, radii_arm, p_arm_s, True)
            elif popt is not None:
                if use_bg_sub:
                    A_f, gamma_f, _f, C_f, _phi = popt
                    r_extrap = _powerlaw_osc_envelope_crossing(
                        A_f, gamma_f, C_f, threshold,
                    )
                else:
                    r_extrap = _find_envelope_crossing(popt, threshold, float(radii_arm[-1]))
                if r_extrap is not None:
                    if r_extrap < float(radii_arm[-1]):
                        r_extrap = float(radii_arm[-1])
                    arm_results[label] = (r_extrap, radii_arm, p_arm_s, False)
                else:
                    arm_results[label] = (float(radii_arm[-1]), radii_arm, p_arm_s, False)
            else:
                arm_results[label] = (float(radii_arm[-1]), radii_arm, p_arm_s, False)

        if max_output_length is not None:
            if arm_results["pos"][0] > max_output_length:
                arm_results["pos"] = (max_output_length, *arm_results["pos"][1:])
            if arm_results["neg"][0] > max_output_length:
                arm_results["neg"] = (max_output_length, *arm_results["neg"][1:])

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
            popt=popt,
            threshold=threshold,
            background_profile=bg_profile_data,
            swath_width=swath_width,
            background_subtracted=use_bg_sub,
        ))

    return lengths
