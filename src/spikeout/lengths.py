"""Measure the length of each diffraction-spike arm via Fraunhofer profile fitting."""

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
        Median-smoothed swath profile along each arm (raw-image values).
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


def _fit_shared_halo(
    arm_data_list,
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

    Parameters
    ----------
    arm_data_list : list of (r, p) tuples
        All arm profiles to use (typically both arms of every spike).
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
    for r, p in arm_data_list:
        if len(r) < 4:
            continue
        p_s = median_filter(p, size=sz, mode='nearest') if sz > 1 else p.copy()
        r_max_arm = float(r[-1])
        r_lo = max(fit_r_min, halo_region_start * r_max_arm)
        r_hi = r_max_fraction * r_max_arm
        mask = (r > r_lo) & (r <= r_hi) & (p_s > 0)
        if mask.sum() >= 3:
            r_all.append(r[mask])
            p_all.append(p_s[mask])

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

    Parameters
    ----------
    r_pos, p_pos : ndarray
        Radii and raw-image swath-profile values for the positive arm.
    r_neg, p_neg : ndarray
        Same for the negative arm.
    blank_r : float
        Radius of any NaN/zero core to exclude from fitting.
    smooth_size : int
        Median-filter window applied before fitting.
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
        Smoothed profiles (returned regardless of fit success).
    """
    sz = max(1, int(smooth_size))
    p_pos_s = median_filter(p_pos, size=sz, mode='nearest') if sz > 1 else p_pos.copy()
    p_neg_s = median_filter(p_neg, size=sz, mode='nearest') if sz > 1 else p_neg.copy()

    fit_r_min = max(blank_r, r_min if r_min is not None else 1.0)

    mask_pos = (
        (r_pos > fit_r_min)
        & (r_pos < r_max_fraction * float(r_pos[-1]))
        & (p_pos_s > 0)
    )
    mask_neg = (
        (r_neg > fit_r_min)
        & (r_neg < r_max_fraction * float(r_neg[-1]))
        & (p_neg_s > 0)
    )

    r_comb = np.concatenate([r_pos[mask_pos], r_neg[mask_neg]])
    p_comb = np.concatenate([p_pos_s[mask_pos], p_neg_s[mask_neg]])

    if len(r_comb) < 6:
        return None, p_pos_s, p_neg_s

    # ── constrained fit: (a, b) only, halo fixed ──────────────────────────
    if shared_halo is not None and False:
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
                #bounds=([0.0, 1e-5], [np.inf, 0.1]),
                maxfev=5000,
            )
            print(a_fit, b_fit, c_s, alpha_s, d_s)
            return np.array([a_fit, b_fit, c_s, alpha_s, d_s]), p_pos_s, p_neg_s
        except (RuntimeError, ValueError, TypeError):
            pass  # fall through to unconstrained fit

    # ── unconstrained fit: all 5 parameters ───────────────────────────────
    p0 = _estimate_p0(r_comb, p_comb)
    bounds = None #([0, 1e-5, 0, 0.1, 0], [np.inf, 0.1, np.inf, 3.0, np.inf])
    
    p0 = np.array([8.78245565e-01, 4.15452670e-03, 2.67775665e+04, 2.34681226e+00,
       1.42098557e-15])
    print("Using fixed initial parameters:", p0)

    try:
        popt, _ = curve_fit(
            _log_fraunhofer_model,
            r_comb,
            np.log10(p_comb),
            p0=p0,
            #bounds=bounds,
            maxfev=10000,
        )
        return popt, p_pos_s, p_neg_s
    except (RuntimeError, ValueError, TypeError) as e:
        print(e)
        return None, p_pos_s, p_neg_s


def _find_envelope_crossing(popt, threshold, r_max):
    """Find the radius where the Fraunhofer envelope crosses *threshold*.

    Uses Brent's method on a bracketed interval.

    Returns the crossing radius, or *None* if the envelope never drops
    below *threshold* within ``[1 px, r_max]``.
    """
    a, b, c, alpha, d = popt
    if threshold <= d:
        return None

    def f(r):
        return _envelope_model(r, a, b, c, alpha, d) - threshold

    r_lo = max(1.0, 1.0 / (np.pi * b))  # near the first sinc² peak
    if f(r_lo) <= 0:
        return float(r_lo)
    if f(r_max) > 0:
        return float(r_max)  # still above threshold at image edge

    try:
        res = root_scalar(f, bracket=[r_lo, r_max], method='brentq')
        return float(res.root) if res.converged else None
    except ValueError:
        return None


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
    r_min=None,
    r_max_fraction=0.7,
    max_radius=None,
    median_subtract=False,
    background_profiles=True,
):
    """Measure the length of each detected spike arm.

    Profiles are extracted along each arm from the **raw image** so the
    full Fraunhofer + halo signal is available for fitting.  Both arms of
    each spike are smoothed with a median filter, then jointly fitted in
    log₁₀ space to the model

        ``f(r) = a · sinc(b r)²  +  c / r^α  +  d``

    where the sinc² term captures the diffraction spike oscillations, the
    power law captures the stellar halo, and ``d`` is the background floor.

    The halo parameters ``(c, alpha, d)`` are the same for all spikes on a
    given star (azimuthal symmetry of the PSF), so they are fitted once from
    all arms combined before the per-spike ``(a, b)`` fit.  This significantly
    improves robustness for short profiles and for stars with many spikes.

    The arm endpoint is where the smoothed profile first drops below a
    background threshold **and** the Fraunhofer envelope also predicts the
    spike has ended (preventing sinc² oscillation troughs from being mistaken
    for the end).  If the spike extends to the profile edge, the envelope is
    used to extrapolate; ``SpikeLengths.converged_pos/neg`` is *False* then.

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
        Arm endpoint threshold: ``background + length_sigma × σ_sky``,
        where ``σ_sky`` is the per-pixel sky noise estimated from
        pixel-to-pixel differences in the outer image annulus.
    centre : (row, col) or *None*
        Star centre.  Auto-detected if *None*.
    radial_bin_width : int
        Annular bin width for background estimation.
    smooth_size : int
        Median-filter window applied to each arm profile before fitting
        and threshold crossing.  Default 10.
    r_min : float or *None*
        Minimum radius included in the log-space fit.  Defaults to the
        blank-core radius (so saturated/NaN cores are excluded).
    r_max_fraction : float
        Upper boundary of the fitting region as a fraction of the profile
        length.  The noisy outer tail (beyond this fraction) is excluded
        from the fit but used for the threshold crossing.  Default 0.7.
    max_radius : float or *None*
        Maximum swath-walk radius (pixels).  Defaults to the image
        diagonal — set smaller to limit profile extraction cost.
    median_subtract : bool
        If *True*, subtract the azimuthal median from the image before
        measuring lengths.  This can help isolate the spike profile from the
        PSF halo, but may not be desirable if the halo is very asymmetric.
    background_profiles : bool
        If *True*, extract background profiles from random angles for sanity check and potential future use.

    Returns
    -------
    lengths : list of `SpikeLengths`
    """
    image = np.asarray(image, dtype=float)

    # Blank-core radius — needed to skip corrupted inner profile segment
    blank_r = _blank_core_radius(image, centre=centre) + 3.0  # add small margin to ensure all blank pixels are excluded

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
        # Background level: median of outer image annulus (far from star).
        bg_level = None  # computed below after cy/cx are set

    if swath_width is None:
        swath_width = max(3.0, min(image.shape) * 0.02)

    cy, cx = centre
    ny, nx = img.shape

    if max_radius is None:
        max_radius = float(np.hypot(ny, nx))

    # ── Background noise via pixel-to-pixel differences ───────────────────
    # Differencing adjacent pixels cancels the smooth PSF halo gradient,
    # leaving only sky noise.  This gives the true per-pixel sigma even for
    # very bright stars where mad_std on the raw image is inflated by the
    # halo signal by 10× or more.
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

    threshold = bg_level + length_sigma * max(sigma_bg, 1e-10)

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
    
    if background_profiles:
        background_profiles = []
        random_angles = np.random.uniform(0, 360, size=40)
        # Pick 6 angles with are > 10 degrees away from any spike arm, and extract background profiles there for sanity check and potential future use.
        for bg_angle in random_angles:
            if np.all(np.abs((bg_angle - result.angles + 180) % 360 - 180) > 10):
                r_bg, p_bg = _extract_swath_profile(
                    img, cx, cy, bg_angle, swath_width, max_radius=max_radius,
                )
                background_profiles.append((r_bg, p_bg, bg_angle))
            if len(background_profiles) >= 6:
                break

    # compute median background profile for sanity check
    if background_profiles:
        r_bg_comb = np.concatenate([r for r, p, a in background_profiles])
        p_bg_comb = np.concatenate([p for r, p, a in background_profiles])
        r_bg_sorted = np.sort(r_bg_comb)
        p_bg_sorted = p_bg_comb[np.argsort(r_bg_comb)]
        sz = max(1, int(smooth_size))
        p_bg_smooth = median_filter(p_bg_sorted, size=sz, mode='nearest') if sz > 1 else p_bg_sorted.copy()
        
    # ── Shared halo fit: (c, alpha, d) from all arms jointly ─────────────
    # The PSF halo is azimuthally symmetric, so these parameters are the
    # same for every spike arm.  Fitting from all arms gives far more data
    # than any single arm and makes the 5-parameter per-spike fit tractable.
    all_arm_data = [
        (r, p)
        for (r_pos, p_pos, r_neg, p_neg, *_) in arm_profiles
        for r, p in [(r_pos, p_pos), (r_neg, p_neg)]
    ]
    shared_halo = _fit_shared_halo(
        all_arm_data,
        blank_r=blank_r,
        smooth_size=smooth_size,
        r_min=r_min,
        r_max_fraction=r_max_fraction,
    )

    # ── Pass 2: per-spike fit and model-aware endpoint detection ──────────
    lengths = []
    for r_pos, p_pos, r_neg, p_neg, x0, y0, angle in arm_profiles:

        popt, p_pos_s, p_neg_s = _fit_spike_profile(
            r_pos, p_pos, r_neg, p_neg,
            blank_r=blank_r,
            smooth_size=smooth_size,
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
                r_extrap = _find_envelope_crossing(popt, threshold, float(radii_arm[-1]))
                if r_extrap is not None:
                    arm_results[label] = (r_extrap, radii_arm, p_arm_s, False)
                else:
                    arm_results[label] = (float(radii_arm[-1]), radii_arm, p_arm_s, False)
            else:
                arm_results[label] = (float(radii_arm[-1]), radii_arm, p_arm_s, False)

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
            background_profile=(r_bg_sorted, p_bg_smooth) if background_profiles else None,
            swath_width=swath_width,
        ))

    return lengths
