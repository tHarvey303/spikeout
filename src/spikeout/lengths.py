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


def _extract_swath_profile(image, x0, y0, angle_deg, swath_width,
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

        if valid.sum() < 2:
            break

        vals = image[iy[valid], ix[valid]]
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size > 0:
            profile[i] = np.median(finite_vals)

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
    """Estimate initial Fraunhofer model parameters from profile data."""
    d0 = max(float(np.percentile(p, 5)), 1e-10)
    a0 = max(float(np.percentile(p, 95)) - d0, d0)

    # b0: sinc²(b·r) = 0.5 near b·r ≈ 0.6 → b ≈ 0.6 / r_half
    p_above = p - d0
    half_level = 0.5 * p_above.max()
    candidates = r[p_above >= half_level]
    r_half = float(candidates[-1]) if len(candidates) else float(r[len(r) // 4])
    b0 = float(np.clip(0.6 / max(r_half, 1.0), 1e-5, 0.05))

    # c0, alpha0: power-law fit to the last 20 % of points
    n_tail = max(3, len(r) // 5)
    r_tail = r[-n_tail:]
    p_tail = np.maximum(p[-n_tail:] - d0, 1e-10)
    try:
        slope, intercept = np.polyfit(np.log(r_tail), np.log(p_tail), 1)
        alpha0 = float(np.clip(-slope, 0.1, 3.0))
        c0 = max(float(np.exp(intercept)), d0)
    except Exception:
        alpha0, c0 = 1.0, a0 * float(r[0])

    return [a0, b0, c0, alpha0, d0]


def _fit_spike_profile(
    r_pos: np.ndarray,
    p_pos: np.ndarray,
    r_neg: np.ndarray,
    p_neg: np.ndarray,
    blank_r: float = 0.0,
    smooth_size: int = 10,
    r_min: Optional[float] = None,
    r_max_fraction: float = 0.7,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Smooth both arm profiles and fit a combined Fraunhofer+halo model.

    Both arms are fit jointly in log₁₀ space to share model parameters,
    giving a more robust estimate than fitting each arm independently.

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

    p0 = _estimate_p0(r_comb, p_comb)
    bounds = ([0, 1e-5, 0, 0.1, 0], [np.inf, 0.1, np.inf, 3.0, np.inf])

    try:
        popt, _ = curve_fit(
            _log_fraunhofer_model,
            r_comb,
            np.log10(p_comb),
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        return popt, p_pos_s, p_neg_s
    except (RuntimeError, ValueError, TypeError):
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


def _find_profile_crossing(radii, profile_smooth, threshold, blank_r=0.0):
    """Return the radius where the smoothed profile first drops below *threshold*.

    Skips the blank-core region (``r ≤ blank_r``) to avoid premature
    termination from NaN/zero-core artefacts in the profile.

    Returns ``(radius, True)`` on success, ``(None, False)`` if the profile
    never crosses within its extent.
    """
    skip_before = int(np.searchsorted(radii, blank_r))
    for j in range(skip_before, len(profile_smooth)):
        if profile_smooth[j] < threshold:
            return float(radii[j]), True
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
):
    """Measure the length of each detected spike arm.

    Profiles are extracted along each arm from the **raw image** so the
    full Fraunhofer + halo signal is available for fitting.  Both arms of
    each spike are smoothed with a median filter, then jointly fitted in
    log₁₀ space to the model

        ``f(r) = a · sinc(b r)²  +  c / r^α  +  d``

    where the sinc² term captures the diffraction spike oscillations, the
    power law captures the stellar halo, and ``d`` is the background floor.

    The arm endpoint is where the smoothed profile first drops below a
    background threshold.  If the spike extends to the profile edge without
    crossing, the upper envelope ``a/(π b r)² + c/r^α + d`` is used to
    extrapolate the crossing radius; ``SpikeLengths.converged_pos/neg`` is
    set to *False* in that case.

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
        Arm endpoint threshold: ``background + length_sigma × σ_MAD``.
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

    Returns
    -------
    lengths : list of `SpikeLengths`
    """
    image = np.asarray(image, dtype=float)

    # Blank-core radius — needed to skip corrupted inner profile segment
    blank_r = _blank_core_radius(image, centre=centre)

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
    else:
        residual = image


    bg_level = 0.0  # residual is already background-subtracted
    sigma_bg = mad_std(residual[np.isfinite(residual)])
    threshold = length_sigma * sigma_bg

    if swath_width is None:
        swath_width = max(3.0, min(image.shape) * 0.02)

    cy, cx = centre
    ny, nx = img.shape

    img_diag = float(np.hypot(*image.shape))

    if max_radius is None:
        max_radius = img_diag

    theta = result.theta
    pk_th = result.peak_theta_indices

    lengths = []
    for i in range(len(result.angles)):
        rho = result.rho_physical[i]
        th_rad = np.deg2rad(theta[pk_th[i]])

        # Closest point on the Radon line → display coords
        x0 = nx / 2.0 + rho * np.cos(th_rad)
        y0 = ny / 2.0 - rho * np.sin(th_rad)

        angle = result.angles[i]

        # Extract raw-image swath profiles for both arms
        r_pos, p_pos = _extract_swath_profile(
            img, x0, y0, angle, swath_width, max_radius=max_radius,
        )
        r_neg, p_neg = _extract_swath_profile(
            img, x0, y0, (angle + 180.0) % 360.0, swath_width,
            max_radius=max_radius,
        )

        # Fit combined Fraunhofer model; get smoothed profiles back
        popt, p_pos_s, p_neg_s = _fit_spike_profile(
            r_pos, p_pos, r_neg, p_neg,
            blank_r=blank_r,
            smooth_size=smooth_size,
            r_min=r_min,
            r_max_fraction=r_max_fraction,
        )

        # Per-arm endpoint: profile crossing first, envelope extrapolation second
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
        ))

    return lengths
