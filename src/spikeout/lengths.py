"""Measure the length of each diffraction-spike arm via swath profiles."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
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
        Swath-median profile along each arm.
    radii_pos, radii_neg : ndarray
        Radial distance array for each arm.
    converged_pos, converged_neg : bool
        True if the arm end was found within the measured profile.
        False means the length was obtained by power-law extrapolation
        and should be treated as an estimate.
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


def _blank_core_radius(image, centre=None):
    """Radius of the NaN/zero patch centred on ``centre``.

    A saturated or masked star core is represented as NaN or exactly zero.
    This function measures how far that patch extends so callers can skip
    the corrupted inner region of swath profiles.

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


def _find_arm_endpoint(profile, threshold, min_run_pixels, skip_before=0):
    """Find the first run of ``min_run_pixels`` consecutive below-threshold points.

    Parameters
    ----------
    skip_before : int
        Profile indices ``< skip_before`` are ignored for run counting.
        Use this to skip the negative/NaN inner region produced by a
        NaN/zero star core after azimuthal-median subtraction.

    Returns
    -------
    endpoint_idx : int
        Index of the arm endpoint in *profile*.
    converged : bool
        True if a qualifying run was found before the end of the profile.
    """
    below = profile < threshold
    run_length = 0
    for j in range(len(below)):
        if j < skip_before:
            run_length = 0  # reset — blank-core region cannot terminate the arm
            continue
        if below[j]:
            run_length += 1
            if run_length >= min_run_pixels:
                return j - int(min_run_pixels) + 1, True
        else:
            run_length = 0
    return len(profile) - 1, False


def _extrapolate_arm_end(radii, profile, threshold, bg_level=0.0,
                         min_fit_radius=0.0, min_fit_points=6):
    """Estimate the arm end by fitting a power-law decay.

    Fits ``profile[i] - bg ≈ A · r[i]^α`` (with α < 0) to the tail of the
    profile in log-log space, then solves for the radius where the fit
    crosses *threshold*.

    Parameters
    ----------
    min_fit_radius : float
        Only use profile points with ``radii >= min_fit_radius``.  Set to
        the blank-core radius to exclude the corrupted inner segment.

    Returns the extrapolated radius, or *None* if the fit is unreliable
    (too few points, non-decaying trend, or non-finite result).
    """
    excess_profile = profile - bg_level
    valid = (
        np.isfinite(excess_profile)
        & (excess_profile > 0)
        & (radii >= min_fit_radius)
    )
    if valid.sum() < min_fit_points:
        return None

    # Use the last min_fit_points valid points (the decay tail)
    tail_idx = np.where(valid)[0][-min_fit_points:]
    r_fit = radii[tail_idx]
    p_fit = excess_profile[tail_idx]

    if np.any(p_fit <= 0):
        return None

    # log(p) = log(A) + alpha * log(r)
    try:
        alpha, log_A = np.polyfit(np.log(r_fit), np.log(p_fit), 1)
    except (np.linalg.LinAlgError, ValueError):
        return None

    # Require a decaying profile
    if alpha >= 0:
        return None

    excess_thresh = threshold - bg_level
    if excess_thresh <= 0:
        return None

    A = np.exp(log_A)
    try:
        r_extrap = (excess_thresh / A) ** (1.0 / alpha)
    except (ZeroDivisionError, OverflowError, FloatingPointError):
        return None

    if not np.isfinite(r_extrap) or r_extrap <= 0:
        return None

    return float(r_extrap)


def _measure_arm_adaptive(
    residual, x0, y0, arm_angle, swath_width,
    threshold, min_run_pixels, bg_level,
    initial_radius, max_radius, growth_factor,
    blank_r=0.0,
):
    """Measure one spike arm with adaptive radius expansion.

    Starts the swath profile at ``initial_radius`` and multiplies by
    ``growth_factor`` until the arm end drops below *threshold*, or
    ``max_radius`` is reached.  At ``max_radius``, a power-law fit to the
    profile tail provides an extrapolated estimate.

    ``blank_r`` is the radius of any NaN/zero core at the star centre.
    Profile points inside this radius are skipped when searching for the
    arm endpoint and when fitting the power-law decay, because azimuthal-
    median subtraction leaves them artificially negative.

    Returns
    -------
    arm_length : float
    radii : ndarray  (from the last trial radius)
    profile : ndarray
    converged : bool
    """
    trial_r = initial_radius
    radii = profile = None

    while True:
        capped_r = min(trial_r, max_radius)
        radii, profile = _extract_swath_profile(
            residual, x0, y0, arm_angle, swath_width,
            max_radius=capped_r,
        )

        # Skip the blank-core segment when searching for the endpoint
        skip_before = int(np.searchsorted(radii, blank_r))
        endpoint_idx, converged = _find_arm_endpoint(
            profile, threshold, min_run_pixels, skip_before=skip_before,
        )

        if converged:
            return float(radii[endpoint_idx]), radii, profile, True

        # Spike still above threshold at the edge — try to grow
        if capped_r >= max_radius:
            # At hard limit: extrapolate from the profile tail, excluding core
            r_extrap = _extrapolate_arm_end(
                radii, profile, threshold, bg_level,
                min_fit_radius=blank_r,
            )
            if r_extrap is not None and r_extrap > radii[-1]:
                return float(min(r_extrap, max_radius)), radii, profile, False
            # Extrapolation failed — return the profile edge as a lower bound
            return float(radii[-1]), radii, profile, False

        trial_r = min(trial_r * growth_factor, max_radius)


def measure_spike_lengths(
    image,
    result,
    swath_width=None,
    length_sigma=2.0,
    min_run_pixels=None,
    centre=None,
    radial_bin_width=2,
    initial_radius=None,
    radius_growth_factor=1.5,
    max_radius=None,
):
    """Measure the length of each detected spike arm.

    Operates on the azimuthal-median-subtracted image so the measurement
    captures where the *spike* fades, not the PSF halo.

    The swath profile is walked out to ``initial_radius`` first.  If the
    spike has not ended by then the walk is extended by ``radius_growth_factor``
    in successive steps until either the end is found or ``max_radius`` is
    reached.  At ``max_radius`` a power-law extrapolation (fitting the
    1/r^α profile tail) provides an estimated endpoint; the corresponding
    ``converged`` flag in `SpikeLengths` is set to *False* so callers can
    distinguish measured from extrapolated lengths.

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
        The profile must drop below ``length_sigma × σ_bg`` to be
        considered "at background".
    min_run_pixels : float or *None*
        Consecutive below-threshold samples required to declare the spike
        has ended.  Default: ``swath_width``.
    centre : (row, col) or *None*
        Star centre.  Auto-detected if *None*.
    radial_bin_width : int
        Annular bin width for the internal azimuthal subtraction.
    initial_radius : float or *None*
        Initial maximum swath-walk radius (pixels).  Default:
        ``min(image.shape) / 4``.  Kept small to avoid the cost of
        computing a long profile when spikes are short.
    radius_growth_factor : float
        Factor by which ``initial_radius`` is multiplied at each step
        when the spike has not yet ended.  Default 1.5.
    max_radius : float or *None*
        Hard upper limit on the walk radius.  Default: image diagonal.

    Returns
    -------
    lengths : list of `SpikeLengths`
    """
    image = np.asarray(image, dtype=float)

    # Measure blank-core radius from the original image before filling NaNs,
    # so the corrupted inner segment of swath profiles can be skipped.
    blank_r = _blank_core_radius(image, centre=centre)

    image = image.copy()
    image[~np.isfinite(image)] = 0.0

    if centre is None:
        centre = find_centre(image)

    model = azimuthal_median(
        image, centre=centre, radial_bin_width=radial_bin_width,
    )
    residual = image - model

    bg_level = 0.0  # residual is already background-subtracted
    sigma_bg = mad_std(residual[np.isfinite(residual)])
    threshold = length_sigma * sigma_bg

    if swath_width is None:
        swath_width = max(3.0, min(image.shape) * 0.02)
    if min_run_pixels is None:
        min_run_pixels = swath_width

    img_diag = float(np.hypot(*image.shape))
    if initial_radius is None:
        # Ensure the first trial window extends well past the blank core so
        # there is real spike signal to measure from the start.
        initial_radius = max(min(image.shape) / 4.0, blank_r + swath_width * 2.0)
    if max_radius is None:
        max_radius = img_diag

    theta = result.theta
    pk_th = result.peak_theta_indices

    lengths = []
    for i in range(len(result.angles)):
        rho = result.rho_physical[i]
        th_rad = np.deg2rad(theta[pk_th[i]])

        # Closest point on Radon line → display coords
        x0 = image.shape[1] / 2.0 + rho * np.cos(th_rad)
        y0 = image.shape[0] / 2.0 - rho * np.sin(th_rad)

        angle = result.angles[i]

        arm_results = {}
        for sign, label in [(1.0, "pos"), (-1.0, "neg")]:
            arm_angle = angle if sign > 0 else (angle + 180.0) % 360.0
            arm_length, radii, profile, converged = _measure_arm_adaptive(
                residual, x0, y0, arm_angle, swath_width,
                threshold, min_run_pixels, bg_level,
                initial_radius, max_radius, radius_growth_factor,
                blank_r=blank_r,
            )
            arm_results[label] = (arm_length, radii, profile, converged)

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
