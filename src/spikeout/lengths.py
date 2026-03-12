"""Measure the length of each diffraction-spike arm via swath profiles."""

import numpy as np
from dataclasses import dataclass
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
    """
    angle_deg: float
    length_pos: float
    length_neg: float
    length_total: float
    profile_pos: np.ndarray
    profile_neg: np.ndarray
    radii_pos: np.ndarray
    radii_neg: np.ndarray


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


def measure_spike_lengths(
    image,
    result,
    swath_width=None,
    length_sigma=2.0,
    min_run_pixels=None,
    centre=None,
    radial_bin_width=2,
):
    """Measure the length of each detected spike arm.

    Operates on the azimuthal-median-subtracted image so the measurement
    captures where the *spike* fades, not the PSF halo.

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

    Returns
    -------
    lengths : list of `SpikeLengths`
    """
    image = np.asarray(image, dtype=float).copy()
    image[~np.isfinite(image)] = 0.0

    if centre is None:
        centre = find_centre(image)

    model = azimuthal_median(
        image, centre=centre, radial_bin_width=radial_bin_width,
    )
    residual = image - model

    sigma_bg = mad_std(residual[np.isfinite(residual)])
    threshold = length_sigma * sigma_bg

    if swath_width is None:
        swath_width = max(3.0, min(image.shape) * 0.02)
    if min_run_pixels is None:
        min_run_pixels = swath_width

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
            radii, profile = _extract_swath_profile(
                residual, x0, y0, arm_angle, swath_width,
            )

            below = profile < threshold
            run_length = 0
            endpoint_idx = len(radii)

            for j in range(len(below)):
                if below[j]:
                    run_length += 1
                    if run_length >= min_run_pixels:
                        endpoint_idx = j - int(min_run_pixels) + 1
                        break
                else:
                    run_length = 0

            arm_length = radii[min(endpoint_idx, len(radii) - 1)]
            arm_results[label] = (arm_length, radii, profile)

        lengths.append(SpikeLengths(
            angle_deg=angle,
            length_pos=arm_results["pos"][0],
            length_neg=arm_results["neg"][0],
            length_total=arm_results["pos"][0] + arm_results["neg"][0],
            profile_pos=arm_results["pos"][2],
            profile_neg=arm_results["neg"][2],
            radii_pos=arm_results["pos"][1],
            radii_neg=arm_results["neg"][1],
        ))

    return lengths
