"""Radon ↔ image-plane coordinate conversions.

skimage's ``radon`` uses a **y-UP** (mathematical) coordinate system::

    x_radon = col − col_centre          (rightward  +)
    y_radon = row_centre − row          (upward     +)

    Projection line:  x cos θ + y sin θ = ρ

When displaying with matplotlib's ``origin='lower'``::

    x_plot = col                         (rightward  +)
    y_plot = row                         (row 0 at bottom, upward +)

Converting between the two::

    col = col_centre + x_radon
    row = row_centre − y_radon           ← y sign flip
"""

import numpy as np

__all__ = [
    "sinogram_rho_to_physical",
    "radon_line_to_image",
    "calculate_star_offset",
]


def calculate_star_offset(radon_peaks):
    """Calculate the (dx, dy) offset of a star from the image centre.

    Uses the fact that all diffraction spikes pass through the star, so their
    Radon (ρ, θ) values satisfy the over-determined linear system
    ``dx cos θ + dy sin θ = ρ``.  Solves via least squares.

    The returned offset is in the Radon y-up frame::

        col_star = col_centre + dx
        row_star = row_centre − dy

    Parameters
    ----------
    radon_peaks : list of (rho, theta) tuples
        ``rho`` is the signed physical perpendicular distance (pixels) from
        the image centre; ``theta`` is in **radians** (Radon projection angle,
        0–π).

    Returns
    -------
    dx, dy : float
        Column and (y-up) row offsets of the star from the image centre.
        Convert to image-plane ``(row, col)`` via
        ``row_star = ny/2 − dy``, ``col_star = nx/2 + dx``.
    """
    A = []
    b = []
    for rho, theta in radon_peaks:
        A.append([np.cos(theta), np.sin(theta)])
        b.append(rho)
    A_mat = np.array(A, dtype=float)
    b_mat = np.array(b, dtype=float)
    offset, _, _, _ = np.linalg.lstsq(A_mat, b_mat, rcond=None)
    return float(offset[0]), float(offset[1])


def sinogram_rho_to_physical(row_indices, n_rho):
    """Convert sinogram row indices to signed physical ρ (pixels).

    skimage centres ρ = 0 at row ``n_rho // 2``.
    """
    return np.asarray(row_indices) - n_rho // 2


def radon_line_to_image(rho_phys, theta_deg, image_shape, pad=0.1):
    """Convert a (ρ, θ) Radon-space line to pixel-coordinate endpoints.

    The returned endpoints are in the coordinate frame used by
    ``matplotlib.imshow(..., origin='lower')``.

    Parameters
    ----------
    rho_phys : float
        Signed perpendicular distance from image centre (pixels).
    theta_deg : float
        Radon projection angle (degrees, 0–180).
    image_shape : (int, int)
        ``(nrows, ncols)`` of the image.
    pad : float
        Fractional extension beyond the image diagonal so lines visually
        reach the edges.

    Returns
    -------
    (x1, y1) : (float, float)
        First endpoint in pixel coordinates.
    (x2, y2) : (float, float)
        Second endpoint in pixel coordinates.
    angle_deg : float
        Direction of the line in the display frame (degrees CCW from +x,
        0–360).
    """
    theta_rad = np.deg2rad(theta_deg)
    yc = image_shape[0] / 2.0
    xc = image_shape[1] / 2.0

    # Closest point on line → plot coords (y-flip)
    x0 = xc + rho_phys * np.cos(theta_rad)
    y0 = yc - rho_phys * np.sin(theta_rad)

    # Line direction in plot coords (y-flip on dy)
    dx = -np.sin(theta_rad)
    dy = -np.cos(theta_rad)

    half_len = np.hypot(*image_shape) * (0.5 + pad)

    x1, y1 = x0 - half_len * dx, y0 - half_len * dy
    x2, y2 = x0 + half_len * dx, y0 + half_len * dy

    angle_deg = np.rad2deg(np.arctan2(dy, dx)) % 360.0
    return (x1, y1), (x2, y2), angle_deg
