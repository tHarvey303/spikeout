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
]


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
