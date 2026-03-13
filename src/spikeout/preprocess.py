"""Image preprocessing to isolate diffraction spikes."""

import numpy as np
from .stats import mad_std
from scipy.ndimage import uniform_filter
from skimage.morphology import disk, opening


__all__ = ["prepare_image", "azimuthal_median", "find_centre"]


def find_centre(image):
    """Detect source centre from the smoothed peak pixel.

    Used when ``centre='auto'`` is passed to `prepare_image`.

    Parameters
    ----------
    image : 2-D array

    Returns
    -------
    centre : (float, float)
        ``(row, col)`` of the detected centre.
    """
    smooth = uniform_filter(np.nan_to_num(image, nan=0.0), size=5)
    peak_idx = np.unravel_index(np.argmax(smooth), smooth.shape)
    return (float(peak_idx[0]), float(peak_idx[1]))


def azimuthal_median(image, centre=None, radial_bin_width=1):
    """Compute the azimuthal (annular) median at each pixel.

    The result is a smooth, radially symmetric model of the PSF core +
    halo.  Subtracting it isolates asymmetric structure like diffraction
    spikes.

    Parameters
    ----------
    image : 2-D array
    centre : (row, col) or *None*
        Centre of the radial profile.  Uses the image centre if *None*.
    radial_bin_width : int
        Width of each annular bin in pixels.

    Returns
    -------
    model : 2-D array  (same shape as *image*)
    """
    ny, nx = image.shape
    if centre is None:
        cy, cx = ny / 2.0, nx / 2.0
    else:
        cy, cx = centre

    Y, X = np.mgrid[:ny, :nx]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    r_int = (R / radial_bin_width).astype(int)
    max_bin = r_int.max() + 1

    model = np.zeros_like(image)
    flat_img = image.ravel()
    flat_r = r_int.ravel()

    for b in range(max_bin):
        mask = flat_r == b
        if mask.any():
            model.ravel()[mask] = np.nanmedian(flat_img[mask])

    return model


def prepare_image(
    image,
    centre='center',
    radial_bin_width=2,
    morph_radius=3,
    sigma_clip=1.5,
    asinh_stretch=None,
    subtract_median=False,
):
    """Preprocess an image to isolate diffraction spikes.

    Steps
    -----
    1. **Azimuthal median subtraction** — removes the radially symmetric
       PSF component (core, halo, sky gradient).
    2. **Sigma clip to zero** — suppresses the noise floor.
    3. **Morphological opening** — erodes compact sources (neighbours,
       hot pixels, cosmic rays) while preserving elongated spikes.
    4. **Asinh scaling** — compresses dynamic range softly so the Radon
       transform receives weighted gradient information rather than a
       binary mask.

    Parameters
    ----------
    image : 2-D array  (NaN-safe)
    centre : ``'center'`` | ``'auto'`` | (row, col)
        Star centre for the azimuthal subtraction.

        * ``'center'`` *(default)* — geometric centre of the cutout
          ``(ny/2, nx/2)``.
        * ``'auto'`` — detected from the smoothed peak pixel via
          `find_centre`; useful when the star is not well centred.
        * ``(row, col)`` — explicit coordinates.
    radial_bin_width : int
        Annular bin width (pixels) for the azimuthal median.
    morph_radius : int
        Radius of the circular structuring element for morphological
        opening.  Set to 0 to skip.
    sigma_clip : float
        Number of robust-σ above zero below which pixels are zeroed.
    asinh_stretch : float or *None*
        Stretch parameter for ``arcsinh(x / stretch)``.  Defaults to the
        median of the positive residual pixels (adaptive).
    subtract_median : bool
        If *True*, perform the azimuthal median subtraction step.  Set to
        *False* to skip directly to sigma clipping, etc.

    Returns
    -------
    prepared : 2-D array  (non-negative)
    """
    image = np.asarray(image, dtype=float).copy()
    image[~np.isfinite(image)] = 0.0

    if centre == 'center':
        ny, nx = image.shape
        centre = (ny / 2.0, nx / 2.0)
    elif centre == 'auto':
        centre = find_centre(image)

    if subtract_median:
        # 1. azimuthal median subtraction
        model = azimuthal_median(
            image, centre=centre, radial_bin_width=radial_bin_width,
        )
        residual = image - model
    else:
        residual = image
    # 2. sigma clip
    sigma = mad_std(residual[np.isfinite(residual)])
    if sigma > 0:
        residual[residual < sigma_clip * sigma] = 0.0
        residual = np.clip(residual, 0, None)

    # 3. morphological opening
    if morph_radius > 0:
        selem = disk(morph_radius)
        residual = opening(residual, selem)

    # 4. asinh stretch
    pos = residual[residual > 0]
    if pos.size == 0:
        return residual

    if asinh_stretch is None:
        asinh_stretch = np.median(pos)
    if asinh_stretch <= 0:
        asinh_stretch = 1.0

    return np.arcsinh(residual / asinh_stretch)
