"""DS9 region file generation for detected diffraction spikes."""

import numpy as np

__all__ = [
    "spike_box_regions",
    "spike_mask",
    "write_ds9_regions",
    "write_catalogue_ds9_regions",
]


def _box_width(length_px, width_fraction, min_width, max_width):
    w = width_fraction * length_px
    if min_width is not None:
        w = max(w, min_width)
    if max_width is not None:
        w = min(w, max_width)
    return w


def spike_mask(
    result,
    image_shape,
    centre=None,
    width_fraction=0.1,
    min_width=5.0,
    max_width=None,
):
    """Boolean mask with spike-contaminated pixels set to *True*.

    The mask is computed by rasterising the same rotated-box geometry as
    `spike_box_regions`, so it is exactly consistent with the DS9 output.

    Requires ``result.lengths`` to be populated
    (run `detect` with ``measure_lengths=True``).

    Parameters
    ----------
    result : SpikeResult
    image_shape : (int, int)
        ``(nrows, ncols)`` of the image.
    centre : (row, col) or None
        Star centre in 0-indexed pixel coordinates.  Defaults to the
        image centre.
    width_fraction, min_width, max_width
        Same semantics as `spike_box_regions`.

    Returns
    -------
    mask : ndarray of bool, shape ``image_shape``
    """
    if result.lengths is None:
        raise ValueError(
            "result.lengths is None; run detect() with measure_lengths=True"
        )

    from skimage.draw import polygon as sk_polygon

    nrows, ncols = image_shape
    if centre is None:
        cx, cy = ncols / 2.0, nrows / 2.0
    else:
        row, col = centre
        cx, cy = float(col), float(row)

    mask = np.zeros((nrows, ncols), dtype=bool)

    for sl in result.lengths:
        angle_rad = np.deg2rad(sl.angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        offset = (sl.length_pos - sl.length_neg) / 2.0
        bx = cx + offset * cos_a
        by = cy + offset * sin_a

        half_len = sl.length_total / 2.0
        half_wid = _box_width(sl.length_total, width_fraction, min_width, max_width) / 2.0

        # 4 corners of the rotated box.
        # Along-spike unit vector: (cos_a, sin_a)
        # Perpendicular unit vector: (-sin_a, cos_a)
        corners_x = np.array([
            bx + half_len * cos_a - half_wid * sin_a,
            bx + half_len * cos_a + half_wid * sin_a,
            bx - half_len * cos_a + half_wid * sin_a,
            bx - half_len * cos_a - half_wid * sin_a,
        ])
        corners_y = np.array([
            by + half_len * sin_a + half_wid * cos_a,
            by + half_len * sin_a - half_wid * cos_a,
            by - half_len * sin_a - half_wid * cos_a,
            by - half_len * sin_a + half_wid * cos_a,
        ])

        # skimage.draw.polygon uses (row, col); in our display frame
        # (origin='lower') y_display == row_index, so no flip needed.
        rr, cc = sk_polygon(corners_y, corners_x, shape=(nrows, ncols))
        mask[rr, cc] = True

    return mask


def _sky_pa(display_angle_deg):
    """Convert a display-frame angle to sky position angle.

    Assumes standard FITS orientation (North up, East left).

    Display angle is CCW from +x (rightward = West).
    PA is measured East of North (CCW on sky).
    """
    a = np.deg2rad(display_angle_deg)
    pa = np.degrees(np.arctan2(-np.cos(a), np.sin(a))) % 180.0
    return pa


def spike_box_regions(
    result,
    image_shape,
    centre=None,
    width_fraction=0.1,
    min_width=5.0,
    max_width=None,
):
    """DS9 box region strings in image pixel coords (1-indexed) for detected spikes.

    Parameters
    ----------
    result : SpikeResult
        Must have ``result.lengths`` populated
        (run `detect` with ``measure_lengths=True``).
    image_shape : (int, int)
        ``(nrows, ncols)`` of the image.
    centre : (row, col) or None
        Star centre in 0-indexed pixel coordinates.  Defaults to the
        image centre.
    width_fraction : float
        Box width perpendicular to the spike as a fraction of the total
        spike length.  Default 0.1 (10 %).
    min_width : float
        Minimum box width in pixels.  Default 5.
    max_width : float or None
        Maximum box width in pixels.  *None* means no cap.

    Returns
    -------
    list of str
        One DS9 ``box(...)`` string per spike.
    """
    if result.lengths is None:
        raise ValueError(
            "result.lengths is None; run detect() with measure_lengths=True"
        )

    nrows, ncols = image_shape
    if centre is None:
        cx, cy = ncols / 2.0, nrows / 2.0
    else:
        row, col = centre
        cx, cy = float(col), float(row)

    regions = []
    for sl in result.lengths:
        angle_rad = np.deg2rad(sl.angle_deg)

        # Offset box centre from star centre to account for asymmetric arms
        offset = (sl.length_pos - sl.length_neg) / 2.0
        bx = cx + offset * np.cos(angle_rad)
        by = cy + offset * np.sin(angle_rad)

        box_len = sl.length_total
        box_wid = _box_width(box_len, width_fraction, min_width, max_width)

        # DS9 image coords are 1-indexed
        ds9_x = bx + 1.0
        ds9_y = by + 1.0
        # box(x, y, width, height, angle): angle is CCW rotation from +x.
        # We put the spike length along the width axis, so angle = spike angle.
        ds9_angle = sl.angle_deg % 180.0

        regions.append(
            f"box({ds9_x:.3f},{ds9_y:.3f},{box_len:.2f},{box_wid:.2f},{ds9_angle:.2f})"
        )

    return regions


def write_ds9_regions(
    path,
    result,
    image_shape,
    centre=None,
    width_fraction=0.1,
    min_width=5.0,
    max_width=None,
    colour="green",
):
    """Write a DS9 region file for a single ``SpikeResult``.

    Regions are written in image (pixel) coordinates (1-indexed).
    Requires ``result.lengths`` to be populated.

    Parameters
    ----------
    path : str or path-like
        Output ``.reg`` file path.
    result : SpikeResult
    image_shape : (int, int)
        ``(nrows, ncols)`` of the image.
    centre : (row, col) or None
        Star centre.  Defaults to the image centre.
    width_fraction : float
        Box width as a fraction of total spike length.  Default 0.1.
    min_width : float
        Minimum box width in pixels.  Default 5.
    max_width : float or None
        Maximum box width in pixels.
    colour : str
        DS9 colour name (e.g. ``'green'``, ``'red'``, ``'cyan'``).
    """
    regions = spike_box_regions(
        result, image_shape,
        centre=centre,
        width_fraction=width_fraction,
        min_width=min_width,
        max_width=max_width,
    )
    _write_reg_file(path, "image", regions, colour)


def write_catalogue_ds9_regions(
    path,
    entries,
    pixel_scale_arcsec,
    width_fraction=0.1,
    min_width=5.0,
    max_width=None,
    colour="green",
    image_pa_deg=0.0,
):
    """Write a DS9 region file for a list of ``CatalogueEntry`` objects.

    Regions are written in sky (FK5) coordinates with sizes in arcseconds,
    so the file can be overlaid on any reprojection of the original image.

    Requires each entry's ``result.lengths`` to be populated
    (run `catalogue_detect` with ``measure_lengths=True``).

    Parameters
    ----------
    path : str or path-like
        Output ``.reg`` file path.
    entries : list of CatalogueEntry
    pixel_scale_arcsec : float
        Pixel scale of the original image (arcseconds per pixel), used to
        convert pixel lengths and offsets to sky units.
    width_fraction : float
        Box width as a fraction of total spike length.  Default 0.1.
    min_width : float
        Minimum box width in **pixels** (converted to arcsec internally).
        Default 5.
    max_width : float or None
        Maximum box width in **pixels**.
    colour : str
        DS9 colour name.
    image_pa_deg : float
        Position angle of the image's +y axis (degrees East of North).
        Default 0 (North up).  Adjust for non-standard orientations.

    Notes
    -----
    The sky position angle is computed assuming standard FITS orientation
    (North up, East left) modified by ``image_pa_deg``.  For images with
    significant rotation, supply the correct ``image_pa_deg`` from the
    WCS CD matrix or ``CROTA`` keyword.
    """
    scale = pixel_scale_arcsec
    regions = []

    for entry in entries:
        if entry.result is None or entry.result.lengths is None:
            if entry.result.lengths is None:
                print(f"Warning: entry {entry} has result.lengths = None; skipping")
            
            continue

        dec_rad = np.deg2rad(entry.dec)

        for sl in entry.result.lengths:
            angle_rad = np.deg2rad(sl.angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Offset from star centre to box centre (asymmetric arms)
            offset_px = (sl.length_pos - sl.length_neg) / 2.0

            # Display frame: +x = West (East-left), +y = North
            d_dec_deg = offset_px * sin_a * scale / 3600.0
            d_ra_deg = -offset_px * cos_a * scale / (3600.0 * np.cos(dec_rad))

            cen_ra = entry.ra + d_ra_deg
            cen_dec = entry.dec + d_dec_deg

            box_len_arcsec = sl.length_total * scale
            box_wid_arcsec = _box_width(
                sl.length_total, width_fraction, min_width, max_width
            ) * scale

            # Sky PA of the spike's long axis, East of North
            sky_pa = (_sky_pa(sl.angle_deg) + image_pa_deg) % 180.0

            # DS9 sky box: box(ra, dec, width", height", angle)
            # Long axis is the height; angle is PA of the height axis from North.
            regions.append(
                f'box({cen_ra:.6f},{cen_dec:.6f},'
                f'{box_wid_arcsec:.2f}",{box_len_arcsec:.2f}",'
                f'{sky_pa:.2f})'
            )

    _write_reg_file(path, "fk5", regions, colour)


def _write_reg_file(path, coordsys, regions, colour):
    header = (
        "# Region file format: DS9 version 4.1\n"
        f'global color={colour} dashlist=8 3 width=1 '
        'font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 "
        "include=1 source=1\n"
        f"{coordsys}\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        for r in regions:
            fh.write(r + "\n")
