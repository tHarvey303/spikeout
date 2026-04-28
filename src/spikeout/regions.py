"""DS9 region file generation and pixel masks for diffraction spikes and halos."""

import re
import numpy as np
from .stats import mad_std, estimate_background

__all__ = [
    "spike_box_regions",
    "spike_mask",
    "write_spike_mask_fits",
    "write_border_mask_fits",
    "write_ds9_regions",
    "write_catalogue_ds9_regions",
    "read_ds9_regions",
    "read_catalogue_ds9_regions",
    "halo_mask",
    "compress_fits_mask_to_bytes",
    "decompress_bytes_to_fits_mask",
    "combine_masks",
    "combine_fits_masks",
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
        if getattr(result, 'corrected_centre', None) is not None:
            row, col = result.corrected_centre
            cx, cy = float(col), float(row)
        else:
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


def _sky_pa_inverse(sky_pa_deg, image_pa_deg=0.0):
    """Inverse of :func:`_sky_pa`: recover display-frame angle in [0, 180) from sky PA.

    The result is ambiguous by 180° (same as the forward function); the
    caller must use the box-centre offset direction to resolve the full
    0–360° angle.
    """
    pa_rad = np.deg2rad((sky_pa_deg - image_pa_deg) % 180.0)
    return float(np.degrees(np.arctan2(np.cos(pa_rad), -np.sin(pa_rad))) % 180.0)


def _offset_sky(ra0_deg, dec0_deg, d_east_arcsec, d_north_arcsec):
    """Exact (ra, dec) for a point offset from (ra0, dec0).

    Uses the gnomonic inverse projection, which is valid at any separation
    and has no 1/cos(dec) singularity near the poles.

    Parameters
    ----------
    ra0_deg, dec0_deg : float
        Reference sky position (degrees).
    d_east_arcsec : float
        Offset East (+) / West (−) in arcseconds.
    d_north_arcsec : float
        Offset North (+) / South (−) in arcseconds.

    Returns
    -------
    ra_deg, dec_deg : float
    """
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)
    xi = np.deg2rad(d_east_arcsec / 3600.0)
    eta = np.deg2rad(d_north_arcsec / 3600.0)
    new_ra = ra0 + np.arctan2(xi, np.cos(dec0) - eta * np.sin(dec0))
    new_dec = np.arctan2(
        np.sin(dec0) + eta * np.cos(dec0),
        np.sqrt(xi ** 2 + (np.cos(dec0) - eta * np.sin(dec0)) ** 2),
    )
    return np.rad2deg(new_ra), np.rad2deg(new_dec)


def _sky_offset(ra0_deg, dec0_deg, ra1_deg, dec1_deg):
    """Exact tangent-plane offsets from (ra0, dec0) to (ra1, dec1).

    Gnomonic forward projection — inverse of :func:`_offset_sky`.

    Returns
    -------
    d_east_arcsec, d_north_arcsec : float
    """
    ra0 = np.deg2rad(ra0_deg);  dec0 = np.deg2rad(dec0_deg)
    ra1 = np.deg2rad(ra1_deg);  dec1 = np.deg2rad(dec1_deg)
    dra = ra1 - ra0
    D = np.sin(dec0) * np.sin(dec1) + np.cos(dec0) * np.cos(dec1) * np.cos(dra)
    xi  = np.cos(dec1) * np.sin(dra) / D
    eta = (np.cos(dec0) * np.sin(dec1) - np.sin(dec0) * np.cos(dec1) * np.cos(dra)) / D
    return np.rad2deg(xi) * 3600.0, np.rad2deg(eta) * 3600.0


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
        if getattr(result, 'corrected_centre', None) is not None:
            row, col = result.corrected_centre
            cx, cy = float(col), float(row)
        else:
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
    verbose=True,
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
    verbose : bool
        If *True*, print warnings about entries with missing length measurements.

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
            if verbose:
                print(f"Warning: entry {entry} has result.lengths = None; skipping")
            continue

        result = entry.result

        # Base sky position for arm measurement.  When recenter_for_lengths was
        # used, arm lengths are measured from corrected_centre (which may differ
        # from the image centre).  star_centre_offset = (dx, dy) gives the
        # Radon-derived pixel offset of the detected star from the image centre:
        #   dx > 0 → star is dx px West of image centre  → d_east = −dx × scale
        #   dy > 0 → star is dy px North of image centre → d_north = +dy × scale
        # The image centre corresponds to entry.ra / entry.dec (catalogue pos).
        if (
            getattr(result, 'corrected_centre', None) is not None
            and getattr(result, 'star_centre_offset', None) is not None
        ):
            dx, dy = result.star_centre_offset
            base_ra, base_dec = _offset_sky(
                entry.ra, entry.dec,
                -dx * scale, dy * scale,
            )
        else:
            base_ra, base_dec = entry.ra, entry.dec

        for sl in result.lengths:
            angle_rad = np.deg2rad(sl.angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Pixel offset from star centre to box centre (asymmetric arms).
            # Display frame: +x = West = −East, +y = North.
            offset_px = (sl.length_pos - sl.length_neg) / 2.0
            d_east_arcsec = -offset_px * cos_a * scale
            d_north_arcsec = offset_px * sin_a * scale

            cen_ra, cen_dec = _offset_sky(base_ra, base_dec, d_east_arcsec, d_north_arcsec)

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

    # Halo aperture circles
    for entry in entries:
        if entry.halo_radius is None:
            continue

        halo_r_arcsec = entry.halo_radius * scale
        regions.append(
            f'circle({entry.ra:.6f},{entry.dec:.6f},{halo_r_arcsec:.2f}")'
        )

    _write_reg_file(path, "fk5", regions, colour)


# ---------------------------------------------------------------------------
# Region file parsing (inverse of the write functions above)
# ---------------------------------------------------------------------------

_BOX_RE = re.compile(
    r'^box\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,"]+?)\s*"?\s*,'
    r'\s*([^,"]+?)\s*"?\s*,\s*([^)]+?)\s*\)$',
    re.IGNORECASE,
)
_CIRC_RE = re.compile(
    r'^circle\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)"]+?)\s*"?\s*\)$',
    re.IGNORECASE,
)
_COORDSYS_TOKENS = frozenset(
    ('image', 'fk5', 'fk4', 'j2000', 'b1950', 'icrs', 'galactic', 'ecliptic')
)


def _detect_coordsys(lines):
    """Return the coordinate-system token from a DS9 region file header."""
    for line in lines:
        s = line.strip().lower()
        if s in _COORDSYS_TOKENS:
            return s
    return None


def _make_spike_lengths(display_angle_0_180, length_total, swath_width, box_offset_px):
    """Reconstruct a SpikeLengths from a box geometry.

    Parameters
    ----------
    display_angle_0_180 : float
        Display-frame box angle in [0, 180) — the 180° ambiguity stored in
        the region file.
    length_total : float
        Total spike length in pixels (box height).
    swath_width : float
        Swath width in pixels (box width).
    box_offset_px : float
        Signed offset of the box centre from the star centre projected along
        ``display_angle_0_180``.  Positive → box shifted toward that angle;
        negative → shifted toward the opposite direction.

    Returns
    -------
    SpikeLengths
    """
    from .lengths import SpikeLengths

    # Resolve 180° ambiguity: positive arm is in the direction of positive offset
    if box_offset_px >= 0:
        angle_deg = float(display_angle_0_180 % 360.0)
    else:
        angle_deg = float((display_angle_0_180 + 180.0) % 360.0)
        box_offset_px = -box_offset_px

    length_pos = max(0.0, length_total / 2.0 + box_offset_px)
    length_neg = max(0.0, length_total / 2.0 - box_offset_px)
    _e = np.array([], dtype=np.float64)
    return SpikeLengths(
        angle_deg=angle_deg,
        length_pos=length_pos,
        length_neg=length_neg,
        length_total=length_total,
        profile_pos=_e, profile_neg=_e,
        radii_pos=_e, radii_neg=_e,
        converged_pos=True, converged_neg=True,
        popt=None,
        threshold=np.nan,
        background_profile=None,
        swath_width=swath_width,
    )


def read_ds9_regions(path, image, centre=None):
    """Read an image-coordinate DS9 region file and reconstruct spike measurements.

    Reverses the output of :func:`write_ds9_regions`: parses each ``box``
    region back into a :class:`~spikeout.lengths.SpikeLengths` and assembles
    a partial :class:`~spikeout.detect.SpikeResult`.

    Fields that cannot be recovered from the region file (``sinogram``,
    ``snr``, ``prepared_image``, profile arrays, etc.) are set to *None*
    or filled with *NaN* / zeros as appropriate.

    Parameters
    ----------
    path : str or path-like
        DS9 region file written by :func:`write_ds9_regions` (image coords).
    image : 2-D array
        Original image — only its shape is used to set the default star centre.
    centre : (row, col) or None
        Star centre in 0-indexed pixel coordinates.  Defaults to the image
        centre.

    Returns
    -------
    SpikeResult
        Partial result with ``angles`` and ``lengths`` populated.

    Raises
    ------
    ValueError
        If the file does not declare image coordinates.
    """
    from .detect import SpikeResult

    image = np.asarray(image, dtype=float)
    nrows, ncols = image.shape
    if centre is None:
        cy, cx = nrows / 2.0, ncols / 2.0
    else:
        cy, cx = float(centre[0]), float(centre[1])

    with open(path) as fh:
        lines = [l.rstrip('\n') for l in fh]

    coordsys = _detect_coordsys(lines)
    if coordsys != 'image':
        raise ValueError(
            f"Expected image-coordinate region file (coordinate system 'image'), "
            f"got {coordsys!r}.  For sky-coordinate files use read_catalogue_ds9_regions."
        )

    lengths = []
    angles_list = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = _BOX_RE.match(line)
        if m is None:
            continue

        ds9_x, ds9_y, box_wid, box_len, ds9_angle = (float(g) for g in m.groups())

        # DS9 image coordinates are 1-indexed; convert to 0-indexed display frame
        # (origin='lower': row 0 at bottom, y increases upward).
        bx = ds9_x - 1.0   # col
        by = ds9_y - 1.0   # row

        angle_rad = np.deg2rad(ds9_angle)
        # Signed projection of (box_centre − star_centre) onto the box axis.
        # This recovers (length_pos − length_neg) / 2 with the correct sign.
        offset_px = (bx - cx) * np.cos(angle_rad) + (by - cy) * np.sin(angle_rad)

        sl = _make_spike_lengths(ds9_angle, box_len, box_wid, offset_px)
        lengths.append(sl)
        angles_list.append(sl.angle_deg)

    n = len(angles_list)
    return SpikeResult(
        angles=np.array(angles_list),
        rho_physical=np.zeros(n),
        snr=np.full(n, np.nan),
        sinogram=None,
        theta=None,
        peak_rho_indices=np.zeros(n, dtype=int),
        peak_theta_indices=np.zeros(n, dtype=int),
        prepared_image=None,
        n_rejected_snr=0,
        lengths=lengths if lengths else None,
    )


def read_catalogue_ds9_regions(
    path,
    entries,
    pixel_scale_arcsec,
    image_pa_deg=0.0,
    match_radius_arcsec=None,
    verbose=True,
):
    """Read a sky-coordinate catalogue DS9 region file and update entries.

    Reverses the output of :func:`write_catalogue_ds9_regions`.  Each ``box``
    region is matched to the nearest entry by angular separation and the
    reconstructed :class:`~spikeout.lengths.SpikeLengths` objects are stored
    in ``entry.result.lengths``.  ``circle`` regions update
    ``entry.halo_radius``.

    Parameters
    ----------
    path : str or path-like
        DS9 region file written by :func:`write_catalogue_ds9_regions`.
    entries : list of CatalogueEntry
        Entries to update in-place.  Matched by nearest sky position.
    pixel_scale_arcsec : float
        Pixel scale of the original image (arcseconds per pixel).  Must
        match the value used when writing the file.
    image_pa_deg : float
        Image position angle used when writing the file.  Must match the
        ``image_pa_deg`` passed to :func:`write_catalogue_ds9_regions`.
        Default 0.
    match_radius_arcsec : float or None
        Maximum angular separation (arcsec) between a region's sky position
        and the entry it is assigned to.  Defaults to the largest box height
        in the file (a safe upper bound on the box-centre offset from the
        star).
    verbose : bool
        If *True*, warn about regions that cannot be matched to any entry.

    Returns
    -------
    entries : list of CatalogueEntry
        The same list, with ``result.lengths`` and ``halo_radius`` updated
        in-place.

    Raises
    ------
    ValueError
        If the file does not declare a recognised sky coordinate system.
    """
    from .detect import SpikeResult

    scale = pixel_scale_arcsec
    _SKY_SYSTEMS = ('fk5', 'fk4', 'j2000', 'b1950', 'icrs')

    with open(path) as fh:
        lines = [l.rstrip('\n') for l in fh]

    coordsys = _detect_coordsys(lines)
    if coordsys not in _SKY_SYSTEMS:
        raise ValueError(
            f"Expected sky-coordinate region file (e.g. 'fk5'), got {coordsys!r}.  "
            f"For image-coordinate files use read_ds9_regions."
        )

    # ── Parse all regions ─────────────────────────────────────────────────
    raw_boxes = []    # (ra, dec, wid_arcsec, len_arcsec, sky_pa_deg)
    raw_circles = []  # (ra, dec, radius_arcsec)
    max_len_arcsec = 0.0

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = _BOX_RE.match(line)
        if m:
            ra, dec, wid, length, pa = (float(g) for g in m.groups())
            raw_boxes.append((ra, dec, wid, length, pa))
            max_len_arcsec = max(max_len_arcsec, length)
            continue
        m = _CIRC_RE.match(line)
        if m:
            ra, dec, r = (float(g) for g in m.groups())
            raw_circles.append((ra, dec, r))

    if match_radius_arcsec is None:
        match_radius_arcsec = max(max_len_arcsec, 10.0)

    if not entries:
        return entries

    # Precompute entry sky positions as arrays for vectorised matching
    ras  = np.deg2rad([e.ra  for e in entries])
    decs = np.deg2rad([e.dec for e in entries])

    def _nearest(ra_deg, dec_deg):
        """Index and separation (arcsec) of the nearest entry."""
        ra_r  = np.deg2rad(ra_deg)
        dec_r = np.deg2rad(dec_deg)
        cos_sep = np.clip(
            np.sin(dec_r) * np.sin(decs) + np.cos(dec_r) * np.cos(decs) * np.cos(ra_r - ras),
            -1.0, 1.0,
        )
        seps = np.degrees(np.arccos(cos_sep)) * 3600.0   # arcsec
        idx = int(np.argmin(seps))
        return idx, seps[idx]

    # ── Match and reconstruct SpikeLengths ────────────────────────────────
    entry_boxes = {i: [] for i in range(len(entries))}

    for ra_b, dec_b, wid_as, len_as, sky_pa in raw_boxes:
        idx, sep = _nearest(ra_b, dec_b)
        if sep > match_radius_arcsec:
            if verbose:
                print(
                    f"Warning: box at ({ra_b:.6f}, {dec_b:.6f}) unmatched "
                    f"(nearest entry {sep:.1f}\" away, threshold {match_radius_arcsec:.1f}\")"
                )
            continue
        entry_boxes[idx].append((ra_b, dec_b, wid_as, len_as, sky_pa))

    for i, entry in enumerate(entries):
        boxes = entry_boxes[i]
        if not boxes:
            continue

        # Use the same base sky position as the writer did
        result = entry.result
        if (
            result is not None
            and getattr(result, 'corrected_centre', None) is not None
            and getattr(result, 'star_centre_offset', None) is not None
        ):
            dx, dy = result.star_centre_offset
            base_ra, base_dec = _offset_sky(entry.ra, entry.dec, -dx * scale, dy * scale)
        else:
            base_ra, base_dec = entry.ra, entry.dec

        lengths_out = []
        angles_out = []

        for ra_b, dec_b, wid_as, len_as, sky_pa in boxes:
            length_total_px = len_as / scale
            swath_width_px  = wid_as / scale

            # Recover display-frame angle (0-180) from stored sky PA
            display_angle = _sky_pa_inverse(sky_pa, image_pa_deg)
            angle_rad = np.deg2rad(display_angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Exact tangent-plane offset from base sky position to box centre.
            # From the writer: d_east = -offset_px * cos_a * scale
            #                  d_north = offset_px * sin_a * scale
            # Inversion via dot product: offset_px = (-d_east * cos_a + d_north * sin_a) / scale
            d_east_as, d_north_as = _sky_offset(base_ra, base_dec, ra_b, dec_b)
            offset_px = (-d_east_as * cos_a + d_north_as * sin_a) / scale

            sl = _make_spike_lengths(display_angle, length_total_px, swath_width_px, offset_px)
            lengths_out.append(sl)
            angles_out.append(sl.angle_deg)

        if result is not None:
            result.lengths = lengths_out
            result.angles = np.array(angles_out)
        else:
            n = len(angles_out)
            entry.result = SpikeResult(
                angles=np.array(angles_out),
                rho_physical=np.zeros(n),
                snr=np.full(n, np.nan),
                sinogram=None,
                theta=None,
                peak_rho_indices=np.zeros(n, dtype=int),
                peak_theta_indices=np.zeros(n, dtype=int),
                prepared_image=None,
                n_rejected_snr=0,
                lengths=lengths_out,
            )

    # ── Match circle regions → halo_radius ───────────────────────────────
    for ra_c, dec_c, r_as in raw_circles:
        idx, sep = _nearest(ra_c, dec_c)
        if sep > match_radius_arcsec:
            if verbose:
                print(
                    f"Warning: circle at ({ra_c:.6f}, {dec_c:.6f}) unmatched "
                    f"(nearest entry {sep:.1f}\" away, threshold {match_radius_arcsec:.1f}\")"
                )
            continue
        entries[idx].halo_radius = r_as / scale

    return entries


def halo_mask(
    image,
    centre=None,
    threshold_nsigma=3.0,
    background_min_r_frac=0.8,
    min_radius=5.0,
    max_radius=None,
    radial_bin_width=1.0,
    smooth_bins=5,
    override_threshold=None,
    radius_factor=1.0,
    n_sectors=8,
    min_sector_pixels=5,
    sector_sigma_clip=2.0,
):
    """Boolean circular mask enclosing the stellar halo.

    Builds a robust radial profile using per-sector medians with outlier
    rejection, then finds the outermost radius at which the profile exceeds
    a background-noise threshold.

    At each radial bin the annulus is divided into ``n_sectors`` angular
    wedges.  The median flux is computed per sector; sectors with too few
    pixels (``< min_sector_pixels``) are skipped.  Outlier sectors —
    those whose median lies more than ``sector_sigma_clip × MAD`` above
    the sector-median distribution — are rejected before the per-bin
    representative value is taken as the median of the surviving sector
    medians.  This makes the profile immune to neighbouring sources, which
    inflate only a small number of sectors.  When fewer than 2 sectors
    survive at a given bin, the function falls back to the plain
    per-annulus median so the profile remains defined.

    The threshold is ``background_level + threshold_nsigma × σ_MAD``, where
    both quantities are estimated from an outer annulus of the image using
    median / MAD statistics, making them robust to contamination.

    Parameters
    ----------
    image : 2-D array
        Input image (NaN-safe).
    centre : (row, col) or None
        Star centre in 0-indexed pixel coordinates.  Defaults to the image
        centre.
    threshold_nsigma : float
        Halo edge is where the azimuthal-median profile drops to
        ``bg_level + threshold_nsigma × σ_MAD``.  Typical values:

            2.0  — lenient, captures faint extended wings
            3.0  — default, good balance
            5.0  — conservative, only the bright core
    background_min_r_frac : float
        Inner radius of the background annulus as a fraction of
        *max_radius*.  Pixels at radius ``> background_min_r_frac ×
        max_radius`` are used for background estimation.  Increase if the
        stellar halo extends to the image edge.  Default 0.8.
    min_radius : float
        Minimum mask radius in pixels.  Prevents the mask from collapsing
        for very faint or unresolved sources.  Default 5.
    max_radius : float or None
        Maximum allowed radius in pixels.  Defaults to the distance from
        the centre to the nearest image edge.
    radial_bin_width : float
        Width of each annular bin in pixels.  Default 1.
    smooth_bins : int
        Window size (in bins) for a 1-D median filter applied to the
        radial profile before thresholding.  Suppresses single-bin
        excursions from noise without blurring the broad radial gradient.
        Default 5.
    override_threshold : float or None
        If not *None*, use this fixed threshold value instead of estimating
        from the image.  Useful when the image is too small to get a good
        background estimate, e.g. a tight cutout around a bright star.
    radius_factor : float
        Optional multiplicative factor applied to the measured halo radius.
        Default 1.0 (no scaling).  Increase to be more conservative in
        masking the halo, at the cost of masking more pixels.
    n_sectors : int
        Number of equal angular sectors to divide each annulus into.
        Higher values give finer angular resolution for neighbour rejection
        but reduce the number of pixels per sector.  Default 8.
    min_sector_pixels : int
        Minimum number of finite pixels a sector must contain to be included
        in the per-bin median.  Sectors below this count are skipped.
        Acts as the guard at small radii where annuli are narrow.  Default 5.
    sector_sigma_clip : float
        Sectors whose median exceeds the median-of-sector-medians by more
        than ``sector_sigma_clip × MAD`` are rejected as contaminated
        (e.g. by a neighbouring source).  Set to a large value (e.g. 10)
        to disable outlier rejection.  Default 2.0.

    Returns
    -------
    mask : ndarray of bool, shape (nrows, ncols)
        True inside the halo aperture.
    radius_px : float
        The measured halo radius in pixels.
    """
    from scipy.ndimage import median_filter as _median_filter

    nrows, ncols = image.shape
    img = np.asarray(image, dtype=float)

    if centre is None:
        cy, cx = nrows / 2.0, ncols / 2.0
    else:
        cy, cx = float(centre[0]), float(centre[1])

    Y, X = np.mgrid[:nrows, :ncols]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    # Angle in [0, 2π) for each pixel, used for sector assignment.
    Theta = np.arctan2(Y - cy, X - cx) % (2.0 * np.pi)

    if max_radius is None:
        max_radius = float(min(cy, cx, nrows - cy, ncols - cx))
    max_radius = max(max_radius, min_radius + 1.0)

    # ── Background estimation ─────────────────────────────────────────────
    if not override_threshold:
        bg_inner = background_min_r_frac * max_radius
        bg_level, bg_sigma = estimate_background(img, cy, cx, bg_inner)
        threshold = bg_level + threshold_nsigma * bg_sigma
    else:
        threshold = override_threshold
        bg_level = 0

    # ── Sector-median radial profile with outlier rejection ───────────────
    bins = np.arange(0.0, max_radius + radial_bin_width, radial_bin_width)
    n_bins = len(bins) - 1
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    r_flat = R.ravel()
    t_flat = Theta.ravel()
    img_flat = img.ravel()
    finite_flat = np.isfinite(img_flat)

    sector_edges = np.linspace(0.0, 2.0 * np.pi, n_sectors + 1)

    profile = np.full(n_bins, bg_level)

    r_bin_all = np.digitize(r_flat, bins) - 1
    s_bin_all = np.digitize(t_flat, sector_edges) - 1
    valid = (
        finite_flat
        & (r_bin_all >= 0) & (r_bin_all < n_bins)
        & (s_bin_all >= 0) & (s_bin_all < n_sectors)
    )

    if valid.any():
        vr = r_bin_all[valid].astype(np.intp)
        vs = s_bin_all[valid].astype(np.intp)
        vi = img_flat[valid]

        group_key = vr * n_sectors + vs
        order = np.argsort(group_key, kind='stable')
        gk_sorted = group_key[order]
        vi_sorted = vi[order]
        vr_sorted = vr[order]

        gs_splits = np.flatnonzero(np.diff(gk_sorted)) + 1
        gs_groups = np.split(vi_sorted, gs_splits)
        gs_keys   = gk_sorted[np.r_[0, gs_splits]]

        gr_splits = np.flatnonzero(np.diff(vr_sorted)) + 1
        gr_groups = np.split(vi_sorted, gr_splits)
        gr_keys   = vr_sorted[np.r_[0, gr_splits]]
        bin_vals  = {int(k): g for k, g in zip(gr_keys, gr_groups)}

        bin_sector_medians = {i: [] for i in range(n_bins)}
        for g, k in zip(gs_groups, gs_keys):
            b = int(k) // n_sectors
            if len(g) >= min_sector_pixels:
                bin_sector_medians[b].append(float(np.median(g)))

        for i in range(n_bins):
            sm_list = bin_sector_medians[i]
            if len(sm_list) < 2:
                bv = bin_vals.get(i)
                if bv is not None and len(bv) >= 3:
                    profile[i] = float(np.median(bv))
                continue

            sm = np.array(sm_list)

            if sector_sigma_clip < 10.0 and len(sm) >= 3:
                centre_val = float(np.median(sm))
                spread = mad_std(sm)
                if spread > 0:
                    keep = sm <= centre_val + sector_sigma_clip * spread
                    sm = sm[keep]

            if len(sm) == 0:
                sm = np.array(sm_list)

            profile[i] = float(np.median(sm))

    # ── Smooth profile ────────────────────────────────────────────────────
    if smooth_bins > 1 and n_bins >= smooth_bins:
        profile_s = _median_filter(profile, size=smooth_bins, mode='nearest')
    else:
        profile_s = profile.copy()

    # ── Find halo radius ──────────────────────────────────────────────────
    halo_r = float(min_radius)
    for i in range(n_bins):
        if profile_s[i] >= threshold:
            halo_r = max(float(r_centers[i]), min_radius)

    if radius_factor != 1.0:
        halo_r *= radius_factor

    # ── Build circular mask ───────────────────────────────────────────────
    mask = R <= halo_r
    return mask, halo_r


def combine_masks(masks, operation='or'):
    """Combine multiple boolean pixel masks into one.

    Operates in-place on a copy of the first mask so no extra intermediate
    arrays are allocated — peak memory is two mask arrays at once regardless
    of how many inputs are given.

    Parameters
    ----------
    masks : sequence of ndarray of bool
        Masks to combine.  All must have the same shape.
    operation : {'or', 'and', 'xor'}
        Pixel-wise logical operation applied sequentially.  'or' (union) is
        the usual choice for contamination masks: a pixel is masked if it is
        masked in *any* input.

    Returns
    -------
    combined : ndarray of bool
    """
    masks = list(masks)
    if not masks:
        raise ValueError("at least one mask is required")

    _ops = {
        'or':  np.ndarray.__ior__,
        'and': np.ndarray.__iand__,
        'xor': np.ndarray.__ixor__,
    }
    if operation not in _ops:
        raise ValueError(f"operation must be 'or', 'and', or 'xor'; got {operation!r}")
    op = _ops[operation]

    result = np.asarray(masks[0], dtype=bool).copy()
    for m in masks[1:]:
        m = np.asarray(m, dtype=bool)
        if m.shape != result.shape:
            raise ValueError(
                f"mask shapes are inconsistent: {result.shape} vs {m.shape}"
            )
        op(result, m)
    return result


def combine_fits_masks(
    input_paths,
    output_path,
    hdu_indices=None,
    operation='or',
    tile_size=4096,
    n_workers=4,
):
    """Combine multiple FITS pixel masks into one using tiled memmap processing.

    Reads each input file tile-by-tile via memory mapping so peak RAM usage
    is proportional to ``tile_size`` rather than the full image, regardless of
    how many files are combined.  All input files must share the same pixel
    dimensions.  The WCS header is taken from the first file.

    Typical use — union of a spike mask and a border mask::

        combine_fits_masks([spike_mask.fits, border_mask.fits], combined.fits)

    Parameters
    ----------
    input_paths : sequence of str or path-like
        FITS mask files to combine.
    output_path : str or path-like
        Destination FITS path.  Overwritten if it already exists.
    hdu_indices : sequence of int or None
        HDU index for each input file.  *None* defaults to 0 for every file.
    operation : {'or', 'and'}
        Pixel-wise operation applied across all inputs.  'or' marks a pixel
        masked if it is masked in *any* input (union); 'and' only if masked
        in *all* inputs (intersection).  Default 'or'.
    tile_size : int
        Side length of each square processing tile in pixels.  Default 4096.
    n_workers : int
        Number of parallel worker threads.  ``-1`` uses all available CPU
        threads.  Default 4.

    Returns
    -------
    None
        The combined mask is written directly to *output_path*.
    """
    try:
        from astropy.io import fits as _fits
        from astropy.wcs import WCS as _WCS
    except ImportError:
        raise ImportError(
            "astropy is required for combine_fits_masks. "
            "Install with: pip install 'spikeout[astropy]'"
        )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kw):
            return it

    input_paths = list(input_paths)
    if not input_paths:
        raise ValueError("at least one input path is required")
    if operation not in ('or', 'and'):
        raise ValueError(f"operation must be 'or' or 'and'; got {operation!r}")
    if hdu_indices is None:
        hdu_indices = [0] * len(input_paths)
    if len(hdu_indices) != len(input_paths):
        raise ValueError("hdu_indices must have the same length as input_paths")

    # ── Read headers, validate dimensions, open memmaps ──────────────────
    header0 = _fits.getheader(input_paths[0], ext=hdu_indices[0])
    full_wcs = _WCS(header0)
    H = int(header0['NAXIS2'])
    W = int(header0['NAXIS1'])

    in_fits = [_fits.open(p, memmap=True) for p in input_paths]
    in_data = []
    for i, (fh, hi) in enumerate(zip(in_fits, hdu_indices)):
        d = fh[hi].data
        if d.shape != (H, W):
            for f in in_fits:
                f.close()
            raise ValueError(
                f"input_paths[{i}] has shape {d.shape}, expected ({H}, {W})"
            )
        in_data.append(d)

    # ── Pre-allocate output FITS ──────────────────────────────────────────
    fill = np.uint8(0) if operation == 'or' else np.uint8(1)
    out_hdu = _fits.PrimaryHDU(data=np.full((H, W), fill, dtype=np.uint8))
    out_hdu.header.update(full_wcs.to_header())
    out_hdu.writeto(output_path, overwrite=True)
    out_fits = _fits.open(output_path, mode='update', memmap=True)
    out_data = out_fits[0].data

    # ── Per-tile worker ───────────────────────────────────────────────────
    def _process_tile(row0, col0):
        row1 = min(row0 + tile_size, H)
        col1 = min(col0 + tile_size, W)

        # Read first tile as the accumulator; keep as uint8 throughout
        tile = in_data[0][row0:row1, col0:col1].astype(np.uint8)

        for d in in_data[1:]:
            chunk = d[row0:row1, col0:col1]
            if operation == 'or':
                tile |= chunk
            else:  # 'and'
                tile &= chunk

        out_data[row0:row1, col0:col1] = tile

    # ── Parallel tile loop ────────────────────────────────────────────────
    tiles = [
        (r, c)
        for r in range(0, H, tile_size)
        for c in range(0, W, tile_size)
    ]
    max_workers = None if n_workers == -1 else n_workers

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_tile, r, c): (r, c) for r, c in tiles}
            for fut in _tqdm(
                as_completed(futures), total=len(futures), desc='Combining masks'
            ):
                fut.result()
    finally:
        out_fits.flush()
        out_fits.close()
        for fh in in_fits:
            fh.close()


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


def write_spike_mask_fits(
    entries,
    image_path,
    output_path,
    hdu_index=0,
    width_fraction=0.1,
    min_width=5.0,
    max_width=None,
    tile_size=4096,
    n_workers=4,
    invert=False,
):
    """Write a full-frame spike + halo mask to a FITS file using tiled processing.

    Rasterises spike boxes and stellar halo circles from a list of
    ``CatalogueEntry`` objects into a ``uint8`` FITS image (1 = masked,
    0 = unmasked).  The output file is pre-allocated as a memory-mapped FITS
    image and tiles are filled in parallel, so peak RAM usage is proportional
    to tile size rather than the full image.

    Spike boxes require ``entry.result.lengths`` to be populated
    (run `catalogue_detect` with ``measure_lengths=True``).  Halo circles are
    drawn whenever ``entry.halo_radius`` is set (requires ``halo_mask_kw`` to
    have been passed to `catalogue_detect`).  Either component is silently
    skipped for entries where the relevant data is absent.

    Parameters
    ----------
    entries : list of CatalogueEntry
        ``.ra`` and ``.dec`` are required.  ``.result.lengths`` drives spike
        boxes; ``.halo_radius`` drives halo circles.
    image_path : str or path-like
        Source FITS image — only the header/WCS are read (no pixel data).
    output_path : str or path-like
        Destination FITS path.  Overwritten if it already exists.
    hdu_index : int
        HDU containing the WCS.  Default 0.
    width_fraction, min_width, max_width
        Box width geometry; identical semantics to `spike_box_regions`.
    tile_size : int
        Side length of each processing tile in pixels.  Default 4096.
    n_workers : int
        Number of parallel worker threads.  ``1`` runs sequentially.
        ``-1`` uses all available CPU threads.  Default 4.
    invert : bool
        If *True*, invert the mask so that 0 = masked and 1 = unmasked.  Default *False*.

    Returns
    -------
    None
        The mask is written directly to *output_path*.
    """
    try:
        from astropy.io import fits as _fits
        from astropy.wcs import WCS as _WCS
        from astropy.coordinates import SkyCoord as _SkyCoord
    except ImportError:
        raise ImportError(
            "astropy is required for write_spike_mask_fits. "
            "Install with: pip install 'spikeout[astropy]'"
        )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from skimage.draw import polygon as sk_polygon, disk as sk_disk
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kw):
            return it

    # ── Read WCS and image dimensions from header only ────────────────────
    header = _fits.getheader(image_path, ext=hdu_index)
    full_wcs = _WCS(header)
    H = int(header['NAXIS2'])
    W = int(header['NAXIS1'])

    # ── Precompute all spike-box corners and halo circles in full-image
    #    pixel coords ────────────────────────────────────────────────────────
    # Boxes: (corners_col, corners_row, col_min, col_max, row_min, row_max)
    # Halos: (cx, cy, radius)  — bounding box stored separately as circles_bb
    boxes = []
    circles = []   # (cx_full, cy_full, radius)
    for entry in entries:
        if entry.error is not None:
            continue

        sky = _SkyCoord(ra=entry.ra, dec=entry.dec, unit='deg')
        px, py = full_wcs.world_to_pixel(sky)
        cx, cy = float(px), float(py)   # col (x), row (y)

        # When recenter_for_lengths was used, arm lengths (and the halo) are
        # measured from corrected_centre, not the cutout image centre.
        # star_centre_offset = (dx, dy) is in the Radon y-up pixel frame:
        #   dx > 0 → star is dx px to the right (col)
        #   dy > 0 → star is dy px upward (North; cy increases northward here)
        result = entry.result
        if (
            result is not None
            and getattr(result, 'corrected_centre', None) is not None
            and getattr(result, 'star_centre_offset', None) is not None
        ):
            dx, dy = result.star_centre_offset
            cx += dx
            cy += dy

        # Halo circle (independent of lengths being populated)
        if entry.halo_radius is not None and entry.halo_radius > 0:
            circles.append((cx, cy, float(entry.halo_radius)))

        if result is None or result.lengths is None:
            continue

        for sl in result.lengths:
            angle_rad = np.deg2rad(sl.angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            offset = (sl.length_pos - sl.length_neg) / 2.0
            bx = cx + offset * cos_a
            by = cy + offset * sin_a

            half_len = sl.length_total / 2.0
            half_wid = _box_width(
                sl.length_total, width_fraction, min_width, max_width
            ) / 2.0

            # 4 corners; x = column, y = row
            cx_box = np.array([
                bx + half_len * cos_a - half_wid * sin_a,
                bx + half_len * cos_a + half_wid * sin_a,
                bx - half_len * cos_a + half_wid * sin_a,
                bx - half_len * cos_a - half_wid * sin_a,
            ])
            cy_box = np.array([
                by + half_len * sin_a + half_wid * cos_a,
                by + half_len * sin_a - half_wid * cos_a,
                by - half_len * sin_a - half_wid * cos_a,
                by - half_len * sin_a + half_wid * cos_a,
            ])
            boxes.append((
                cx_box, cy_box,
                cx_box.min(), cx_box.max(),
                cy_box.min(), cy_box.max(),
            ))

    # ── Pre-allocate output FITS via memmap ───────────────────────────────
    out_hdu = _fits.PrimaryHDU(data=np.zeros((H, W), dtype=np.uint8))
    out_hdu.header.update(full_wcs.to_header())
    out_hdu.writeto(output_path, overwrite=True)
    out_fits = _fits.open(output_path, mode='update', memmap=True)
    out_data = out_fits[0].data

    if not boxes and not circles:
        out_fits.flush()
        out_fits.close()
        return

    # Vectorised bounding boxes for fast per-tile intersection tests
    bb = np.array([
        (col_min, col_max, row_min, row_max)
        for _, _, col_min, col_max, row_min, row_max in boxes
    ], dtype=np.float64) if boxes else np.empty((0, 4), dtype=np.float64)

    # Circles stored as (cx, cy, r); bounding box is trivially ±r
    circ_arr = np.array(circles, dtype=np.float64) \
        if circles else np.empty((0, 3), dtype=np.float64)

    # ── Per-tile worker ───────────────────────────────────────────────────
    def _process_tile(row0, col0):
        row1 = min(row0 + tile_size, H)
        col1 = min(col0 + tile_size, W)
        th = row1 - row0
        tw = col1 - col0

        tile = np.zeros((th, tw), dtype=np.uint8)
        any_hit = False

        # Spike boxes
        if bb.shape[0]:
            hit = (
                (bb[:, 0] < col1) & (bb[:, 1] > col0) &
                (bb[:, 2] < row1) & (bb[:, 3] > row0)
            )
            for idx in np.nonzero(hit)[0]:
                local_cols = boxes[idx][0] - col0
                local_rows = boxes[idx][1] - row0
                rr, cc = sk_polygon(local_rows, local_cols, shape=(th, tw))
                tile[rr, cc] = 1
                any_hit = True

        # Halo circles
        if circ_arr.shape[0]:
            cx_arr, cy_arr, r_arr = circ_arr[:, 0], circ_arr[:, 1], circ_arr[:, 2]
            hit_c = (
                (cx_arr + r_arr > col0) & (cx_arr - r_arr < col1) &
                (cy_arr + r_arr > row0) & (cy_arr - r_arr < row1)
            )
            for idx in np.nonzero(hit_c)[0]:
                # sk_disk centre is (row, col)
                rr, cc = sk_disk(
                    (cy_arr[idx] - row0, cx_arr[idx] - col0),
                    r_arr[idx],
                    shape=(th, tw),
                )
                tile[rr, cc] = 1
                any_hit = True

        if invert:
            tile = 1 - tile

        if any_hit:
            out_data[row0:row1, col0:col1] |= tile

    # ── Parallel tile loop ────────────────────────────────────────────────
    tiles = [
        (r, c)
        for r in range(0, H, tile_size)
        for c in range(0, W, tile_size)
    ]
    max_workers = None if n_workers == -1 else n_workers

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_tile, r, c): (r, c) for r, c in tiles}
        for fut in _tqdm(
            as_completed(futures), total=len(futures), desc='Writing spike mask'
        ):
            fut.result()   # propagate any worker exception

    out_fits.flush()
    out_fits.close()

def compress_fits_mask_to_bytes(image_path, output_path, hdu_index=0):
    """Compress a FITS mask to a byte-packed .npz file with the smallest footprint.
    Store dimensions as metadata in the .npz file for later reconstruction.

    Examples of use:
    >>> write_spike_mask_fits(entries, image_path, output_path)
    >>> write_border_mask_fits(image_path, output_path, edge_distance_px)
    >>> compress_fits_mask_to_bytes(output_path, compressed_output_path)

    Examples of reconstruction:
    >>> npz = np.load(compressed_output_path)
    >>> packed_mask = npz['mask']
    >>> height = npz['height']
    >>> width = npz['width']
    >>> mask = np.unpackbits(packed_mask)[:height * width].reshape((height, width))

    """
    from astropy.io import fits as _fits
    from astropy.wcs import WCS as _WCS
    with _fits.open(image_path, memmap=True) as fits:
        mask_data = fits[hdu_index].data.astype(bool)
        H, W = mask_data.shape
        # Pack bits into bytes; the last byte may have unused bits if W is not a multiple of 8
        packed_mask = np.packbits(mask_data, axis=-1)
        # store WCS as string metadata in the .npz file for later use if needed
        header = fits[hdu_index].header
        wcs = _WCS(header)
        wcs_header_str = wcs.to_header_string()

        np.savez_compressed(output_path, mask=packed_mask, height=H, width=W, wcs_header=wcs_header_str)

def decompress_bytes_to_fits_mask(npz_path, output_fits_path):
    """Decompress a byte-packed .npz file back to a FITS mask file.

    Examples of use:
    >>> decompress_bytes_to_fits_mask(compressed_output_path, reconstructed_fits_path)

    """
    from astropy.io import fits as _fits
    from astropy.wcs import WCS as _WCS
    npz = np.load(npz_path)
    packed_mask = npz['mask']
    height = npz['height']
    width = npz['width']
    wcs_header_str = str(npz['wcs_header'])

    # create header from the stored WCS string, then create WCS object from that header
    wcs_header = _fits.Header.fromstring(wcs_header_str)
    wcs = _WCS(wcs_header)

    # Unpack bits and reshape to original dimensions
    mask_data = np.unpackbits(packed_mask)[:height * width].reshape((height, width)).astype(np.uint8)

    # Write to FITS with the original WCS header
    hdu = _fits.PrimaryHDU(data=mask_data)
    hdu.header.update(wcs.to_header())
    hdu.writeto(output_fits_path, overwrite=True)
    


def write_border_mask_fits(
    image_path,
    output_path,
    edge_distance_px,
    hdu_index=0,
    tile_size=4096,
    n_workers=4,
):
    """Write a border-proximity mask to a FITS file using tiled EDT processing.

    Pixels within *edge_distance_px* pixels of the image border (defined as
    the outermost extent of valid — non-NaN, non-zero — data) are set to 1.
    Internal holes such as dead pixels or small NaN regions are filled before
    border detection so they do not generate spurious interior edge masks.

    The distance is computed via ``scipy.ndimage.distance_transform_edt`` on
    the border mask.  Each tile is processed with a halo of at least
    *edge_distance_px* pixels so that EDT results are accurate right up to
    tile boundaries.

    Parameters
    ----------
    image_path : str or path-like
        Source FITS image (memmap-read; only the validity mask is loaded).
    output_path : str or path-like
        Destination FITS path.  Overwritten if it already exists.
    edge_distance_px : int or float
        Mask pixels closer than this many pixels to the image border.
    hdu_index : int
        HDU containing the image data.  Default 0.
    tile_size : int
        Side length of each processing tile in pixels.  Default 4096.
    n_workers : int
        Number of parallel worker threads.  ``-1`` uses all available
        CPU threads.  Default 4.

    Returns
    -------
    None
        The mask is written directly to *output_path*.
    """
    try:
        from astropy.io import fits as _fits
        from astropy.wcs import WCS as _WCS
    except ImportError:
        raise ImportError(
            "astropy is required for write_border_mask_fits. "
            "Install with: pip install 'spikeout[astropy]'"
        )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from scipy.ndimage import distance_transform_edt, binary_fill_holes
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kw):
            return it

    edge_px = int(np.ceil(edge_distance_px))

    # ── Read header and memmap image data ─────────────────────────────────
    header = _fits.getheader(image_path, ext=hdu_index)
    H = int(header['NAXIS2'])
    W = int(header['NAXIS1'])
    full_wcs = _WCS(header)

    image_fits = _fits.open(image_path, memmap=True)
    image_data = image_fits[hdu_index].data

    # ── Build global border mask (full image, boolean — much smaller than
    #    the pixel data).  Fills internal holes so dead pixels / small NaN
    #    regions don't generate spurious interior edge strips.  ────────────
    print("Building global border mask …")
    valid = np.isfinite(image_data) & (image_data != 0)
    # binary_fill_holes requires the full array; result is a compact bool array
    border_mask = ~binary_fill_holes(valid)   # True = exterior / invalid
    del valid

    # ── Pre-allocate output FITS ──────────────────────────────────────────
    out_hdu = _fits.PrimaryHDU(data=np.zeros((H, W), dtype=np.uint8))
    out_hdu.header.update(full_wcs.to_header())
    out_hdu.writeto(output_path, overwrite=True)
    out_fits = _fits.open(output_path, mode='update', memmap=True)
    out_data = out_fits[0].data

    # ── Per-tile worker ───────────────────────────────────────────────────
    def _process_tile(row0, col0):
        row1 = min(row0 + tile_size, H)
        col1 = min(col0 + tile_size, W)
        th = row1 - row0
        tw = col1 - col0

        # Padded slice for EDT context (clamped to image bounds)
        pr0 = max(row0 - edge_px, 0);  pr1 = min(row1 + edge_px, H)
        pc0 = max(col0 - edge_px, 0);  pc1 = min(col1 + edge_px, W)
        r_off = row0 - pr0
        c_off = col0 - pc0

        chunk_border = border_mask[pr0:pr1, pc0:pc1]

        if not chunk_border.any():
            # Entirely interior — no border pixels even in the padded region
            return

        if chunk_border.all():
            out_data[row0:row1, col0:col1] = 1
            return

        # EDT: distance from valid data (i.e. from ~chunk_border)
        dist = distance_transform_edt(~chunk_border)
        edge_padded = (dist < edge_px).astype(np.uint8)

        # Force mask on true image-boundary strips (pixels right at the array
        # edge have no exterior context from the padded slice, so EDT cannot
        # see beyond the image; mark them explicitly).
        if pr0 == 0: edge_padded[:edge_px, :]  = 1
        if pr1 == H: edge_padded[-edge_px:, :] = 1
        if pc0 == 0: edge_padded[:, :edge_px]  = 1
        if pc1 == W: edge_padded[:, -edge_px:]  = 1

        out_data[row0:row1, col0:col1] = \
            edge_padded[r_off:r_off + th, c_off:c_off + tw]

    # ── Parallel tile loop ────────────────────────────────────────────────
    tiles = [
        (r, c)
        for r in range(0, H, tile_size)
        for c in range(0, W, tile_size)
    ]
    max_workers = None if n_workers == -1 else n_workers

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_tile, r, c): (r, c) for r, c in tiles}
        for fut in _tqdm(
            as_completed(futures), total=len(futures), desc='Writing border mask'
        ):
            fut.result()

    out_fits.flush()
    out_fits.close()
    image_fits.close()
