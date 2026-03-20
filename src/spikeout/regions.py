"""DS9 region file generation and pixel masks for diffraction spikes and halos."""

import numpy as np
from .stats import mad_std, estimate_background

__all__ = [
    "spike_box_regions",
    "spike_mask",
    "write_spike_mask_fits",
    "write_border_mask_fits",
    "write_ds9_regions",
    "write_catalogue_ds9_regions",
    "halo_mask",
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
            print(f"Warning: entry {entry} has result.lengths = None; skipping")
            if entry.result is not None and entry.lengths is None:
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

    # if we have halo measurements, add a circle region for the halo aperture
    for entry in entries:
        if entry.halo_radius is None:
            continue

        halo_r_arcsec = entry.halo_radius * scale
        regions.append(
            f'circle({entry.ra:.6f},{entry.dec:.6f},{halo_r_arcsec:.2f}")'
        )

    _write_reg_file(path, "fk5", regions, colour)


def halo_mask(
    image,
    centre=None,
    threshold_nsigma=3.0,
    background_min_r_frac=0.6,
    min_radius=5.0,
    max_radius=None,
    radial_bin_width=1.0,
    smooth_bins=5,
):
    """Boolean circular mask enclosing the stellar halo.

    Builds a robust azimuthal-median radial profile (immune to diffraction
    spikes and neighbouring sources) and finds the outermost radius at which
    the profile exceeds a background-noise threshold.

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
        stellar halo extends to the image edge.  Default 0.6.
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
        excursions from noise or a neighbour source occupying a small
        fraction of an annulus.  Default 5.

    Returns
    -------
    mask : ndarray of bool, shape (nrows, ncols)
        True inside the halo aperture.
    radius_px : float
        The measured halo radius in pixels.

    Notes
    -----
    The azimuthal median is computed independently at each radial bin, so
    a neighbouring source affects only bins where it occupies more than
    half the annulus area — typically none for sources well separated from
    the target.  For sources very close to the target, increase
    *threshold_nsigma* or provide a custom *centre*.
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

    if max_radius is None:
        max_radius = float(min(cy, cx, nrows - cy, ncols - cx))
    max_radius = max(max_radius, min_radius + 1.0)

    # ── Background estimation ─────────────────────────────────────────────
    # Use sep-based estimator (masks halo + neighbouring sources) when
    # available; falls back to pixel-to-pixel MAD otherwise.
    bg_inner = background_min_r_frac * max_radius
    bg_level, bg_sigma = estimate_background(img, cy, cx, bg_inner)
    threshold = bg_level + threshold_nsigma * bg_sigma

    # ── Azimuthal-median radial profile ───────────────────────────────────
    # Median per concentric annulus; a single-source neighbour biases the
    # result only when it covers > 50 % of the annulus — very unlikely for
    # r much larger than the neighbour's PSF.
    bins = np.arange(0.0, max_radius + radial_bin_width, radial_bin_width)
    n_bins = len(bins) - 1
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    r_flat = R.ravel()
    img_flat = img.ravel()
    finite_flat = np.isfinite(img_flat)

    # Initialise profile to bg_level so empty/sparse bins don't trigger a
    # spurious detection.
    profile = np.full(n_bins, bg_level)
    for i in range(n_bins):
        in_bin = (r_flat >= bins[i]) & (r_flat < bins[i + 1]) & finite_flat
        if in_bin.sum() >= 3:
            profile[i] = float(np.median(img_flat[in_bin]))

    # ── Smooth profile ────────────────────────────────────────────────────
    # 1-D median filter kills single-bin spikes from noise or a bright
    # neighbour crossing one annulus, without blurring the broad radial
    # gradient that carries the halo extent information.
    if smooth_bins > 1 and n_bins >= smooth_bins:
        profile_s = _median_filter(profile, size=smooth_bins, mode='nearest')
    else:
        profile_s = profile.copy()

    # ── Find halo radius ──────────────────────────────────────────────────
    # Walk outward; keep updating halo_r as long as the profile is above
    # threshold.  The last update gives the outermost extent of the halo.
    halo_r = float(min_radius)
    for i in range(n_bins):
        if profile_s[i] >= threshold:
            halo_r = max(float(r_centers[i]), min_radius)

    # ── Build circular mask ───────────────────────────────────────────────
    mask = R <= halo_r
    return mask, halo_r


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

        # Halo circle (independent of lengths being populated)
        if entry.halo_radius is not None and entry.halo_radius > 0:
            circles.append((cx, cy, float(entry.halo_radius)))

        if entry.result is None or entry.result.lengths is None:
            continue

        for sl in entry.result.lengths:
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
