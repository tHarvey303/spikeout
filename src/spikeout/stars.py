"""Star catalogue utilities: fetching, proper-motion correction, and filtering."""

import warnings
import numpy as np

__all__ = ["fetch_gaia_stars"]


def _proximity_filter(table, mag_col, sep_arcsec):
    """Remove the fainter source from each close pair in *table*.

    For every pair of rows whose propagated sky positions are within
    *sep_arcsec* of each other, the row with the larger (fainter) value in
    *mag_col* is removed.  Rows with NaN magnitudes are treated as infinitely
    faint.  The brighter source of each pair is always retained.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Must contain ``ra_epoch``, ``dec_epoch`` (degrees) and *mag_col*.
    mag_col : str
        Column name used to compare brightness.
    sep_arcsec : float
        Angular separation threshold in arcseconds.

    Returns
    -------
    `~astropy.table.Table`
        Filtered table (a new masked view, not a copy of the full array).
    int
        Number of rows removed.
    """
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        raise ImportError("astropy is required for proximity filtering.")

    if len(table) <= 1:
        return table, 0

    if isinstance(table, str):
        from astropy.table import Table
        table = Table.read(table)

    coords = SkyCoord(
        ra=np.array(table["ra_epoch"], dtype=float) * u.deg,
        dec=np.array(table["dec_epoch"], dtype=float) * u.deg,
    )
    idx1, idx2, _, _ = coords.search_around_sky(
        coords, seplimit=sep_arcsec * u.arcsec,
    )
    # Each unordered pair once; self-matches excluded by idx1 < idx2
    pair_mask = idx1 < idx2
    idx1, idx2 = idx1[pair_mask], idx2[pair_mask]

    raw_mags = np.array(table[mag_col], dtype=float)
    mags = np.where(np.isfinite(raw_mags), raw_mags, np.inf)

    remove = np.zeros(len(table), dtype=bool)
    for i, j in zip(idx1, idx2):
        if mags[i] <= mags[j]:
            remove[j] = True
        else:
            remove[i] = True

    n_removed = int(remove.sum())
    return table[~remove], n_removed


def fetch_gaia_stars(
    fits_path,
    epoch,
    hdu_index=0,
    scale_extra=0.1,
    mag_limit=None,
    mag_col="phot_g_mean_mag",
    gaia_row_limit=10_000_000,
    min_separation_arcsec=None,
    output_path=None,
    verbose=True,
    cred=None,
):
    """Fetch Gaia DR3 stars covering a FITS image footprint and apply proper
    motion correction to a target epoch.

    The image footprint is expanded by *scale_extra* (as a fraction of each
    axis) on all sides before querying Gaia, so stars near the edges are
    included.  Proper motions, parallaxes, and radial velocities are used to
    propagate coordinates from the Gaia DR3 reference epoch (J2016.0) to the
    requested *epoch*.  Missing astrometric values are zero-filled before
    propagation.

    The returned table contains all Gaia DR3 columns plus:

    ``ra_epoch``, ``dec_epoch``
        Proper-motion-corrected sky coordinates at *epoch* (degrees).
    ``x_pix``, ``y_pix``
        Pixel coordinates in the input image (0-indexed).

    Parameters
    ----------
    fits_path : str or path-like
        Path to the FITS image.  Only the header/WCS are read.
    epoch : str or `~astropy.time.Time`
        Target epoch for proper-motion correction, e.g. ``'2025-01-01'``
        or ``Time('2025-01-01')``.
    hdu_index : int
        HDU containing the WCS.  Default 0.
    scale_extra : float
        Fractional padding added to each image axis before querying.
        Default 0.1 (10 % per side).
    mag_limit : float or None
        If given, only rows with ``mag_col < mag_limit`` are returned.
    mag_col : str
        Magnitude column used for *mag_limit*.  Default ``'phot_g_mean_mag'``.
    gaia_row_limit : int
        Maximum rows returned by the Gaia TAP query.  Default 10 000 000.
    min_separation_arcsec : float or None
        If given, sources with a brighter neighbour (in ``mag_col``) within
        this angular distance are removed.  For each pair of sources closer
        than the threshold, the fainter one is dropped; the brighter is kept.
        Sources with no magnitude measurement (NaN) are treated as the
        faintest possible and will be dropped in favour of any measured
        neighbour.  Applied after the ``mag_limit`` cut.  Default *None*
        (no proximity filtering).
    output_path : str or path-like or None
        If provided, the table is written to this FITS path (overwriting any
        existing file).
    verbose : bool
        Print progress messages.  Default True.

    Returns
    -------
    `~astropy.table.Table`
        Gaia DR3 source table with propagated coordinates and pixel positions.

    Raises
    ------
    ImportError
        If ``astroquery`` or ``astropy`` are not installed.
    """
    try:
        from astropy.io import fits as _fits
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord, Distance
        from astropy.time import Time
        from astropy.table import Table
        from astropy.utils.exceptions import AstropyWarning
        import astropy.units as u
    except ImportError:
        raise ImportError(
            "astropy is required for fetch_gaia_stars. "
            "Install with: pip install astropy"
        )
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        raise ImportError(
            "astroquery is required for fetch_gaia_stars. "
            "Install with: pip install astroquery"
        )

    warnings.filterwarnings("ignore", category=AstropyWarning)

    target_time = Time(epoch) if not isinstance(epoch, Time) else epoch

    # ── Image footprint ───────────────────────────────────────────────────
    header = _fits.getheader(fits_path, ext=hdu_index)
    wcs = WCS(header)
    nx = float(header["NAXIS1"])
    ny = float(header["NAXIS2"])

    pad_x = scale_extra * nx
    pad_y = scale_extra * ny
    vertices_pix = [
        (-pad_x,        -pad_y),
        (-pad_x,        ny + pad_y),
        (nx + pad_x,    ny + pad_y),
        (nx + pad_x,    -pad_y),
    ]
    vertices_sky = wcs.all_pix2world(vertices_pix, 0)

    if cred is not None:
        print("Logging in to Gaia archive with provided credentials …")
        Gaia.login(**cred)

    # ── Gaia TAP query ────────────────────────────────────────────────────
    Gaia.ROW_LIMIT = gaia_row_limit

    poly_pts = " ".join(
        f"POINT('ICRS', {v[0]}, {v[1]})," for v in vertices_sky
    ).rstrip(",")

    adql = f"""
        SELECT source_id, ra, dec, {mag_col},
               classlabel_dsc_joint, vari_best_class_name,
               pmra, pmdec, parallax, radial_velocity, ref_epoch
        FROM gaiadr3.gaia_source
        LEFT OUTER JOIN gaiadr3.galaxy_candidates USING (source_id)
        WHERE 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            POLYGON('ICRS', {poly_pts}))
    """


    adql = f"""
        SELECT source_id, gs.ra, gs.dec, gs.{mag_col}, gs.pmra, 
            gs.pmdec, gs.parallax, gs.parallax_error, 
            gs.radial_velocity, gs.ref_epoch, gs.ruwe,
            gc.classlabel_dsc_joint, gc.vari_best_class_name,
            gc.radius_sersic, gc.n_sersic, gc.ellipticity_sersic, gc.redshift_ugc
        FROM gaiadr3.gaia_source AS gs
        LEFT OUTER JOIN gaiadr3.galaxy_candidates AS gc USING (source_id)
        WHERE 1 = CONTAINS(
            POINT('ICRS', gs.ra, gs.dec),
            POLYGON('ICRS', {poly_pts}))
    """

    print(adql)

    if verbose:
        print("Querying Gaia DR3 …")
    job = Gaia.launch_job_async(adql)
    gaia_table = job.get_results()
    if verbose:
        print(f"  {len(gaia_table)} sources returned.")

    # ── Proper-motion propagation ─────────────────────────────────────────
    ra_deg  = np.array(gaia_table["ra"],  dtype=float)
    dec_deg = np.array(gaia_table["dec"], dtype=float)

    pmra  = np.nan_to_num(np.array(gaia_table["pmra"],  dtype=float)) * (u.mas / u.yr)
    pmdec = np.nan_to_num(np.array(gaia_table["pmdec"], dtype=float)) * (u.mas / u.yr)

    plx = np.nan_to_num(np.array(gaia_table["parallax"], dtype=float))
    plx = np.where(plx > 0, plx, 0.01)   # Distance requires positive parallax

    rv = np.nan_to_num(np.array(gaia_table["radial_velocity"], dtype=float)) * (u.km / u.s)

    ref_epoch = np.nan_to_num(
        np.array(gaia_table["ref_epoch"], dtype=float), nan=2016.0
    )

    coords = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        distance=Distance(parallax=plx * u.mas),
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=rv,
        obstime=Time(ref_epoch, format="jyear"),
    )
    propagated = coords.apply_space_motion(new_obstime=target_time)

    gaia_table["ra_epoch"]  = propagated.ra.deg
    gaia_table["dec_epoch"] = propagated.dec.deg

    x_pix, y_pix = wcs.all_world2pix(
        propagated.ra.deg, propagated.dec.deg, 0
    )
    gaia_table["x_pix"] = x_pix
    gaia_table["y_pix"] = y_pix

    # ── Fix masked string columns for FITS serialisation ──────────────────
    for col in ("classlabel_dsc_joint", "vari_best_class_name"):
        if col not in gaia_table.colnames:
            continue
        if hasattr(gaia_table[col], "filled"):
            gaia_table[col] = gaia_table[col].filled("")
        gaia_table[col] = gaia_table[col].astype("<U20")

    # ── Optional magnitude filter ─────────────────────────────────────────
    if mag_limit is not None and mag_col in gaia_table.colnames:
        gaia_table = gaia_table[
            np.array(gaia_table[mag_col], dtype=float) < mag_limit
        ]
        if verbose:
            print(f"  {len(gaia_table)} sources after {mag_col} < {mag_limit} cut.")

    # ── Optional proximity filter (remove fainter of each close pair) ────
    if min_separation_arcsec is not None:
        gaia_table, n_removed = _proximity_filter(
            gaia_table, mag_col, min_separation_arcsec,
        )
        if verbose:
            print(
                f"  {len(gaia_table)} sources after proximity filter "
                f"({min_separation_arcsec} arcsec; {n_removed} neighbours removed)."
            )

    # ── Optional write ────────────────────────────────────────────────────
    if output_path is not None:
        gaia_table.write(str(output_path), overwrite=True, format="fits")
        if verbose:
            print(f"  Written to {output_path}")

    return gaia_table
