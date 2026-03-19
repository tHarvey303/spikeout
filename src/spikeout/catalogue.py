"""Catalogue-mode spike detection over a list of sky positions."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
import warnings


from .detect import detect, SpikeResult
from .regions import halo_mask as _halo_mask

__all__ = ["CatalogueEntry", "catalogue_detect", "catalogue_summary", "plot_catalogue"]

warnings.filterwarnings("ignore", category=UserWarning, module="scipy")


@dataclass
class CatalogueEntry:
    """Detection result for a single catalogue source.

    Attributes
    ----------
    ra, dec : float
        Sky coordinates of the source (degrees).
    cutout : ndarray or None
        Raw image cutout centred on the source.  *None* if extraction failed.
    result : SpikeResult or None
        Spike-detection output.  *None* if detection failed or was skipped.
    error : str or None
        Error message if extraction or detection raised an exception.
    wcs : WCS or None
        Astropy WCS for the cutout.  Populated when extraction succeeds;
        useful for accurate sky-coordinate region export.
    halo_mask : ndarray or None
        Boolean mask of the stellar halo, same shape as ``cutout``.
        Populated when ``halo_mask_kw`` is passed to `catalogue_detect`.
    halo_radius : float or None
        Radius (pixels) of the halo mask.
    """
    ra: float
    dec: float
    cutout: Optional[np.ndarray]
    result: Optional[SpikeResult]
    error: Optional[str] = None
    wcs: Optional[object] = None
    halo_mask: Optional[np.ndarray] = None
    halo_radius: Optional[float] = None

    def __repr__(self) -> str:
        if self.error:
            status = f"error='{self.error[:40]}'"
        elif self.result is not None and len(self.result.angles) > 0:
            status = f"n_spikes={len(self.result.angles)}"
        else:
            status = "no_spikes"
        if self.halo_radius is not None:
            status += f", halo_r={self.halo_radius:.1f}px"
        return f"CatalogueEntry(ra={self.ra:.4f}, dec={self.dec:.4f}, {status})"


def catalogue_detect(
    coords,
    image_path,
    cutout_size=256,
    hdu_index=0,
    n_jobs=1,
    halo_mask_kw=None,
    full_array=None,
    measure_lengths=False,
    length_kw=None,
    **detect_kw,
) -> List[CatalogueEntry]:
    """Run spike detection on a list of sky positions in a FITS image.

    The FITS file is opened with memory-mapping so only the pixels that
    fall inside the requested cutouts are read from disk.

    Parameters
    ----------
    coords : sequence of (ra, dec) in degrees, or `~astropy.coordinates.SkyCoord`
        Sky positions to process.
    image_path : str or path-like
        Path to the FITS image.
    cutout_size : int or sequence of int
        Side length of each square cutout in pixels.  Pass a sequence of
        the same length as ``coords`` to use a different size per source.
    hdu_index : int
        Index of the HDU containing the image data and WCS.
    n_jobs : int
        Number of parallel worker threads for the detection step.
        ``1`` (default) runs sequentially.  ``-1`` uses all available
        CPU threads.  Cutout extraction is always sequential because the
        FITS file is memory-mapped.
    halo_mask_kw : dict or *None*
        If provided, `~spikeout.regions.halo_mask` is called on each
        cutout with these keyword arguments and the result is stored in
        ``CatalogueEntry.halo_mask`` / ``CatalogueEntry.halo_radius``.
        Pass an empty dict ``{}`` to use all defaults.  *None* (default)
        skips halo masking entirely.
    full_array : 2-D array or *None*
        Pre-opened memory-mapped full image array (e.g. ``hdul[0].data``
        with ``memmap=True``).  When provided together with
        ``measure_lengths=True``, spike arms that extend beyond the
        cutout are measured against the full image, so the reported
        lengths are not clipped at the cutout boundary.
    measure_lengths : bool
        If *True*, measure spike arm lengths after detection.  When
        ``full_array`` is given the two-stage extraction is used (cutout
        first, then full image for unconstrained arms).  Lengths are
        stored in ``entry.result.lengths``.
    length_kw : dict or *None*
        Extra keyword arguments forwarded to
        `~spikeout.lengths.measure_spike_lengths`.
    **detect_kw
        Forwarded to `~spikeout.detect.detect`.

    Returns
    -------
    list of `CatalogueEntry`
        One entry per input coordinate, in the same order.  Entries where
        extraction or detection failed have ``result=None`` and a
        non-empty ``error`` string.
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.nddata import Cutout2D
        from astropy.coordinates import SkyCoord
    except ImportError:
        raise ImportError(
            "astropy is required for catalogue_detect. "
            "Install with: pip install 'spikeout[astropy]'"
        )

    if not hasattr(coords, 'ra'):
        coords = SkyCoord(
            [c[0] for c in coords],
            [c[1] for c in coords],
            unit='deg',
        )

    if hasattr(cutout_size, '__len__') and len(cutout_size) != len(coords):
        raise ValueError(
            f"cutout_size has {len(cutout_size)} elements but coords has "
            f"{len(coords)}"
        )

    sizes = (
        [(s, s) for s in cutout_size]
        if hasattr(cutout_size, '__len__')
        else [(cutout_size, cutout_size)] * len(coords)
    )

    from .lengths import measure_spike_lengths as _measure_spike_lengths

    # ── Phase 1: extract cutouts (sequential, file must stay open) ────────
    # raw items: (ra, dec, cutout_data, cutout_wcs, px_col, px_row, error)
    raw = []
    with fits.open(image_path, memmap=True) as hdul:
        hdu = hdul[hdu_index]
        image_wcs = WCS(hdu.header)
        data = hdu.data  # memmap — pixels are not read until sliced

        for sky, size in zip(coords, sizes):
            try:
                # pixel position of source in the full array
                px, py = image_wcs.world_to_pixel(sky)
                co = Cutout2D(
                    data, sky, size, wcs=image_wcs,
                    mode='partial', fill_value=np.nan,
                    copy=True,
                )
                raw.append((
                    float(sky.ra.deg), float(sky.dec.deg),
                    co.data.astype(float), co.wcs,
                    float(px), float(py),  # full-array col, row
                    None,
                ))
            except Exception as exc:
                raw.append((
                    float(sky.ra.deg), float(sky.dec.deg),
                    None, None, None, None, str(exc),
                ))

    # ── Phase 2: run detect (optionally parallel) ─────────────────────────
    entries = []
    _lkw = length_kw or {}

    def _run_one(cutout_data):
        """Run detect (and optionally halo_mask) on one cutout."""
        result = detect(cutout_data, **detect_kw)
        if halo_mask_kw is not None:
            hmask, hradius = _halo_mask(cutout_data, **(halo_mask_kw or {}))
        else:
            hmask, hradius = None, None
        return result, hmask, hradius

    def _apply_lengths(cutout_data, result, px_col, px_row):
        """Measure spike lengths, using full_array if provided."""
        if result is None or len(result.angles) == 0:
            return
        kw = dict(_lkw)
        if full_array is not None and px_col is not None:
            kw.setdefault('full_array', full_array)
            kw.setdefault('centre_col_full', px_col)
            kw.setdefault('centre_row_full', px_row)
        result.lengths = _measure_spike_lengths(cutout_data, result, **kw)

    if n_jobs == 1:
        for ra, dec, cutout_data, cutout_wcs, px_col, px_row, error in tqdm(
            raw, desc="Detecting spikes"
        ):
            if error is not None:
                entries.append(CatalogueEntry(
                    ra=ra, dec=dec, cutout=None, result=None, error=error,
                ))
            else:
                try:
                    result, hmask, hradius = _run_one(cutout_data)
                    if measure_lengths:
                        _apply_lengths(cutout_data, result, px_col, px_row)
                    entries.append(CatalogueEntry(
                        ra=ra, dec=dec, cutout=cutout_data,
                        result=result, wcs=cutout_wcs,
                        halo_mask=hmask, halo_radius=hradius,
                    ))
                except Exception as exc:
                    entries.append(CatalogueEntry(
                        ra=ra, dec=dec, cutout=cutout_data,
                        result=None, error=str(exc), wcs=cutout_wcs,
                    ))
    else:
        from concurrent.futures import ThreadPoolExecutor
        max_workers = None if n_jobs == -1 else n_jobs

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_one, cd) if error is None else None
                for _, _, cd, _, _, _, error in raw
            ]

        for (ra, dec, cd, cwcs, px_col, px_row, error), future in tqdm(
            zip(raw, futures), total=len(raw), desc="Detecting spikes"
        ):
            if error is not None:
                entries.append(CatalogueEntry(
                    ra=ra, dec=dec, cutout=None, result=None, error=error,
                ))
            elif future is None:
                entries.append(CatalogueEntry(
                    ra=ra, dec=dec, cutout=cd, result=None,
                    error="cutout extraction failed",
                ))
            else:
                try:
                    result, hmask, hradius = future.result()
                    if measure_lengths:
                        _apply_lengths(cd, result, px_col, px_row)
                    entries.append(CatalogueEntry(
                        ra=ra, dec=dec, cutout=cd,
                        result=result, wcs=cwcs,
                        halo_mask=hmask, halo_radius=hradius,
                    ))
                except Exception as exc:
                    entries.append(CatalogueEntry(
                        ra=ra, dec=dec, cutout=cd,
                        result=None, error=str(exc), wcs=cwcs,
                    ))

    return entries


def catalogue_summary(entries) -> dict:
    """Summarise detection results across a list of ``CatalogueEntry`` objects.

    Parameters
    ----------
    entries : list of CatalogueEntry

    Returns
    -------
    dict with keys:

    ``total``
        Number of input entries.
    ``processed``
        Entries where extraction succeeded (no error).
    ``failed``
        Entries where extraction or detection raised an exception.
    ``with_spikes``
        Processed entries that have at least one detected spike.
    ``no_spikes``
        Processed entries with no detected spikes.
    ``n_spikes``
        Total number of individual spike detections.
    ``angles``
        1-D array of all detected angles (degrees).
    ``snr``
        1-D array of SNR values for all detected spikes.
    ``lengths``
        1-D array of total lengths (pixels) for all measured spikes.
        Empty if no entry has ``result.lengths`` populated.
    ``halo_radii``
        1-D array of halo radii (pixels) for all entries where halo
        masking was run.  Empty if no entry has ``halo_radius`` set.
    """
    total = len(entries)
    failed = sum(1 for e in entries if e.error is not None)
    processed = total - failed
    with_spikes = sum(
        1 for e in entries
        if e.error is None and e.result is not None and len(e.result.angles) > 0
    )

    all_angles = np.array([
        a for e in entries if e.result is not None for a in e.result.angles
    ])
    all_snr = np.array([
        s for e in entries if e.result is not None for s in e.result.snr
    ])
    all_lengths = np.array([
        sl.length_total
        for e in entries
        if e.result is not None and e.result.lengths is not None
        for sl in e.result.lengths
    ])
    all_halo_radii = np.array([
        e.halo_radius for e in entries if e.halo_radius is not None
    ])

    return {
        "total": total,
        "processed": processed,
        "failed": failed,
        "with_spikes": with_spikes,
        "no_spikes": processed - with_spikes,
        "n_spikes": len(all_angles),
        "angles": all_angles,
        "snr": all_snr,
        "lengths": all_lengths,
        "halo_radii": all_halo_radii,
    }


def plot_catalogue(
    entries,
    ncols=5,
    panel_size=3.0,
    vmin_pct=1,
    vmax_pct=99.5,
):
    """Grid plot of cutouts with detected spike lines overlaid.

    Each panel shows the raw cutout image with detected spike lines drawn
    over it.  Failed entries are shown as dark panels with the error message.

    Parameters
    ----------
    entries : list of `CatalogueEntry`
    ncols : int
        Number of columns in the grid.
    panel_size : float
        Size of each panel in inches.
    vmin_pct, vmax_pct : float
        Percentile cuts for image stretch.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
    """
    import matplotlib.pyplot as plt
    from .geometry import radon_line_to_image

    n = len(entries)
    if n == 0:
        raise ValueError("No entries to plot.")

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).ravel()

    for idx, entry in tqdm(enumerate(entries), total=len(entries)):
        ax = axes_flat[idx]

        if entry.cutout is None or entry.error:
            ax.set_facecolor('0.15')
            msg = entry.error or 'extraction failed'
            if len(msg) > 60:
                msg = msg[:57] + '...'
            ax.text(
                0.5, 0.5, msg,
                ha='center', va='center', transform=ax.transAxes,
                fontsize=6, color='salmon',
            )
        else:
            finite = entry.cutout[np.isfinite(entry.cutout)]
            vmin, vmax = np.percentile(finite, [vmin_pct, vmax_pct]) \
                if finite.size > 0 else (0, 1)
            
            finite = entry.cutout[np.isfinite(entry.cutout)]
            from matplotlib.colors import Normalize, LogNorm
            vmin, vmax = np.percentile(finite, [vmin_pct, vmax_pct]) \
                if finite.size > 0 else (0, 1)
            
            vmin = vmax/1e3
            ax.imshow(entry.cutout, origin='lower', cmap='gray',
                      norm=LogNorm(vmin=vmin, vmax=vmax))

            ax.set_xlim(0, entry.cutout.shape[1])
            ax.set_ylim(0, entry.cutout.shape[0])

            if entry.halo_radius is not None:
                from matplotlib.patches import Circle
                cx = entry.cutout.shape[1] / 2.0
                cy = entry.cutout.shape[0] / 2.0
                ax.add_patch(Circle(
                    (cx, cy), entry.halo_radius,
                    fill=False, edgecolor='cyan', lw=0.8, alpha=0.7,
                ))

            res = entry.result
            if res is not None and len(res.angles) > 0:
                colors = plt.cm.Set1(np.linspace(0, 1, len(res.angles)))
                for i in range(len(res.angles)):
                    rho = res.rho_physical[i]
                    th = res.theta[res.peak_theta_indices[i]]
                    (x1, y1), (x2, y2), angle = radon_line_to_image(
                        rho, th, entry.cutout.shape,
                    )
                    ax.plot([x1, x2], [y1, y2],
                            color='red', lw=1.0, alpha=0.85, linestyle='dotted')
                    '''
                    ax.text(
                        0.02, 0.98 - i * 0.13,
                        f"{angle:.0f}\u00b0  SNR={res.snr[i]:.1f}",
                        transform=ax.transAxes,
                        fontsize=6, color=colors[i], va='top',
                    '''
            elif res is not None:
                ax.text(
                    0.5, 0.02, 'no spikes',
                    ha='center', va='bottom', transform=ax.transAxes,
                    fontsize=6, color='white', alpha=0.6,
                )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"RA={entry.ra:.4f}  Dec={entry.dec:.4f}",
            fontsize=6,
        )

    for idx in range(n, nrows * ncols):
        axes_flat[idx].axis('off')

    return fig
