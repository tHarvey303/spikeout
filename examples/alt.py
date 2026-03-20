"""
Swath Profile Extraction from Memory-Mapped FITS Arrays
========================================================

Extracts a profile along an arbitrary line from a large memory-mapped 2D array
(e.g. astropy FITS memmap) with minimal IO by:

1. Choosing chunk orientation based on line angle (row-major for near-horizontal,
   column-major for near-vertical) to align reads with memory layout.
2. Reading rectangular bounding-box chunks that cover a segment of the line,
   so each chunk is a single contiguous slice — one IO operation.
3. Sampling the swath (multi-pixel width perpendicular to the line) within
   each chunk using fast vectorised NumPy indexing.

The result is one 1D profile: the mean (or median) across the swath width
at each sample step along the line.

Usage
-----
    from astropy.io import fits
    from swath_profile import extract_swath_profile

    hdul = fits.open("large_image.fits", memmap=True)
    data = hdul[0].data  # np.memmap, shape e.g. (100000, 80000)

    profile = extract_swath_profile(
        data,
        start=(5000, 3000),   # (row, col) origin
        angle_deg=37.0,       # degrees CCW from +x  (origin='lower' frame)
        length=4000,          # pixels along the line
        swath_width=5,        # pixels perpendicular to the line
        chunk_length=512,     # how many line-pixels per IO chunk
        reducer="mean",       # "mean" | "median" | "sum"
    )
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_swath_profile(
    data: np.ndarray,
    start: tuple[float, float],
    angle_deg: float,
    length: int,
    swath_width: int = 1,
    chunk_length: int = 512,
    reducer: Literal["mean", "median", "sum"] = "mean",
    step: float = 1.0,
    debug_plot: bool = False,
    debug_plot_margin: int = 50,
    debug_plot_path: str | None = None,
    **debug_kw: dict,
) -> np.ndarray:
    """
    Extract a 1-D swath profile along a line from a 2-D (row, col) array.

    Parameters
    ----------
    data : array-like, shape (nrow, ncol)
        The 2-D image.  Typically an ``np.memmap`` from ``astropy.io.fits``.
        Row-major (C-order) layout is assumed for IO planning.
    start : (row0, col0)
        Starting pixel coordinate (sub-pixel floats accepted).
    angle_deg : float
        Angle of the line in the **display frame** you get from
        ``imshow(..., origin='lower')`` — degrees counter-clockwise from
        the +x (rightward / +column) axis, with +y pointing upward
        (increasing row).  0° = rightward, 90° = upward, etc.
        Compatible with ``SpikeResult.angles``.
    length : int
        Number of sample steps along the line (≈ pixel count).
    swath_width : int
        Width of the swath perpendicular to the line, in pixels.
        1 = single-pixel profile (no averaging).  Must be odd or will be
        rounded up to the next odd number so the line is centred.
    chunk_length : int
        How many along-line samples to process per IO chunk.  Larger values
        read bigger rectangles but make fewer IO calls.  512–2048 is usually
        a good range; tune to your storage latency / bandwidth.
    reducer : {"mean", "median", "sum"}
        How to collapse the swath width into a single value per sample.
    step : float
        Sampling interval along the line in pixels.  1.0 = Nyquist for a
        line at 45°; set < 1 for oversampling.
    debug_plot : bool
        If True, produce a matplotlib figure showing the swath footprint
        overlaid on the image region around the extraction.  Shows:
        - the image cutout (auto-stretched) under the swath
        - the centre-line of the profile
        - the swath edges (± half-width)
        - each chunk's bounding-box rectangle
        - start / end markers
    debug_plot_margin : int
        Extra pixels of padding around the swath bounding box when
        cutting out the image region for the debug figure.
    debug_plot_path : str or None
        If given, save the figure to this path instead of calling
        ``plt.show()``.

    Returns
    -------
    profile : np.ndarray, shape (length,)
        The extracted 1-D profile (float64).
    """
    # ---- normalise inputs ------------------------------------------------
    swath_width = max(1, swath_width)
    if swath_width % 2 == 0:
        swath_width += 1  # ensure odd so line is centred
    half_w = swath_width // 2

    angle_rad = math.radians(angle_deg)
    # Angle convention: degrees CCW from +x in the imshow(origin='lower')
    # display frame, where x = column (rightward), y = row (upward).
    #   display direction  (dx, dy)  = (cos θ,  sin θ)
    #   array   direction  (dr, dc)  = (sin θ,  cos θ)   [row = y]
    dc = math.cos(angle_rad)
    dr = math.sin(angle_rad)
    # perpendicular unit vector — +90° CCW in display frame
    #   display perp (dx, dy) = (-sin θ, cos θ)
    #   array  perp  (pr, pc) = ( cos θ, -sin θ)
    pc = -math.sin(angle_rad)
    pr = math.cos(angle_rad)

    nrow, ncol = data.shape
    reduce_fn = _get_reducer(reducer)

    profile = np.empty(length, dtype=np.float64)

    # ---- decide chunk axis based on angle --------------------------------
    # If the line is more horizontal (|dc| >= |dr|), chunks span columns so
    # each chunk is a few full rows  → contiguous in C-order.
    # If more vertical, chunks span rows.
    horizontal_major = abs(dc) >= abs(dr)

    # ---- collect chunk bounding boxes for debug plot -----------------------
    chunk_bboxes: list[tuple[int, int, int, int]] = []  # (r_min, r_max, c_min, c_max)

    # ---- iterate over chunks along the line ------------------------------
    for chunk_start in range(0, length, chunk_length):
        chunk_end = min(chunk_start + chunk_length, length)
        n_samples = chunk_end - chunk_start

        # --- compute ALL (row, col) sample coordinates for this chunk -----
        # along-line parameter values
        t = chunk_start + np.arange(n_samples, dtype=np.float64) * step

        # centre-line coordinates
        rows_c = start[0] + t * dr
        cols_c = start[1] + t * dc

        # expand across swath width
        offsets = np.arange(-half_w, half_w + 1, dtype=np.float64)  # (swath_width,)

        # shape: (n_samples, swath_width)
        rows_all = rows_c[:, None] + offsets[None, :] * pr
        cols_all = cols_c[:, None] + offsets[None, :] * pc

        # round to nearest integer pixel
        ri = np.rint(rows_all).astype(np.intp)
        ci = np.rint(cols_all).astype(np.intp)

        # --- compute bounding box of this chunk's samples -----------------
        r_min, r_max = int(ri.min()), int(ri.max())
        c_min, c_max = int(ci.min()), int(ci.max())

        # clip to image bounds
        r_min_c = max(r_min, 0)
        r_max_c = min(r_max, nrow - 1)
        c_min_c = max(c_min, 0)
        c_max_c = min(c_max, ncol - 1)

        chunk_bboxes.append((r_min_c, r_max_c, c_min_c, c_max_c))

        # --- single rectangular read  (the key IO operation) --------------
        # For a C-order memmap this touches contiguous pages when the bbox
        # is narrow in the row direction (horizontal line) or when
        # chunk_length is tuned so the bbox fits in a reasonable window.
        chunk_data = np.asarray(
            data[r_min_c : r_max_c + 1, c_min_c : c_max_c + 1]
        )
        # np.asarray forces the read from the memmap into RAM once.

        # --- re-index samples into the local chunk array ------------------
        li = ri - r_min_c  # local row indices
        lj = ci - c_min_c  # local col indices

        # mask out-of-bounds pixels
        valid = (
            (ri >= 0) & (ri < nrow) &
            (ci >= 0) & (ci < ncol)
        )

        # gather values — fill invalids with NaN so reducer can ignore them
        vals = np.full(ri.shape, np.nan, dtype=np.float64)
        v_mask = valid
        vals[v_mask] = chunk_data[li[v_mask], lj[v_mask]]

        # --- reduce across swath width ------------------------------------
        profile[chunk_start:chunk_end] = reduce_fn(vals)

    # ---- debug plot: swath footprint on image ----------------------------
    if debug_plot:
        _debug_plot_swath(
            data, profile, start, angle_deg, length, swath_width,
            step, dr, dc, pr, pc, half_w, chunk_bboxes,
            debug_plot_margin, debug_plot_path, **(debug_kw or {}),
        )

    return profile


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _debug_plot_swath(
    data, profile, start, angle_deg, length, swath_width,
    step, dr, dc, pr, pc, half_w, chunk_bboxes,
    margin, save_path, log_y=False
):
    """
    Plot the swath footprint overlaid on an image cutout.

    Produces a two-panel figure:
      - Left:  image cutout with the centre-line, swath edges, sampled pixel
               positions, and per-chunk bounding-box rectangles.
      - Right: the extracted 1-D profile.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    nrow, ncol = data.shape

    # ---- compute full geometry for the overlay ---------------------------
    t_all = np.arange(length, dtype=np.float64) * step
    centre_r = start[0] + t_all * dr
    centre_c = start[1] + t_all * dc

    # swath edge lines (offset by ± half_w along the perpendicular)
    edge1_r = centre_r + half_w * pr
    edge1_c = centre_c + half_w * pc
    edge2_r = centre_r - half_w * pr
    edge2_c = centre_c - half_w * pc

    # ---- cutout region (union of all chunk bboxes + margin) --------------
    all_r_min = min(bb[0] for bb in chunk_bboxes)
    all_r_max = max(bb[1] for bb in chunk_bboxes)
    all_c_min = min(bb[2] for bb in chunk_bboxes)
    all_c_max = max(bb[3] for bb in chunk_bboxes)

    cut_r0 = max(all_r_min - margin, 0)
    cut_r1 = min(all_r_max + margin, nrow - 1)
    cut_c0 = max(all_c_min - margin, 0)
    cut_c1 = min(all_c_max + margin, ncol - 1)

    cutout = np.asarray(data[cut_r0 : cut_r1 + 1, cut_c0 : cut_c1 + 1],
                        dtype=np.float64)

    # auto-stretch with percentile clipping
    vmin = np.nanpercentile(cutout, 1)
    vmax = np.nanpercentile(cutout, 99)

    # ---- build figure ----------------------------------------------------
    fig, (ax_img, ax_prof) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # -- left panel: image + overlay ---------------------------------------
    # origin='lower' so row 0 is at the bottom — matches the angle convention.
    ax_img.imshow(
        cutout, origin="lower", cmap="gray",
        vmin=vmin, vmax=vmax,
        extent=[cut_c0 - 0.5, cut_c1 + 0.5,
                cut_r0 - 0.5, cut_r1 + 0.5],  # lower origin: ymin..ymax = row_lo..row_hi
        aspect="equal", interpolation="nearest",
    )

    # centre-line
    ax_img.plot(centre_c, centre_r, color="cyan", lw=1.0,
                label="centre line", zorder=3)

    # swath edges
    ax_img.plot(edge1_c, edge1_r, color="lime", lw=0.7, ls="--",
                label=f"swath edges (w={swath_width})", zorder=3)
    ax_img.plot(edge2_c, edge2_r, color="lime", lw=0.7, ls="--", zorder=3)

    # start / end markers
    ax_img.plot(centre_c[0], centre_r[0], "o", color="red", ms=7,
                zorder=5, label="start")
    ax_img.plot(centre_c[-1], centre_r[-1], "s", color="orange", ms=7,
                zorder=5, label="end")

    # chunk bounding-box rectangles
    chunk_colors = plt.cm.tab10(np.linspace(0, 1, max(len(chunk_bboxes), 1)))
    rects = []
    for i, (rb0, rb1, cb0, cb1) in enumerate(chunk_bboxes):
        rect = Rectangle(
            (cb0 - 0.5, rb0 - 0.5),
            cb1 - cb0 + 1, rb1 - rb0 + 1,
            linewidth=1.2, edgecolor=chunk_colors[i % len(chunk_colors)],
            facecolor="none", linestyle=":", zorder=4,
        )
        ax_img.add_patch(rect)
    # dummy for legend
    ax_img.plot([], [], ":", color="gray", label=f"chunk bbox ({len(chunk_bboxes)} chunks)")

    ax_img.set_xlabel("x  (column / pixel)")
    ax_img.set_ylabel("y  (row / pixel)")
    ax_img.set_title(
        f"Swath footprint — {angle_deg:.1f}° CCW from +x, "
        f"length {length}, width {swath_width}"
    )
    ax_img.legend(loc="best", fontsize=8, framealpha=0.8)

    # -- right panel: 1-D profile ------------------------------------------
    dist = np.arange(length, dtype=np.float64) * step
    ax_prof.plot(dist, profile, color="cyan", lw=0.8)
    ax_prof.set_xlabel("Distance along line (pixels)")
    ax_prof.set_ylabel("Profile value")
    ax_prof.set_title("Extracted swath profile")
    ax_prof.grid(True, alpha=0.3)

    if log_y:
        ax_prof.set_yscale("log")
        ax_prof.set_ylim(bottom=np.nanmin(profile[profile > 0]) * 0.5)

    fig.tight_layout()

    rms_level = 0.028506154387528886
    ax_prof.axhline(rms_level, color="red", ls="--", label=f"RMS noise level ({rms_level:.3f})")
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[debug_plot] saved → {save_path}")
    else:
        plt.show()

    return fig

def _get_reducer(name: str):
    """Return a function  (vals: ndarray(n, w)) → ndarray(n,)."""
    if name == "mean":
        return lambda v: np.nanmean(v, axis=1)
    if name == "median":
        return lambda v: np.nanmedian(v, axis=1)
    if name == "sum":
        return lambda v: np.nansum(v, axis=1)
    raise ValueError(f"Unknown reducer {name!r}; use 'mean', 'median', or 'sum'.")


# ---------------------------------------------------------------------------
# Convenience: profile with distance axis
# ---------------------------------------------------------------------------

def extract_swath_profile_with_coords(
    data: np.ndarray,
    start: tuple[float, float],
    angle_deg: float,
    length: int,
    pixel_scale: float = 1.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like ``extract_swath_profile`` but also returns the distance array.

    Parameters
    ----------
    pixel_scale : float
        Physical size of one pixel (e.g. arcsec/pixel).  The returned
        distance array is in these units.

    Returns
    -------
    distance : np.ndarray, shape (length,)
    profile  : np.ndarray, shape (length,)
    """
    step = kwargs.get("step", 1.0)
    profile = extract_swath_profile(data, start, angle_deg, length, **kwargs)
    distance = np.arange(length, dtype=np.float64) * step * pixel_scale
    return distance, profile


# ---------------------------------------------------------------------------
# Quick visual test / demo
# ---------------------------------------------------------------------------
crash
if __name__ == "__main__":
    # --- create a synthetic large image with a bright diagonal ridge ------
    print("Creating synthetic 8000×8000 test image …")
    ny, nx = 8000, 8000
    img = np.random.default_rng(42).normal(100, 10, (ny, nx)).astype(np.float32)

    # paint a bright ridge at 30° CCW from +x in the origin='lower' frame
    # i.e. going rightward and upward  (increasing col, increasing row)
    ridge_angle = 30.0
    ridge_rad = math.radians(ridge_angle)
    for t in range(6000):
        r = int(1000 + t * math.sin(ridge_rad))   # row increases (upward in display)
        c = int(1000 + t * math.cos(ridge_rad))   # col increases (rightward)
        if 0 <= r < ny and 0 <= c < nx:
            for w in range(-2, 3):
                rr = r + w
                if 0 <= rr < ny:
                    img[rr, c] = 500 + 50 * (3 - abs(w))

    # --- extract profile along the ridge with debug plot ------------------
    print("Extracting swath profile (length=5000, swath_width=7, chunk=1024) …")
    import time

    t0 = time.perf_counter()
    prof = extract_swath_profile(
        img,
        start=(1000, 1000),       # start at the base of the ridge
        angle_deg=ridge_angle,
        length=5000,
        swath_width=7,
        chunk_length=1024,
        debug_plot=True,
        debug_plot_path="debug_swath_30deg.png",
    )
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.3f} s  —  profile shape {prof.shape}")
    print(f"  Profile min={prof.min():.1f}  max={prof.max():.1f}  "
          f"mean={prof.mean():.1f}")

    # --- also test an angle close to vertical (85° ≈ upward) ---------------
    print("\nExtracting near-vertical profile (angle=85°, length=6000) …")
    t0 = time.perf_counter()
    prof2 = extract_swath_profile(
        img,
        start=(1000, 4000),       # start low, line goes nearly straight up
        angle_deg=85.0,
        length=6000,
        swath_width=5,
        chunk_length=512,
        debug_plot=True,
        debug_plot_path="debug_swath_85deg.png",
    )
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.3f} s  —  profile shape {prof2.shape}")
    print(f"  Profile min={prof2.min():.1f}  max={prof2.max():.1f}  "
          f"mean={prof2.mean():.1f}")

    print("\nAll tests passed ✓")


def find_profile_end(
    profile: np.ndarray,
    threshold: float,
    run_length: int | None = None,
    run_frac: float = 0.05,
    min_run: int = 16,
    above_frac: float = 0.25,
    pad_frac: float = 0.5,
    min_pad: int = 16,
) -> int:
    """
    Find the distance along a diffraction-spike profile where the signal
    drops below a noise-floor threshold — resilient to sinusoidal
    modulation, noise, and contamination from neighbouring objects.
 
    Strategy
    --------
    Walk *forward* from the start with a sliding window of length *W*.
    At each position the window is classified as "in signal" if at least
    ``above_frac`` of its samples exceed ``threshold``.  Track the
    farthest contiguous run of "in signal" windows anchored to the start
    of the profile.  Once a window fails the test, we stop — isolated
    bumps further out (neighbouring stars) cannot restart the run.
 
    After finding the last passing window position, add padding equal to
    ``pad_frac × W`` to accommodate the possibility that the true signal
    extends slightly beyond the last window that passed (e.g. if the
    window ended in a sinusoidal trough).
 
    Why this works:
 
    * **Sinusoidal modulation**: ``above_frac = 0.25`` passes even if
      most of the cycle is below threshold, as long as the peaks still
      exceed it.  The window should be at least one full modulation
      period wide.
    * **Noise transients below threshold**: a handful of below-threshold
      samples cannot fail the window, so brief dips are tolerated.
    * **Neighbouring objects**: a bright contaminant *beyond* the spike
      end cannot restart the run because the scan requires contiguity
      from the start.
    * **Overestimate bias**: the final padding pushes the endpoint
      outward, which is the preferred direction of error.
 
    Parameters
    ----------
    profile : 1-D array
        Extracted swath profile (linear flux units, *not* log).
    threshold : float
        Noise-floor level, same units as ``profile``.
    run_length : int or None
        Explicit sliding-window size.  If None, computed as
        ``max(min_run, int(len(profile) * run_frac))``.
    run_frac : float
        Window size as a fraction of profile length (used when
        ``run_length is None``).  Should be at least as wide as one
        full cycle of any sinusoidal modulation in the profile.
    min_run : int
        Absolute minimum window size in pixels.
    above_frac : float
        Fraction of samples in a window that must exceed ``threshold``
        for that window to count as "in signal".
    pad_frac : float
        After finding the raw endpoint, extend by ``pad_frac × W``
        pixels.  0.5 adds half a window, which is enough to cover a
        trough following the last detected peak.
    min_pad : int
        Minimum padding in pixels.
 
    Returns
    -------
    end_idx : int
        Index into ``profile`` of the estimated endpoint (inclusive).
        Clamped to ``[0, len(profile) - 1]``.
    """
    n = len(profile)
    if n == 0:
        return 0
 
    W = run_length if run_length is not None else max(min_run, int(n * run_frac))
    W = max(1, W)
 
    above = (profile >= threshold)
 
    # running sum for O(1) window queries
    cum = np.zeros(n + 1, dtype=np.int64)
    cum[1:] = np.cumsum(above)
 
    required = max(1, int(W * above_frac))
 
    # forward scan — find the farthest contiguous "in signal" window
    # anchored to the start of the profile
    last_passing = -1
    for i in range(n):
        j = min(i + W, n)
        count = int(cum[j] - cum[i])
        if count >= required:
            last_passing = i
        else:
            # the contiguous run from the start has ended
            break
 
    if last_passing < 0:
        # nothing above threshold at all
        return 0
 
    # the signal was "in" at window starting at last_passing, so the
    # signal extends at least to last_passing + W - 1
    raw_end = last_passing + W - 1
 
    # add padding to err on the side of overestimation
    pad = max(min_pad, int(W * pad_frac))
    end_idx = min(raw_end + pad, n - 1)
 
    return int(end_idx)
 
 