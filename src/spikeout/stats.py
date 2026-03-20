"""Robust statistics utilities.

Provides `mad_std` with an optional astropy fallback so the package works
in environments where astropy is not installed.
"""

import numpy as np

__all__ = ["mad_std", "estimate_background"]

def _mad_std_fallback(data, ignore_nan=True):
    """Median absolute deviation scaled to match Gaussian σ."""
    data = np.asarray(data).ravel()
    if ignore_nan:
        data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0
    median = np.median(data)
    return 1.4826 * np.median(np.abs(data - median))


try:
    from astropy.stats import mad_std as _astropy_mad_std

    def mad_std(data, ignore_nan=True):
        """Robust σ via MAD (wraps astropy, safe for all-NaN input)."""
        result = _astropy_mad_std(data, ignore_nan=ignore_nan)
        if not np.isfinite(result):
            return 0.0
        return float(result)

except ImportError:
    mad_std = _mad_std_fallback


# Module-level cached dilation structuring element (10-px disk).
# Built once at import time rather than rebuilt for every source.
_DILATE_RADIUS = 10
_dy, _dx = np.ogrid[-_DILATE_RADIUS:_DILATE_RADIUS + 1,
                    -_DILATE_RADIUS:_DILATE_RADIUS + 1]
_DILATE_STRUCT = (_dx ** 2 + _dy ** 2 <= _DILATE_RADIUS ** 2)


def estimate_background(img, cy, cx, inner_r, _R=None):
    """Estimate background level and per-pixel RMS, masking the inner region
    (stellar halo) and neighbouring sources.

    Uses ``sep`` (Source Extractor Python) when available, with a
    segmentation-based neighbour mask.  Falls back to pixel-to-pixel
    MAD differences when ``sep`` is not installed.

    Parameters
    ----------
    img : 2-D array
        Image to analyse (NaN-safe).
    cy, cx : float
        Centre of the star.
    inner_r : float
        Inner exclusion radius in pixels — all pixels at r < inner_r
        are treated as halo and excluded from noise estimation.
    _R : 2-D array or *None*
        Pre-computed distance array ``sqrt((X-cx)²+(Y-cy)²)``.  Computed
        internally when *None*.

    Returns
    -------
    bg_level : float
    sigma_bg : float
    """
    ny, nx = img.shape
    if _R is None:
        Y_g, X_g = np.ogrid[:ny, :nx]
        R = np.sqrt((X_g - cx) ** 2 + (Y_g - cy) ** 2)
    else:
        R = _R

    try:
        import sep
        from scipy.ndimage import binary_dilation

        work = np.ascontiguousarray(img, dtype=np.float64)
        nan_mask = ~np.isfinite(work)
        work[nan_mask] = 0.0

        halo_mask = (R < inner_r).astype(np.uint8)
        halo_mask[nan_mask] = 1

        try:
            bkg = sep.Background(work, mask=halo_mask, bw=64, bh=64,
                                 fw=3, fh=3)
        except Exception:
            bkg = sep.Background(work, mask=halo_mask)

        bkg_img = bkg.back()
        rms_img = bkg.rms()

        residual = work - bkg_img
        residual[halo_mask.astype(bool)] = 0.0

        try:
            _, seg = sep.extract(residual, thresh=3.0, err=rms_img,
                                 segmentation_map=True)
            source_mask = seg > 0
        except Exception:
            source_mask = np.zeros(img.shape, dtype=bool)

        full_mask = halo_mask.astype(bool) | source_mask | nan_mask
        full_mask = binary_dilation(full_mask, structure=_DILATE_STRUCT)
        outer = R >= inner_r
        good = outer & ~full_mask

        if good.sum() >= 20:
            bg_level = float(np.median(bkg_img[good]))
            sigma_bg = float(np.median(rms_img[good]))
        else:
            bg_level = float(bkg.globalback)
            sigma_bg = float(bkg.globalrms)

        return bg_level, max(sigma_bg, 1e-10)

    except ImportError:
        pass

    # ── Fallback: pixel-to-pixel difference estimator ─────────────────────
    dx_diff = np.diff(img, axis=1)
    dy_diff = np.diff(img, axis=0)
    noise_dx = dx_diff[R[:, :-1] >= inner_r]
    noise_dy = dy_diff[R[:-1, :] >= inner_r]
    noise_all = np.concatenate([noise_dx, noise_dy])
    noise_all = noise_all[np.isfinite(noise_all)]

    if noise_all.size >= 20:
        sigma_bg = float(mad_std(noise_all)) / np.sqrt(2)
    else:
        sigma_bg = float(mad_std(img[np.isfinite(img)]))

    outer_px = img[R >= inner_r]
    outer_px = outer_px[np.isfinite(outer_px)]
    bg_level = float(np.median(outer_px)) if outer_px.size >= 10 \
        else float(np.median(img[np.isfinite(img)]))

    return bg_level, max(sigma_bg, 1e-10)
