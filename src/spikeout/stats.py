"""Robust statistics utilities.

Provides `mad_std` with an optional astropy fallback so the package works
in environments where astropy is not installed.
"""

import numpy as np

__all__ = ["mad_std"]

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
