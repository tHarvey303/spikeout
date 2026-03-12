"""Tests for spikeout.stats."""

import numpy as np
import pytest
from spikeout.stats import mad_std


class TestMadStd:
    """Tests for the mad_std robust standard deviation estimator."""

    def test_gaussian_recovery(self, rng):
        """mad_std of Gaussian data should approximate the true σ."""
        data = rng.normal(0, 3.0, 100_000)
        assert mad_std(data) == pytest.approx(3.0, rel=0.05)

    def test_zero_for_constant(self):
        """Constant array → σ = 0."""
        assert mad_std(np.ones(100)) == 0.0

    def test_ignores_nan(self):
        """NaNs should be silently ignored."""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        result = mad_std(data, ignore_nan=True)
        assert np.isfinite(result)
        assert result > 0

    def test_empty_after_nan_filter(self):
        """All-NaN → 0."""
        assert mad_std(np.full(10, np.nan)) == 0.0

    def test_outlier_robustness(self, rng):
        """A few extreme outliers should not blow up the estimate."""
        data = rng.normal(0, 1.0, 10_000)
        data[:5] = 1e6  # extreme outliers
        assert mad_std(data) == pytest.approx(1.0, rel=0.1)
