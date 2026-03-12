"""Tests for spikeout.plotting."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from spikeout.detect import detect
from spikeout.plotting import plot_diagnostics


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestPlotDiagnostics:
    """Tests for the diagnostic plot function."""

    def test_returns_figure(self, star_centred):
        """Should return a matplotlib Figure."""
        fig = plot_diagnostics(star_centred)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_precomputed_result(self, star_centred):
        """Should accept a precomputed SpikeResult."""
        result = detect(star_centred, min_snr=3.0)
        fig = plot_diagnostics(star_centred, result)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_lengths(self, star_centred):
        """Should produce a 2-row figure when lengths are measured."""
        result = detect(star_centred, measure_lengths=True, min_snr=3.0)
        fig = plot_diagnostics(star_centred, result)
        assert isinstance(fig, matplotlib.figure.Figure)
        # Should have 2 rows of axes
        axes = fig.get_axes()
        assert len(axes) > 4  # more than just the top row

    def test_no_spikes_no_crash(self, pure_noise):
        """Should handle zero detections gracefully."""
        result = detect(pure_noise, min_snr=5.0)
        assert len(result.angles) == 0
        fig = plot_diagnostics(pure_noise, result)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_detect_kw_forwarded(self, star_centred):
        """Extra kwargs should be forwarded to detect()."""
        fig = plot_diagnostics(
            star_centred, min_snr=3.0, morph_radius=0,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_four_spikes_plot(self, star_four_spikes):
        """Should handle more than 2 spikes."""
        result = detect(
            star_four_spikes, min_snr=2.0, min_peak_separation_deg=15.0,
        )
        fig = plot_diagnostics(star_four_spikes, result)
        assert isinstance(fig, matplotlib.figure.Figure)
