"""spikeout — detect, measure, and mask diffraction spikes."""

from .detect import detect, SpikeResult
from .lengths import measure_spike_lengths, SpikeLengths
from .preprocess import prepare_image, azimuthal_median, find_centre
from .geometry import radon_line_to_image, sinogram_rho_to_physical
from .plotting import plot_diagnostics
from .stats import mad_std

__all__ = [
    # core
    "detect",
    "SpikeResult",
    # lengths
    "measure_spike_lengths",
    "SpikeLengths",
    # preprocessing
    "prepare_image",
    "azimuthal_median",
    "find_centre",
    # geometry
    "radon_line_to_image",
    "sinogram_rho_to_physical",
    # plotting
    "plot_diagnostics",
]
