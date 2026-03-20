"""spikeout — detect, measure, and mask diffraction spikes."""

from .detect import detect, SpikeResult
from .lengths import measure_spike_lengths, SpikeLengths
from .preprocess import prepare_image, azimuthal_median, find_centre
from .geometry import radon_line_to_image, sinogram_rho_to_physical
from .plotting import plot_diagnostics
from .catalogue import catalogue_detect, catalogue_summary, plot_catalogue, CatalogueEntry, catalogue_halo
from .regions import spike_box_regions, spike_mask, write_ds9_regions, write_catalogue_ds9_regions, halo_mask
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
    # catalogue
    "catalogue_detect",
    "catalogue_halo",
    "catalogue_summary",
    "plot_catalogue",
    "CatalogueEntry",
    # regions
    "spike_box_regions",
    "spike_mask",
    "write_ds9_regions",
    "write_catalogue_ds9_regions",
    "halo_mask",
]
