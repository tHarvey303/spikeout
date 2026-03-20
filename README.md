# spikeout

Detect, measure, and mask diffraction spikes in astronomical images.

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

Catalogue mode, FITS I/O, and WCS support require astropy:

```bash
pip install ".[astropy]"
```

For robust background estimation (recommended), install `sep`:

```bash
pip install sep
```

Gaia star fetching requires astroquery:

```bash
pip install astroquery
```

## Example

Example of spikeout results on a real Euclid Q1 NISP image, showing detected
spikes and the Radon sinogram.  The diagnostic plot shows the preprocessed
image, sinogram restricted to the central ρ band, and angular profile with
detected peaks.

<img src="examples/example.png" alt="Example of spikeout results" width="800"/>


## Quick start

```python
import spikeout

# Detect spikes
result = spikeout.detect(cutout_data)

# Tune for your data
result = spikeout.detect(
    cutout_data,
    min_peak_separation_deg=20.0,   # minimum angle between spikes
    morph_radius=0,                 # skip morphological filtering
    peak_prominence=0.8,            # aggressive initial peak selection
    min_snr=5.0,                    # reject insignificant spikes
    max_rho_fraction=0.1,           # lines must pass near centre
)

# Access results
print(result.angles)        # image-plane angles (degrees)
print(result.snr)           # SNR of each detected spike
print(result.rho_physical)  # perpendicular offset from centre (pixels)

# With length measurement
result = spikeout.detect(cutout_data, measure_lengths=True)
for spike in result.lengths:
    print(f"{spike.angle_deg:.1f}°: {spike.length_total:.0f} px total")
    print(f"  pos arm: {spike.length_pos:.0f} px, neg arm: {spike.length_neg:.0f} px")
    if not spike.converged_pos or not spike.converged_neg:
        print("  (arm reached cutout boundary)")

# Diagnostic plot
fig = spikeout.plot_diagnostics(cutout_data, result)
```

## Features

### Spike detection

`detect()` returns a `SpikeResult` with:

| Attribute | Description |
|---|---|
| `angles` | Display-frame spike angles (degrees) |
| `snr` | Angular-profile SNR of each spike |
| `rho_physical` | Perpendicular offset from image centre (pixels) |
| `lengths` | Per-arm length measurements (when `measure_lengths=True`) |
| `sinogram` | Full Radon sinogram |
| `prepared_image` | Preprocessed image fed to the Radon transform |

### Length measurement

Spike arm lengths are measured from swath profiles: a narrow band of pixels
perpendicular to the spike direction is sampled at each step along the arm,
and the median across the band is taken.  The profile is smoothed with a
median filter before endpoint detection.

```python
from spikeout import measure_spike_lengths

lengths = measure_spike_lengths(
    image, result,
    swath_width=7,        # perpendicular sampling width (pixels)
    length_sigma=2.0,     # threshold: bg + 2 × σ_sky
)

for sl in lengths:
    print(f"{sl.angle_deg:.1f}°  total={sl.length_total:.0f} px  "
          f"(+{sl.length_pos:.0f} / -{sl.length_neg:.0f})")
```

For large images where a spike may extend beyond the cutout, pass the
full memory-mapped array so arm extraction continues past the cutout boundary:

```python
import astropy.fits as fits

with fits.open("image.fits", memmap=True) as hdul:
    lengths = measure_spike_lengths(
        cutout, result,
        full_array=hdul[0].data,
        centre_row_full=star_row_full,
        centre_col_full=star_col_full,
    )
```

### Spike masking and DS9 regions

```python
# Boolean pixel mask of spike-contaminated pixels
mask = spikeout.spike_mask(result, image.shape)

# Write DS9 region file (pixel coordinates)
spikeout.write_ds9_regions("spikes.reg", result, image.shape)
```

### Stellar halo mask

`halo_mask()` finds the radius where the stellar halo falls to
`threshold_nsigma × σ_MAD` above the background, using a robust azimuthal
median profile (resilient to neighbours and diffraction spikes).

```python
mask, radius_px = spikeout.halo_mask(
    cutout_data,
    threshold_nsigma=3.0,
    min_radius=5.0,
)
```

`CatalogueEntry.halo_mask` / `CatalogueEntry.halo_radius` are populated
automatically when `halo_mask_kw` is passed to `catalogue_detect()`.

### Full-frame FITS mask output

For large survey images, spike boxes and halo circles can be rasterised
directly into a FITS file using tile-based processing so peak RAM usage stays
proportional to the tile size rather than the full image.

```python
from spikeout.regions import write_spike_mask_fits, write_border_mask_fits

# Spike boxes + halo circles → uint8 FITS mask
write_spike_mask_fits(
    entries,
    image_path="image.fits",
    output_path="spike_mask.fits",
    width_fraction=0.1,
    tile_size=4096,
    n_workers=8,
)

# Pixels within N px of the image border → uint8 FITS mask
write_border_mask_fits(
    image_path="image.fits",
    output_path="border_mask.fits",
    edge_distance_px=1200,   # e.g. 120 arcsec at 0.1 arcsec/px
    tile_size=4096,
    n_workers=8,
)
```

### Catalogue mode

Process many sources in a FITS image in one call.  Cutouts are extracted with
memory-mapping so only the required pixels are read.

```python
from astropy.coordinates import SkyCoord
from spikeout.catalogue import catalogue_detect, catalogue_halo, combine_entries

coords = SkyCoord([ra1, ra2, ...], [dec1, dec2, ...], unit="deg")

# Detect spikes and measure halos for bright stars
spike_entries = catalogue_detect(
    coords,
    "image.fits",
    cutout_size=256,
    measure_lengths=True,       # run swath-profile length measurement
    full_array=image_data,      # memmap array for unconstrained arm extraction
    n_jobs=-1,                  # use all CPU threads
    halo_mask_kw={},            # also run halo_mask with defaults
    min_snr=5.0,
)

# Halo masking only for a second list of sources (no spike detection)
halo_entries = catalogue_halo(
    faint_coords,
    "image.fits",
    halo_mask_kw={"threshold_nsigma": 2.0},
)

# Combine and write a single full-frame mask
all_entries = combine_entries(spike_entries, halo_entries)
write_spike_mask_fits(all_entries, "image.fits", "mask.fits")

summary = spikeout.catalogue_summary(spike_entries)
print(f"{summary['with_spikes']} / {summary['total']} sources have spikes")
print(f"median halo radius: {np.median(summary['halo_radii']):.1f} px")

fig = spikeout.plot_catalogue(spike_entries)

# Write sky-coordinate DS9 regions for all sources
spikeout.write_catalogue_ds9_regions(
    "catalogue_spikes.reg", spike_entries, pixel_scale_arcsec=0.1
)
```

### Gaia star fetching

`fetch_gaia_stars` queries Gaia DR3 over the footprint of a FITS image and
applies proper-motion propagation to a target epoch.  The returned table
includes epoch-corrected sky coordinates and pixel positions.

```python
from spikeout.stars import fetch_gaia_stars

stars = fetch_gaia_stars(
    "image.fits",
    epoch="2025-01-01",
    scale_extra=0.1,        # pad footprint by 10 % per side
    mag_limit=20.0,         # only return G < 20
    output_path="gaia_stars.fits",
)

# Feed directly into catalogue_halo / write_spike_mask_fits
from astropy.coordinates import SkyCoord
coords = SkyCoord(
    ra=stars["ra_epoch"], dec=stars["dec_epoch"], unit="deg"
)
```

## How it works

### Preprocessing

1. **Azimuthal median subtraction** — removes the radially symmetric PSF
   (core + halo), isolating asymmetric structure like diffraction spikes.
2. **Sigma clipping** — zeros pixels below `sigma_clip × σ_MAD`, suppressing
   the noise floor.
3. **Morphological opening** — erodes compact sources (neighbours, hot pixels,
   cosmic rays) while preserving elongated spikes.
4. **Asinh scaling** — compresses dynamic range softly.  The stretch scale is
   computed from the inner 50 % of the image radius so bright off-centre
   sources do not compress the central spike signal.

### Detection

The **Radon transform** projects the preprocessed image along many angles.
Diffraction spikes produce bright peaks in the sinogram at their corresponding
(ρ, θ) coordinates.

Peak detection is restricted to the **central ρ band** (`|ρ| ≤ max_rho_px`)
so that off-centre bright sources never influence the threshold, peak
positions, ρ assignments, or SNR estimates.  For saturated stars with a
NaN/zero core the band is automatically widened by the blank-core radius so
that spike arms — which peak at the edge of the saturated region — are still
detected.

Two quality filters reject false positives:

- **ρ filter** — lines must pass within `max_rho_fraction × min(shape)/2`
  pixels of the star centre.  Automatically extended for saturated cores.
- **SNR filter** — each peak's height in the angular profile must exceed
  `min_snr × σ_MAD` above the median.

### Length measurement

Each spike arm is measured by extracting a **swath profile**: at each step
along the arm, pixels within `swath_width / 2` of the spike axis are sampled
and the median (or mean) is taken, giving a 1-D flux profile in per-pixel
units.  The profile is smoothed with a 1-D median filter before endpoint
detection.

The **endpoint detector** uses a sliding window: the window advances while a
sufficient fraction (`above_frac`) of samples exceed the noise threshold; the
first failing window terminates the arm.  A padding step prevents sinusoidal
fluctuations (from PSF structure) from cutting the arm too early.

The **noise threshold** is `bg_level + length_sigma × σ_sky`.  When `sep` is
installed, background and RMS are estimated from a `sep.Background` mesh fit
with the stellar halo and neighbouring sources masked via a segmentation map,
giving a robust per-pixel noise floor.  Without `sep`, pixel-to-pixel MAD
differences in an outer annulus are used as a fallback.

**Two-stage extraction**: the arm is first measured within the cutout.  If it
reaches the cutout boundary and a `full_array` is provided, extraction
continues in the full memory-mapped image, so reported lengths are never
clipped at the cutout edge.

### Halo masking

`halo_mask()` builds a robust azimuthal median radial profile (1-pixel-wide
annuli, median per bin), smooths it with a 1-D median filter, and finds the
outermost radius where the profile exceeds `bg + threshold_nsigma × σ_MAD`.
The azimuthal median is resilient to any single source occupying less than
50 % of an annulus.  When `sep` is available, the background and RMS used for
the threshold are estimated with the same sep-based estimator as the length
measurement module.

### Full-frame mask writing

`write_spike_mask_fits` and `write_border_mask_fits` both use a tiled
processing strategy: the output FITS is pre-allocated as a zeroed
memory-mapped file, then filled tile by tile in parallel (via
`ThreadPoolExecutor`).  Tiles never overlap so workers write to disjoint
memory regions without locks.

For the **border mask**, each tile is padded by `edge_distance_px` before
running `distance_transform_edt` so EDT results are accurate at tile
boundaries.  A global `binary_fill_holes` pass ensures internal NaN/zero
regions (dead pixels, gaps) are not mistaken for image borders.

## API reference

### Core

| Function / class | Description |
|---|---|
| `detect(image, ...)` | Detect spikes via Radon transform |
| `SpikeResult` | Detection output dataclass |
| `measure_spike_lengths(image, result, ...)` | Measure per-arm spike lengths via swath profiles |
| `SpikeLengths` | Length measurement dataclass |

### Preprocessing

| Function | Description |
|---|---|
| `prepare_image(image, ...)` | Full preprocessing pipeline |
| `azimuthal_median(image, ...)` | Azimuthal median PSF model |
| `find_centre(image)` | Auto-detect source centre |

### Masking and regions

| Function | Description |
|---|---|
| `halo_mask(image, ...)` | Circular stellar halo mask from radial profile |
| `spike_mask(result, shape, ...)` | Boolean pixel mask of spike arms |
| `spike_box_regions(result, shape, ...)` | DS9 box region strings |
| `write_ds9_regions(path, result, ...)` | Write DS9 region file (pixel coords) |
| `write_catalogue_ds9_regions(path, entries, ...)` | Write DS9 region file (sky coords) |
| `write_spike_mask_fits(entries, image, output, ...)` | Tiled full-frame spike + halo FITS mask |
| `write_border_mask_fits(image, output, edge_px, ...)` | Tiled full-frame border proximity FITS mask |

### Catalogue mode

| Function / class | Description |
|---|---|
| `catalogue_detect(coords, path, ...)` | Batch spike detection on a FITS image |
| `catalogue_halo(coords, path, ...)` | Batch halo measurement only (no spike detection) |
| `combine_entries(*lists)` | Concatenate entry lists for joint mask writing |
| `CatalogueEntry` | Per-source result dataclass |
| `catalogue_summary(entries)` | Aggregate statistics across entries |
| `plot_catalogue(entries, ...)` | Grid plot of cutouts with spike overlays |

### Star catalogues

| Function | Description |
|---|---|
| `fetch_gaia_stars(fits_path, epoch, ...)` | Query Gaia DR3 over a FITS footprint with proper-motion correction |

### Plotting and geometry

| Function | Description |
|---|---|
| `plot_diagnostics(image, result, ...)` | Multi-panel diagnostic figure |
| `radon_line_to_image(rho, theta, shape)` | (ρ, θ) → pixel endpoints |
| `sinogram_rho_to_physical(indices, n_rho)` | Sinogram row index → physical ρ |
| `mad_std(data)` | Robust σ via median absolute deviation |
