# spikeout

Detect, measure, and mask diffraction spikes in astronomical images.

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Example
Here is an example of spikeout results on a real Euclid Q1 NISP image, showing the detected spikes and sinogram. The diagnostic plot shows the preprocessed image, the Radon sinogram, and the angular profile with detected peaks.

<img src="examples/example.png"  alt="Example of spikeout results" width="800"/>


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

# Diagnostic plot
fig = spikeout.plot_diagnostics(cutout_data, result)
```

## How it works

### Preprocessing

1. **Azimuthal median subtraction** removes the radially symmetric PSF
   (core + halo), isolating asymmetric structure.
2. **Morphological opening** erodes compact sources (neighbours, hot pixels etc)
   while preserving elongated spikes.
3. **Asinh scaling** compresses dynamic range softly — the Radon transform 
   gets weighted gradient information.

### Detection

The **Radon transform** projects the preprocessed image along many angles.
Diffraction spikes produce bright peaks in the sinogram at their
corresponding (ρ, θ) coordinates.  Peaks are detected in the full 2-D
sinogram so results are independent of source centring.

Two quality filters reject false positives:

- **ρ filter** — lines must pass within `max_rho_fraction` of the image
  centre (since cutouts are centred on the target star).
- **SNR filter** — each peak's height in the angular profile must exceed
  `min_snr × σ_MAD` above the median.  This naturally rejects featureless
  images (galaxies, faint stars without visible spikes).

### Length measurement

Each spike arm is traced outward from the line's closest point to the
image centre.  A perpendicular swath is sampled at each step (median over
the band width) to build a 1-D radial profile.  The spike endpoint is
where this profile drops below `length_sigma × σ_background` for a
sustained run of consecutive pixels.
