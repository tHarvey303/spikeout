"""Diagnostic plots for spike detection and length measurement."""

import numpy as np
import matplotlib.pyplot as plt
from .stats import mad_std

from .detect import detect
from .geometry import radon_line_to_image
from .preprocess import azimuthal_median

__all__ = ["plot_diagnostics"]


def plot_diagnostics(image, result=None, max_rho_fraction=0.1, **detect_kw):
    """Multi-panel diagnostic figure.

    Layout
    ------
    Row 1: preprocessed image | sinogram | angular profile | overlaid image
    Row 2 (if lengths measured): one panel per spike showing arm profiles

    Parameters
    ----------
    image : 2-D array
    result : `~spikeout.detect.SpikeResult` or *None*
        If *None*, detection is run with *detect_kw*.
    max_rho_fraction : float
        Passed through if detection is run.
    **detect_kw
        Forwarded to `~spikeout.detect.detect`.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
    """
    if result is None:
        result = detect(
            image, max_rho_fraction=max_rho_fraction, **detect_kw,
        )

    image = np.asarray(image, dtype=float)
    sinogram = result.sinogram
    theta = result.theta
    n_rho = sinogram.shape[0]
    pk_rho = result.peak_rho_indices
    pk_th = result.peak_theta_indices
    centre_row = n_rho // 2
    max_rho_px = max_rho_fraction * min(image.shape) / 2.0

    has_lengths = result.lengths is not None and len(result.lengths) > 0
    n_spikes = len(result.angles)

    if has_lengths:
        nrows_fig = 2
        ncols_fig = max(4, n_spikes)
    else:
        nrows_fig = 1
        ncols_fig = 4

    fig, axes = plt.subplots(
        nrows_fig, ncols_fig,
        figsize=(5.5 * ncols_fig, 5 * nrows_fig),
        constrained_layout=True,
    )
    if nrows_fig == 1:
        axes = axes[np.newaxis, :]

    # ── Row 1 ────────────────────────────────────────────────────────────

    # panel 1: preprocessed image
    ax = axes[0, 0]
    ax.imshow(result.prepared_image, origin="lower", cmap="gray")
    ax.set(title="Preprocessed (Radon input)")

    # panel 2: sinogram + ρ band + SNR annotations
    ax = axes[0, 1]
    ax.imshow(
        sinogram, extent=(0, 180, 0, n_rho), aspect="auto", cmap="inferno",
    )
    band_lo = n_rho - (centre_row + max_rho_px)
    band_hi = n_rho - (centre_row - max_rho_px)
    ax.axhspan(
        band_lo, band_hi, color="cyan", alpha=0.12,
        label=f"|ρ| ≤ {max_rho_px:.0f} px",
    )
    if len(pk_th) > 0:
        ax.scatter(
            theta[pk_th], n_rho - pk_rho,
            facecolors="none", edgecolors="cyan", s=140, linewidths=1.5,
            zorder=5, label=f"{len(pk_th)} accepted",
        )
    for ti in pk_th:
        ax.axvline(theta[ti], color="cyan", ls="--", lw=0.8, alpha=0.5)

    parts = ["Radon sinogram"]
    if result.n_rejected_snr > 0:
        parts.append(f"{result.n_rejected_snr} SNR-rej")
    ax.set(
        xlabel="Projection angle (°)", ylabel="ρ  (pixels)",
        title="  |  ".join(parts),
    )
    ax.legend(loc="upper right", fontsize=7)

    # panel 3: angular profile with SNR labels
    ax = axes[0, 2]
    max_profile = np.max(sinogram, axis=0)
    ax.plot(theta, max_profile, "k-", lw=1)
    if len(pk_th) > 0:
        for j, ti in enumerate(pk_th):
            ax.axvline(theta[ti], color="cyan", ls="--", lw=1, alpha=0.7)
            ax.annotate(
                f"SNR={result.snr[j]:.1f}",
                xy=(theta[ti], max_profile[ti]),
                xytext=(5, 8), textcoords="offset points",
                fontsize=7, color="cyan",
            )
        ax.scatter(
            theta[pk_th], max_profile[pk_th], color="cyan", s=60, zorder=5,
        )
    ax.set(
        xlabel="Projection angle (°)", ylabel="Peak intensity along ρ",
        title="Angular profile (max over ρ)",
    )

    # panel 4: original image with spike lines
    ax = axes[0, 3]
    finite = image[np.isfinite(image)]
    vmin, vmax = np.percentile(finite, [1, 99.5])
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    colors = plt.cm.Set1(np.linspace(0, 1, max(n_spikes, 1)))

    for i in range(n_spikes):
        rho = result.rho_physical[i]
        th = theta[pk_th[i]]
        (x1, y1), (x2, y2), angle = radon_line_to_image(
            rho, th, image.shape,
        )
        color = colors[i % len(colors)]

        label = f"{angle:.1f}° SNR={result.snr[i]:.1f}"
        if has_lengths:
            label += f" ({result.lengths[i].length_total:.0f} px)"

        ax.plot([x1, x2], [y1, y2], ls="--", lw=1.5, color=color,
                label=label)

        if has_lengths:
            sl = result.lengths[i]
            th_rad = np.deg2rad(th)
            x0 = image.shape[1] / 2.0 + rho * np.cos(th_rad)
            y0 = image.shape[0] / 2.0 - rho * np.sin(th_rad)
            a_rad = np.deg2rad(angle)
            for arm_len, sign in [(sl.length_pos, 1), (sl.length_neg, -1)]:
                ex = x0 + sign * arm_len * np.cos(a_rad)
                ey = y0 + sign * arm_len * np.sin(a_rad)
                ax.plot(ex, ey, "o", color=color, ms=6, mew=1.5, mfc="none")

    ax.set(
        xlim=(0, image.shape[1]), ylim=(0, image.shape[0]),
        title="Detected spikes",
    )
    ax.legend(loc="upper right", fontsize=7)

    for j in range(4, ncols_fig):
        axes[0, j].axis("off")

    # ── Row 2: swath profiles ────────────────────────────────────────────
    if has_lengths:
        sigma_bg = mad_std(image - azimuthal_median(image))

        for i in range(ncols_fig):
            ax = axes[1, i]
            if i >= n_spikes:
                ax.axis("off")
                continue

            sl = result.lengths[i]
            color = colors[i % len(colors)]

            ax.plot(sl.radii_pos, sl.profile_pos, "-", color=color,
                    lw=1.2, label=f"+ arm ({sl.length_pos:.0f} px)")
            ax.plot(sl.radii_neg, sl.profile_neg, "--", color=color,
                    lw=1.2, label=f"− arm ({sl.length_neg:.0f} px)")

            ax.axhline(2.0 * sigma_bg, color="grey", ls=":", lw=1,
                       label="2σ background")

            ax.axvline(sl.length_pos, color=color, ls="-", lw=0.8, alpha=0.5)
            ax.axvline(sl.length_neg, color=color, ls="--", lw=0.8, alpha=0.5)

            ax.set(
                xlabel="Radius (pixels)", ylabel="Swath median flux",
                title=f"Spike @ {sl.angle_deg:.1f}°",
            )
            ax.legend(fontsize=7)

    return fig
