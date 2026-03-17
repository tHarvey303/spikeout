"""Diagnostic plots for spike detection and length measurement."""

import numpy as np
import matplotlib.pyplot as plt
from .stats import mad_std

from .detect import detect
from .geometry import radon_line_to_image, sinogram_rho_to_physical
from .preprocess import azimuthal_median
from .lengths import _fraunhofer_model, _envelope_model, _log_sinc2_model

__all__ = ["plot_diagnostics"]


def plot_diagnostics(image, result=None, max_rho_fraction=0.1, show_swath=False, **detect_kw):
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
    show_swath : bool
        If *True* and lengths were measured, overlay the swath extraction
        regions on the image panel as semi-transparent rectangles.
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
    # Use the actual acceptance band stored in the result (includes blank_r
    # for saturated stars) rather than recomputing from max_rho_fraction alone.
    max_rho_px = result.max_rho_px if result.max_rho_px > 0 \
        else max_rho_fraction * min(image.shape) / 2.0

    # Reconstruct the ρ-restricted sinogram for the angular profile panel.
    all_rho_phys = sinogram_rho_to_physical(np.arange(n_rho), n_rho)
    rho_central = np.abs(all_rho_phys) <= max_rho_px
    sinogram_central = sinogram * rho_central[:, np.newaxis]

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
    # Use sinogram_central (|ρ| ≤ max_rho_px) to match what the detector
    # actually used — off-centre sources are excluded from this profile.
    ax = axes[0, 2]
    max_profile = np.max(sinogram_central, axis=0)
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
        title=f"Angular profile (|ρ| ≤ {max_rho_px:.0f} px)",
    )

    # panel 4: original image with spike lines
    ax = axes[0, 3]
    finite = image[np.isfinite(image)]
    vmin, vmax = np.percentile(finite, [1, 99.5])
    from matplotlib.colors import Normalize, LogNorm
    if vmin <= 0:
        vmin = vmax/1e3
    norm = LogNorm(vmin=vmin, vmax=vmax)
    ax.imshow(image, origin="lower", cmap="gray", norm=norm)

    colors = plt.cm.Set1(np.linspace(0, 1, max(n_spikes, 1)))

    for i in range(n_spikes):
        rho = result.rho_physical[i]
        th = theta[pk_th[i]]
        (x1, y1), (x2, y2), angle = radon_line_to_image(
            rho, th, image.shape,
        )
        color = colors[i % len(colors)]
        color = 'red'

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

            if show_swath and sl.swath_width > 0:
                from matplotlib.patches import Polygon as MplPolygon
                # Perpendicular unit vector (rotated 90° from along direction)
                px = -np.sin(a_rad)
                py = np.cos(a_rad)
                half_w = sl.swath_width / 2.0
                # Rectangle corners: from start of pos arm to end, and neg arm
                r_start_pos = float(sl.radii_pos[0]) if len(sl.radii_pos) else 0.0
                r_start_neg = float(sl.radii_neg[0]) if len(sl.radii_neg) else 0.0
                far_pos_x = x0 + sl.length_pos * np.cos(a_rad)
                far_pos_y = y0 + sl.length_pos * np.sin(a_rad)
                far_neg_x = x0 - sl.length_neg * np.cos(a_rad)
                far_neg_y = y0 - sl.length_neg * np.sin(a_rad)
                near_pos_x = x0 + r_start_pos * np.cos(a_rad)
                near_pos_y = y0 + r_start_pos * np.sin(a_rad)
                near_neg_x = x0 - r_start_neg * np.cos(a_rad)
                near_neg_y = y0 - r_start_neg * np.sin(a_rad)
                # Draw each arm's swath as a separate semi-transparent rectangle
                for (x_near, y_near), (x_far, y_far) in [
                    ((near_pos_x, near_pos_y), (far_pos_x, far_pos_y)),
                    ((near_neg_x, near_neg_y), (far_neg_x, far_neg_y)),
                ]:
                    corners = np.array([
                        [x_near + half_w * px, y_near + half_w * py],
                        [x_near - half_w * px, y_near - half_w * py],
                        [x_far  - half_w * px, y_far  - half_w * py],
                        [x_far  + half_w * px, y_far  + half_w * py],
                    ])
                    patch = MplPolygon(
                        corners, closed=True,
                        facecolor=color, edgecolor=color,
                        alpha=0.15, lw=0.8, zorder=3,
                    )
                    ax.add_patch(patch)

    ax.set(
        xlim=(0, image.shape[1]), ylim=(0, image.shape[0]),
        title="Detected spikes",
    )
    ax.legend(loc="upper right", fontsize=7)

    for j in range(4, ncols_fig):
        axes[0, j].axis("off")

    # ── Row 2: swath profiles ────────────────────────────────────────────
    if has_lengths:
        for i in range(ncols_fig):
            ax = axes[1, i]
            if i >= n_spikes:
                ax.axis("off")
                continue

            sl = result.lengths[i]
            color = colors[i % len(colors)]

            r_all = np.concatenate([sl.radii_pos, sl.radii_neg])
            r_dense = np.linspace(max(r_all.min(), 0.5), r_all.max(), 400)

            if sl.background_subtracted:
                # ── Background-subtracted mode ────────────────────────────
                # Profiles are already signal (bg removed); use symlog so
                # near-zero and slightly negative noise values are visible.
                linthresh = max(sl.threshold, 1e-6) if sl.threshold > 0 else 1e-4

                ax.set_yscale("symlog", linthresh=linthresh)

                ax.plot(sl.radii_pos, sl.profile_pos,
                        "-", color=color, lw=1.2,
                        label=f"+ arm ({sl.length_pos:.0f} px"
                              + ("" if sl.converged_pos else ", extrap") + ")")
                ax.plot(sl.radii_neg, sl.profile_neg,
                        "--", color=color, lw=1.2,
                        label=f"− arm ({sl.length_neg:.0f} px"
                              + ("" if sl.converged_neg else ", extrap") + ")")

                # Background profile shown for reference
                if sl.background_profile is not None:
                    r_bg, p_bg = sl.background_profile
                    ax.plot(r_bg, p_bg, ":", color="green", lw=1.0, alpha=0.7,
                            label="Background (ref)")

                # Fitted sinc² model and its envelope
                if sl.popt is not None:
                    a, b = sl.popt[0], sl.popt[1]
                    fit_vals = a * np.sinc(b * r_dense) ** 2
                    env_vals = a / (np.pi * b * r_dense) ** 2
                    ax.plot(r_dense, fit_vals,
                            "-", color="0.35", lw=1.2, alpha=0.8,
                            label=f"sinc² fit (a={a:.2g}, b={b:.2g})")
                    ax.plot(r_dense, env_vals,
                            ":", color="0.35", lw=1.0, alpha=0.7,
                            label="Envelope")

                ylabel = "Signal flux (bg subtracted)"

            else:
                # ── Full Fraunhofer model mode ────────────────────────────
                ax.semilogy(sl.radii_pos, np.maximum(sl.profile_pos, 1e-30),
                            "-", color=color, lw=1.2,
                            label=f"+ arm ({sl.length_pos:.0f} px"
                                  + ("" if sl.converged_pos else ", extrap") + ")")
                ax.semilogy(sl.radii_neg, np.maximum(sl.profile_neg, 1e-30),
                            "--", color=color, lw=1.2,
                            label=f"− arm ({sl.length_neg:.0f} px"
                                  + ("" if sl.converged_neg else ", extrap") + ")")

                # Fix ylims to avoid autoscaling to the noise floor in long arms
                low = 0.1 * sl.threshold if sl.threshold > 0 else 1e-4
                ax.set_ylim(low, ax.get_ylim()[1])

                if sl.background_profile is not None:
                    r_bg, p_bg = sl.background_profile
                    ax.semilogy(r_bg, np.maximum(p_bg, 1e-30), ":", color="green",
                                lw=1.0, alpha=0.7, label="Background profile")

                if sl.popt is not None:
                    fit_vals = _fraunhofer_model(r_dense, *sl.popt)
                    env_vals = _envelope_model(r_dense, *sl.popt)
                    ax.semilogy(r_dense, np.maximum(fit_vals, 1e-30),
                                "-", color="0.35", lw=1.0, alpha=0.7,
                                label="Fraunhofer fit")
                    ax.semilogy(r_dense, np.maximum(env_vals, 1e-30),
                                ":", color="0.35", lw=1.0, alpha=0.7,
                                label="Envelope")
                    # Halo + background component
                    halo_vals = sl.popt[2] / r_dense ** sl.popt[3] + sl.popt[4]
                    ax.semilogy(r_dense, np.maximum(halo_vals, 1e-30),
                                "--", color="0.35", lw=1.0, alpha=0.7,
                                label="Halo + bg")

                ylabel = "Swath median flux"

            # Detection threshold — common to both modes
            if sl.threshold > 0:
                ax.axhline(sl.threshold, color="grey", ls=":", lw=1,
                           label=f"threshold ({sl.threshold:.2g})")

            # Arm endpoint markers (filled = converged, open = extrapolated)
            for arm_len, ls, converged in [
                (sl.length_pos, "-", sl.converged_pos),
                (sl.length_neg, "--", sl.converged_neg),
            ]:
                ax.axvline(arm_len, color=color, ls=ls, lw=0.8, alpha=0.5)
                if sl.threshold > 0:
                    marker = "o" if converged else "^"
                    ax.plot(arm_len, sl.threshold, marker,
                            color=color, ms=6,
                            mfc=color if converged else "none", mew=1.5)

            ax.set(
                xlabel="Radius (pixels)", ylabel=ylabel,
                title=f"Spike @ {sl.angle_deg:.1f}°",
            )
            ax.legend(fontsize=12)

    return fig
