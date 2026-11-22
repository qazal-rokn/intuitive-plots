#!/usr/bin/env python3
"""Reproduce the three illustrative APE statistics plots.

This is a teaching example, NOT measured APE data or a calibrated APE
skew-normal likelihood. For clarity, the toy model uses a quadratic
log-likelihood, q0 = max(mu_hat/sigma_mu, 0)**2, and its ASYMPTOTIC
one-sided mapping p_local = scipy.stats.norm.sf(sqrt(q0)).

Run: python plot_mu_q0_false_alarm.py
Dependency: numpy matplotlib scipy pillow
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Reproducible file output on servers and laptops
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(27)
    f = np.arange(1, 61)
    sigma = 1.3 + 0.45 * (1 + np.sin(f / 7))
    mu = rng.normal(0, sigma)
    mu[12] = 3.1 * sigma[12]
    mu[28] = 4.1 * sigma[28]
    mu[45] = 5.4 * sigma[45]

    # Example ONLY: this identity is not exact for APE's skew-normal likelihood.
    z = np.maximum(mu / sigma, 0)
    q0 = z ** 2
    p_local = norm.sf(z)
    p5 = norm.sf(5)

    fig1, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(f, mu, marker="o", markersize=3, linewidth=1.4,
            label=r"Best fit $\hat{\mu}(f)$")
    ax.fill_between(f, -sigma, sigma, alpha=0.15,
                    label=r"Illustrative $\pm\sigma_\mu(f)$")
    ax.axhline(0, linewidth=1, color="gray")
    ax.annotate(r"$\hat{\mu}=5.4\sigma_\mu$", xy=(46, mu[45]),
                xytext=(35, mu[45] + 1.5), arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="Frequency bin (illustrative)",
           ylabel=r"Best-fit strength $\hat{\mu}$",
           title=r"1. Best-fit signal strength across frequency")
    ax.legend(loc="upper left")
    fig1.tight_layout()
    fig1.savefig(output_dir / "ape_mu_spectrum.png", dpi=190)

    fig2, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(f, q0, marker="o", markersize=3, linewidth=1.4)
    ax.axhline(25, linestyle="--", label=r"Local 5$\sigma$: $q_0=25$")
    ax.scatter([46], [q0[45]], s=70, zorder=5)
    ax.annotate(r"$q_0=5.4^2=29.16$", xy=(46, q0[45]),
                xytext=(30, 33), arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="Frequency bin (illustrative)",
           ylabel=r"Likelihood-ratio statistic $q_0$",
           title=r"2. $q_0(f)$ from the local likelihood at each frequency",
           ylim=(0, 38))
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "ape_q0_spectrum.png", dpi=190)

    fig3, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(f, p_local, marker="o", markersize=3, linewidth=1.4)
    ax.set_yscale("log")
    ax.axhline(p5, linestyle="--",
               label=r"Local 5$\sigma$ tail: $p=2.87\times10^{-7}$")
    ax.scatter([46], [p_local[45]], s=70, zorder=5)
    ax.annotate(fr"$p={p_local[45]:.2g}$", xy=(46, p_local[45]),
                xytext=(33, 2e-9), arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="Frequency bin (illustrative)",
           ylabel="Local false-alarm probability (log scale)",
           title="3. Noise-only tail probability at each frequency",
           ylim=(1e-10, 1))
    ax.legend(loc="lower left")
    fig3.tight_layout()
    fig3.savefig(output_dir / "ape_pvalue_spectrum.png", dpi=190)

    # Same 3 plots, convenient to view as one tall image on mobile.
    from PIL import Image
    files = [output_dir / name for name in (
        "ape_mu_spectrum.png", "ape_q0_spectrum.png", "ape_pvalue_spectrum.png")]
    imgs = [Image.open(path).convert("RGB") for path in files]
    canvas = Image.new("RGB", (max(im.width for im in imgs), sum(im.height for im in imgs)), "white")
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(output_dir / "APE_three_statistical_plots.png")
    for im in imgs:
        im.close()
    plt.close("all")
    print(f"Saved 4 figures to {output_dir.resolve()}")
    print(f"Bin 46: mu/sigma=5.4, q0={q0[45]:.2f}, p={p_local[45]:.3g}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
