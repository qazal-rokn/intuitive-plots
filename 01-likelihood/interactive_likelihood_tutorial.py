"""Interactive, self-contained tutorial for the scalar-dark-matter likelihood.

Run: python interactive_likelihood_tutorial.py
Requirements: numpy, scipy, matplotlib

Educational example only. This script does not read or change APE data or code.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.optimize import minimize
from scipy.stats import skewnorm


# ------------- The same illustrative example as our discussion -------------
FREQUENCY_HZ = 100.0
BACKGROUND_PSD = 10.0
MEASURED_PSD = np.array([12.0, 14.0, 13.0])
RECORDINGS = ("A", "B", "C")
COLORS = ("#2563eb", "#ea580c", "#16a34a")

# Teaching example: parameters of the fitted residual distribution.
# They are NOT derived from real APE data.
SKEW_SHAPE = 3.0       # alpha
SKEW_LOCATION = -0.08 # loc, in log-PSD residual units
SKEW_SCALE = 0.12     # scale, in log-PSD residual units

MU_MIN, MU_MAX = 0.0, 6.0


def predicted_psd(mu):
    """Unit signal response: P_predicted = B + mu."""
    return BACKGROUND_PSD + float(mu)


def residuals_for_mu(mu):
    """Residuals are in log-PSD space, matching the original model's idea."""
    return np.log(MEASURED_PSD / predicted_psd(mu))


def log_likelihood(mu):
    """Sum log probability *densities*, one per recording."""
    residuals = residuals_for_mu(mu)
    return float(np.sum(skewnorm.logpdf(
        residuals, a=SKEW_SHAPE, loc=SKEW_LOCATION, scale=SKEW_SCALE
    )))


def find_best_mu():
    """Coarse trial scan, then Nelder-Mead refinement (tutorial: mu >= 0)."""
    trials = np.linspace(MU_MIN, MU_MAX, 601)
    seed = trials[np.argmax([log_likelihood(mu) for mu in trials])]
    fit = minimize(
        lambda x: -log_likelihood(x[0]),
        x0=np.array([seed]),
        method="Nelder-Mead",
        bounds=[(MU_MIN, MU_MAX)],
        options={"xatol": 1e-11, "fatol": 1e-11, "maxiter": 2000},
    )
    if not fit.success:
        raise RuntimeError(f"Nelder-Mead did not converge: {fit.message}")
    return float(fit.x[0])


def build_figure():
    best_mu = find_best_mu()
    best_logL = log_likelihood(best_mu)
    mu_grid = np.linspace(MU_MIN, MU_MAX, 601)
    rel_likelihood = np.exp(np.array([log_likelihood(m) for m in mu_grid]) - best_logL)
    residual_grid = np.linspace(-0.42, 0.42, 600)
    skew_pdf = skewnorm.pdf(
        residual_grid, a=SKEW_SHAPE, loc=SKEW_LOCATION, scale=SKEW_SCALE
    )

    fig = plt.figure(figsize=(10.5, 11.4), constrained_layout=False)
    fig.suptitle("Interactive likelihood tutorial | one frequency: 100 Hz", fontsize=15, weight="bold")
    gs = fig.add_gridspec(3, 1, left=0.12, right=0.96, top=0.92, bottom=0.24, hspace=0.52)
    ax_psd, ax_pdf, ax_lkl = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    # Plot 1: PSD observations versus trial prediction.
    y_positions = np.array([3, 2, 1], dtype=float)
    prediction_line = ax_psd.axvline(13, color="#7c3aed", ls="--", lw=2,
                                     label="Prediction (background + mu)")
    mismatch_lines = []
    for y, measured, name, color in zip(y_positions, MEASURED_PSD, RECORDINGS, COLORS):
        (mismatch,) = ax_psd.plot([measured, 13], [y, y], color=color, lw=4, alpha=0.35)
        mismatch_lines.append(mismatch)
        ax_psd.scatter([measured], [y], s=92, color=color, zorder=5, label=f"Recording {name}: PSD {measured:g}")
    ax_psd.set_xlim(9.7, 16.3)
    ax_psd.set_ylim(0.5, 3.5)
    ax_psd.set_yticks(y_positions, ["A", "B", "C"])
    ax_psd.set_xlabel("PSD (illustrative units)")
    ax_psd.set_title("1. Compare each measured PSD with the predicted PSD", loc="left")
    ax_psd.grid(axis="x", alpha=0.2)
    ax_psd.legend(loc="upper right", fontsize=8, ncol=2)

    # Plot 2: fitted skew-normal residual distribution and moving residuals.
    ax_pdf.plot(residual_grid, skew_pdf, color="#7c3aed", lw=2.6, label="Fitted skew-normal PDF")
    ax_pdf.axvline(0, color="0.5", ls=":", lw=1)
    pdf_stems, pdf_points = [], []
    for name, color in zip(RECORDINGS, COLORS):
        (stem,) = ax_pdf.plot([], [], color=color, ls="--", lw=1.5, alpha=0.8)
        (dot,) = ax_pdf.plot([], [], "o", color=color, ms=8, label=f"Recording {name}")
        pdf_stems.append(stem)
        pdf_points.append(dot)
    ax_pdf.set_xlim(-0.42, 0.42)
    ax_pdf.set_ylim(0, skew_pdf.max()*1.19)
    ax_pdf.set_xlabel(r"Residual $r_i=\ln(P_i/P_{\mathrm{predicted}})$")
    ax_pdf.set_ylabel("Probability density (not probability)")
    ax_pdf.set_title("2. Evaluate each residual on the skew-normal curve", loc="left")
    ax_pdf.legend(loc="upper right", fontsize=8)
    ax_pdf.grid(alpha=0.17)
    detail_text = ax_pdf.text(0.02, 0.95, "", transform=ax_pdf.transAxes, va="top",
                              ha="left", fontsize=9.5, family="monospace",
                              bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.94))

    # Plot 3: combined, maximum-normalized likelihood as a function of mu.
    ax_lkl.plot(mu_grid, rel_likelihood, color="#7c3aed", lw=2.7)
    ax_lkl.axvline(best_mu, color="#16a34a", ls="--", lw=1.8,
                   label=fr"Best fit $\hat{{\mu}}={best_mu:.3f}$")
    chosen_line = ax_lkl.axvline(3, color="#ea580c", ls=":", lw=1.7)
    (chosen_dot,) = ax_lkl.plot([], [], "o", color="#ea580c", ms=9,
                                 label="Your trial mu")
    ax_lkl.set_xlim(MU_MIN, MU_MAX)
    ax_lkl.set_ylim(0, 1.08)
    ax_lkl.set_xlabel(r"Trial signal strength $\mu$")
    ax_lkl.set_ylabel(r"Relative likelihood $\mathcal{L}(\mu)/\mathcal{L}(\hat{\mu})$")
    ax_lkl.set_title("3. Multiply densities (or sum log densities); find the peak", loc="left")
    ax_lkl.legend(loc="upper right", fontsize=9)
    ax_lkl.grid(alpha=0.2)

    # One real, draggable Matplotlib slider controls all three plots.
    slider_ax = fig.add_axes([0.16, 0.088, 0.60, 0.035])
    slider = Slider(slider_ax, r"Signal $\mu$", MU_MIN, MU_MAX, valinit=3.0,
                    valstep=0.05, color="#7c3aed")
    best_ax = fig.add_axes([0.79, 0.088, 0.15, 0.04])
    best_button = Button(best_ax, "Go to best fit")

    info = fig.text(0.50, 0.16, "", ha="center", fontsize=10, weight="bold")
    fig.text(0.5, 0.027,
             "Educational PSDs and skew-normal parameters, not APE data. "
             "Original code can also scan negative mu and use different models for recordings.",
             ha="center", fontsize=8, color="0.4")

    def update(mu):
        mu = float(mu)
        model_psd = predicted_psd(mu)
        r = residuals_for_mu(mu)
        density = skewnorm.pdf(r, a=SKEW_SHAPE, loc=SKEW_LOCATION, scale=SKEW_SCALE)
        rel = np.exp(log_likelihood(mu) - best_logL)

        prediction_line.set_xdata([model_psd, model_psd])
        for line, measured, y in zip(mismatch_lines, MEASURED_PSD, y_positions):
            line.set_data([measured, model_psd], [y, y])
        for stem, dot, rv, pv in zip(pdf_stems, pdf_points, r, density):
            stem.set_data([rv, rv], [0, pv])
            dot.set_data([rv], [pv])
        detail_text.set_text("Recording   residual   PDF density\n" +
                             "\n".join(f"{name:>5s}     {rv:+8.3f}    {pv:7.3f}"
                                       for name, rv, pv in zip(RECORDINGS, r, density)))
        chosen_line.set_xdata([mu, mu])
        chosen_dot.set_data([mu], [rel])
        info.set_text(f"Selected mu = {mu:.2f}  |  predicted PSD = {model_psd:.2f}  |  "
                      f"relative likelihood = {rel:.3f}  |  best mu = {best_mu:.3f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    best_button.on_clicked(lambda _event: slider.set_val(best_mu))
    update(slider.val)
    # Keep widget references alive to prevent Matplotlib garbage collection.
    fig._tutorial_controls = (slider, best_button)
    return fig


if __name__ == "__main__":
    build_figure()
    plt.show()
