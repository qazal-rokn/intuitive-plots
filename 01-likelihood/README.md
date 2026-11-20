# Interactive likelihood tutorial

A self-contained, interactive introduction to maximum-likelihood fitting for a hypothetical narrowband signal in a power spectral density (PSD).

## What you'll see

The same adjustable signal strength, `mu`, drives three synchronized plots:

1. **PSD measurements and model:** three illustrative recordings (12, 14, 13 at 100 Hz), a fixed background of 10, and a model prediction of `10 + mu`.
2. **Skew-normal residual distribution:** how the three log-PSD residuals move along the assumed noise distribution as `mu` changes.
3. **Combined relative likelihood:** the product of the residual probability densities, visualized relative to its maximum.

The interactive **Go to best fit** button finds the optimum using a trial search followed by Nelder–Mead optimization.

## Run as a Python script

Install Python 3 and dependencies:

```bash
python -m pip install -r requirements.txt
```

Run from this folder:

```bash
python interactive_likelihood_tutorial.py
```

A Matplotlib window should appear with a `mu` slider.

## Run as a notebook

Open `interactive_likelihood_tutorial.ipynb` in JupyterLab or Jupyter Notebook, install the same dependencies plus the interactive backend:

```bash
python -m pip install -r requirements.txt ipympl notebook
```

Execute the notebook cells. If the interactive backend is unavailable, follow the notebook's backend instructions.

## Scientific scope

**Teaching example only.** The PSDs, background, response factor, and skew-normal parameters are illustrative, not actual APE data. The likelihood follows the same general residual/log-density combination as the original scalar-dark-matter code, but the optimizer search range here is deliberately simplified. This script does not read, import, or modify the APE analysis pipeline.

The tutorial is kept in this repository specifically to be separate from `APE-analysis-code`.
