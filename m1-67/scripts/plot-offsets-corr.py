import numpy as np
from astropy.table import Table
import typer
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pandas as pd


def robust_fit(data: pd.DataFrame, xlabel: str, ylabel: str):
    """Fit a robust linear model to the data."""
    # First, sort the data on the exogenous variable so that plotting will work better
    sorted_data = data.sort_values(xlabel)
    #  Take the exogenous variable as x column from the data and add a constant term
    X = sm.add_constant(sorted_data[xlabel])
    #  Take the endogenous variable as y column from the data
    y = sorted_data[ylabel]
    #  Fit the model
    fit = sm.RLM(y, X).fit()
    return fit


def plot_robust_fit(
    ax,
    data: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    nsigma: float = 1.0,
    color: str = "k",
):
    """
    Plot a robust linear model fit to the data plus the confidence
    region.

    This is basically the same as what seaborn.regplot does, but also
    gives the parameters of the fit, which seaborn refuses to do
    """
    fit = robust_fit(data, xlabel, ylabel)
    x = fit.model.exog[:, 1]
    xmin, xmax = np.min(x), np.max(x)
    ymodel = fit.fittedvalues
    # Fitted straight line
    intercept, slope = fit.params
    # Errors on fit
    e_intercept, e_slope = fit.bse
    # Range of models corresponding to +/- N-sigma errors
    odd = np.array([1, -1])
    ymodels = np.stack(
        [
            np.dot(fit.model.exog, fit.params + nsigma * fit.bse),
            np.dot(fit.model.exog, fit.params - nsigma * fit.bse),
            np.dot(fit.model.exog, fit.params + nsigma * odd * fit.bse),
            np.dot(fit.model.exog, fit.params - nsigma * odd * fit.bse),
        ],
        axis=0,
    )
    # Upper and lower bounds of the model
    ymodel_max = np.max(ymodels, axis=0)
    ymodel_min = np.min(ymodels, axis=0)

    ax.plot(x, ymodel, color=color, linestyle="solid")
    ax.fill_between(x, ymodel_min, ymodel_max, color=color, alpha=0.2, linewidth=0)

    offset_string = rf"offset = $({intercept:+.1f} \pm {e_intercept:.1f})$ mas"
    slope_string = rf"slope = $({slope:+.3f} \pm {e_slope:.3f}) \times 10^{{-3}}$"

    ax.text(0.05, 0.9, offset_string, fontsize="small", transform=ax.transAxes)
    ax.text(0.05, 0.8, slope_string, fontsize="small", transform=ax.transAxes)

    return fit


def main(
    filename: str,
    max_sep: float = 200.0,
    alpha: float = 1.0,
):
    data = Table.read(filename).to_pandas()
    ra_med = np.median(data["d RA, mas"])
    dec_med = np.median(data["d Dec, mas"])
    plot_kws = {}
    if alpha < 1.0:
        plot_kws["alpha"] = alpha
        # If points are transparent, don't draw the edges
        plot_kws["linewidth"] = 0
    x_vars = ["RA, arcsec", "Dec, arcsec"]
    y_vars = ["d RA, mas", "d Dec, mas"]
    grid = sns.pairplot(
        data=data,
        x_vars=x_vars,
        y_vars=y_vars,
        plot_kws=plot_kws,
    )
    results = []
    for j, y_var in enumerate(y_vars):
        for i, x_var in enumerate(x_vars):
            ax = grid.axes[j, i]
            fit = plot_robust_fit(ax, data, x_var, y_var, color="r", nsigma=2.0)
            center = fit.params[0]
            ax.set_ylim(center - max_sep, center + max_sep)
            ax.axhline(center, color="k", linestyle="dashed")
            results.append(fit.summary())

    grid.figure.suptitle(filename, y=1.01, va="baseline")

    figfile = filename.replace(".ecsv", "-CORR.pdf")
    fitfile = filename.replace(".ecsv", "-CORR.txt")
    grid.savefig(figfile, bbox_inches="tight")
    with open(fitfile, "w") as f:
        for result in results:
            f.write(result.as_text())
            f.write("\n")

    print(figfile, end="")


if __name__ == "__main__":
    typer.run(main)
