import numpy as np
from astropy.table import Table
import typer
from matplotlib import pyplot as plt
import seaborn as sns


def main(
    filename: str,
    max_sep: float = 200.0,
    alpha: float = 1.0,
):
    tab = Table.read(filename)
    ra_med = np.median(tab["d RA, mas"])
    dec_med = np.median(tab["d Dec, mas"])
    plot_kws = {}
    if alpha < 1.0:
        plot_kws["alpha"] = alpha
        # If points are transparent, don't draw the edges
        plot_kws["linewidth"] = 0
    grid = sns.pairplot(
        data=tab.to_pandas(),
        x_vars=["RA, arcsec", "Dec, arcsec"],
        y_vars=["d RA, mas", "d Dec, mas"],
        plot_kws=plot_kws,
    )
    grid.figure.suptitle(filename, y=1.01, va="baseline")

    for ax in grid.axes[0, :]:
        ax.set_ylim(ra_med - max_sep, ra_med + max_sep)
        ax.axhline(ra_med, color="r", linestyle="dashed")
    for ax in grid.axes[1, :]:
        ax.set_ylim(dec_med - max_sep, dec_med + max_sep)
        ax.axhline(dec_med, color="r", linestyle="dashed")

    figfile = filename.replace(".ecsv", "-CORR.pdf")
    grid.savefig(figfile, bbox_inches="tight")
    print(figfile, end="")


if __name__ == "__main__":
    typer.run(main)
