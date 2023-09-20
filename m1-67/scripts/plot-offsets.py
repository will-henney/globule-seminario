import numpy as np
from astropy.table import Table
import typer
from matplotlib import pyplot as plt
import seaborn as sns


def main(
    filename: str,
    max_sep: float = 200.0,
    palette: str = "flare",
):
    tab = Table.read(filename)
    ra = tab["d RA, mas"]
    dec = tab["d Dec, mas"]
    r = tab["Radius, arcsec"]
    ra_med = np.median(ra)
    dec_med = np.median(dec)
    cmap = sns.color_palette(palette, as_cmap=True)
    g = sns.JointGrid(
        data=tab.to_pandas(),
        x="d RA, mas",
        y="d Dec, mas",
        hue="Radius, arcsec",
        height=4,
        palette=palette,
    )
    g.plot_joint(sns.scatterplot)
    g.hue = None  # Trick seaborn into ignoring the hue for the marinal plots
    g.plot_marginals(sns.histplot, color=cmap(0.5))
    # scat = ax.scatter(ra, dec, linewidths=0, s=50, c=r, alpha=0.5)
    # fig.colorbar(scat, ax=ax, location="right", label="Radius from center, arcsec")
    ax = g.ax_joint
    ax.axvline(0.0, color="k", lw=1, linestyle="dashed")
    ax.axhline(0.0, color="k", lw=1, linestyle="dashed")
    ax.scatter(ra_med, dec_med, marker="+", s=400, c="k")
    ax.set(
        xlim=[ra_med - max_sep, ra_med + max_sep],
        ylim=[dec_med - max_sep, dec_med + max_sep],
    )
    g.figure.suptitle(filename, y=1.01, va="baseline", fontsize="small")
    figfile = filename.replace(".ecsv", ".pdf")
    g.figure.savefig(figfile, bbox_inches="tight")
    print(figfile, end="")


if __name__ == "__main__":
    typer.run(main)
