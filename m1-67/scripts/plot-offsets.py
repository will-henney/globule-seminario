import numpy as np
from astropy.table import Table
import typer
from matplotlib import pyplot as plt
import seaborn as sns


def main(
    filename: str,
    max_sep: float = 200.0,
):
    tab = Table.read(filename)
    ra = tab["d RA, mas"]
    dec = tab["d Dec, mas"]
    r = tab["Radius, arcsec"]
    limits = [-max_sep, max_sep]
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(ra, dec, linewidths=0, s=50, c=r)
    fig.colorbar(scat, ax=ax, location="right", label="Radius from center, arcsec")
    ax.axvline(0.0, color="k", lw=1, linestyle="dashed")
    ax.axhline(0.0, color="k", lw=1, linestyle="dashed")
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="displacement RA, milliarcsec",
        ylabel="displacement DEC, milliarcsec",
    )
    ax.set_aspect("equal")
    ax.set_title(filename)
    figfile = filename.replace(".ecsv", ".pdf")
    fig.savefig(figfile, bbox_inches="tight")
    print(figfile, end="")


if __name__ == "__main__":
    typer.run(main)
