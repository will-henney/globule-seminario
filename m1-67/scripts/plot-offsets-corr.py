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
    grid = sns.pairplot(
        data=tab.to_pandas(),
        x_vars=["RA, arcsec", "Dec, arcsec"],
        y_vars=["d RA, mas", "d Dec, mas"],
    )
    grid.set(ylim=[-max_sep, max_sep])
    grid.figure.suptitle(filename, y=1.01, va="baseline")
    figfile = filename.replace(".ecsv", "-CORR.pdf")
    grid.savefig(figfile, bbox_inches="tight")
    print(figfile, end="")


if __name__ == "__main__":
    typer.run(main)
