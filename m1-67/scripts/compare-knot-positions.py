"""
Compare the positions of the knots from two different images
"""

import numpy as np
import typer
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

ORIGIN = SkyCoord.from_name("wr124", cache=True)


def main(
    prefix1: str,
    prefix2: str,
    coordname: str = "Center",
    maxdelta: float = 0.45,
    xyshift1: tuple[float, float] = (0.0, 0.0),
):
    # Load the data
    table1 = QTable.read(f"{prefix1}-knot-fluxes.ecsv")
    table2 = QTable.read(f"{prefix2}-knot-fluxes.ecsv")
    # Check that have the same knots in both tables
    assert np.all(table1["label"] == table2["label"])

 
    # Apply optional shift to the first table
    #
    # Use a klugey way to get round astropy bugs in spherical_offsets_by
    ra, dec = table1[coordname].ra, table1[coordname].dec
    table1[coordname] = SkyCoord(
        ra + np.cos(dec) * xyshift1[0] * u.arcsec,
        dec + xyshift1[1] * u.arcsec,
        frame="icrs",
    )

    # This is the way it should be done, but there is a bug in astropy
    # < 6.1 (which requires python 3.10 or later)
    #
    # table1[coordname] = table1[coordname].spherical_offsets_by(
    #     xyshift1[0] * u.arcsec, xyshift1[1] * u.arcsec
    # )

    # Find ra, dec offsets of first set of knots from star
    x, y = ORIGIN.spherical_offsets_to(table1[coordname])
    # Find ra, dec offsets of second set of knots from first set
    dx, dy = table1[coordname].spherical_offsets_to(table2[coordname])

    plot_tab = pd.DataFrame(
        {
            "x": x.arcsec,
            "y": y.arcsec,
            "dx": dx.arcsec,
            "dy": dy.arcsec,
            "label": table1["label"],
        }
    )
    # Plot the offsets
    grid = sns.pairplot(
        data=plot_tab,
        x_vars=["x", "y"],
        y_vars=["dx", "dy"],
        plot_kws=dict(alpha=0.2),
    )
    grid.fig.suptitle(f"{prefix1} - {prefix2}")
    for ax in grid.axes.flat:
        ax.axhline(0, color="gray", ls="--")
        ax.axvline(0, color="gray", ls="--")
        ax.set_ylim(-maxdelta, maxdelta)
    plotfile = f"{prefix1}-{prefix2}-knot-offsets.pdf"
    grid.savefig(plotfile)
    print(f"Saved plot to {plotfile}")


if __name__ == "__main__":
    typer.run(main)
