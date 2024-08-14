"""
Compare the radial separations of the knots from two different images
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

    r1 = table1[coordname].separation(ORIGIN)
    r2 = table2[coordname].separation(ORIGIN)
    dr = r2 - r1
    
    # Find ra, dec offsets of first set of knots from star
    x, y = ORIGIN.spherical_offsets_to(table1[coordname])
    # Find ra, dec offsets of second set of knots from first set
    dx, dy = table1[coordname].spherical_offsets_to(table2[coordname])

    xlabel = "Radial displacement from star, $R$, arcsec"
    ylabel = r"Radial offset between peaks, $\delta R$, arcsec"
    plot_tab = pd.DataFrame(
        {
            xlabel: r1.arcsec,
            ylabel: dr.arcsec,
            "label": table1["label"],
        }
    )
    # Plot the offsets
    grid = sns.jointplot(
        data=plot_tab,
        x=xlabel,
        y=ylabel,
    )
    grid.fig.suptitle(f"{prefix1} - {prefix2}", va="bottom")
    grid.refline(y=0)
    grid.ax_joint.set_ylim(-maxdelta/2, maxdelta)
    plotfile = f"{prefix1}-{prefix2}-knot-seps.pdf"
    grid.savefig(plotfile, bbox_inches="tight")
    print(f"Saved plot to {plotfile}")

    savetab = QTable(
        {
            "label": table1["label"],
            "PA": table1["PA"],
            "R": r1.to(u.arcsec),
            "dR": dr.to(u.arcsec),
        }
    )
    tabfile = plotfile.replace(".pdf", ".ecsv")
    savetab.write(tabfile, overwrite=True)
    # plot_tab.to_csv(tabfile, index=False)
    print(f"Saved data to {tabfile}")

if __name__ == "__main__":
    typer.run(main)
