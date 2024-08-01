"""
Find the positions of the cores of the knots from a given image

Approximate knot positions are taken from a region file. 
"""

import numpy as np
from astropy.io import fits
import typer
from pathlib import Path
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
import astropy.units as u
from astropy.nddata import Cutout2D
import regions as rg


ORIGIN = SkyCoord.from_name("wr124", cache=True)


def get_knot_table(regionfile: str) -> QTable:
    regs = rg.Regions.read(regionfile, format="ds9")
    # Select only those regions that are circle points, since those are the ones that are the globules
    regs = [_ for _ in regs if hasattr(_, "center") and _.visual.get("marker") == "o"]
    knot_table = QTable(
        [
            {"ICRS": r.center, "Isolated": r.visual.get("markeredgewidth") == 3}
            for r in regs
        ]
    )
    knot_table["PA"] = ORIGIN.position_angle(knot_table["ICRS"]).to(u.deg)
    knot_table["Sep"] = knot_table["ICRS"].separation(ORIGIN).to(u.arcsec)
    knot_table["PA"].info.format = ".2f"
    knot_table["Sep"].info.format = ".2f"
    return knot_table


class SourceCutout:
    """Small image cut out around a given source"""

    def __init__(self, pdata, hdu, size=3 * u.arcsec, initial_core=0.2 * u.arcsec):
        self.center = pdata["ICRS"]
        # PA of source wrt star
        self.pa_source = pdata["PA"]
        # and PA of star wrt source
        self.pa_star = Angle(self.pa_source + 180 * u.deg).wrap_at(360 * u.deg)
        self.sep = pdata["Sep"]
        self.label = f"PA{int(np.round(self.pa_source.value)):03d}"
        self.label += f"-R{int(np.round(10 * self.sep.value)):03d}"
        self.is_isolated = pdata["Isolated"]
        self.size = size
        self.cutout = Cutout2D(
            hdu.data,
            position=self.center,
            size=size,
            wcs=WCS(hdu),
            copy=True,
        )
        self.image = self.cutout.data
        self.wcs = self.cutout.wcs
        ny, nx = self.image.shape
        self.x, self.y = np.meshgrid(np.arange(nx), np.arange(ny))
        self.image_coords = self.wcs.pixel_to_world(self.x, self.y)
        # Radius and PA of each pixel with respect to the NOMINAL center
        self.r = self.center.separation(self.image_coords)
        self.pa = self.center.position_angle(self.image_coords)
        # Default mask has max radius of half the cutout size

        # We use a more generous initial radius for the core, so we
        # can guarantee that we will enclose the neutral emission peak
        self.set_mask(r_out=self.size / 2, r_core=initial_core)

        self.find_bary_center()
        # Recalculate r, pa, and masks wrt the barycenter
        self.r = self.barycenter.separation(self.image_coords)
        self.pa = self.barycenter.position_angle(self.image_coords)
        self.set_mask(r_out=self.size / 2)

        # And do photometry
        self.bright_peak = np.max(self.image[self.coremask])
        # Take the 10% centile as estimate of BG value
        self.bright_bg = np.percentile(
            self.image[self.mask],
            10,
        )
        # BG-subtracted of the core region
        self.flux_core = np.sum((self.image - self.bright_bg)[self.coremask])
        # BG-subtracted part of the outer part within the mask but excluding core
        self.flux_halo = np.sum(
            (self.image - self.bright_bg)[(~self.coremask) & self.mask]
        )

    def __repr__(self):
        return f"SourceCutout({self.label})"

    def set_mask(
        self,
        r_out=1.0 * u.arcsec,
        r_in=0.2 * u.arcsec,
        r_core=0.2 * u.arcsec,
        mu_min=0.5,
    ):
        cth = np.cos((self.pa - self.pa_star))
        self.mask = (self.r <= r_out) & ((cth >= mu_min) | (self.r <= r_in))
        self.coremask = self.r <= r_in

    def find_bary_center(self):
        """Refine estimate of center by using flux-weighted mean
        position within the core mask
        """
        m = self.coremask

        # Original version was flux-weighted mean
        # xbary = np.average(self.x[m], weights=self.image[m])
        # ybary = np.average(self.y[m], weights=self.image[m])

        # New version is just the peak pixel
        index = np.argmax(self.image[m], axis=None)
        xbary = self.x[m][index]
        ybary = self.y[m][index]

        self.barycenter = self.wcs.pixel_to_world(xbary, ybary)


def main(
    imagefile: str,
    regionfile: str = "../../m1-67-globules.reg",
):
    # Read the image
    hdu = fits.open(imagefile)[0]

    # Read the region file
    knot_table = get_knot_table(regionfile)

    cutouts = [SourceCutout(source, hdu) for source in knot_table]
    cutouts = sorted(cutouts, key=lambda x: x.sep.value)

    # Save the barycenters as a region file
    outfile = imagefile.replace(".fits", "-knot-coords.reg")
    regs = rg.Regions(
        [
            rg.CircleSkyRegion(
                center=cutout.barycenter,
                radius=0.2 * u.arcsec,
                meta={
                    "text": cutout.label,
                },
            )
            for cutout in cutouts
        ]
    )
    regs.write(outfile, format="ds9", overwrite=True)
    print(f"Saved barycenters to {outfile}")


if __name__ == "__main__":
    typer.run(main)
