"""
Find size and fluxes within a given aperture

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
from astropy.modeling import models, fitting
from astropy.modeling.models import Const2D, Gaussian2D

FITTER = fitting.LevMarLSQFitter()
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

    def cutout_around_point(self, hdu, center):
        """Get a new cutout around a given point in the image

        Together with all other necessary arrays for each pixel in the
        cutout, such as pixel coordinates, sky coordinates, radii from center,
        position angle

        """
        self.cutout = Cutout2D(
            hdu.data,
            position=self.center,
            size=self.size,
            wcs=WCS(hdu),
            copy=True,
        )
        self.image = self.cutout.data
        self.wcs = self.cutout.wcs
        ny, nx = self.image.shape
        self.x, self.y = np.meshgrid(np.arange(nx), np.arange(ny))
        self.image_coords = self.wcs.pixel_to_world(self.x, self.y)
        # Radius and PA of each pixel with respect to the NOMINAL center
        self.r = self.center.separation(self.image_coords).to(u.arcsec)
        self.pa = self.center.position_angle(self.image_coords)
        # Find the size of pixels in arcsec
        cdelt = self.wcs.wcs.get_cdelt()
        assert abs(cdelt[0]) == abs(cdelt[1]), "Pixels are not square"
        self.pixel_scale = (abs(cdelt[0]) * u.deg).to(u.arcsec)

    def __init__(self, pdata, hdu, size=0.6 * u.arcsec, core_radius=0.2 * u.arcsec):
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
        self.cutout_around_point(hdu, self.center)
        self.set_masks(r_out=1.5 * core_radius, r_core=core_radius)

        # First pass is to find pixel of peak
        self.find_peak_center()
        # Recalculate everything wrt the peakcenter
        self.cutout_around_point(hdu, self.peakcenter)
        self.set_masks(r_out=1.5 * core_radius, r_core=core_radius)

        # Refine center by flux-weighted mean
        self.find_bary_center()

        # And do photometry
        # For BG take the median in outer ring
        self.bright_bg = np.median(self.image[self.bgmask])
        # And median absolute deviation
        self.mad_bg = np.median(np.abs(self.image[self.bgmask] - self.bright_bg))
        # BG-subtracted flux of the core region
        self.flux_core = np.sum(self.image[self.coremask] - self.bright_bg)
        # Should we correct the flux for any star mask NaN pixels?
        # TODO: perhaps better to interpolate away the holes where the stars were?

        # Peak brightness
        self.bright_peak = np.max(self.image[self.coremask] - self.bright_bg)
        # Effective area of peak is number of pixels at peak brightness that would give the same flux
        self.eff_area = self.flux_core / self.bright_peak
        # And the effective radius, assuming a circle
        self.reff = np.sqrt(self.eff_area / np.pi) * self.pixel_scale
        # Also, save the pixel filling fraction. If this gets near 1,
        # then we need to increase the core radius
        self.fillfrac = self.eff_area / np.sum(self.coremask)

        # Also calculate rms radius weighted by flux (I imagine this will not be reliable)
        self.rms_radius = np.sqrt(
            np.average(
                self.r[self.coremask] ** 2,
                weights=self.image[self.coremask] - self.bright_bg,
            )
        )

    def __repr__(self):
        return f"SourceCutout({self.label})"

    def set_masks(
        self,
        r_out=0.3 * u.arcsec,
        r_core=0.2 * u.arcsec,
    ):
        cth = np.cos((self.pa - self.pa_star))
        self.coremask = self.r <= r_core
        self.bgmask = (self.r <= r_out) & ~self.coremask
        # Also mask out nans
        self.starmask = np.isfinite(self.image)
        # Save the fraction of NaN pixels in the core
        self.nan_frac = np.sum((~self.starmask) & self.coremask) / np.sum(self.coremask)
        self.bgmask &= self.starmask
        self.coremask &= self.starmask

    def find_bary_center(self):
        """Refine estimate of center by using flux-weighted mean
        position within the core mask
        """
        m = self.coremask
        # Original version was flux-weighted mean
        xbary = np.average(self.x[m], weights=self.image[m])
        ybary = np.average(self.y[m], weights=self.image[m])
        self.barycenter = self.wcs.pixel_to_world(xbary, ybary)

    def find_peak_center(self):
        """Find position of peak pixel within the core mask"""
        m = self.coremask
        # New version is just the peak pixel
        index = np.argmax(self.image[m], axis=None)
        xpeak = self.x[m][index]
        ypeak = self.y[m][index]
        self.peakcenter = self.wcs.pixel_to_world(xpeak, ypeak)


def main(
    imagefile: str,
    regionfile: str = "../../m1-67-globules.reg",
    starfile: str = "combo-A-stars.fits",
    star_absolute_threshold: float = 2.0,
    star_relative_threshold: float = 0.2,
    mask_radius_arcsec: float = 0.2,
):
    # Read the image
    hdu = fits.open(imagefile)[0]

    # Mask out pixels dominated by stars
    shdu = fits.open(starfile)[0]
    star_mask = (shdu.data > star_absolute_threshold) & (
        shdu.data > star_relative_threshold * hdu.data
    )
    hdu.data[star_mask] = np.nan

    # Save the masked image
    mfile = imagefile.replace(".fits", "-nostars.fits")
    hdu.writeto(mfile, overwrite=True)
    print(f"Saved masked image (no stars) to {mfile}")

    # Read the region file
    knot_table = get_knot_table(regionfile)

    cutouts = [
        SourceCutout(
            source,
            hdu,
            size=3 * mask_radius_arcsec * u.arcsec,
            core_radius=mask_radius_arcsec * u.arcsec,
        )
        for source in knot_table
    ]
    cutouts = sorted(cutouts, key=lambda x: x.sep.value)

    # Save the coordinates and fluxes as a table
    flux_table = QTable(
        [
            {
                "label": cutout.label,
                "Peak Center": cutout.peakcenter,
                "r_eff": cutout.reff,
                "r_rms": cutout.rms_radius,
                "Bright Peak": cutout.bright_peak,
                "Core flux": cutout.flux_core,
                "Bright BG": cutout.bright_bg,
                "MAD BG": cutout.mad_bg,
                "Fill fraction": cutout.fillfrac,
                "NaN fraction": cutout.nan_frac,
            }
            for cutout in cutouts
        ]
    )
    outfile2 = imagefile.replace(".fits", "-knot-peak-stats.ecsv")
    flux_table.write(outfile2, format="ascii.ecsv", overwrite=True)
    print(f"Saved peak coordinates, sizes, photometry to {outfile2}")

    # Save the barycenters as a region file
    outfile = imagefile.replace(".fits", "-knot-peak-sizes.reg")
    regs = rg.Regions(
        [
            rg.CircleSkyRegion(
                center=cutout.peakcenter,
                radius=cutout.reff if np.isfinite(cutout.reff) else 0.2 * u.arcsec,
                meta={
                    "text": cutout.label,
                },
            )
            for cutout in cutouts
        ]
    )
    regs.write(outfile, format="ds9", overwrite=True)
    print(f"Saved peak regions to {outfile}")


if __name__ == "__main__":
    typer.run(main)
