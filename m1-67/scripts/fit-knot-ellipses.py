"""
Fit an elliptical Gaussian to the brigthess profile of the cores of the knots from a given image

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

        # First pass is to find pixel of peak
        self.find_peak_center()
        # Recalculate r, pa, and masks wrt the peakcenter
        self.r = self.peakcenter.separation(self.image_coords)
        self.pa = self.peakcenter.position_angle(self.image_coords)
        # Recalculate mask around peak center, allowing 360 deg coverage for the outer mask
        self.set_mask(r_out=self.size / 2, mu_min=-1.0)
        # Refine center by flux-weighted mean
        self.find_bary_center()

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

        # And fit an ellipse to the core
        self.fit_elliptical_peak()

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
        # Also mask out nans
        self.mask &= np.isfinite(self.image)
        self.coremask &= np.isfinite(self.image)

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

    def fit_elliptical_peak(self):
        """Fit a 2D Gaussian to the peak"""
        # Find pixel coords of barycenter as initial guess
        x0, y0 = self.wcs.world_to_pixel(self.barycenter)
        # Initial guess for sigma is 3 pixels, or about 0.1 arcsec
        sig0 = 3.0
        g = Gaussian2D(
            amplitude=self.bright_peak,
            x_mean=x0,
            y_mean=y0,
            x_stddev=sig0,
            y_stddev=sig0,
            theta=self.pa_source.deg,
        )
        g.theta.fixed = True
        g.x_stddev.bounds = (1.0, None)
        g.y_stddev.bounds = (1.0, None)

        c = Const2D(self.bright_bg)
        c.amplitude.bounds = (0.1 * self.bright_bg, 10 * self.bright_bg)

        model = g + c
        m = self.mask & np.isfinite(self.image)
        self.fitted = FITTER(model, self.x[m], self.y[m], self.image[m])
        self.ellipse_center = self.wcs.pixel_to_world(
            self.fitted.x_mean_0.value, self.fitted.y_mean_0.value
        )
        # Find the sigma in arcsec
        cdelt = self.wcs.wcs.get_cdelt()
        assert abs(cdelt[0]) == abs(cdelt[1]), "Pixels are not square"
        pixel_scale = (abs(cdelt[0]) * u.deg).to(u.arcsec)
        self.ellipse_sigma_x = self.fitted.x_stddev_0.value * pixel_scale
        self.ellipse_sigma_y = self.fitted.y_stddev_0.value * pixel_scale
        self.ellipse_theta = self.fitted.theta_0.value
        self.ellipse_bright = self.fitted.amplitude_0.value
        self.ellipse_bg = self.fitted.amplitude_1.value


def main(
    imagefile: str,
    regionfile: str = "../../m1-67-globules.reg",
    starfile: str = "combo-A-stars.fits",
    star_absolute_threshold: float = 2.0,
    star_relative_threshold: float = 0.2,
    mask_radius_arcsec: float = 0.75,
):
    # Read the image
    hdu = fits.open(imagefile)[0]

    # Mask out pixels dominated by stars
    shdu = fits.open(starfile)[0]
    star_mask = (shdu.data > star_absolute_threshold) & (
        shdu.data > star_relative_threshold * hdu.data
    )
    hdu.data[star_mask] = np.nan

    # Read the region file
    knot_table = get_knot_table(regionfile)

    cutouts = [
        SourceCutout(source, hdu, size=2 * mask_radius_arcsec * u.arcsec)
        for source in knot_table
    ]
    cutouts = sorted(cutouts, key=lambda x: x.sep.value)

    # Save the barycenters as a region file
    outfile = imagefile.replace(".fits", "-knot-ellipses.reg")
    regs = rg.Regions(
        [
            rg.EllipseSkyRegion(
                center=cutout.ellipse_center,
                width=cutout.ellipse_sigma_x * 2.0,
                height=cutout.ellipse_sigma_y * 2.0,
                angle=cutout.ellipse_theta * u.deg,
                meta={
                    "text": cutout.label,
                },
            )
            for cutout in cutouts
        ]
    )
    regs.write(outfile, format="ds9", overwrite=True)
    print(f"Saved ellipse regions to {outfile}")

    # Save the coordinates and fluxes as a table
    flux_table = QTable(
        [
            {
                "label": cutout.label,
                "Ellipse Center": cutout.ellipse_center,
                "Ellipse Sigma X": cutout.ellipse_sigma_x,
                "Ellipse Sigma Y": cutout.ellipse_sigma_y,
                "Ellipse Theta": cutout.ellipse_theta,
                "Ellipse Bright Peak": cutout.ellipse_bright,
                "Ellipse Bright BG": cutout.ellipse_bg,
            }
            for cutout in cutouts
        ]
    )
    outfile2 = imagefile.replace(".fits", "-knot-ellipses.ecsv")
    flux_table.write(outfile2, format="ascii.ecsv", overwrite=True)
    print(f"Saved ellipse coordinates, sizes, angles, brightnesses to {outfile2}")


if __name__ == "__main__":
    typer.run(main)
