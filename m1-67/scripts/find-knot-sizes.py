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
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

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

    def __init__(
        self, pdata, hdu, size=0.6 * u.arcsec, core_radius=0.2 * u.arcsec, bg_frac=0.2
    ):
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
        self.set_masks(r_out=(1 + bg_frac) * core_radius, r_core=core_radius)

        # First pass is to find pixel of peak
        self.find_peak_center()
        # Recalculate everything wrt the peakcenter
        self.cutout_around_point(hdu, self.peakcenter)
        self.set_masks(r_out=(1 + bg_frac) * core_radius, r_core=core_radius)

        # Refine center by flux-weighted mean
        self.find_bary_center()
        # And also subpixel refined peak
        self.find_subpixel_center()

        # And do photometry
        # For BG take the median in outer ring
        self.bright_bg = np.median(self.image[self.bgmask])
        # And median absolute deviation
        self.mad_bg = np.median(np.abs(self.image[self.bgmask] - self.bright_bg))
        # BG-subtracted flux of the core region
        # after first interpolating away the holes where the stars were
        patched_image = interpolate_replace_nans(self.image, Gaussian2DKernel(2))
        # We use the full core mask here, since we have gotten rid of
        # the nans, and we want to make sure we do not underestimate
        # the flux
        self.flux_core = np.sum(patched_image[self.coremask_full] - self.bright_bg)

        # Peak brightness
        self.bright_peak = np.max(self.image[self.coremask] - self.bright_bg)
        # Effective area of peak is number of pixels at peak brightness that would give the same flux
        self.eff_area = self.flux_core / self.bright_peak
        # And the effective radius, assuming a circle
        self.reff = np.sqrt(self.eff_area / np.pi) * self.pixel_scale
        # Also, save the pixel filling fraction. If this gets near 1,
        # then we need to increase the core radius
        self.fillfrac = self.eff_area / np.sum(self.coremask_full)

        # Also calculate rms radius weighted by flux
        self.rms_radius = np.sqrt(
            np.average(
                self.r[self.coremask] ** 2,
                weights=self.image[self.coremask] - self.bright_bg,
            )
        )

        # And finally, fit a gaussian
        self.fit_gaussian_peak()

    def __repr__(self):
        return f"SourceCutout({self.label})"

    def set_masks(
        self,
        r_out=0.24 * u.arcsec,
        r_core=0.2 * u.arcsec,
    ):
        cth = np.cos((self.pa - self.pa_star))
        self.coremask = self.r <= r_core
        self.bgmask = (self.r <= r_out) & ~self.coremask
        # Also mask out nans
        self.starmask = np.isfinite(self.image)
        # Save the fraction of NaN pixels in the core
        self.nan_frac = np.sum((~self.starmask) & self.coremask) / np.sum(self.coremask)
        # Save the complete version of the core mask
        self.coremask_full = self.coremask.copy()
        # But then mask out the NaNs for most purposes
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
        self.xpeak = self.x[m][index]
        self.ypeak = self.y[m][index]
        self.peakcenter = self.wcs.pixel_to_world(self.xpeak, self.ypeak)

    def find_subpixel_center(self):
        """Refine estimate of center by using flux-weighted mean
        position in 3x3 region around peak pixel
        """
        slices_3x3 = slice(self.ypeak - 1, self.ypeak + 2), slice(
            self.xpeak - 1, self.xpeak + 2
        )
        # Original version was flux-weighted mean
        self.xsubpix = np.average(self.x[slices_3x3], weights=self.image[slices_3x3])
        self.ysubpix = np.average(self.y[slices_3x3], weights=self.image[slices_3x3])
        if not np.isfinite(self.xsubpix) or not np.isfinite(self.ysubpix):
            # If the subpixel center is not finite, fall back to the peak center
            self.xsubpix, self.ysubpix = self.xpeak, self.ypeak
        # assert np.abs(self.xsubpix - self.xpeak) <= 0.5
        # assert np.abs(self.ysubpix - self.ypeak) <= 0.5
        self.subpixcenter = self.wcs.pixel_to_world(self.xsubpix, self.ysubpix)

    def fit_gaussian_peak(self):
        """Fit a 2D circular Gaussian to the peak"""
        # Use subpixel center as initial guess
        x0, y0 = self.xsubpix, self.ysubpix
        # Initial guess for sigma is based on larger of rms or effective radius
        sig0 = (
            max(self.rms_radius, self.reff) / self.pixel_scale / np.sqrt(2)
        )  # Make sure it does not have units!
        if not np.isfinite(sig0):
            # If the rms radius is not finite, fall back to unity
            sig0 = 1.0
        g = Gaussian2D(
            amplitude=self.bright_peak,
            x_mean=x0,
            y_mean=y0,
            x_stddev=sig0,
            y_stddev=sig0,
            theta=0.0,
        )
        g.amplitude.bounds = (0.0, None)
        g.theta.fixed = True
        g.x_stddev.bounds = (1.0, 5 * sig0)
        # Enforce circularity
        g.y_stddev.tied = lambda model: model.x_stddev

        # Fit to the union of the core and background masks
        m = (self.coremask | self.bgmask) & np.isfinite(self.image)
        # Fit to the image minus the background because we have
        # eliminated the BG component of the model (this is because I
        # suspect it was encouraging large gaussian widths by reducing
        # the BG intensity)
        self.fitted = FITTER(g, self.x[m], self.y[m], self.image[m] - self.bright_bg)
        self.gauss_center = self.wcs.pixel_to_world(
            self.fitted.x_mean.value, self.fitted.y_mean.value
        )
        self.gauss_sigma = self.fitted.x_stddev.value * self.pixel_scale
        self.gauss_bright = self.fitted.amplitude.value


def main(
    imagefile: str,
    regionfile: str = "../../m1-67-globules.reg",
    starfile: str = "combo-A-stars.fits",
    star_absolute_threshold: float = 2.0,
    star_relative_threshold: float = 0.2,
    mask_radius_arcsec: float = 0.2,
    bg_ring_width_frac: float = 0.2,
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
            bg_frac=bg_ring_width_frac,
        )
        for source in knot_table
    ]
    cutouts = sorted(cutouts, key=lambda x: x.sep.value)

    # Save the coordinates and fluxes as a table
    flux_table = QTable(
        [
            {
                "label": cutout.label,
                "Peak Center": cutout.subpixcenter,
                "Gauss Center": cutout.gauss_center,
                "r_eff": cutout.reff,
                "r_rms": cutout.rms_radius,
                "Gauss sigma": cutout.gauss_sigma,
                "Bright Peak": cutout.bright_peak,
                "Bright Gauss": cutout.gauss_bright,
                "Core flux": cutout.flux_core,
                "Bright BG": cutout.bright_bg,
                "MAD BG": cutout.mad_bg,
                "Fill fraction": cutout.fillfrac,
                "NaN fraction": cutout.nan_frac,
                "Peak SNR": cutout.bright_peak / cutout.mad_bg,
            }
            for cutout in cutouts
        ]
    )
    outfile2 = imagefile.replace(".fits", "-knot-peak-stats.ecsv")
    flux_table.write(outfile2, format="ascii.ecsv", overwrite=True)
    print(f"Saved peak coordinates, sizes, photometry to {outfile2}")

    # Save the r_eff and peak positions as a region file
    outfile = imagefile.replace(".fits", "-knot-peak-reff.reg")
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
    print(f"Saved effective radii of peaks to {outfile}")

    # Repeat for the rms radii
    outfile = imagefile.replace(".fits", "-knot-peak-rrms.reg")
    regs = rg.Regions(
        [
            rg.CircleSkyRegion(
                center=cutout.peakcenter,
                radius=(
                    cutout.rms_radius
                    if np.isfinite(cutout.rms_radius)
                    else 0.2 * u.arcsec
                ),
                meta={
                    "text": cutout.label,
                },
            )
            for cutout in cutouts
        ]
    )
    regs.write(outfile, format="ds9", overwrite=True)
    print(f"Saved rms radii of peaks to {outfile}")

    # Repeat for the gaussian fits
    outfile = imagefile.replace(".fits", "-knot-gauss-fits.reg")
    regs = rg.Regions(
        [
            rg.CircleSkyRegion(
                center=cutout.gauss_center,
                radius=(
                    cutout.gauss_sigma
                    if np.isfinite(cutout.gauss_sigma)
                    else 0.2 * u.arcsec
                ),
                meta={
                    "text": cutout.label,
                },
            )
            for cutout in cutouts
        ]
    )
    regs.write(outfile, format="ds9", overwrite=True)
    print(f"Saved gauss fit regions to {outfile}")


if __name__ == "__main__":
    typer.run(main)
