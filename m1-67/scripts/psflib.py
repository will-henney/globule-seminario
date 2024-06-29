"""
Tools for fitting and removing PSFs from images.


"""

import numpy as np
import lmfit
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.convolution import (
    interpolate_replace_nans, 
    Box2DKernel, 
    Gaussian2DKernel, 
    convolve
)
from astropy.stats import sigma_clip
from regions import PixCoord, CirclePixelRegion, Regions
from reproject import reproject_interp
import webbpsf
from scipy import ndimage as ndi
import skimage as ski
from pathlib import Path

class StarImage():
    """A point source (star) in an image"""
    def __init__(self, im, i0, j0):
        self.im = im 
        self.i0 = i0
        self.j0 = j0

    def cutout(self, size=15):
        return Cutout2D(self.im, (self.i0, self.j0), size=size, copy=True, mode="partial")

    def fit_gaussian(self, initial_sigma=1.5, stamp_size=15, fit=fitting.LMLSQFitter()):
        """
        Fit a 2D Gaussian to a star in an image to find its approximate centroid.
        """
        # Get pixel coordinates of the full image
        yfull, xfull = np.indices(self.im.shape)
        # Cutout stamp around the peak pixel
        imcutout = self.cutout(size=stamp_size)
        xcutout = Cutout2D(xfull, (self.i0, self.j0), size=stamp_size, copy=True, mode="partial")
        ycutout = Cutout2D(yfull, (self.i0, self.j0), size=stamp_size, copy=True, mode="partial")
        # Initialize gaussian model
        g0 = models.Gaussian2D(
            amplitude=self.im[self.j0, self.i0], 
            x_mean=self.i0, y_mean=self.j0, 
            x_stddev=initial_sigma, y_stddev=initial_sigma,
        )
        # Fit model to non-NaN pixels
        m = np.isfinite(imcutout.data)
        g = fit(g0, xcutout.data[m], ycutout.data[m], imcutout.data[m])
        # Stuff extra attributes onto the cut
        self.gaussian = g
        self.gaussian_fit = fit
        self.gaussian_cutout = imcutout
        return g
        
def oversampled_centered_cutout(im, wcsi, xcenter, ycenter, oversample=8, size=15):
    """Produce a cutout from `im` of `size` x `size` pixels oversampled by `oversample`"""
    # Set up the output oversampled cutout wcs
    wcso = WCS()
    wcso.pixel_shape = oversample * size, oversample * size
    wcso.wcs.cdelt = [1 / oversample, 1 / oversample]
    wcso.wcs.crpix = (np.array(wcso.pixel_shape) + 1) / 2
    wcso.wcs.crval = xcenter, ycenter
    wcso.wcs.cunit = ["pixel", "pixel"]
    wcso.wcs.cname = ["x", "y"]
    wcso.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]

    # Get cutout of original pixel grid
    # We add borders of 1 pixel to make sure oversampled grid is fully enveloped
    cutout = Cutout2D(
        im, 
        (xcenter, ycenter), 
        size=size + 2, wcs=wcsi, copy=True, mode="partial",
    )    

    # Reproject onto oversampled grid
    imo = reproject_interp(
        (cutout.data, cutout.wcs), 
        wcso, 
        wcso.pixel_shape, 
        order="nearest-neighbor",
        return_footprint=False,
    )

    # retutn the oversampled image and the original cutout data
    return imo, cutout.data

def coarse_cutout_from_oversampled_psf(psfim, im, wcsi, xcenter, ycenter, oversample=8, order="bilinear"):
    """Reproject `psfim` back to coarse image `im`, centered on `xcenter`, `ycenter`
    
    Also requires coarse-pixel WCS `wcsi` for original image and `oversample` factor.
    Returns cutout that encloses `psfim` on the original image."""

    # Set up the output oversampled cutout wcs
    wcso = WCS()
    wcso.pixel_shape = psfim.shape
    assert wcso.pixel_shape[0] == wcso.pixel_shape[1], "psfim must be square"
    assert wcso.pixel_shape[0] % oversample == 0, "psfim size must be integer multiple of oversample factor"
    size = wcso.pixel_shape[0] // oversample
    wcso.wcs.cdelt = [1 / oversample, 1 / oversample]
    wcso.wcs.crpix = (np.array(wcso.pixel_shape) + 1) / 2
    wcso.wcs.crval = xcenter, ycenter
    wcso.wcs.cunit = ["pixel", "pixel"]
    wcso.wcs.cname = ["x", "y"]
    wcso.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]

    # Get cutout of original pixel grid
    cutout = Cutout2D(
        im, 
        (xcenter, ycenter), 
        size=size, wcs=wcsi, copy=True, mode="partial",
    )    

    # Reproject PSF from oversampled grid back to cutout
    cutout.psfdata = reproject_interp(
        (psfim, wcso), 
        cutout.wcs, 
        (size, size), 
        order=order,
        return_footprint=False,
    )

    # return the cutout with psf attached
    return cutout

def fit_star_centroid(im, i0, j0, initial_sigma=1.5, stamp_size=15, fit=fitting.LMLSQFitter()):
    """
    Fit a 2D Gaussian to a star in an image to find its approximate centroid.
    """
    # Get pixel coordinates of the full image
    yfull, xfull = np.indices(im.shape)
    # Cutout stamp around the peak pixel
    imcutout = Cutout2D(im, (i0, j0), size=stamp_size, copy=True, mode="partial")
    xcutout = Cutout2D(xfull, (i0, j0), size=stamp_size, copy=True, mode="partial")
    ycutout = Cutout2D(yfull, (i0, j0), size=stamp_size, copy=True, mode="partial")
    # Initialize gaussian model
    g0 = models.Gaussian2D(
        amplitude=im[j0, i0], 
        x_mean=i0, y_mean=j0, 
        x_stddev=initial_sigma, y_stddev=initial_sigma,
    )
    # Fit model to non-NaN pixels
    m = np.isfinite(imcutout.data)
    g = fit(g0, xcutout.data[m], ycutout.data[m], imcutout.data[m])
    # Stuff extra attributes onto the cutout stamp
    imcutout.xdata = xcutout.data
    imcutout.ydata = ycutout.data
    # Save image of the fitted model (note we do not use g.render())
    imcutout.fit_data = g(imcutout.xdata, imcutout.ydata)
    return {"cutout": imcutout, "model": g}
