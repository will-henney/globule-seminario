"""
Fit an oversampled psf to a star image

The PSF can be read from a file, previously generated either by an
optical model (for example, the OVERDIST extension saved from webbpsf)
or by empirically combining many star images.

The PSF will be shifted, and optionally linearly distorted, and
smoothed in order to obtain the best fit.  Portions of the star image
can be masked out, for instance to find a partial fit to one
particular diffraction spike
"""

import numpy as np
import lmfit
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.convolution import (
    interpolate_replace_nans,
    Box2DKernel,
    Gaussian2DKernel,
    convolve,
)
from reproject import reproject_interp

DEFAULT_MODEL_DICT = {
    "norm": 1,
    "x0": 0,
    "y0": 0,
    "theta": 0,
    "smoothing": 0,
}


class PixelImage:
    """
    A two-dimensional image with WCS information that describes the pixel coordinates
    """

    def __init__(self, data: np.ndarray, wcs: WCS = None):
        self.data = data
        assert len(data.shape) == 2
        if wcs is not None:
            self.wcs = wcs
        else:
            # We set CRPIX = 1 and CRVAL = 0, so it gives us python-style 0-based pixel numbering.
            self.wcs = WCS()
            self.wcs.pixel_shape = self.data.shape
            self.wcs.wcs.crpix = [1, 1]
            self.wcs.wcs.cunit = ["pixel", "pixel"]
            self.wcs.wcs.cname = ["x", "y"]
            self.wcs.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]
        self.j_indices, self.i_indices = np.indices(self.data.shape)


class OversampledPSF:
    """A PSF that is oversampled by a certain factor with respect to a target image

    The PSF has WCS information that allows it to be reprojected onto
    an image at a particular location and orientation

    """

    def __init__(self, data: np.ndarray, oversample: int = 4):
        assert len(data.shape) == 2
        assert data.shape[0] == data.shape[1]
        self.data = data
        self.oversample = oversample
        self.size_detector = data.shape[0] // oversample
        self.center = (0.0, 0.0)
        self.theta = 0.0
        self.smoothing = 0.0
        self.wcs = WCS()
        self.wcs.pixel_shape = self.data.shape
        self.wcs.wcs.cdelt = [1 / self.oversample, 1 / self.oversample]
        self.wcs.wcs.crpix = (np.array(self.wcs.pixel_shape) + 1) / 2
        self.wcs.wcs.crval = self.center
        self.wcs.wcs.cunit = ["pixel", "pixel"]
        self.wcs.wcs.cname = ["x", "y"]
        self.wcs.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]

    def set_center(self, x0, y0):
        self.center = x0, y0
        self.wcs.wcs.crval = self.center

    def set_rotation(self, theta: float):
        """
        Adjust the rotation of the PSF in the image plane

        Counterclockwise rotation by theta in degrees
        """
        self.theta = theta
        cth = np.cos(theta * u.deg)
        sth = np.sin(theta * u.deg)
        self.wcs.wcs.pc = np.array([[cth, -sth], [sth, cth]])

    def apply_smoothing(self, smoothing: float):
        """
        Apply a Gaussian smoothing to the PSF

        Smoothed version is stored in self.sdata
        """
        self.smoothing = smoothing
        if smoothing == 0.0:
            self.sdata = self.data
        else:
            sigma = smoothing * self.oversample
            kernel = Gaussian2DKernel(x_stddev=sigma)
            self.sdata = convolve(self.data, kernel)


class Peak:
    """
    A peak within an image that can be fitted by a model PSF
    """
    # Valid options for reproject_order include "nearest-neighbor", "bilinear", "bicubic"
    # Note that bilinear can give NaNs in the model psf, not sure why
    reproject_order = "bicubic"

    def __init__(self, skyimage: PixelImage, x0: float, y0: float, psf: OversampledPSF):
        self.skyimage = skyimage
        self.x0 = x0
        self.y0 = y0
        self.psf = psf
        self.psf.set_center(x0, y0)
        # Cutout of the sky image that fully encloses the PSF
        self.skycutout = Cutout2D(
            data=skyimage.data,
            position=self.psf.center,
            size=self.psf.size_detector,
            wcs=self.skyimage.wcs,
            mode="partial",
        )
        self.obs_im = self.skycutout.data
        self.mask = np.isfinite(self.skycutout.data)
        # Set initial psf array on the sky grid without any brightness scaling or bg
        self.update_model_image(bscale=1.0, x=self.x0, y=self.y0, bg=0.0)
        # Initial BG guess as 1% percentile
        self.bg0 = np.percentile(self.obs_im[self.mask], 1)
        # Initial guess at the brightness scaling
        self.bscale0 = np.max(self.obs_im[self.mask] - self.bg0) / np.max(
            self.mod_im[self.mask]
        )
        # Apply initial brightness scaling to projected psf
        self.mod_im *= self.bscale0

    def update_model_image(
        self,
        bscale: float,
        x: float,
        y: float,
        bg: float = None,
        theta: float = 0.0,
        smoothing: float = 0.0,
    ):
        """
        Reproject PSF to the sky grid with shift (x, y) and brightness scaling (bscale)

        Plus optional uniform background (bg), rotation (theta), and smoothing

        Sets the mod_im attribute with the reprojected scaled PSF data array
        """
        if bg is None:
            # Allow for pulling the estimated background from the sky image
            bg = self.bg0
        self.psf.set_center(x, y)
        if theta != 0:
            self.psf.set_rotation(theta)
        self.psf.apply_smoothing(smoothing)
        self.mod_im = reproject_interp(
            (self.psf.sdata, self.psf.wcs),
            self.skycutout.wcs,
            self.skycutout.wcs.pixel_shape,
            order=self.reproject_order,
            return_footprint=False,
        )
        self.mod_im *= bscale
        self.mod_im += bg
        # Update the mask to exclude NaNs in the reprojected PSF, just in case
        self.mask = np.isfinite(self.obs_im) & np.isfinite(self.mod_im)


def residual(pars, peak: Peak):
    """
    Compute the residual between the sky cutout and the model PSF
    """
    parvals = pars.valuesdict()
    peak.update_model_image(**parvals)
    assert np.all(np.isfinite(peak.obs_im[peak.mask]))
    assert np.all(np.isfinite(peak.mod_im[peak.mask]))
    # The residual needs to be a flattened array
    return (peak.obs_im - peak.mod_im)[peak.mask]


def fit_peak(peak: Peak, allow_rotation=False, allow_smoothing=False):
    """
    Fit a model PSF to a peak in an image
    """
    # Set up initial values of parameters
    params = lmfit.create_params(
        bscale=peak.bscale0,
        bg=peak.bg0,
        x=peak.x0,
        y=peak.y0,
        theta=0,
        smoothing=0,
    )

    # Set bounds on parameters
    params["x"].set(min=peak.x0 - 0.5, max=peak.x0 + 0.5)
    params["y"].set(min=peak.y0 - 0.5, max=peak.y0 + 0.5)

    if allow_rotation:
        params["theta"].set(min=-5, max=5)
    else:
        params["theta"].set(value=0, vary=False)

    if allow_smoothing:
        params["smoothing"].set(min=0, max=2)
    else:
        params["smoothing"].set(value=0, vary=False)

    # Create Minimizer object
    minner = lmfit.Minimizer(residual, params, fcn_args=(peak,))
    # do the fit
    result = minner.minimize(method="leastsq")
    return result
