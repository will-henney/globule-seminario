"""
Factored out utility functions for working with data in the wr124 images

"""

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve
from astropy.wcs import WCS, WCSCOMPARE_ANCILLARY
from astropy.io import fits

import re

# Regular expression to extract filter wavelength from string such as "f150w"
F_PATTERN = re.compile(
    r"""
    \bf                         # Skip leading f
    (?P<wave>\d{3,4})           # Extract 3 or 4 digits as group "wave"
    [wmn]\b                     # Skip trailing w, m or n
    """,
    re.IGNORECASE | re.VERBOSE,
)


def sub_bg(image, bg):
    """Subtract background and set to NaN all negative values"""
    image_new = image - bg
    image_new[image_new <= 0.0] = np.nan
    return image_new


def guess_filter_wave(fits_file_name: str) -> float:
    """
    We do not have filter information in the header, so we have to
    parse the filename to find it
    """
    # Finally a use for regexps
    try:
        wave = int(F_PATTERN.search(fits_file_name).group("wave"))
    except ValueError:
        # Failed to find wavelength
        return None
    if "jwst" in fits_file_name.lower():
        # All wavelengths in unit of 10 nm or 0.01 micron
        return wave * 0.01
    else:
        # Wavelength in nm
        return wave * 0.001


def find_psf_fwhm(fits_file_name: str) -> float:
    """
    Approximation to the PSF FWHM in arcseconds

    This only takes into account the core of the PSF, not the wings,
    but it should be good enough for matching the resolutions of  list ifofro
    different filters
    """
    wave = guess_filter_wave(fits_file_name)
    if "jwst" in fits_file_name.lower():
        # JWST PSF FWHM in arcseconds Calibrated from table in
        # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
        return 0.066 * (wave / 2.0)
    elif "hst" in fits_file_name.lower():
        # HST PSF FWHM in arcseconds, just from lambda / D
        return 0.06765492 * (wave / 0.656)
    else:
        raise ValueError("Unknown telescope")


def smooth_by_fwhm(image, fwhm):
    """Smooth image by a Gaussian kernel with FWHM in pixels"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    kernel = Gaussian2DKernel(sigma)
    if sigma > 5.0:
        # Use FFT for large kernels
        return convolve_fft(image, kernel)
    else:
        # Use direct convolution for small kernels
        return convolve(image, kernel)


def find_common_pixel_scale(hdus: list[fits.ImageHDU]) -> float:
    """Check that WCS are identical and return pixel scale in arc seconds"""

    # Check that the WCS is identical in the  list ifofro images (necessary so
    # that we can take a ratio without interpolation)
    wlist = [WCS(hdu.header) for hdu in hdus]

    w_a = wlist[0]
    for w_b in wlist[1:]:
        # We need to use the ANCILLARY flag to make sure that the
        # comparison ignores irrelevant differences such as DATE-OBS
        assert w_a.wcs.compare(
            w_b.wcs, cmp=WCSCOMPARE_ANCILLARY, tolerance=1e-6
        ), "WCS are not identical"

    # Find the pixel scale in arc seconds
    cdelt = w_a.wcs.get_cdelt()
    # Check that the pixels are square
    assert abs(cdelt[0]) == abs(cdelt[1]), "Pixels are not square"
    pixel_scale = abs(cdelt[0]) * 3600.0
    return pixel_scale


def find_extra_pixel_sigma(
    file_list: list[str],
    pixel_scale: float,
    match_psf_to: float = None,
    debug: bool = False,
) -> tuple[float, float]:
    """
    Find the extra smoothing necessary to match the PSF FWHM of  list of images
    """

    # Find wavelength of each image
    waves = [guess_filter_wave(f) for f in file_list]
    for wave in waves:
        if wave is None:
            raise ValueError("Failed to guess filter wavelength")
    if debug:
        print("Wavelengths:", waves)
    # Find PSF FWHM for each image
    fwhms = [find_psf_fwhm(f) for f in file_list]
    if match_psf_to is None:
        # Match to the largest FWHM
        fwhm_match = max(fwhms)
    else:
        # Or match to a specific filter given on command line
        fwhm_match = find_psf_fwhm(match_psf_to)
    if debug:
        print(
            "PSF FWHM (pixels):",
            *[f"{f / pixel_scale:.1f}" for f in fwhms],
            f"Match to = {fwhm_match / pixel_scale:.1f}",
        )
    # Quadrature subtraction to find how much to smooth each image (in pixels)
    extras = [np.sqrt(fwhm_match**2 - fwhm**2) / pixel_scale for fwhm in fwhms]
    if debug:
        print("Extra smoothing (pixels):", extras)
    return extras
