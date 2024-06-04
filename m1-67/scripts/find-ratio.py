import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve
from astropy.wcs import WCS, WCSCOMPARE_ANCILLARY

import typer
import re
import sys

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
    but it should be good enough for matching the resolutions of two
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


def main(
    file_a: str,
    file_b: str,
    outfile: str,
    bg_a: float = 0,
    bg_b: float = 0,
    match_psf: bool = False,
    match_psf_to: str = None,
    debug: bool = False,
):
    """Find the ratio A/B of two fits images after subtracting
    respective backgrounds

    Optionally (`--match-psf`) match the PSF of the two images by
    smoothing the images.  By default, smooth to the largest PSF FWHM
    of the two images.  Optionally, match to a specific filter given
    on the command line with, for instance, `--match-psf-to
    jwst-f335m`.
    """
    hdu_a = fits.open(file_a)[0]
    hdu_b = fits.open(file_b)[0]

    # Check that the WCS is identical in the two images (necessary so
    # that we can take a ratio without interpolation)
    w_a = WCS(hdu_a.header)
    w_b = WCS(hdu_b.header)
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
    if debug:
        print(f"Pixel scale: {pixel_scale:.4f} arcseconds")

    if match_psf:
        # Find wavelength of each image
        wave_a = guess_filter_wave(file_a)
        wave_b = guess_filter_wave(file_b)
        if wave_a is None or wave_b is None:
            raise ValueError("Failed to guess filter wavelength")
        if debug:
            print(f"Wavelengths: Filter A = {wave_a:.2f} Filter B = {wave_b:.2f}")
        # Find PSF FWHM for each image
        fwhm_a = find_psf_fwhm(file_a)
        fwhm_b = find_psf_fwhm(file_b)
        if match_psf_to is None:
            # Match to the largest FWHM
            fwhm_match = max(fwhm_a, fwhm_b)
        else:
            # Or match to a specific filter given on command line
            fwhm_match = find_psf_fwhm(match_psf_to)
        if debug:
            print(
                "PSF FWHM (pixels):",
                f"Filter A = {fwhm_a / pixel_scale:.1f}",
                f"Filter B = {fwhm_b / pixel_scale:.1f}",
                f"Match to = {fwhm_match / pixel_scale:.1f}",
            )
        # Quadrature subtraction to find how much to smooth each image (in pixels)
        extra_a = np.sqrt(fwhm_match**2 - fwhm_a**2) / pixel_scale
        extra_b = np.sqrt(fwhm_match**2 - fwhm_b**2) / pixel_scale
        # Smooth the images
        if extra_a > 0:
            hdu_a.data = smooth_by_fwhm(hdu_a.data, extra_a)
        if extra_b > 0:
            hdu_b.data = smooth_by_fwhm(hdu_b.data, extra_b)

    ratio = sub_bg(hdu_a.data, bg_a) / sub_bg(hdu_b.data, bg_b)

    hdu_a.data = ratio
    hdu_a.writeto(outfile, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
