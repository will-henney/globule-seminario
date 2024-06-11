import numpy as np
from astropy.io import fits
import typer
import sys

from wrutils import (
    sub_bg,
    smooth_by_fwhm,
    find_common_pixel_scale,
    find_extra_pixel_sigma,
)


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

    pixel_scale = find_common_pixel_scale([hdu_a, hdu_b])

    if debug:
        print(f"Pixel scale: {pixel_scale:.4f} arcseconds")

    if match_psf:
        extra_a, extra_b = find_extra_pixel_sigma(
            [file_a, file_b], pixel_scale, match_psf_to, debug
        )
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
