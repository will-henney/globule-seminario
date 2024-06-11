import numpy as np
from astropy.io import fits
import typer
from typing import Optional
from typing_extensions import Annotated
import sys

from wrutils import (
    sub_bg,
    smooth_by_fwhm,
    find_common_pixel_scale,
    find_extra_pixel_sigma,
)


def main(
    file: Annotated[
        Optional[list[str]],
        typer.Option(
            help=(
                "Fits file of image to include in linear combination. "
                "Must be used at least once."
            )
        ),
    ] = None,
    coeff: Annotated[
        Optional[list[float]],
        typer.Option(
            help=(
                "Coefficient of this image in the linear combination. "
                "There must be one --coeff for each --file argument."
            )
        ),
    ] = None,
    bg: Annotated[
        Optional[list[float]],
        typer.Option(
            help=(
                "Background value for this image. "
                "There must be one --bg for each --file argument."
            )
        ),
    ] = None,
    outfile: str = "linear-combo.fits",
    match_psf: bool = False,
    match_psf_to: str = None,
    debug: bool = False,
):
    """Find the linear combination of a sequence of fits images
    after subtracting respective backgrounds

    Optionally (`--match-psf`) match the PSF of the two images by
    smoothing the images.  By default, smooth to the largest PSF FWHM
    of the two images.  Optionally, match to a specific filter given
    on the command line with, for instance, `--match-psf-to
    jwst-f335m`.
    """
    assert len(file) > 0, "Need at least one --file argument"
    assert (
        len(file) == len(coeff) == len(bg)
    ), "Need one --coeff and one --bg for every --file argument"
    sys.exit()
    hdu_list = [fits.open(f)[0] for f in file]

    pixel_scale = find_common_pixel_scale(hdu_list)

    if debug:
        print(f"Pixel scale: {pixel_scale:.4f} arcseconds")

    if match_psf:
        extras = find_extra_pixel_sigma(
            file, pixel_scale, match_psf_to, debug
        )
        # Smooth the images
        for extra, hdu in zip(extras, hdu_list):
            if extra > 0:
                hdu.data = smooth_by_fwhm(hdu.data, extra)
    result = np.zeros_like(hdu_list[0].data)
    for hdu, c, b in zip(hdu_list, coeff, bg):
        result += c * sub_bg(hdu.data, b)
    
    hdu_list[0].data = result
    hdu_list[0].writeto(outfile, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
