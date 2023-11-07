import numpy as np
from astropy.io import fits
import typer


def sub_bg(image, bg):
    """Subtract background and set to NaN all negative values"""
    image_new = image - bg
    image_new[image_new <= 0.0] = np.nan
    return image_new


def main(file_a: str, file_b: str, outfile: str, bg_a: float = 0, bg_b: float = 0):
    """Find the ratio A/B of two fits images after subtracting respective backgrounds"""
    hdu_a = fits.open(file_a)[0]
    hdu_b = fits.open(file_b)[0]
    ratio = sub_bg(hdu_a.data, bg_a) / sub_bg(hdu_b.data, bg_b)

    hdu_a.data = ratio
    hdu_a.writeto(outfile, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
