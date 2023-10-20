import numpy as np
from astropy.io import fits
import typer


def main(file_a: str, file_b: str, outfile: str, bg_a: float = 0, bg_b: float = 0):
    """Find the ratio of two bg-subtracted fits images"""
    hdu_a = fits.open(file_a)[0]
    hdu_b = fits.open(file_b)[0]
    ratio = (hdu_a.data - bg_a) / (hdu_b.data - bg_b)

    hdu_a.data = ratio
    hdu_a.writeto(outfile, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
