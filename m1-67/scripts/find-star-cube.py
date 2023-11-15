from __future__ import annotations
from typing import Optional
import typer

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u


def main(
    prefix: str,
    mid_pass_suffix: str = "median-cont-021-median-csub-151",
    star_name: str = "wr124",
    radius_arcsec: float = 1.0,
    hdu_key: str = "SCI",
    hdu_index: Optional[int] = None,
):
    """Fit and remove stellar psf from spectral cube"""
    cube_path = Path(f"{prefix}-{mid_pass_suffix}.fits")

    hdulist = fits.open(cube_path)
    if hdu_index is not None:
        hdu = hdulist[hdu_index]
    else:
        hdu = hdulist[hdu_key]

    # The data cube
    cube = hdu.data
    nv, ny, nx = cube.shape

    # Construct coordinate array
    origin = SkyCoord.from_name(star_name)
    wcs = WCS(hdu.header)
    coords = wcs.celestial.pixel_to_world(*np.meshgrid(range(nx), range(ny)))
    radii = origin.separation(coords)
    # All pixels within a certain readius of the star
    star_mask = radii < radius_arcsec * u.arcsec

    star_spec = np.mean(cube)


if __name__ == "__main__":
    typer.run(main)
