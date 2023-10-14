from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import typer
import yaml


@u.quantity_input
def transform_header(
    hdr: fits.Header,
    origin: SkyCoord,
    offset: u.deg,
    matrix=None,
) -> fits.Header:
    """Apply coordinate transformation to WCS in a FITS header"""

    # Original WCS
    wcs = WCS(hdr)
    # New WCS starts same as original
    new_wcs = wcs.deepcopy()

    # Move reference pixel to the origin
    new_wcs.wcs.crpix = origin.to_pixel(wcs, origin=1)
    new_wcs.wcs.crval = origin.ra.deg, origin.dec.deg

    # Shift CRVAL by the offset
    new_wcs.wcs.crval += offset.to_value(u.deg)

    if matrix is not None:
        # Update CD or PC matrix for rotation, scale, and shear
        try:
            # HST images have CD matrix
            cd = WCS(hdr).wcs.cd
            new_wcs.wcs.cd = np.dot(matrix, new_wcs.wcs.cd)
        except AttributeError:
            # JWST images have PC matrix together with CDELT
            # But what happens when PC matrix does not sum to 1?
            new_wcs.wcs.pc = np.dot(matrix, new_wcs.wcs.pc)

    return new_wcs.to_header()


def main(
    fits_file_prefix: str,
    yaml_file_prefix: str,
    output_suffix: str = "align",
    offset_only: bool = False,
    object_name: str = "wr124",
):
    """Apply astrometric correction to a FITS image"""

    # Take origin as canonical coordinates of object
    origin = SkyCoord.from_name(object_name)

    # Read the FITS HDU list
    hdulist = fits.open(f"{fits_file_prefix}.fits")

    # Read transform from yaml file
    with open(f"{yaml_file_prefix}-OFFSETS-TRANSFORM.yaml") as f:
        transform = yaml.safe_load(f)

    # Calculate offset vector
    offset = np.array(transform["coeff_0"]) * u.milliarcsecond
    print(f"Offset: {offset}")

    # Calculate transform matrix
    if offset_only:
        matrix = None
    else:
        matrix = np.array(transform["coeff_1"]) + np.eye(2)
    print(f"Matrix: {matrix}")

    for hdu in hdulist:
        if not type(hdu) in [fits.ImageHDU, fits.CompImageHDU]:
            # Skip anything that is not an image
            continue
        if not hdu.name in [
            "SCI",
        ]:
            # Only do SCI for now since other have no WCS in JWST files
            continue
        print(hdu.name)
        print(WCS(hdu.header))
        hdu.header.update(transform_header(hdu.header, origin, offset, matrix))
        print(WCS(hdu.header))

    # Write the output file
    hdulist.writeto(f"{fits_file_prefix}-{output_suffix}.fits", overwrite=True)


if __name__ == "__main__":
    typer.run(main)
