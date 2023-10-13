from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
import typer
import yaml


def transform_header(
    hdr: fits.Header,
    offset,
    matrix=None,
) -> fits.Header:
    new_hdr = hdr.copy()

    # Update CRVAL for offset
    new_hdr["CRVAL1"] = hdr["CRVAL1"] + offset[0]
    new_hdr["CRVAL2"] = hdr["CRVAL2"] + offset[1]

    if matrix is None:
        return new_hdr

    # Update CD or PC matrix for rotation, scale, and shear
    try:
        # HST images have CD matrix
        cd = WCS(hdr).wcs.cd
        new_cd = np.dot(matrix, cd)
        (new_hdr["CD1_1"], new_hdr["CD1_2"]), (
            new_hdr["CD2_1"],
            new_hdr["CD2_2"],
        ) = new_cd
    except AttributeError:
        # JWST images have PC matrix together with CDELT
        # But what happens when PC matrix does not sum to 1?
        pc = WCS(hdr).wcs.pc
        new_pc = np.dot(matrix, pc)
        (new_hdr["PC1_1"], new_hdr["PC1_2"]), (
            new_hdr["PC2_1"],
            new_hdr["PC2_2"],
        ) = new_pc

    return new_hdr


def main(
    fits_file_prefix: str,
    yaml_file_prefix: str,
    output_suffix: str = "align",
    offset_only: bool = False,
    object_name: str = "wr124",
):
    """Apply astrometric correction to a FITS image"""

    # Read the FITS HDU list
    hdulist = fits.open(f"{fits_file_prefix}.fits")

    # Read transform from yaml file
    with open(f"{yaml_file_prefix}-OFFSETS-TRANSFORM.yaml") as f:
        transform = yaml.safe_load(f)

    # Calculate offset vector
    offset = np.array(transform["coeff_0"]) * u.milliarcsecond.to(u.deg)
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
        hdu.header.update(transform_header(hdu.header, offset, matrix))
        print(WCS(hdu.header))

    # Write the output file
    hdulist.writeto(f"{fits_file_prefix}-{output_suffix}.fits", overwrite=True)


if __name__ == "__main__":
    typer.run(main)
