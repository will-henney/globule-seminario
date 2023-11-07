from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import typer
import astropy.units as u
from astropy.coordinates import SkyCoord


def get_first_image_hdu(hdulist: fits.HDUList) -> fits.ImageHDU:
    """Return the first HDU with a 2D image"""
    for hdu in hdulist:
        if hdu.data is not None and len(hdu.data.shape) == 2:
            return hdu
    raise ValueError("No HDU with data array found")


@u.quantity_input
def get_reference_hdu(
    origin: SkyCoord,
    angular_size: u.deg,
    pixel_scale: u.deg,
) -> fits.PrimaryHDU:
    """Return a FITS hdu for a square ra-dec grid centered on origin with blank image"""

    # Compute the size of the image in pixels
    n = int(round(float(angular_size / pixel_scale)))

    # Create a WCS object
    wcs = WCS(naxis=2)
    # Set the pixel scale
    pixel_scale_deg = pixel_scale.to_value(u.deg)
    wcs.wcs.cdelt = np.array([-pixel_scale_deg, pixel_scale_deg])
    # Set the PC matrix to identity
    wcs.wcs.pc = np.eye(2)
    # Set the origin
    wcs.wcs.crpix = (n + 1) / 2.0, (n + 1) / 2.0
    # Set the reference pixel
    wcs.wcs.crval = np.array([origin.ra.deg, origin.dec.deg])
    # Set the coordinate system
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Return the HDU
    return fits.PrimaryHDU(
        header=wcs.to_header(),
        data=np.zeros((n, n), dtype=np.float32),
    )


def main(
    input_file: str,
    output_file: str,
    object_name: str = "wr124",
    pixel_scale_arcsec: float = 0.1,
    angular_size_arcmin: float = 3.0,
):
    """Reproject to ra-dec grid centered on object_name"""

    # Take origin as canonical coordinates of object
    origin = SkyCoord.from_name(object_name)

    if not input_file.endswith(".fits"):
        input_file += ".fits"
    if not output_file.endswith(".fits"):
        output_file += ".fits"

    # Read the FITS HDU list
    hdu1 = get_first_image_hdu(fits.open(input_file))
    # Get the reference HDU
    hdu0 = get_reference_hdu(
        origin, angular_size_arcmin * u.arcmin, pixel_scale_arcsec * u.arcsec
    )

    #
    # Reproject hdu1 to WCS of hdu0
    #
    # First copy the original header to get the non-WCS keywords
    header2 = hdu1.header.copy()
    # But we need to purge any existing CD and PC keywords
    del header2["CD*"]
    del header2["PC*"]
    # And any SIP distortion keywords
    del header2["A_*"]
    del header2["B_*"]

    # Then update the WCS keywords
    header2.update(hdu0.header)
    # Then reproject the data
    data2 = reproject_interp(hdu1, hdu0.header, return_footprint=False).astype(
        np.float32
    )
    # Then create a new HDU
    hdu2 = fits.PrimaryHDU(
        header=header2,
        data=data2,
    )
    # Save reprojected file
    hdu2.writeto(output_file, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
