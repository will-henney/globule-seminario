from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import typer
import yaml
import warnings


@u.quantity_input
def transform_header(
    hdr: fits.Header,
    origin: SkyCoord,
    offset: u.deg,
    scale: float = 1.0,
    tan_theta: float = 0.0,
) -> fits.Header:
    """Apply coordinate transformation to WCS in a FITS header"""

    # New WCS starts same as original
    new_wcs = WCS(hdr, fix=False)

    # Move reference pixel to the origin
    new_wcs.wcs.crpix = origin.to_pixel(new_wcs, origin=1)
    new_wcs.wcs.crval = origin.ra.deg, origin.dec.deg

    # Shift CRVAL by the offset
    new_wcs.wcs.crval += offset.to_value(u.deg)

    # Deal with non-unit scale
    if scale != 1.0:
        if new_wcs.wcs.has_cd():
            new_wcs.wcs.cd *= scale
        else:
            # If the WCS uses PC instead of CD, then scale is applied to CDELT
            new_wcs.wcs.cdelt *= scale

    # Deal with non-zero rotation
    if tan_theta != 0.0:
        # Calculate rotation matrix
        theta = np.arctan(tan_theta)
        matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        # Update CD or PC matrix for rotation
        if new_wcs.wcs.has_cd():
            new_wcs.wcs.cd = np.dot(matrix, new_wcs.wcs.cd)
        else:
            new_wcs.wcs.pc = np.dot(matrix, new_wcs.wcs.pc)

    return new_wcs.to_header(relax=True)


def scale_and_rotate_from_residual_slopes(
    slopes,
    eslopes,
    nsigma: float = 1.0,
) -> tuple[float, float]:
    """Calculate scale and rotation from slopes of linear fit to residuals"""

    # Extract coefficients
    [[xx, xy], [yx, yy]] = slopes
    [[exx, exy], [eyx, eyy]] = eslopes

    # Even combo of diagonal elements give differences in scale ...
    scale_minus_one = (xx + yy) / 2
    scale_error = np.sqrt(exx**2 + eyy**2)
    # ... and odd combo gives on-axis shear
    onaxis_shear = (xx - yy) / 2
    onaxis_shear_error = np.sqrt(exx**2 + eyy**2)

    # Odd combo of off-diagonal elements give rotation ...
    tan_theta = (xy - yx) / 2
    tan_theta_error = np.sqrt(exy**2 + eyx**2)
    # ... and even combo gives off-axis shear
    offaxis_shear = (xy + yx) / 2
    offaxis_shear_error = np.sqrt(exy**2 + eyx**2)

    # Warn if there is significant shear, but otherwise ignore it
    if np.abs(onaxis_shear) > nsigma * onaxis_shear_error:
        print(
            f"WARNING: On-axis shear {onaxis_shear:.6f}",
            f"(significant at {onaxis_shear / onaxis_shear_error:.3f} sigma)",
        )
    if np.abs(offaxis_shear) > nsigma * offaxis_shear_error:
        print(
            f"WARNING: Off-axis shear {offaxis_shear:.6f}",
            f"(significant at {offaxis_shear / offaxis_shear_error:.3f} sigma)",
        )

    if np.abs(tan_theta) > nsigma * tan_theta_error:
        print(
            f"Rotation {tan_theta:.6f}",
            f"(significant at {tan_theta / tan_theta_error:.3f} sigma)",
        )
    else:
        tan_theta = 0.0

    if np.abs(scale_minus_one) > nsigma * scale_error:
        scale = scale_minus_one + 1
        print(
            f"Scale {scale:.6f}",
            f"(significant at {scale_minus_one / scale_error:.3f} sigma)",
        )
    else:
        scale = 1.0

    return scale, tan_theta


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

    # Calculate scale and rotation
    if offset_only:
        scale, tan_theta = 1.0, 0.0
    else:
        scale, tan_theta = scale_and_rotate_from_residual_slopes(
            transform["coeff_1"],
            transform["e_coeff_1"],
        )
    print(f"Scale: {scale:.6f}, Rotation: {tan_theta:.6f}")

    with warnings.catch_warnings():
        # Ignore annoying warnings about CDELT being ignored
        warnings.filterwarnings(
            "ignore",
            "cdelt will be ignored since cd is present",
            RuntimeWarning,
        )
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
            print(WCS(hdu.header, fix=False))
            hdu.header.update(
                transform_header(hdu.header, origin, offset, scale, tan_theta)
            )
            print(WCS(hdu.header, fix=False))

    # Write the output file
    hdulist.writeto(
        f"{fits_file_prefix}-{output_suffix}.fits",
        overwrite=True,
        output_verify="silentfix",
    )


if __name__ == "__main__":
    typer.run(main)
