from astropy.table import QTable
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import typer


Gaia.ROW_LIMIT = 10000  # Set the row limit for returned data


def main(
    object_name: str = "ngc 346",
    search_radius_arcsec: float = 60,
):
    """Get Gaia catalog of stars around a given object name."""
    c0 = SkyCoord.from_name(object_name)
    job = Gaia.cone_search_async(c0, radius=search_radius_arcsec * u.arcsec)
    table = job.get_results()
    cols = [
        "source_id",
        "ra",
        "dec",
        "parallax",
        "parallax_error",
        "pmra",
        "pmdec",
        "radial_velocity",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
    ]
    fname = f"gaia-sources-{object_name.replace(' ', '')}.ecsv"
    table[cols].write(fname, format="ascii.ecsv", overwrite=True)


if __name__ == "__main__":
    typer.run(main)
