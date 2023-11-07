"""
Obtain filter throughput curves for JWST and other observatories
"""

import numpy as np
from astropy import units as u
from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.instrument_factory import InstrumentFactory


@u.quantity_input
def get_filter_throughput(
    wave: u.micron,
    filtername: str,
    telescope: str = "jwst",
    instrument: str = "miri",
    mode: str = "imaging",
):
    """
    Return effective transmission of filter `filtername` as a function
    of wavelength `wave`, which should be an `astropy.units.Quantity`
    in any units that can be converted to microns.  Defaults to MIRI
    imaging with JWST, but that can be changed by specifying
    `telescope`, `instrument`, or `mode`.
    """
    calculation = build_default_calc(telescope, instrument, mode)
    config = calculation["configuration"]
    config["instrument"]["filter"] = filtername

    # create a configured instrument
    instrument_factory = InstrumentFactory(config=config)

    # get the throughput of the instrument over the desired wavelength range
    eff = instrument_factory.get_total_eff(wave.to_value(u.micron))

    return eff


if __name__ == "__main__":
    # Simple test of throughput
    wave = [16, 18, 22] * u.micron
    eff = get_filter_throughput(wave, "f1800w")
    print("Transmission of f1800w")
    print(list(zip(wave, eff)))
