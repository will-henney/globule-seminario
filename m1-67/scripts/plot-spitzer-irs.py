import numpy as np
from astropy.table import Table
import astropy.units as u
from matplotlib import pyplot as plt
from filter_throughput import get_filter_throughput

fig, ax = plt.subplots(figsize=(15, 5))
for spec in "SH", "SL1", "SL2", "LH", "LL1", "LL2":
    tab = Table.read(f"wr124_{spec}_complete.tbl", format="ascii.ipac")
    w = tab["WAVELENGTH"]
    # (F_nu / wave) is proportional to SED
    f = tab["FLUX"] / w
    if "H" in spec:
        f /= 10
        lw = 0.4
    else:
        lw = 1.0
    ax.plot(w, f, lw=lw)

wave = np.linspace(5, 25, 500) * u.micron
for fname in "f770w", "f1130w", "f1280w", "f1800w":
    eff = get_filter_throughput(wave, fname)
    wmean = np.average(wave.value, weights=eff)
    ax.fill_between(wave.value, 4 * eff, alpha=0.2)
    ax.text(wmean, 0.1, fname, ha="center")
ax.set(
    xlim=[5, 21],
    ylim=[0, 2.0],
    xlabel="Wavelength, micron",
    ylabel=(
        r"Brightness SED, $\nu F_\nu = \lambda F_\lambda$,"
        "\n"
        r"or Filter throughput, $T_\lambda$,"
        "\n"
        "(Arbitrary units)"
    ),
    # yscale="log",
    # xscale="log",
)
ax.set_title("M1-67: Spitzer IRS spectra compared with JWST MIRI filters")
ax.set_xticks(range(5, 23))
ax.grid(axis="x", which="major", linewidth=0.5)
ax.minorticks_on()
ax.grid(axis="x", which="minor", linewidth=0.1)
figfile = "irs-spec.pdf"
fig.savefig(figfile)
print(figfile, end="")
