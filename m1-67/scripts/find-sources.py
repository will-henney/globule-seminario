import sys
import numpy as np
from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.table import Table
from regions import PixCoord, CirclePixelRegion, Regions

try:
    fname = sys.argv[1]
    fwhm = float(sys.argv[2])
    thresh = float(sys.argv[3])
except:
    sys.exit(f"Usage: {sys.argv[0]} FITSNAME FWHM THRESH")

hdu, = fits.open(fname)

# Set parameters for source finding
daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh)
# Look for the sources
sources = daofind(hdu.data)

# String to indicate the parameters used in the finding
param_id = f"fwhm{fwhm:.1f}-thresh{int(thresh)}"
# Write the source table to a CSV file
tabname = fname.replace(".fits", f"-sources-{param_id}.ecsv")
for col in sources.colnames:  
    sources[col].info.format = '%.8g'
sources.write(tabname, overwrite=True)
print("Source list saved to", tabname) 

# Write the x, y coordinates to a DS9 region file
regs = Regions([CirclePixelRegion(center=PixCoord(x ,y), radius=0.5 * fwhm)
                for x, y in sources["xcentroid", "ycentroid"]])
regname = tabname.replace(".ecsv", ".reg")
regs.write(regname, format="ds9", overwrite=True)
print("Region file saved to", regname)
