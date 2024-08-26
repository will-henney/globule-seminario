---
jupyter:
  jupytext:
    formats: md,ipynb,py:light
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import typer
from astropy.table import Table, QTable, join
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

from matplotlib import pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
```

```python
ORIGIN = SkyCoord.from_name("wr124", cache=True)
```

```python
from pathlib import Path
```

```python
datapath = Path.cwd().parent / "data" / "reproject"
```

```python
prefix = "wr124-hst-2008-f656n-radec"
table = QTable.read(datapath / f"{prefix}-knot-fluxes.ecsv")
```

```python
table["Peak minus BG"] = table["Bright Peak"] - table["Bright BG"]
```

```python
table
```

```python
so.Plot.config.theme.update(sns.axes_style("whitegrid"))
so.Plot.config.theme.update(sns.plotting_context("talk"))
so.Plot.config.display["format"] = "svg"
```

```python
(
    so.Plot(table.to_pandas(), x="Sep", y="Core Flux", color="PA")
    .add(so.Dot())
    .layout(size=(8, 6), engine="constrained")
)
```

```python
fig, ax = plt.subplots(figsize=(6, 4))
g = (
    so.Plot(table.to_pandas(), x="Sep", y="Bright Peak")
    .on(ax)
    .add(so.Dot(), color="PA")
    .scale(x="log", y="log")
    .limit(x=(2, 70.0), y=(0.05, 5.0))
    .layout(engine="constrained")
    .plot()
)
bmax = 2.2
rmin, rmax = 17.0, 70.0
ax.axhline(bmax, lw=2, ls=":")
ax.plot([rmin/3, rmax], [bmax * 3**2, bmax * (rmin/rmax)**2], lw=2, ls=":")
for i in 30, 45, 60, 75:
    cosi = np.cos(i * u.deg)
    ax.plot([cosi * rmin, cosi * rmax], [np.sqrt(cosi) * bmax, np.sqrt(cosi) * bmax * (rmin/rmax)**2], lw=1, ls="--")
sns.despine()
...;
```

```python
fig, ax = plt.subplots(figsize=(6, 4))
g = (
    so.Plot(table.to_pandas(), x="Sep", y="Peak minus BG")
    .on(ax)
    .add(so.Dot(), color="PA")
    .scale(x="log", y="log")
    .limit(x=(2, 70.0), y=(0.03, 3.0))
    .layout(engine="constrained")
    .plot()
)
bmax = 1.8
rmin, rmax = 17.0, 70.0
ax.axhline(bmax, lw=2, ls=":")
ax.plot([rmin/3, rmax], [bmax * 3**2, bmax * (rmin/rmax)**2], lw=2, ls=":")
for i in 30, 45, 60, 75:
    cosi = np.cos(i * u.deg)
    ax.plot([cosi * rmin, cosi * rmax], [np.sqrt(cosi) * bmax, np.sqrt(cosi) * bmax * (rmin/rmax)**2], lw=1, ls="--")
sns.despine()
...;
```

This shows that the max brightness envelope is roughly constant for projected separations less than about 20 arcsec, and thereafter falls like $r^{-2}$ (blue dotted lines). But only about 0.3 decades in radius. 

The dashed lines show effects of varying the inclination $|i|$ to 30, 45, 60, 75 deg. Effects on both the separation and the brightness (assuming a $\cos^{1/2} i$ dependence). Of course, for high inclination, this will bottom out for face-on globules ($i < 0$) as we will see the apex region with no limb-brightening as a floor on the max brightness. But for tail-on globules ($i > 0$), we will only see the limb-brightened rim, so the $\cos^{1/2} i$ behavior will continue indefinitely. 

So, it looks like we should be able to derive the inclination of individual knots from this diagram. 

- [ ] And compare them with the kinematics-derived inclinations
- [ ] We should do simple models of the brightness profiles
- [ ] And we should color according to the spatial groups
- [ ] We need to work out the theoretical curves for the peak brightnesses. This might be best done by expressing the brightness in terms of the mean surface brightness of the nebula, and the separations in terms of the Stromgren radius. That would avoid having to know the distance to the nebula or the ionizing luminosity, but it does assume that the diffuse nebula is ionization bounded in at least some directions, so we can know the Stromgren radius. 

There are two more factors that might reduce the brightnesses and so complicate the assignment of inclination based on this diagram. 

1. Small globules will have the peak reduced by the psf (beam dilution). 
2. Globules in a group that are slightly further from the star may be shadowed by those that are closer in.

Both these could be ameliorated by considering a subset of knots that are large and at the forefront of their respective group. 

***Actually, all this is flawed because I have not subtracted the BG brightness*** Although now I have corrected that and it does not make a lot of difference


```python
(
    so.Plot(table.to_pandas(), x="Core Flux", y="Bright Peak")
    .add(so.Dot(), color="PA", pointsize="Sep")
    .add(so.Line(), so.PolyFit())
    .layout(size=(8, 6), engine="constrained")
)
```

```python
(
    so.Plot(table.to_pandas(), x="Bright BG", y="Bright Peak", color="PA", pointsize="Sep")
    .add(so.Dot())
    .layout(size=(8, 6), engine="constrained")
)
```

```python
prefix = "combo-D-neutral"
ntable = QTable.read(datapath / f"{prefix}-knot-fluxes.ecsv")
```

```python
ntable
```

```python
fig, ax = plt.subplots(figsize=(6, 4))
g = (
    so.Plot(ntable.to_pandas(), x="Sep", y="Bright Peak")
    .on(ax)
    .add(so.Dot(), color="PA")
    .scale(x="log", y="log")
    .limit(x=(2, 70.0), y=(0.1, 100.0))
    .layout(engine="constrained")
    .plot()
)
bmax = 10
power_law = 2
rmin, rmax = 17.0, 70.0
ax.axhline(bmax, lw=2, ls=":")
ax.plot([rmin/3, rmax], [bmax * 3**power_law, bmax * (rmin/rmax)**power_law], lw=2, ls=":")
for i in 30, 45, 60, 75:
    cosi = np.cos(i * u.deg)
    ax.plot([cosi * rmin, cosi * rmax], [bmax/cosi, (bmax/cosi) * (rmin/rmax)**power_law], lw=1, ls="--")
sns.despine()
...;
```

With the neutral emission, the variation with inclination ahould be very diffferent. If the knots have long tails, then the path length through the tail should increase as $1/\cos i$.  Although this would saturate for high inclinations when you run out of tail. And also would be lessened by the fact that the heads are brighter than the tails. 

So, if all the globules were at the same radius, then this would give brightness going as projected radius as $1/R$, which is more or less what we see for the general trend. 

However, this is completely inconsistent with the good positive correlation between PAH and Ha brightness. It would predict a **negative** correlation, so it cannot be the dominant effect. 

```python
(
    so.Plot(ntable.to_pandas(), x="Core Flux", y="Bright Peak", color="PA", pointsize="Sep")
    .add(so.Dot())
    .layout(size=(8, 6), engine="constrained")
)
```

```python
ntable.remove_columns("Center	Peak	PA	Sep	".split())
```

```python
ntable
```

```python
etable = QTable.read(datapath / f"{prefix}-knot-ellipses.ecsv")
etable
```

```python

```

```python

```

```python
df = join(join(table, ntable, keys="label", table_names=["ha", "pah"]), etable, keys="label").to_pandas()
```

```python
df.columns
```

```python
interesting_columns = [	"Sep",	"Core Flux_ha", "Bright Peak_ha", "Core Flux_pah", "Bright Peak_pah", "Ellipse Sigma X", "Ellipse Bright Peak", "Peak minus BG"]
df[interesting_columns].describe()
```

```python
cdf = df[interesting_columns].corr()
sol = (cdf.where(np.triu(np.ones(cdf.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=False, key=np.abs))
sol.head(25)
```

So the highest correlations are between the flux and the brightness, suggesting that there is very little variation in the sizes. 

However, this might be just because the core mask has a fixed small radius. Also, now that we have included the ellipse table, we see that there is a correlation between ellipse size and ellipse brightness, but strangely not between ellipse size and the pixel-derived Bright Peak_pah

```python
(
    so.Plot(df, x="Bright Peak_ha", y="Bright Peak_pah")
    .add(so.Dot(), color="PA", pointsize="Sep")
    .add(so.Line(), so.PolyFit(order=1))
    .scale(x="log", y="log")
    .limit(x=(0.01, 25.0), y=(0.08, 200.0))
    .layout(size=(8, 8), engine="constrained")
)
```

```python
(
    so.Plot(df, x="Peak minus BG", y="Bright Peak_pah")
    .add(so.Dot(), color="PA", pointsize="Sep")
    .add(so.Line(), so.PolyFit(order=1))
    .scale(x="log", y="log")
    .limit(x=(0.01, 25.0), y=(0.08, 200.0))
    .layout(size=(8, 8), engine="constrained")
)
```

This is the correlation between Ha peak brightness and neutral peak brightness, which is pretty tight. This is strong evidence that they are both varying due to the same cause, which is presumably incident radiation field strength, falling as $R^{-2}$ with true distance from the star. 

However, the relation is non-linear, with a variation of nearly two orders of magnitude in PAH, but only just over an order of magnitude in H alpha.

```python
(
    so.Plot(df, x="Core Flux_ha", y="Core Flux_pah")
    .add(so.Dot(), color="PA", pointsize="Sep")
    .add(so.Line(), so.PolyFit(order=1))
    .scale(x="log", y="log")
    .limit(x=(0.8, 2000), y=(5.0, 12500))
    .layout(size=(8, 8), engine="constrained")
)
```

```python
(
    so.Plot(df, x="Ellipse Bright Peak", y="Bright Peak_pah")
    .add(so.Dot(), color="PA", pointsize="Sep")
    .add(so.Line(), so.PolyFit(order=1))
    .scale(x="log", y="log")
    .limit(x=(0.08, 200.0), y=(0.08, 200.0))
    .layout(size=(8, 8), engine="constrained")
)
```

```python
(
    so.Plot(df, x="Ellipse Bright Peak", y="Ellipse Sigma X")
    .add(so.Dot(), color="PA", pointsize="Sep")
    .add(so.Line(), so.PolyFit(order=1))
    .scale(x="log", y="log")
    .limit(x=(0.08, 200.0), y=(None, None))
    .layout(size=(8, 8), engine="constrained")
)
```

```python
table.show_in_notebook()
```

```python
etable.show_in_notebook()
```

```python

```
