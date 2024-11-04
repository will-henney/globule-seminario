# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import uncertainties
from uncertainties import unumpy as unp
import astropy.units as u
import numpy as np

# We want to check the interaction between uncertainties (unumpy) and astropy.units

x = unp.uarray(1 + np.arange(4), np.ones((4)))
xx = unp.uarray(1 + np.arange(4), np.ones((4)))
x

x**2, x * xx

x / x, xx / x

y = x * (1.0 * u.cm)
yy = xx * (1.0 * u.cm)

y, yy

y**2, y * yy

y / y, yy / y

z = y * yy

uncertainties.correlation_matrix([y.value[3], z.value[3]])

unp.nominal_values(z.value), unp.std_devs(z.value)

# Test with functions

xxx = unp.degrees(unp.arctan(x))

xxx

xxx = unp.degrees(unp.arctan(y.value))

y.value

unp.nominal_values(xxx)


