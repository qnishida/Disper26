"""Frequency-argument conversions.

The raw disper26py bindings mix conventions: Romberg forwards take period [s],
grid Hessians take frequency [Hz], layer-cake kernels / Hessians / mode counts
take omega [rad/s]. Every call site in this package converts through here so
the package-level API is uniformly period [s].
"""

import math

TWO_PI = 2.0 * math.pi


def period_to_freq(period_s: float) -> float:
    return 1.0 / period_s


def period_to_omega(period_s: float) -> float:
    return TWO_PI / period_s


def omega_to_period(omega: float) -> float:
    return TWO_PI / omega
