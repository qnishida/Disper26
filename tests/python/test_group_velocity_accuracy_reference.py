"""Independent identities for the diagnostic high-precision reference.

At 60 decimal digits, require 45 digits of absolute agreement (15 guard
digits). These are reference checks, not relaxed production tolerances.
"""
from mpmath import mp

from group_velocity_accuracy_reference import layer_matrix


def assert_matrix_close(actual, expected):
    assert mp.norm(actual - mp.matrix(expected)) < mp.mpf("1e-45")


def test_love_zero_vertical_wavenumber_limit_and_derivatives():
    with mp.workdps(60):
        p = [2, 2, 1, 2, mp.mpf(".5"), mp.mpf(".5")]
        h, w, c = mp.mpf(".1"), mp.mpf(1), mp.mpf(".5")
        f = lambda ww, cc: layer_matrix(p, h, ww, cc, "Love")
        assert_matrix_close(f(w, c), [[1, -mp.mpf(".2")], [0, 1]])
        assert_matrix_close(mp.diff(lambda x: f(x, c), w), mp.zeros(2))
        assert_matrix_close(mp.diff(lambda x: f(w, x), c),
                            [[-mp.mpf(".08"), mp.mpf(2)/375],
                             [mp.mpf(".8"), -mp.mpf(".08")]])


def test_fluid_zero_vertical_wavenumber_limit_and_derivatives():
    with mp.workdps(60):
        p = [2, mp.mpf(".5"), mp.mpf(".5"), mp.mpf(".5"), 0, 0]
        h, w, c = mp.mpf(".1"), mp.mpf(1), mp.mpf(".5")
        f = lambda ww, cc: layer_matrix(p, h, ww, cc, "Rayleigh")
        assert_matrix_close(f(w, c), [[1, 0], [-mp.mpf(".2"), 1]])
        assert_matrix_close(mp.diff(lambda x: f(x, c), w),
                            [[0, 0], [-mp.mpf(".4"), 0]])
        assert_matrix_close(mp.diff(lambda x: f(w, x), c),
                            [[-mp.mpf(".08"), mp.mpf(".8")],
                             [mp.mpf(2)/375, -mp.mpf(".08")]])


def test_rayleigh_compound_propagator_semigroup_near_vs():
    with mp.workdps(60):
        p = [mp.mpf("2.6"), 60, 20, 55, 25, 26]
        w, c, h = 2*mp.pi, mp.sqrt(p[4]/p[0]), mp.mpf(".4")
        half = layer_matrix(p, h/2, w, c, "Rayleigh")
        assert_matrix_close(layer_matrix(p, 0, w, c, "Rayleigh"), mp.eye(5))
        assert_matrix_close(layer_matrix(p, h, w, c, "Rayleigh"), half*half)
