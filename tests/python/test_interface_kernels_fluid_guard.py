#!/usr/bin/env python3
"""Interface derivatives are rejected for models with fluid layers.

The interface jump uses the solid spatial Hamiltonian (sigma_zr^2 / L), which
has no fluid form; before this guard both Layer-cake and GRID models returned
nonfinite values while get_capabilities reported the operation as available.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import disper26 as d26  # noqa: E402

MESSAGE = "Interface derivatives with fluid layers are not supported"


def ocean_layer_cake():
    return d26.create_layer_cake_model(
        [d26.create_fluid_parameters(1.5, 1.03), d26.create_isotropic_parameters(3.2, 5.8, 2.6)],
        [4.0, 20.0], d26.create_isotropic_parameters(4.5, 8.1, 3.35))


def solid_layer_cake():
    return d26.create_layer_cake_model(
        [d26.create_isotropic_parameters(3.2, 5.8, 2.6)],
        [20.0], d26.create_isotropic_parameters(4.5, 8.1, 3.35))


def test_layer_cake_fluid_interface_kernels_are_rejected():
    model = ocean_layer_cake()
    rows = [c for c in d26.get_capabilities(model) if c.operation == "interface_kernels"]
    assert rows and all(not c.supported and c.reason == MESSAGE for c in rows)
    solution = d26.compute_dispersion(model, period=7.0)
    try:
        d26.compute_interface_kernels(solution)
    except Exception as error:
        assert MESSAGE in str(error)
    else:
        raise AssertionError("fluid interface kernels must be rejected")


def test_solid_interface_kernels_remain_available():
    model = solid_layer_cake()
    rows = [c for c in d26.get_capabilities(model) if c.operation == "interface_kernels"]
    assert rows and all(c.supported for c in rows)
    solution = d26.compute_dispersion(model, period=7.0)
    values = d26.compute_interface_kernels(solution).gradients["phase_velocity"]
    assert len(values) == 1


if __name__ == "__main__":
    test_layer_cake_fluid_interface_kernels_are_rejected()
    test_solid_interface_kernels_remain_available()
    print("OK")
