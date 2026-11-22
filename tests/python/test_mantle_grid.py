from pathlib import Path
import sys

import numpy as np
import pytest
import disper26

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inversion"))
from disper26_inv.model import ForwardGridModel
from disper26_inv.prem import load_prem_spherical, refine_forward_grid


@pytest.mark.parametrize("loader,options,limit", [
    (disper26.load_prem_spherical, {}, 2891.),
    (load_prem_spherical, {}, 2891.),
    (load_prem_spherical, {"ocean_mode": "drop"}, 2888.),
])
def test_internal_fluid_boundary(loader, options, limit):
    def load(depth):
        return loader(max_depth_km=depth, max_spacing_km=None, **options)
    assert load(limit - .1).depths[-1] < 2891.
    model = load(limit)
    assert model.depths[-1] == 2891.
    vs = (np.sqrt(model.halfspace.L / model.halfspace.rho)
          if hasattr(model, "halfspace") else model.vsv[-1])
    assert vs == pytest.approx(7.22538)
    with pytest.raises(ValueError, match="mantle"):
        load(np.nextafter(limit, np.inf))


def test_refinement_preserves_only_flagged_pairs():
    def model(depths, flags):
        return ForwardGridModel(depths, *([np.ones(len(depths))] * 6), is_discontinuity=flags)
    m = model([0, 10, 10, 20], [False, True, False, False])
    for spacing in (0., 1e-6):
        refined = refine_forward_grid(m, [5, 5, 10], min_spacing_km=spacing)
        np.testing.assert_array_equal(refined.depths, [0, 5, 10, 10, 20])
        np.testing.assert_array_equal(refined.is_discontinuity, [0, 0, 1, 0, 0])
    assert refine_forward_grid(m, [5, 5.01], min_spacing_km=.1).n_nodes == 5
    for depths, flags in [([0, 5, 5], [0, 0, 0]), ([0, 5, 5, 5], [0, 1, 1, 0]),
                          ([0, 5, 10], [0, 1, 0]), ([0, 5], [0, 1])]:
        with pytest.raises(ValueError, match="node .*depth"):
            model(depths, flags)
