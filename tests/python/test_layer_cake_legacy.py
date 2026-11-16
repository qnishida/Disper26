"""The terminology change preserves numerical and model-metadata contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import disper26 as api
import disper26py as raw


@pytest.mark.parametrize('wave', ['Rayleigh', 'Love'])
@pytest.mark.parametrize('basis', ['isotropic', 'velocity_vti', 'stiffness'])
def test_legacy_model_numerical_parity(wave, basis):
    fields = dict(vs=[3., 4.5], vp=[5.2, 8.], rho=[2.5, 3.3])
    if basis != 'isotropic':
        fields = dict(rho=fields['rho'], vph=[5.3, 8.2], vpv=fields['vp'],
                      vsh=[3.1, 4.6], vsv=fields['vs'], eta=[.9, 1.1])
        if basis == 'stiffness':
            materials = [api.create_velocity_vti_parameters(*values)
                         for values in zip(*fields.values())]
            fields = {key: [getattr(m, key) for m in materials]
                      for key in ('rho', 'A', 'C', 'F', 'L', 'N')}
    models = [api.create_model([0., 10.], kind=kind, **fields)
              for kind in ('layer_cake', 'pancake')]
    outputs = []
    for model, kind in zip(models, ('layer_cake', 'pancake')):
        assert model.kind == kind
        solution = api.compute_dispersion(model, period=10., wave=wave,
                                          outputs=('phase_velocity', 'group_velocity', 'hv_ratio'))
        assert solution.model.kind == kind
        outputs.append((solution, api.compute_kernels(solution, basis='stiffness'),
                        api.compute_hessian(solution, basis='stiffness')))
    a, b = outputs
    # Aliases select the identical arithmetic, so existing array-constructor
    # parity tests' exact comparisons also apply here (no tolerance relaxation).
    assert (a[0].phase_velocity, a[0].group_velocity, a[0].hv_ratio) == (
        b[0].phase_velocity, b[0].group_velocity, b[0].hv_ratio)
    np.testing.assert_array_equal(a[1].gradient, b[1].gradient)
    np.testing.assert_array_equal(a[2].hessian, b[2].hessian)


@pytest.mark.parametrize('kind', ['layer_cake', 'pancake'])
@pytest.mark.parametrize('case', ['solid', 'fluid', 'halfspace'])
def test_legacy_constructors_and_transforms(kind, case):
    upper = api.create_isotropic_parameters(3., 5.2, 2.5)
    half = api.create_isotropic_parameters(4.5, 8., 3.3)
    if case == 'fluid':
        upper = api.create_fluid_parameters(1.5, 1.03)
    layers, h = ([], []) if case == 'halfspace' else ([upper], [10.])
    create = getattr(api, 'create_' + kind + '_model')
    model = create(layers, h, half, geometry=api.Geometry('spherical_ba97'), name='legacy test')
    flat = api.earth_flattening_transformation(model)
    assert model.kind == flat.kind == kind
    corrected = api.apply_mineos_physical_dispersion(flat, [1000.]*(len(h)+1),
                    [500.]*(len(h)+1), reference_period=1., period=10.)
    assert corrected.kind == kind and corrected.name == model.name
    assert api.compute_dispersion(corrected, period=10.).model.kind == kind


def test_all_lowlevel_legacy_signatures_and_defaults():
    names = [name for name in dir(raw) if name.endswith('_pancake')]
    assert len(names) == 16
    for old in names:
        new = old.replace('pancake', 'layer_cake')
        old_doc, new_doc = getattr(raw, old).__doc__, getattr(raw, new).__doc__
        assert 'Deprecated alias. Use ' + new in old_doc
        assert old_doc.splitlines()[0].replace(old, new) == new_doc.splitlines()[0]


def test_lowlevel_curve_alias_numerical_parity():
    upper = raw.create_isotropic_parameters(3., 5.2, 2.5)
    half = raw.create_isotropic_parameters(4.5, 8., 3.3)
    for wave in ('Rayleigh', 'Love'):
        args = ([upper], [10.], half, [10., 8.], wave)
        new = raw.compute_dispersion_curve_layer_cake(*args)
        old = raw.compute_dispersion_curve_pancake(*args)
        assert new.keys() == old.keys()
        for key in new:
            np.testing.assert_equal(new[key], old[key])


def test_snapshot_dispatch_only_falls_back_for_missing_name(monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'paper/scripts/generate_kernel_figures.py'
    spec = importlib.util.spec_from_file_location('layer_cake_generator_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = lambda *args, **kwargs: ('legacy', args, kwargs)
    new = lambda *args, **kwargs: ('modern', args, kwargs)
    monkeypatch.setattr(module, 'dp', SimpleNamespace(compute_dispersion_curve_pancake=old), raising=False)
    assert module.layer_cake_curve(1, wave='Love') == ('legacy', (1,), {'wave': 'Love'})
    module.dp.compute_dispersion_curve_layer_cake = new
    assert module.layer_cake_curve(1)[0] == 'modern'
    def fails(*args, **kwargs):
        raise ValueError('calculation error')
    module.dp.compute_dispersion_curve_layer_cake = fails
    with pytest.raises(ValueError, match='calculation error'):
        module.layer_cake_curve(1)


@pytest.mark.parametrize('kind', ['layer_cake', 'pancake'])
def test_invalid_input_contract(kind):
    fields = dict(vs=[3., 4.5], vp=[5.2, 8.], rho=[2.5, 3.3])
    for depths, interpolation in [([0., 0.], None), ([0., 10.], 'constant')]:
        with pytest.raises(api.Error) as error:
            api.create_model(depths, kind=kind, interpolation=interpolation, **fields)
        assert error.value.code == 'invalid_argument'
