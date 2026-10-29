"""Native-MX component protocol and representation gates; no GPU imports."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT/'scripts/research/ds41_native_mx_components.py'
    spec = importlib.util.spec_from_file_location('_native_mx_components_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cfg():
    return {'native_mx_options': {'repeats': 7, 'projection': 'down', 'actual_rows': [34, 1, 3, 5, 9, 32, 98]}}


def test_options_exact():
    assert load().options(cfg())['repeats'] == 7


@pytest.mark.parametrize('bad', [None, {}, {'extra': 0}, {'repeats': 3}, {'projection': 'gate'},
    {'actual_rows': [34, True, 3, 5, 9, 32, 98]}, {'actual_rows': [1, 3, 5, 9, 32, 34, 98]}])
def test_options_refuse_unknown_changed_or_bool_shapes(bad):
    value = cfg()
    if bad is None: value = {}
    elif not bad: value['native_mx_options'] = {}
    else: value['native_mx_options'].update(bad)
    with pytest.raises(ValueError): load().options(value)


def inventory():
    return [(dict(kind='original', rows=row, expert=i), {'fixture': row})
            for i, row in enumerate((1, 3, 5, 9, 32, 34, 98))]


def test_actual_inventory_m34_first_no_subsets_or_q1():
    cases = inventory()+[(dict(kind='q1', rows=34), {})]
    got = load().select_cases(cases)
    assert [case['rows'] for case, _ in got] == [34, 1, 3, 5, 9, 32, 98]
    assert got[0][1] is cases[5][1]


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'bool'])
def test_actual_inventory_refuses_missing_or_ambiguous_shapes(bad):
    cases = inventory()
    if bad == 'missing': cases.pop()
    elif bad == 'duplicate': cases.append(cases[0])
    else: cases[0][0]['rows'] = True
    with pytest.raises(ValueError): load().select_cases(cases)


def test_half_gate_preserves_signed_zero_and_actual_values():
    value = np.array([[0., -0., 2**-24, -.375, .28125]], np.float32)
    result = load().half_roundtrip(np, value)
    assert result['bitwise_exact'] and result['finite']
    assert result['input_sha256'] == result['restored_sha256']


@pytest.mark.parametrize('value', [np.float32(.1), np.float32(2**-25), np.float32(2**20)])
def test_half_gate_does_not_silently_round(value):
    assert load().half_roundtrip(np, np.array([[value]], np.float32))['bitwise_exact'] is False


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_half_gate_refuses_nonfinite(value):
    with pytest.raises(ValueError): load().half_roundtrip(np, np.array([[value]], np.float32))


def test_coefficient_gate_checks_all_rows_bounded_tiles():
    q = np.full((35, 4), 0x76543210, np.uint32)
    scales = np.full((35, 1), 127, np.uint8)
    result = load().coefficient_gate(np, q, scales)
    assert result['eligible'] and result['inspected_coefficients'] == 35*32
    assert result['maximum_decoded_cpu_tile_bytes'] == 16*32*4


def test_coefficient_gate_refuses_ff_semantic_mismatch_and_nonhalf_range():
    q = np.full((1, 4), 0x77777777, np.uint32)
    assert not load().coefficient_gate(np, q, np.array([[255]], np.uint8))['eligible']
    result = load().coefficient_gate(np, q, np.array([[150]], np.uint8))
    assert not result['eligible'] and result['nonfinite_words'] == 32


def test_bounded_fp64_gate_allows_small_reduction_difference_refuses_geometry_error():
    module = load()
    q = np.full((2, 4), 0x11111111, np.uint32)
    scales = np.full((2, 1), 127, np.uint8)
    x = np.ones((3, 32), np.float32)
    bounds, metadata = module.dot_bounds(np, x, q, scales, input_rows=(0, 2), output_rows=(0, 1))
    assert bounds.shape == (2, 2) and metadata['eps'] == 2**-23
    truth = np.full((2, 2), 16., np.float64)
    candidate = np.nextafter(truth.astype(np.float32), np.float32(np.inf))
    assert module._error(np, truth, candidate, bounds)['within_bound']
    assert not module._error(np, truth, np.full((2, 2), 17., np.float32), bounds)['within_bound']


def test_bounded_fp64_gate_handles_cancellation_and_zero_absolute_floor():
    module = load()
    q = np.full((1, 4), 0x11111111, np.uint32)
    scales = np.full((1, 1), 127, np.uint8)
    x = np.tile(np.array([1., -1.], np.float32), (1, 16))
    bounds, _ = module.dot_bounds(np, x, q, scales, input_rows=(0,), output_rows=(0,))
    assert bounds[0, 0] > 0
    assert module._error(np, np.zeros((1, 1)), np.zeros((1, 1), np.float32), bounds)['within_bound']
    zero_bounds, _ = module.dot_bounds(np, x*0, q, scales, input_rows=(0,), output_rows=(0,))
    assert zero_bounds[0, 0] == np.finfo(np.float32).tiny


def row():
    return dict(capture={'rows': 34}, reference_exact=True, coefficient_identity=True,
        coefficient_gate={'eligible': True}, half_roundtrip={'finite': True, 'bitwise_exact': True},
        native_dispatched=True, native_finite=True, native_stable=True, canary_passed=True,
        native_raw={'exact': False}, native_owner={'exact': False}, fp64_oracle={'reference': {'within_bound': True}, 'native': {'within_bound': True}})


def test_new_reduction_difference_is_not_automatic_engineering_or_quality_failure():
    result = load().verdicts([row()])
    assert all(result.values())
    assert 'quality' not in result and 'raw_exact' not in result


@pytest.mark.parametrize('field', ['reference_exact', 'coefficient_identity', 'native_dispatched', 'native_finite', 'native_stable', 'canary_passed'])
def test_engineering_gates_remain_required(field):
    item = row()
    item[field] = False
    assert not all(load().verdicts([item]).values())


def test_m34_dispatch_required_and_empty_inventory_never_passes():
    item = row()
    item['capture']['rows'] = 32
    assert not load().verdicts([item])['native_dispatched']
    assert not any(load().verdicts([]).values())


def test_actual_fp64_bound_remains_required_without_raw_parity():
    item = row()
    item['fp64_oracle']['native']['within_bound'] = False
    assert not load().verdicts([item])['actual_fp64_bound']


def test_no_gpu_or_device_import_at_module_import():
    module = load()
    tree = ast.parse(Path(module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import): assert all(not name.name.startswith(('mlx', 'Metal')) for name in node.names)
        if isinstance(node, ast.ImportFrom): assert not (node.module or '').startswith(('mlx', 'Metal'))
    text = Path(module.__file__).read_text()
    assert 'MTLCreateSystemDefaultDevice' not in text and 'MTLDevice' not in text
