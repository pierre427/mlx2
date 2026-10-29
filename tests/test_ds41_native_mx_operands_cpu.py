import importlib.util
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/research/ds41_native_mx_operands.py'
spec = importlib.util.spec_from_file_location('native_mx_operands_cpu', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_no_mlx_import():
    subprocess.run([sys.executable, '-c',
        "import runpy,sys;runpy.run_path(sys.argv[1]);assert 'mlx.core' not in sys.modules", str(PATH)], check=True)


def test_half_roundtrip_signed_zero_exact_and_loss():
    x = np.asarray([[0., -0., .5, -6.]], np.float32)
    assert m.half_roundtrip(np, x)['exact']
    x[0, 2] = np.float32(.5001)
    r = m.half_roundtrip(np, x)
    assert r['changed_words'] == 1 and r['first_difference']['coordinate'] == [0, 2]


def test_half_roundtrip_overflow_is_not_admitted():
    r = m.half_roundtrip(np, np.asarray([[1e10]], np.float32))
    assert not r['exact'] and not r['half_finite']


def test_installed_e8m0_specials_not_standard_nan_assumption():
    s = np.asarray([0, 1, 127, 254, 255], np.uint8)
    assert m.decode_scales(np, s).view(np.uint32).tolist() == [0x00400000, 0x00800000, 0x3f800000, 0x7f000000, 0x7f800000]


def test_all_nibbles_and_row_scale_stride():
    q = np.asarray([[0x76543210, 0xfedcba98, 0x76543210, 0xfedcba98],
                    [0x76543210, 0xfedcba98, 0x76543210, 0xfedcba98]], np.uint32)
    s = np.asarray([[127], [128]], np.uint8)
    a = m.decode_rows(np, q, s, (0, 1))
    expected = np.asarray([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]*2,np.float32)
    assert np.array_equal(a[0].view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(a[1], expected*2)


def test_oracle_uses_original_input_and_f64_output():
    q = np.full((2, 4), 0x22222222, np.uint32)
    s = np.asarray([[127], [128]], np.uint8)
    x = np.full((3, 32), np.float32(.5001), np.float32)
    result = m.oracle(np, x, q, s, output_rows=(0, 1), input_rows=(2, 0))
    assert result.dtype == np.float64 and result.shape == (2, 2)
    assert result[0, 0] == float(x[0, 0])*32
    assert result[0, 1] == result[0, 0]*2


@pytest.mark.parametrize('rows', [(), (-1,), (2,), tuple(range(17))])
def test_output_extent_refuses(rows):
    with pytest.raises(ValueError):
        m.decode_rows(np, np.zeros((2,4),np.uint32), np.zeros((2,1),np.uint8), rows)


def test_decode_geometry_refuses_padding_assumption():
    with pytest.raises(ValueError):
        m.decode_rows(np, np.zeros((2,4),np.uint32), np.zeros((2,2),np.uint8), (0,))
