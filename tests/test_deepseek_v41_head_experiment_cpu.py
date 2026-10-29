"""CPU contracts for head experiment preparation; no full checkpoint loaded."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest

from mlx2.runtime.models import deepseek_v41_head_experiment as head


@pytest.fixture(autouse=True)
def forbid_gpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **k: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def test_full_geometry_byte_accounting_and_rows():
    reference, direct = head.plan(head.REFERENCE), head.plan(head.DIRECT)
    assert reference['weight_shape'] == direct['weight_shape'] == (129280, 5120)
    assert reference['coefficient_bytes'] == 2647654400
    assert direct['coefficient_bytes'] == 1323827200
    assert reference['persistent_companion_bytes'] == direct['persistent_companion_bytes'] == 0
    assert head.plan(head.REFERENCE, 99)['output_shape'] == (1, 99, 129280)
    for profile, rows in ((head.DIRECT, 2), (head.REFERENCE, 0), (head.REFERENCE, True), ('bad', 1)):
        with pytest.raises(ValueError): head.plan(profile, rows)


def test_preparation_retains_one_owner_and_casts_reference_once(monkeypatch):
    # Small explicit synthetic geometry tests ownership and numerical consumer
    # independently; production full-vocabulary geometry is tested above.
    monkeypatch.setattr(head, 'WEIGHT_SHAPE', (3, 4))
    coefficients = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4), mx.bfloat16)
    direct = head.prepare(coefficients, profile=head.DIRECT, checkpoint='synthetic-head@tiny', source_revision='test-source')
    reference = head.prepare(coefficients, profile=head.REFERENCE, checkpoint='synthetic-head@tiny', source_revision='test-source')
    assert direct.coefficients is coefficients and direct.coefficients.dtype == mx.bfloat16
    assert reference.coefficients.dtype == mx.float32
    x = mx.array([[[1., 2., 3., 4.]]], mx.float32)
    expected = np.asarray(x) @ np.asarray(coefficients.astype(mx.float32)).T
    np.testing.assert_array_equal(np.asarray(head.project(x, reference)), expected)
    from mlx2.runtime.gguf import source_dense
    calls = []
    def project(value, weight, kind, *, algorithm):
        calls.append((value, weight, kind, algorithm)); return value @ weight.astype(mx.float32).T
    monkeypatch.setattr(source_dense, 'project', project)
    np.testing.assert_array_equal(np.asarray(head.project(x, direct)), expected)
    assert calls[0][1] is coefficients and calls[0][2:] == ('head', 'gemv')
    assert direct.receipt()['checkpoint'] == 'synthetic-head@tiny'
    assert direct.receipt()['qualification'] == 'unqualified'
    with pytest.raises(ValueError, match='identity'): head.project(x, replace(direct, implementation='stale'))


def test_full_head_preparation_refuses_crops_identity_and_dtype_before_allocation():
    for weight in (SimpleNamespace(shape=(129279, 5120), dtype=mx.bfloat16),
                   SimpleNamespace(shape=(129280, 5120), dtype=mx.float32)):
        with pytest.raises(ValueError, match='full original BF16'):
            head.prepare(weight, profile=head.DIRECT, checkpoint='synthetic@geometry', source_revision='test')
    weight = SimpleNamespace(shape=(129280, 5120), dtype=mx.bfloat16)
    for checkpoint, revision in (('', 'test'), ('family', 'test'), ('target@fingerprint', '')):
        with pytest.raises(ValueError, match='identities'):
            head.prepare(weight, profile=head.DIRECT, checkpoint=checkpoint, source_revision=revision)
