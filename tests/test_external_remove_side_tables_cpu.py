"""K1: removing a lane drops its per-uid draft-phase side tables."""

import mlx.core as mx
import pytest

from mlx2.runtime.external_speculative import _LAST_ALTERNATIVE_FIELDS
from tests.test_external_dflash2_cpu import generator, tiny


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_remove_pops_last_alternative_entries_for_the_removed_uid_only():
    model, draft = tiny()
    batch = generator(model, draft)
    uid, other = batch.insert([[1, 2, 3], [4, 5, 6]], max_tokens=[4, 4])
    for name in _LAST_ALTERNATIVE_FIELDS:
        setattr(batch, name, {uid: object(), other: object()})
    batch.remove([uid])
    for name in _LAST_ALTERNATIVE_FIELDS:
        table = getattr(batch, name)
        assert uid not in table, name
        assert other in table, name
    batch.close()
