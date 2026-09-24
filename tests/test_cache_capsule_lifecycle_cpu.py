"""Run capsule ownership code with host arrays and no MLX imports."""

import ast
import gc
import sys
import threading
import weakref
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def capsule(monkeypatch):
    path = Path(__file__).parents[1] / "src/mlx2/runtime/cache_capsule.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("mlx") for alias in node.names)
        )
        and not (isinstance(node, ast.ImportFrom) and node.level)
    ]
    module = ModuleType("_capsule_lifecycle_cpu")
    # A distinct sentinel type keeps NumPy buffers host-only to the stage gate.
    module.mx = SimpleNamespace(array=type("DeviceArray", (), {}), bfloat16=object())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 - local production AST only
    return module


def source_for(capsule):
    keys = np.zeros((1, 1, 3, 2), dtype=np.float16)
    return capsule.KVCachePlaneSource(
        0, "source", ("exact",), keys, keys.copy(), 3, 2,
        ("layout",), threading.get_ident(),
    )


def product_for(capsule, source, **changes):
    payload = capsule.KVCacheCapsulePayload(
        np.repeat(source.keys, 2, axis=0), np.repeat(source.values, 2, axis=0),
        source.offset, source.generation, source.source_id,
        source.compatibility_signature, source.layout_fingerprint, "external",
    )
    return capsule.CacheCapsuleProduct(replace(payload, **changes))


class Adapter:
    def __init__(self, capsule):
        self.capsule = capsule
        self.adoptions = 0
        self.discards = []

    def stage(self, _source):
        return {"host": True}

    def build(self, staged):
        return staged

    def adopt(self, _built, source):
        self.adoptions += 1
        return product_for(self.capsule, source)

    def discard(self, built):
        self.discards.append(built)


def test_cancelled_ticket_never_adopts_discarded_host_backing(capsule):
    adapter = Adapter(capsule)
    pool = capsule.CacheCapsulePool(
        capsule.CacheCapsuleGeneration(), adapter=adapter, enabled=True,
    )
    ticket = pool.submit(source_for(capsule))
    ticket.future.result(timeout=1)
    assert ticket.cancel()
    assert adapter.discards == [{"host": True}]
    with pytest.raises(capsule.CacheCapsuleError, match="pending"):
        ticket.await_adopt(timeout_s=1, fallback=None)
    assert adapter.adoptions == 0
    pool.close()


def test_adopted_ticket_cannot_adopt_a_second_time(capsule):
    adapter = Adapter(capsule)
    pool = capsule.CacheCapsulePool(
        capsule.CacheCapsuleGeneration(), adapter=adapter, enabled=True,
    )
    ticket = pool.submit(source_for(capsule))
    receipt = ticket.await_adopt(timeout_s=1, fallback=None)
    try:
        with pytest.raises(capsule.CacheCapsuleError, match="pending"):
            ticket.await_adopt(timeout_s=1, fallback=None)
        assert adapter.adoptions == 1
        assert not ticket.dispose_completed()
        assert adapter.discards == []
    finally:
        receipt.owner.release()
        pool.close()


def test_cancellation_during_wait_never_adopts_completed_backing(capsule):
    adapter = Adapter(capsule)
    pool = capsule.CacheCapsulePool(
        capsule.CacheCapsuleGeneration(), adapter=adapter, enabled=True,
    )
    ticket = pool.submit(source_for(capsule))
    ticket.future.result(timeout=1)
    original_result = ticket.future.result

    def result_then_close(timeout=None):
        result = original_result(timeout)
        ticket.future.result = original_result
        # Mimic another thread closing the pool while await waited on build.
        pool.close()
        return result

    ticket.future.result = result_then_close
    with pytest.raises(capsule.CacheCapsuleError, match="pending"):
        ticket.await_adopt(timeout_s=1, fallback=None)
    assert adapter.adoptions == 0
    assert adapter.discards == [{"host": True}]


def test_backend_must_preserve_source_offset(capsule):
    source = source_for(capsule)
    pool = capsule.CacheCapsulePool(capsule.CacheCapsuleGeneration(), enabled=True)
    with pytest.raises(capsule.CacheCapsuleError, match="offset"):
        pool._accept(product_for(capsule, source, offset=2), source, "external", None)


def test_released_owner_drops_source_arrays_even_if_receipt_survives(capsule):
    source = source_for(capsule)
    original_keys = weakref.ref(source.keys)
    pool = capsule.CacheCapsulePool(capsule.CacheCapsuleGeneration(), enabled=True)
    receipt = pool._accept(product_for(capsule, source), source, "external", None)
    del source
    assert original_keys() is not None
    receipt.owner.release()
    gc.collect()
    assert original_keys() is None
    with pytest.raises(capsule.CacheCapsuleOwnerReleased):
        receipt.owner.lease()


def test_raw_bit_verification_checks_payload_instead_of_backend_claim(capsule):
    source = source_for(capsule)
    source = replace(source, raw_digest=capsule._raw_digest(source.keys, source.values))
    claimed_digest = capsule._raw_digest(source.keys, source.values, source.target_batch)
    product = product_for(capsule, source, raw_digest=claimed_digest)
    product.payload.keys[0, 0, 0, 0] = 1
    pool = capsule.CacheCapsulePool(
        capsule.CacheCapsuleGeneration(), enabled=True, verify_raw_bits=True,
    )
    with pytest.raises(capsule.CacheCapsuleError, match="checksum"):
        pool._accept(product, source, "external", None)


def test_closed_pool_rejects_synchronous_build_before_materialization(capsule):
    pool = capsule.CacheCapsulePool(capsule.CacheCapsuleGeneration(), enabled=True)
    pool.close()
    builds = []
    source = source_for(capsule)
    pool._build = lambda *_args: builds.append(True) or product_for(capsule, source)
    with pytest.raises(capsule.CacheCapsuleError, match="closed"):
        pool.prepare(source, primary="cpu", fallback=None)
    assert builds == []


@pytest.mark.parametrize("failure", ["timeout", "error", "synchronous", "adopt"])
def test_close_during_build_or_adopt_releases_product_before_publication(capsule, failure):
    source = source_for(capsule)
    released = []
    backing = SimpleNamespace(release=lambda: released.append("backing"))
    adapter = Adapter(capsule)
    pool = capsule.CacheCapsulePool(
        capsule.CacheCapsuleGeneration(), adapter=adapter, enabled=True,
    )

    def build_then_close(*_args):
        pool.close()
        return capsule.CacheCapsuleProduct(
            product_for(capsule, source, backend="cpu").payload, backing,
        )

    pool._build = build_then_close
    with pytest.raises(capsule.CacheCapsuleError, match="closed"):
        if failure == "synchronous":
            pool.prepare(source, primary="cpu")
        elif failure == "adopt":
            def adopt_then_close(_built, _source):
                pool.close()
                return capsule.CacheCapsuleProduct(product_for(capsule, source).payload, backing)

            adapter.adopt = adopt_then_close
            pool.prepare(source, primary="external", fallback=None, timeout_s=1)
        else:
            future = Future()
            if failure == "error":
                future.set_exception(RuntimeError("injected build failure"))
            ticket = capsule.CacheCapsuleTicket(pool, source, {}, future)
            pool._tickets.add(ticket)
            ticket.await_adopt(timeout_s=0, fallback="cpu")
    assert released == ["backing"]
    assert pool.counters["primary_successes"] == 0
    assert not pool._tickets
