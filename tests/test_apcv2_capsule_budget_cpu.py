"""Exercise APCv2 resident accounting with host-only cache entries."""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def apc_class():
    path = Path(__file__).parents[1] / "src/mlx2/runtime/apc_v2.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body
        if (isinstance(node, ast.Import) and all(
            not alias.name.startswith("mlx") for alias in node.names
        ))
        or (isinstance(node, ast.ImportFrom) and not node.level)
        or (isinstance(node, ast.ClassDef) and node.name in {
            "APCv2", "_CapsuleCapacityReservation",
        })
    ]
    scope = {"PrefixIndex": object, "mx": SimpleNamespace(clear_cache=lambda: None)}
    exec(compile(tree, str(path), "exec"), scope)  # noqa: S102 - local production AST only
    return scope["APCv2"]


def owner(apc_class, *, resident=0, reserved=60, disk=False):
    apc = apc_class.__new__(apc_class)
    apc._apc_lock = threading.RLock()
    apc.max_bytes = 100
    apc.max_size = apc.max_interior_entries = 8
    apc._n_bytes = resident
    apc._capsule_reserved_bytes = reserved
    apc._capsule_capacity = {
        "reservation_rejections": 0, "reservations": 0, "reserved_bytes_peak": reserved,
    }
    apc._idle_disk_dir = "host-scratch" if disk else None
    apc._enforce_count_pool_locked = lambda **_kwargs: None
    apc._count_pools_fit_locked = lambda: True
    apc._enforce_disk_limit_locked = lambda **_kwargs: None
    apc._entry_disk_pinned_locked = lambda *_args: False
    apc.removed = []
    entry = SimpleNamespace(nbytes=resident)
    apc._pressure_candidates_locked = lambda **_kwargs: (
        [(0, 0, "key", [1], entry)] if apc._n_bytes else []
    )

    def remove(_key, _tokens, victim, **_kwargs):
        apc._n_bytes -= victim.nbytes
        apc.removed.append(victim)
        return True

    apc._drop_entry_locked = remove
    apc._spill_entry_locked = remove
    return apc


@pytest.mark.parametrize("disk", [False, True])
def test_entry_publication_cannot_consume_reserved_capsule_capacity(apc_class, disk):
    apc = owner(apc_class, resident=50, disk=disk)
    assert apc._enforce_entry_limits_locked()
    assert apc._n_bytes + apc._capsule_reserved_bytes <= apc.max_bytes
    assert len(apc.removed) == 1


def test_publication_does_not_double_charge_reserved_capacity(apc_class):
    apc = owner(apc_class, resident=30, reserved=60, disk=True)
    assert apc._enforce_entry_limits_locked()
    assert apc._n_bytes == 30 and apc.removed == []


def test_capsule_reservation_spill_preserves_original_snapshot_cap(apc_class):
    apc = owner(apc_class, resident=95, reserved=0, disk=True)
    spill = apc._spill_entry_locked
    caps = []

    def record_cap(*args, **kwargs):
        caps.append(kwargs["hard_cap"])
        return spill(*args, **kwargs)

    apc._spill_entry_locked = record_cap
    reservation = apc.reserve_capsule_bytes(10)
    assert reservation is not None
    assert caps == [100]
    reservation.release()


def test_restore_defers_when_capsules_hold_capacity_without_reclaiming(apc_class):
    apc = owner(apc_class, resident=10, disk=True)
    assert apc._reserve_restore_bytes_locked(50, entry=object()) is False
    assert apc.max_bytes == 100
    assert apc._n_bytes == 10 and apc.removed == []


def test_restore_reclaims_against_combined_resident_and_capsule_budget(apc_class):
    apc = owner(apc_class, resident=20, disk=True)
    assert apc._reserve_restore_bytes_locked(30, entry=object()) is True
    assert apc.max_bytes == 100
    assert apc._n_bytes + apc._capsule_reserved_bytes + 30 <= apc.max_bytes
    assert len(apc.removed) == 1


def test_additional_capsule_reservation_is_charged_exactly_once(apc_class):
    apc = owner(apc_class, resident=10, disk=True)
    reservation = apc.reserve_capsule_bytes(20)
    assert reservation is not None
    assert apc._n_bytes == 10 and apc.removed == []
    assert apc._capsule_reserved_bytes == 80
    reservation.release()
    reservation.release()
    assert apc._capsule_reserved_bytes == 60


def test_oversize_restore_still_rejects_a_snapshot_exceeding_total_cap(apc_class):
    apc = owner(apc_class)
    with pytest.raises(ValueError, match="resident byte cap"):
        apc._reserve_restore_bytes_locked(101, entry=object())
