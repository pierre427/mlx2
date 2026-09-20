"""Concurrent multi-LoRA serving: per-row adapters inside one physical batch.

Design (see docs/SERVING.md "Concurrent multi-LoRA"):

* ``MultiLoRALinear`` wraps an explicit ``Linear``/``QuantizedLinear`` key and
  owns stacked slot tensors ``lora_a: (S, in, R)`` and ``lora_b: (S, R, out)``
  with the adapter scale folded into ``lora_b``.  Slot 0 is the base model and
  is permanently zero.  The per-row delta is
  ``gather_mm(gather_mm(x, A, rhs=ids), B, rhs=ids)`` -- the Punica/S-LoRA
  "gathered BGMV" computed with MLX's ``gather_mm``.  A batch whose rows are
  all base skips the delta entirely.
* ``MultiLoRAManager`` owns the host registry (validated tensors), the resident
  slot table with pin counts and LRU eviction of unpinned adapters, the
  uid->slot row map, and default-on sync-free mechanism counters.  The ordinary
  generator seams call :func:`bind_lora_rows` with the batch uids right before
  each model forward and :func:`clear_lora_rows` after it, so no model code and
  no scheduler model-name branching is involved.

The row-mapping-before-forward pattern follows vLLM's LoRA mapping/punica
wrapper; the slot/LRU residency follows S-LoRA.  Mechanism is original MLX code.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

MULTI_LORA_RECEIPT_SCHEMA = "mlx2.multi-lora.v1"
_MANAGER_ATTRIBUTE = "_mlx2_multi_lora"


def multi_lora_manager(model):
    """Return the manager attached to ``model``, or None (feature off)."""
    return getattr(model, _MANAGER_ATTRIBUTE, None) if model is not None else None


def bind_lora_rows(model, uids):
    """Publish per-row slot ids for the next forward.  Returns the manager."""
    manager = multi_lora_manager(model)
    if manager is not None:
        manager.bind_rows(uids)
    return manager


def clear_lora_rows(manager):
    if manager is not None:
        manager.clear_rows()


_COUNTER_NAMES = (
    "forwards", "base_only_forwards", "mixed_forwards", "single_adapter_forwards",
    "rows_base", "rows_adapter", "slot_hits", "slot_loads", "slot_evictions",
    "slot_deferred", "registrations", "unregistrations", "structural_wraps",
    "materializations",
)


class _RowState:
    """Per-model row binding shared by every wrapped module (plain object)."""

    __slots__ = ("ids", "rows", "delta_applications", "delta_skips")

    def __init__(self):
        self.ids = None  # mx.array (B,) uint32, or None when all rows are base
        self.rows = None  # bound row count, or None when nothing is bound
        self.delta_applications = 0
        self.delta_skips = 0


def _linear_dims(linear):
    from mlx import nn

    output_dims, packed_input_dims = linear.weight.shape
    if isinstance(linear, nn.QuantizedLinear):
        return packed_input_dims * 32 // linear.bits, output_dims
    return packed_input_dims, output_dims


def _default_lora_dtype(linear):
    from mlx import nn

    if isinstance(linear, nn.QuantizedLinear):
        return linear.scales.dtype
    return linear.weight.dtype


def _make_wrapper_class():
    import mlx.core as mx
    from mlx import nn

    class MultiLoRALinear(nn.Module):
        def __init__(self, base, *, slots, rank, dtype, rows):
            super().__init__()
            input_dims, output_dims = _linear_dims(base)
            self.base = base
            self.lora_a = mx.zeros((slots, input_dims, rank), dtype=dtype)
            self.lora_b = mx.zeros((slots, rank, output_dims), dtype=dtype)
            self._rows = rows

        @property
        def input_dims(self):
            return self.lora_a.shape[1]

        @property
        def output_dims(self):
            return self.lora_b.shape[2]

        def __call__(self, x):
            y = self.base(x)
            rows = self._rows
            ids = rows.ids
            if ids is None:
                rows.delta_skips += 1
                return y
            squeeze = x.ndim == 2
            value = x[:, None, :] if squeeze else x
            if value.ndim != 3 or value.shape[0] != ids.shape[0]:
                # Fail closed: a forward whose leading dimension is not the
                # bound batch rows would apply adapters to the wrong rows.
                raise RuntimeError(
                    "multi-LoRA row binding does not match the forward batch: "
                    f"input {tuple(x.shape)} vs {ids.shape[0]} bound rows"
                )
            value = value.astype(self.lora_a.dtype)
            hidden = mx.gather_mm(value, self.lora_a, None, ids)
            delta = mx.gather_mm(hidden, self.lora_b, None, ids)
            if squeeze:
                delta = delta[:, 0, :]
            rows.delta_applications += 1
            return y + delta.astype(y.dtype)

    return MultiLoRALinear


_WRAPPER = None


def multi_lora_linear_class():
    global _WRAPPER
    if _WRAPPER is None:
        _WRAPPER = _make_wrapper_class()
    return _WRAPPER


@dataclass
class RegisteredAdapter:
    name: str
    path: str
    fingerprint: str
    rank: int
    scale: float
    keys: tuple
    tensors: dict  # key -> (A (in, rank), B_scaled (rank, out))
    nbytes: int
    lora_int_id: int | None = None


class SlotUnavailable(RuntimeError):
    """Every resident slot is pinned by a live row."""


class MultiLoRAManager:
    """Registry, slot residency (pin + LRU), row binding and counters."""

    def __init__(self, model, *, max_loras, max_lora_rank, dtype=None):
        if isinstance(max_loras, bool) or not isinstance(max_loras, int) or not 1 <= max_loras <= 64:
            raise ValueError("max_loras must be an integer from 1 to 64")
        if (
            isinstance(max_lora_rank, bool)
            or not isinstance(max_lora_rank, int)
            or not 1 <= max_lora_rank <= 1024
        ):
            raise ValueError("max_lora_rank must be an integer from 1 to 1024")
        if multi_lora_manager(model) is not None:
            raise ValueError("model already carries a multi-LoRA manager")
        self.model = model
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.dtype = dtype
        self.lock = threading.RLock()
        self.rows = _RowState()
        self.registry: dict[str, RegisteredAdapter] = {}
        self.wrapped: dict[str, object] = {}
        # slot index (1..max_loras) -> adapter name; LRU order = OrderedDict order
        self.resident: OrderedDict[int, str] = OrderedDict()
        self.pins: dict[int, int] = defaultdict(int)
        self.uid_slots: dict[int, int] = {}
        self._bound_key = None
        self._map_version = 0
        # Pre-seeded so a concurrent status() copy never sees a resize.
        self.counts = defaultdict(int, {name: 0 for name in _COUNTER_NAMES})
        self.rows_by_adapter = defaultdict(int)
        object.__setattr__(model, _MANAGER_ATTRIBUTE, self)

    # ----------------------------------------------------------------- config
    @property
    def slots(self):
        return self.max_loras + 1

    def settings(self):
        return {
            "max_loras": self.max_loras,
            "max_lora_rank": self.max_lora_rank,
            "schema": MULTI_LORA_RECEIPT_SCHEMA,
        }

    def detach(self):
        """Restore the original modules and detach from the model."""
        from mlx.utils import tree_unflatten

        with self.lock:
            if any(self.pins.values()):
                raise ValueError("cannot detach multi-LoRA while rows are pinned")
            originals = [(key, module.base) for key, module in self.wrapped.items()]
            if originals:
                self.model.update_modules(tree_unflatten(originals))
            self.wrapped.clear()
            try:
                object.__delattr__(self.model, _MANAGER_ATTRIBUTE)
            except AttributeError:
                pass

    # --------------------------------------------------------------- registry
    def prepare(self, name, path, *, lora_int_id=None):
        """Validate an adapter directory; return (adapter, keys needing a wrap).

        Pure host work with no MLX operations (safe on an HTTP thread while the
        generation worker runs): the config is parsed, the safetensors header
        is checked for exact key coverage and shapes, and the file content is
        hashed into the adapter fingerprint.  Tensors are materialized lazily
        by the generation worker on first slot load.
        """
        from mlx import nn

        from .lora import read_lora_config

        if not isinstance(name, str) or not 1 <= len(name) <= 128:
            raise ValueError("lora_name must contain 1 to 128 characters")
        path, config = read_lora_config(path)
        if config["rank"] > self.max_lora_rank:
            raise ValueError(
                f"LoRA rank {config['rank']} exceeds max_lora_rank {self.max_lora_rank}"
            )
        weights_path = path / "adapters.safetensors"
        payload = weights_path.read_bytes()
        header = _safetensors_header(payload)
        digest = hashlib.sha256()
        digest.update(
            repr((config["rank"], config["scale"], tuple(sorted(config["keys"])))).encode()
        )
        digest.update(payload)
        fingerprint = digest.hexdigest()
        expected = {
            f"{key}.{leaf}" for key in config["keys"] for leaf in ("lora_a", "lora_b")
        }
        if set(header) != expected:
            missing = sorted(expected - set(header))
            extra = sorted(set(header) - expected)
            raise ValueError(
                f"LoRA tensor coverage mismatch; missing={missing}, extra={extra}"
            )
        modules = dict(self.model.named_modules())
        wrapper = multi_lora_linear_class()
        new_keys = []
        nbytes = 0
        for key in config["keys"]:
            module = modules.get(key)
            if isinstance(module, wrapper):
                base = module.base
            elif isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                base = module
                new_keys.append(key)
            else:
                raise ValueError(
                    f"LoRA key {key!r} is not a Linear/QuantizedLinear module"
                )
            input_dims, output_dims = _linear_dims(base)
            a_shape = tuple(header[f"{key}.lora_a"]["shape"])
            b_shape = tuple(header[f"{key}.lora_b"]["shape"])
            if a_shape != (input_dims, config["rank"]) or b_shape != (
                config["rank"],
                output_dims,
            ):
                raise ValueError(
                    f"LoRA tensor {key!r} has shapes {a_shape}/{b_shape}, expected "
                    f"{(input_dims, config['rank'])}/{(config['rank'], output_dims)}"
                )
            itemsize = _dtype_itemsize(self.dtype or _default_lora_dtype(base))
            nbytes += itemsize * config["rank"] * (input_dims + output_dims)
        adapter = RegisteredAdapter(
            name=name,
            path=str(path),
            fingerprint=fingerprint,
            rank=config["rank"],
            scale=config["scale"],
            keys=tuple(config["keys"]),
            tensors={},
            nbytes=nbytes,
            lora_int_id=lora_int_id,
        )
        return adapter, tuple(new_keys)

    def _materialize(self, adapter):
        """Load and scale-fold the adapter tensors (generation worker only)."""
        import mlx.core as mx

        if adapter.tensors:
            return
        weights = mx.load(str(Path(adapter.path) / "adapters.safetensors"))
        tensors = {}
        for key in adapter.keys:
            module = self.wrapped[key]
            dtype = module.lora_a.dtype
            a = weights[f"{key}.lora_a"].astype(dtype)
            b = (weights[f"{key}.lora_b"].astype(mx.float32) * adapter.scale).astype(dtype)
            tensors[key] = (a, b)
        mx.eval(list(tensors.values()))
        adapter.tensors = tensors
        self.counts["materializations"] += 1

    def wrap_keys(self, keys):
        """Structural edit: wrap new module keys.  Caller guarantees no forward
        is in flight (the engine drains through its exclusive operation)."""
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        if not keys:
            return 0
        wrapper = multi_lora_linear_class()
        modules = dict(self.model.named_modules())
        replacements = []
        with self.lock:
            for key in keys:
                if key in self.wrapped:
                    continue
                base = modules[key]
                module = wrapper(
                    base,
                    slots=self.slots,
                    rank=self.max_lora_rank,
                    dtype=self.dtype or _default_lora_dtype(base),
                    rows=self.rows,
                )
                replacements.append((key, module))
                self.wrapped[key] = module
            if replacements:
                self.model.update_modules(tree_unflatten(replacements))
                mx.eval([(m.lora_a, m.lora_b) for _, m in replacements])
                # Adapters already resident must also be written into the new
                # modules (their slot entries are zero for keys they do not
                # cover, which is the correct value).
                self.counts["structural_wraps"] += len(replacements)
        return len(replacements)

    def commit(self, adapter):
        with self.lock:
            if adapter.name in self.registry:
                raise ValueError(f"LoRA adapter {adapter.name!r} is already loaded")
            missing = [key for key in adapter.keys if key not in self.wrapped]
            if missing:
                raise RuntimeError("LoRA keys must be wrapped before registration")
            self.registry[adapter.name] = adapter
            self.counts["registrations"] += 1
        return self.describe(adapter.name)

    def register(self, name, path, *, lora_int_id=None):
        """Validate, wrap (no drain -- callers that serve traffic must use the
        engine path), and register.  Convenience for tests and offline use."""
        adapter, new_keys = self.prepare(name, path, lora_int_id=lora_int_id)
        self.wrap_keys(new_keys)
        return self.commit(adapter)

    def unregister(self, name):
        with self.lock:
            if name not in self.registry:
                raise ValueError("LoRA adapter is not loaded")
            for slot, resident in list(self.resident.items()):
                if resident == name:
                    if self.pins.get(slot):
                        raise ValueError(
                            f"LoRA adapter {name!r} is in use by live requests"
                        )
                    # No MLX work here (HTTP thread): the freed slot keeps
                    # stale values, but no row can map to an unpinned,
                    # non-resident slot, and the next load overwrites it.
                    del self.resident[slot]
            del self.registry[name]
            self.counts["unregistrations"] += 1

    def describe(self, name):
        adapter = self.registry[name]
        return {
            "name": adapter.name,
            "fingerprint": adapter.fingerprint,
            "rank": adapter.rank,
            "keys": list(adapter.keys),
            "nbytes": adapter.nbytes,
        }

    def lookup(self, name):
        with self.lock:
            adapter = self.registry.get(name) if isinstance(name, str) else None
            return None if adapter is None else (adapter.name, adapter.fingerprint)

    # --------------------------------------------------------------- residency
    def _write_slot(self, slot, adapter):
        import mlx.core as mx

        updated = []
        for key, module in self.wrapped.items():
            pair = adapter.tensors.get(key) if adapter is not None else None
            a_slot = mx.zeros(module.lora_a.shape[1:], dtype=module.lora_a.dtype)
            b_slot = mx.zeros(module.lora_b.shape[1:], dtype=module.lora_b.dtype)
            if pair is not None:
                a, b = pair
                rank = a.shape[1]
                a_slot = mx.concatenate(
                    [a.astype(a_slot.dtype), a_slot[:, rank:]], axis=1
                )
                b_slot = mx.concatenate(
                    [b.astype(b_slot.dtype), b_slot[rank:, :]], axis=0
                )
            # Functional rebuild: forwards already enqueued keep the old arrays.
            module.lora_a = mx.concatenate(
                [module.lora_a[:slot], a_slot[None], module.lora_a[slot + 1 :]]
            )
            module.lora_b = mx.concatenate(
                [module.lora_b[:slot], b_slot[None], module.lora_b[slot + 1 :]]
            )
            updated.append((module.lora_a, module.lora_b))
        mx.eval(updated)

    def acquire(self, name):
        """Pin a resident slot for ``name``.  Returns ``(slot, residency)``.

        Raises :class:`SlotUnavailable` when every slot is pinned.
        """
        with self.lock:
            adapter = self.registry.get(name)
            if adapter is None:
                raise ValueError(f"LoRA adapter {name!r} is not loaded")
            for slot, resident in self.resident.items():
                if resident == name:
                    self.resident.move_to_end(slot)
                    self.pins[slot] += 1
                    self.counts["slot_hits"] += 1
                    return slot, "hit"
            free = [
                slot
                for slot in range(1, self.slots)
                if slot not in self.resident
            ]
            residency = "loaded"
            if free:
                slot = free[0]
            else:
                victims = [slot for slot in self.resident if not self.pins.get(slot)]
                if not victims:
                    self.counts["slot_deferred"] += 1
                    raise SlotUnavailable(
                        "every multi-LoRA slot is pinned by a live request"
                    )
                slot = victims[0]  # least recently used unpinned
                del self.resident[slot]
                self.counts["slot_evictions"] += 1
                residency = "evicted"
            self._materialize(adapter)
            self._write_slot(slot, adapter)
            self.resident[slot] = name
            self.pins[slot] += 1
            self.counts["slot_loads"] += 1
            return slot, residency

    def release(self, slot):
        with self.lock:
            if self.pins.get(slot, 0) <= 0:
                raise RuntimeError(f"multi-LoRA slot {slot} is not pinned")
            self.pins[slot] -= 1

    def bind_uid(self, uid, slot):
        with self.lock:
            self.uid_slots[uid] = slot
            self._map_version += 1

    def unbind_uid(self, uid):
        with self.lock:
            if self.uid_slots.pop(uid, None) is not None:
                self._map_version += 1

    # -------------------------------------------------------------- row binding
    def bind_rows(self, uids):
        # One uncontended lock per forward (not per layer); no device sync.
        with self.lock:
            self._bind_rows_locked(uids)

    def _bind_rows_locked(self, uids):
        import mlx.core as mx

        key = (self._map_version, tuple(uids))
        rows = self.rows
        self.counts["forwards"] += 1
        if key != self._bound_key:
            slots = [self.uid_slots.get(uid, 0) for uid in uids]
            self._bound_key = key
            self._bound_slots = slots
            self._bound_ids = (
                mx.array(slots, dtype=mx.uint32) if any(slots) else None
            )
        slots = self._bound_slots
        rows.ids = self._bound_ids
        rows.rows = len(slots)
        if rows.ids is None:
            self.counts["base_only_forwards"] += 1
            self.counts["rows_base"] += len(slots)
            return
        adapters = 0
        for slot in slots:
            if slot:
                adapters += 1
                self.rows_by_adapter[self.resident.get(slot, "?")] += 1
        self.counts["rows_adapter"] += adapters
        self.counts["rows_base"] += len(slots) - adapters
        if adapters != len(slots) or len({s for s in slots}) > 1:
            self.counts["mixed_forwards"] += 1
        else:
            self.counts["single_adapter_forwards"] += 1

    def clear_rows(self):
        self.rows.ids = None
        self.rows.rows = None

    # ------------------------------------------------------------------ status
    def reserved_bytes(self):
        return sum(m.lora_a.nbytes + m.lora_b.nbytes for m in self.wrapped.values())

    def status(self):
        with self.lock:
            return {
                "enabled": True,
                **self.settings(),
                "registered": sorted(self.registry),
                "resident": {
                    str(slot): name for slot, name in self.resident.items()
                },
                "pinned_slots": sum(1 for v in self.pins.values() if v),
                "wrapped_modules": len(self.wrapped),
                "reserved_bytes": self.reserved_bytes(),
                "registered_bytes": sum(a.nbytes for a in self.registry.values()),
                "counts": {
                    **dict(self.counts),
                    "delta_applications": self.rows.delta_applications,
                    "delta_skips": self.rows.delta_skips,
                },
                "rows_by_adapter": dict(self.rows_by_adapter),
            }


_SAFETENSORS_ITEMSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
    "I8": 1, "U8": 1, "BOOL": 1,
}


def _safetensors_header(payload):
    import json
    import struct

    if len(payload) < 8:
        raise ValueError("LoRA safetensors file is truncated")
    (length,) = struct.unpack("<Q", payload[:8])
    if length > len(payload) - 8 or length > 100 << 20:
        raise ValueError("LoRA safetensors header is malformed")
    try:
        header = json.loads(payload[8 : 8 + length])
    except ValueError as error:
        raise ValueError("LoRA safetensors header is malformed") from error
    if not isinstance(header, dict):
        raise ValueError("LoRA safetensors header is malformed")
    header.pop("__metadata__", None)
    for key, entry in header.items():
        if (
            not isinstance(entry, dict)
            or entry.get("dtype") not in _SAFETENSORS_ITEMSIZE
            or not isinstance(entry.get("shape"), list)
        ):
            raise ValueError(f"LoRA safetensors entry {key!r} is malformed")
    return header


def _dtype_itemsize(dtype):
    return getattr(dtype, "size", 4)


def lora_apc_scope(media_fingerprint, lora_fingerprint):
    """Compose the APCv2 per-request scope: media plus adapter identity."""
    if not lora_fingerprint:
        return media_fingerprint
    return f"{media_fingerprint or ''}|lora:{lora_fingerprint}"


def write_adapter(path, *, keys, dims, rank, scale=2.0, seed=0, dtype=None):
    """Write a random LoRA adapter directory (tests / GPU bench fixtures).

    ``dims`` maps key -> (input_dims, output_dims).
    """
    import json

    import mlx.core as mx

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "lora_parameters": {
                    "rank": rank,
                    "scale": scale,
                    "dropout": 0.0,
                    "keys": list(keys),
                },
            }
        )
    )
    subkeys = mx.random.split(mx.random.key(seed), 2 * len(keys))
    weights = {}
    for index, name in enumerate(keys):
        input_dims, output_dims = dims[name]
        ka, kb = subkeys[2 * index], subkeys[2 * index + 1]
        a = mx.random.normal((input_dims, rank), key=ka) * (1.0 / input_dims**0.5)
        b = mx.random.normal((rank, output_dims), key=kb) * 0.05
        if dtype is not None:
            a, b = a.astype(dtype), b.astype(dtype)
        weights[f"{name}.lora_a"] = a
        weights[f"{name}.lora_b"] = b
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    return path
