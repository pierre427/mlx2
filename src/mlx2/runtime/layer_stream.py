"""Bounded whole-layer residency over proven safetensors bytes.

Adapters choose decoder blocks; the cache never knows model names. Layer zero
is always resident. A cache miss reads a complete block, and outputs plus
mutated request state are evaluated before any block can be evicted. The
budget includes cached arrays and the host/device overlap of the next read;
OS page cache, allocator caches, activations and request state are excluded.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from .weight_stream import (
    DEFAULT_READ_WORKERS,
    READ_TIMEOUT_SECONDS,
    ProjectionReader,
    StreamingUnavailable,
    StreamStats,
    WorkingSetTooSmall,
    _close_read_pool,
    _location_entry,
    _ManagerBase,
    _manifest,
    _module_parent,
    _ReadPool,
    _swap_module,
    is_mtp_path,
    storage_owner,
)


@dataclass
class LayerStreamPlan:
    layers: int
    excluded_bytes: int
    max_layer_bytes: int
    ceiling_bytes: int
    manifest_sha256: str
    resident_paths: tuple[str, ...]
    retained_tensor_paths: tuple[str, ...]
    retained_tensor_bytes: int
    load_phase: str = "pre_materialization"
    resident_remainder_bytes: int | None = None

    def as_dict(self):
        return {
            **asdict(self),
            "reserved_bytes": self.ceiling_bytes,
            "load_peak_bounded": True,
            "layer_zero_resident": True,
            "eviction": "retain_then_bypass",
            "transient_bound_bytes": 2 * self.max_layer_bytes,
        }


def _set_weights(module, arrays):
    for key, value in arrays.items():
        parent, leaf = _module_parent(module, key)
        parent[leaf] = value


def _eval_state(value):
    """Finish request state as well as the residual before releasing weights."""
    import mlx.core as mx

    if isinstance(value, (list, tuple)):
        for item in value:
            _eval_state(item)
    elif isinstance(value, dict):
        for item in value.values():
            _eval_state(item)
    elif getattr(value, "state", None) is not None:
        mx.eval(value.state)


_LAYER_CLASS = None


def _layer_class():
    global _LAYER_CLASS
    if _LAYER_CLASS is not None:
        return _LAYER_CLASS
    import mlx.core as mx
    from mlx import nn

    class StreamedDecoderLayer(nn.Module):
        def __init__(self, manager, path, block, retained):
            super().__init__()
            # A non-module holder prevents Module traversal from retaining or
            # rewriting this private, reloadable tree as resident parameters.
            object.__setattr__(
                self,
                "_stream_slot",
                SimpleNamespace(manager=manager, path=path, block=block),
            )
            self.resident_weights = retained

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                # Only metadata access is supported outside a forward. Methods
                # such as mixed() would bypass the ownership/evaluation window.
                slot = object.__getattribute__(self, "_stream_slot")
                value = getattr(slot.block, name)
                if callable(value):
                    raise StreamingUnavailable(
                        f"streamed layer method {name!r} is unsupported"
                    )
                return value

        def __call__(self, *args, **kwargs):
            slot = self._stream_slot
            with slot.manager.acquire(slot.path):
                out = slot.block(*args, **kwargs)
                mx.eval(out)
                _eval_state(args)
                _eval_state(kwargs)
                return out

    _LAYER_CLASS = StreamedDecoderLayer
    return _LAYER_CLASS


class LayerStreamManager(_ManagerBase):
    mode = "dense_layers"

    def __init__(self, *, plan, stats, pool, shards, blocks, readers, streamed_keys):
        self.plan, self.stats = plan, stats
        self._pool, self._handles = pool, shards
        self.blocks, self.readers = blocks, readers
        self.streamed_keys = tuple(streamed_keys)
        self.forced_stock = {"whole_layer_ordinary": len(blocks)}
        self._cache = OrderedDict()
        self._active = None
        self.hits = self.evictions = self.bypasses = self.serving_bypasses = 0
        self._failed = False

    def reserved_bytes(self):
        return self.plan.ceiling_bytes

    def live_bytes(self):
        return sum(self.readers[path].nbytes for path in self._cache)

    def _evict(self, path):
        _set_weights(self.blocks[path], dict.fromkeys(self.readers[path].specs))
        del self._cache[path]
        self.stats.remove_resident(self.readers[path].nbytes)
        self.evictions += 1
        self.stats.bump("evictions")

    @contextmanager
    def acquire(self, path):
        if self._handles is None or self._failed:
            raise StreamingUnavailable("dense layer cache is closed or failed")
        if self._active is not None:
            raise StreamingUnavailable("dense layer cache requires serial forwards")
        self._active = path
        temporary = False
        try:
            # Dense inference scans all layers each token. Plain LRU would
            # evict a useful earlier block on every miss and thrash the entire
            # cache. Reserve a progress window and bypass cache admission for
            # the tail of the scan, preserving already-cached blocks.
            retain_limit = self.reserved_bytes() - 2 * self.plan.max_layer_bytes
            while self._cache and self.live_bytes() > retain_limit:
                self._evict(next(iter(self._cache)))
            if path in self._cache:
                self.hits += 1
                self.stats.bump("hits")
                self._cache.move_to_end(path)
            else:
                reader = self.readers[path]
                # Read buffers and device arrays coexist until load completes.
                while (
                    self._cache
                    and self.live_bytes() + 2 * reader.nbytes > self.reserved_bytes()
                ):
                    self._evict(next(iter(self._cache)))
                arrays = reader.load()
                _set_weights(self.blocks[path], arrays)
                temporary = self.live_bytes() + reader.nbytes > retain_limit
                self.bypasses += int(temporary)
                if self.stats.phase == "serve":
                    self.serving_bypasses += int(temporary)
                self._cache[path] = True
                self.stats._group_transient = 0
                self.stats.add_resident(reader.nbytes)
                del arrays
            yield self.blocks[path]
        except BaseException:
            # A failed forward can leave lazy state depending on its arrays.
            # Retain it and refuse reuse until the route owner closes it.
            self._failed = True
            raise
        finally:
            if temporary and not self._failed:
                self._evict(path)
            self._active = None

    def counters(self):
        result = {
            "dense_stream_page_ins_total": self.stats.page_ins,
            "dense_stream_page_in_bytes_total": self.stats.page_in_bytes,
            "dense_stream_load_page_ins_total": self.stats.load_page_ins,
            "dense_stream_load_page_in_bytes_total": self.stats.load_page_in_bytes,
            "dense_stream_peak_tracked_bytes": self.stats.peak_tracked_bytes,
            "dense_layer_cache_hits_total": self.stats.hits,
            "dense_layer_evictions_total": self.stats.evictions,
            "dense_layer_cache_bypasses_total": self.serving_bypasses,
        }
        if self._pool is not None:
            result.update(self._pool.counters())
        return result

    def receipt(self):
        return {
            **super().receipt(),
            "layer_zero_resident": True,
            "resident_paths": list(self.plan.resident_paths),
            "retained_tensor_paths": list(self.plan.retained_tensor_paths),
            "retained_tensor_bytes": self.plan.retained_tensor_bytes,
            "cached_layers": list(self._cache),
            "live_bytes": self.live_bytes(),
            "cache_policy": "retain_then_bypass",
            "failed": self._failed,
        }

    def close(self):
        if self._active is not None:
            raise StreamingUnavailable("cannot close an active dense layer cache")
        _close_read_pool(self._pool, self._handles)
        self._pool = self._handles = None
        for path in list(self._cache):
            self._evict(path)


def trunk_layer_targets(model, *, layers_prefix):
    """Adapter-selected decoder list, excluding its first block and MTP."""
    return [
        path
        for path, _ in model.named_modules()
        if path.startswith(layers_prefix)
        and not is_mtp_path(path)
        and path[len(layers_prefix) :].isdigit()
        and int(path[len(layers_prefix) :]) > 0
    ]


def discover_layer_targets(model):
    """Resolve one decoder sequence by object identity, never by family name."""
    modules = dict(model.named_modules())
    layers = getattr(model, "layers", None)
    if isinstance(layers, (list, tuple)) and len(layers) > 1:
        candidates = []
        for path, module in modules.items():
            if not path.endswith(".0") or module is not layers[0] or is_mtp_path(path):
                continue
            prefix = path[:-1]
            if all(
                modules.get(prefix + str(i)) is layer for i, layer in enumerate(layers)
            ):
                candidates.append(prefix)
    else:
        sequences = {}
        for path, module in modules.items():
            parent, _, index = path.rpartition(".")
            if parent and index.isdigit() and not is_mtp_path(path):
                sequences.setdefault(parent, {})[int(index)] = module
        candidates = [
            parent + "."
            for parent, sequence in sequences.items()
            if len(sequence) > 1
            and set(sequence) == set(range(len(sequence)))
            and len({type(m) for m in sequence.values()}) == 1
        ]
    if len(candidates) != 1:
        raise StreamingUnavailable(
            "dense streaming needs one unambiguous decoder sequence"
        )
    prefix = candidates[0]
    return sorted(
        [
            path
            for path in modules
            if path.startswith(prefix)
            and path[len(prefix) :].isdigit()
            and int(path[len(prefix) :]) > 0
        ],
        key=lambda path: int(path[len(prefix) :]),
    )


def _ordinary_linear_tensor(modules, layer_path, key):
    """Page stock linear arrays; retain norms, convolutions and state constants.

    Unknown/custom storage stays resident and is charged to the remainder.
    Transformed stock-linear arrays still fail source proof, never silently
    receive an offload credit. Subclass math is not inferred from a name.
    """
    from mlx import nn

    relative, _, _ = key.rpartition(".")
    parent = layer_path + ("." + relative if relative else "")
    return type(modules.get(parent)) in (nn.Linear, nn.QuantizedLinear)


def install_layer_streaming(
    model,
    *,
    targets,
    shards,
    resolver,
    ceiling_bytes,
    read_workers=DEFAULT_READ_WORKERS,
    resident_tensor=None,
):
    """Install only after resolving every layer; never page decoder layer 0."""
    from mlx.utils import tree_flatten

    from .models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    pool = None
    try:
        if not targets or len(set(targets)) != len(targets):
            raise StreamingUnavailable(
                "dense layer targets must be nonempty and unique"
            )
        modules = dict(model.named_modules())
        if any(
            isinstance(m, (QuantizedSwitchLinear, SwitchLinear))
            for m in modules.values()
        ):
            raise StreamingUnavailable(
                "dense layer streaming cannot offload routed expert modules"
            )
        blocks, specs, retained, resident_paths = {}, {}, {}, set()
        for path in targets:
            parent, _, leaf = path.rpartition(".")
            if not parent or not leaf.isdigit() or int(leaf) == 0 or is_mtp_path(path):
                raise StreamingUnavailable(
                    f"invalid offloaded layer {path!r}; layer 0 must stay resident"
                )
            resident_paths.add(parent + ".0")
            block = modules.get(path)
            if block is None or any(other.startswith(path + ".") for other in targets):
                raise StreamingUnavailable(
                    f"invalid or overlapping decoder layer {path!r}"
                )
            tensors = tree_flatten(block.parameters())
            if not tensors:
                raise StreamingUnavailable(f"decoder layer {path!r} has no parameters")
            blocks[path] = block
            specs[path] = {}
            retained[path] = {}
            for key, value in tensors:
                if not _ordinary_linear_tensor(modules, path, key) or (
                    resident_tensor is not None and resident_tensor(path, key)
                ):
                    retained[path][key] = value
                    continue
                relative, _, name = key.rpartition(".")
                specs[path][key] = resolver(
                    path + ("." + relative if relative else ""), name, value
                )
        if any(path not in modules for path in resident_paths):
            raise StreamingUnavailable("dense layer list has no resident layer 0")
        for path, table in specs.items():
            if not table:
                raise StreamingUnavailable(
                    f"decoder layer {path!r} has no streamable parameters"
                )
        sizes = {
            path: sum(loc.nbytes for loc in table.values())
            for path, table in specs.items()
        }
        largest = max(sizes.values())
        if type(ceiling_bytes) is not int or ceiling_bytes < 2 * largest:
            raise WorkingSetTooSmall(
                f"dense layer budget must hold twice the largest layer ({2 * largest} B)"
            )
        manifest = _manifest(
            [
                [path, key, [_location_entry(loc)], 0]
                for path, table in specs.items()
                for key, loc in sorted(table.items())
            ],
            shards,
        )
        pool = _ReadPool(
            max_workers=read_workers,
            thread_name_prefix="layer-stream",
            timeout=READ_TIMEOUT_SECONDS,
            owner=storage_owner("weights", shards.root),
        )
        stats = StreamStats(phase="load")
        readers = {
            path: ProjectionReader(table, handles=shards, pool=pool, stats=stats)
            for path, table in specs.items()
        }
        manager = LayerStreamManager(
            plan=LayerStreamPlan(
                len(targets),
                sum(sizes.values()),
                largest,
                ceiling_bytes,
                manifest,
                tuple(sorted(resident_paths)),
                tuple(f"{p}.{k}" for p, table in retained.items() for k in table),
                sum(v.nbytes for table in retained.values() for v in table.values()),
            ),
            stats=stats,
            pool=pool,
            shards=shards,
            blocks=blocks,
            readers=readers,
            streamed_keys=[
                f"{path}.{key}" for path, table in specs.items() for key in table
            ],
        )
        cls = _layer_class()
        for path, block in blocks.items():
            _set_weights(block, dict.fromkeys(specs[path]))
            _swap_module(model, path, cls(manager, path, block, retained[path]))
        return manager
    except BaseException:
        _close_read_pool(pool, shards)
        raise
