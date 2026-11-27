"""Tree-verify executors: one topology, transaction, revision binder and choice.

Every AST target-tree executor verifies a tile of parent-first rows (row 0 is
the anchor, ``parents[r] < r``) against a request's committed cache, then
publishes exactly one root path.  The pieces that do not depend on the model
live here, parameterised by spec fields instead of copied per executor:

* :class:`TopologySpec` / :class:`TreeTopology` -- validation, root paths,
  depths, depth levels, ancestor masks and ancestry history windows.  The row
  cap, root rule and parent type rule are spec fields set to each executor's
  current value (see ``LEGACY_TOPOLOGIES``); unifying them would change which
  trees an executor accepts and is a proposal, not a migration.
* :class:`TreeLaneTransaction` -- validate every lane's path, prepare every
  lane, publish all prepared descriptors, and close (and release lane
  material) in ``finally`` whatever happens.
* :class:`SourceBoundRevision` -- a per-executor source manifest bound at
  construction and revalidated stat-then-hash on the hot path.
* :func:`tree_executor_choice` / :func:`build_tree_executor` -- the
  serial-reference versus packed-provider selection each adapter repeated.
* :class:`TreeMixerOps` -- the per-mixer interface (branch a parent state, run
  the tile rows, prepare a path's commit, publish).  The commit halves for KV,
  recurrent (KDA/GDN/short-conv) terminal state, Mamba replay and descriptor
  snapshots are implemented here as thin adapters; the forward math stays in
  each model's tree module and is reached through these adapters only.

Original mlx2 code; no model names appear below.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import mlx.core as mx

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

ROOT_SINGLE_ROW0 = "single_root_row0"   # row 0 parent -1, every other row 0 <= p < r
ROOT_ANY_NEGATIVE = "any_negative"      # any negative parent is a root; else p < r

TYPE_STRICT_INT = "strict_int"                    # type(p) is int, every row
TYPE_STRICT_INT_EXCEPT_ROOT = "strict_int_except_root"  # row 0 only compared to -1
TYPE_COERCE_INT = "coerce_int"                    # int(p), errors propagate


@dataclass(frozen=True)
class TopologySpec:
    """Which parent arrays an executor accepts, and what it says otherwise."""

    max_rows: int | None = 16
    min_rows: int = 1
    root_rule: str = ROOT_SINGLE_ROW0
    type_rule: str = TYPE_STRICT_INT
    rows_error: str = "tree tile must contain 1..16 rows"
    root_error: str = "tree parents must form one parent-first rooted tree"
    order_error: str | None = None      # None: root_error
    path_error: str = "tree commit must follow one complete root path"

    def __post_init__(self):
        if self.root_rule not in (ROOT_SINGLE_ROW0, ROOT_ANY_NEGATIVE):
            raise ValueError(f"unknown tree root rule {self.root_rule!r}")
        if self.type_rule not in (
            TYPE_STRICT_INT, TYPE_STRICT_INT_EXCEPT_ROOT, TYPE_COERCE_INT
        ):
            raise ValueError(f"unknown tree parent type rule {self.type_rule!r}")

    @classmethod
    def from_config(cls, config: Any = None, **overrides) -> "TopologySpec":
        """Topology is a property of the executor, not of the checkpoint.

        Checkpoint configs carry no tree fields; this exists for the common
        technology shape and accepts ``max_tree_rows`` when a config sets it.
        """
        from .dispatch import spec_from_config

        return spec_from_config(cls, config or {}, {"max_rows": "max_tree_rows"}, **overrides)


def validate_parents(parents: Sequence[Any], spec: TopologySpec) -> tuple[int, ...]:
    """Validate ``parents`` under ``spec``; return them as a tuple.

    Pass one checks the row count, parent types and the root rule over every
    row (``root_error``); pass two checks that each non-root parent precedes
    its row (``order_error``).  The accepted set equals each legacy copy's.
    """
    values = tuple(parents)
    rows = len(values)
    if rows < spec.min_rows or (spec.max_rows is not None and rows > spec.max_rows):
        raise ValueError(spec.rows_error)
    if spec.type_rule == TYPE_COERCE_INT:
        values = tuple(int(parent) for parent in values)
    elif spec.type_rule == TYPE_STRICT_INT:
        if any(type(parent) is not int for parent in values):
            raise ValueError(spec.root_error)
    elif spec.type_rule == TYPE_STRICT_INT_EXCEPT_ROOT:
        if any(type(parent) is not int for parent in values[1:]):
            raise ValueError(spec.root_error)
    if spec.root_rule == ROOT_SINGLE_ROW0:
        if rows and (values[0] != -1 or any(parent < 0 for parent in values[1:])):
            raise ValueError(spec.root_error)
    order = spec.order_error or spec.root_error
    for row, parent in enumerate(values):
        if not parent < 0 and parent >= row:
            raise ValueError(order)
    return values


class TreeTopology:
    """Validated parent-first rows and everything derived from them."""

    __slots__ = ("spec", "parents", "paths", "depths", "_levels", "_windows", "_parents_array")

    def __init__(self, parents: Sequence[Any], spec: TopologySpec):
        self.spec = spec
        self.parents = validate_parents(parents, spec)
        paths: list[tuple[int, ...]] = []
        for row, parent in enumerate(self.parents):
            paths.append((row,) if parent < 0 else (*paths[parent], row))
        self.paths = tuple(paths)
        self.depths = tuple(len(path) - 1 for path in self.paths)
        self._levels = None
        self._windows: dict = {}
        self._parents_array = None

    def __len__(self):
        return len(self.parents)

    # -- derived views --------------------------------------------------------

    def path_lists(self) -> list[list[int]]:
        """Root paths as fresh lists (the legacy list-of-lists shape)."""
        return [list(path) for path in self.paths]

    @property
    def levels(self) -> list[list[int]]:
        """Rows grouped by depth, shallowest first."""
        if self._levels is None:
            self._levels = [
                [row for row, depth in enumerate(self.depths) if depth == level]
                for level in range(max(self.depths, default=-1) + 1)
            ]
        return self._levels

    def chain_prefix(self) -> list[bool]:
        """Rows whose root path is ``0..depth`` (a contiguous prefix view)."""
        return [list(path) == list(range(len(path))) for path in self.paths]

    def leaf_representatives(self) -> list[int]:
        """A descendant leaf whose path contains each row, including the leaf itself."""
        leaves = list(range(len(self.parents)))
        for row in range(len(self.parents) - 1, -1, -1):
            parent = self.parents[row]
            if parent >= 0:
                leaves[parent] = leaves[row]
        return leaves

    def ancestor_rows(self) -> list[list[bool]]:
        """``[row][column]``: column is ``row`` or one of its ancestors."""
        width = len(self.parents)
        rows = [[False] * width for _ in range(width)]
        for row, path in enumerate(self.paths):
            for node in path:
                rows[row][node] = True
        return rows

    def parents_array(self) -> mx.array:
        if self._parents_array is None:
            self._parents_array = mx.array(self.parents, dtype=mx.int32)
        return self._parents_array

    def history_rows(self, state_rows: int, window: int, *, include_self: bool) -> list[list[int]]:
        """Each node's last ``window`` inputs along its true ancestry.

        Indices address ``[state rows; tile rows]``: ``0..state_rows-1`` is the
        committed history, ``state_rows + r`` is tile row ``r``.  Siblings never
        appear in each other's windows.
        """
        windows = []
        for path in self.paths:
            rows = path if include_self else path[:-1]
            history = [*range(state_rows), *(state_rows + row for row in rows)]
            windows.append(history[-window:] if window else [])
        return windows

    def history_array(self, state_rows: int, window: int, *, include_self: bool) -> mx.array:
        key = (state_rows, window, include_self)
        cached = self._windows.get(key)
        if cached is None:
            cached = mx.array(self.history_rows(state_rows, window, include_self=include_self),
                              dtype=mx.int32)
            self._windows[key] = cached
        return cached

    # -- publication ----------------------------------------------------------

    def is_root_path(self, path: Sequence[Any], *, allow_empty: bool = False) -> bool:
        path = list(path)
        if not path:
            return allow_empty
        return (
            all(type(row) is int and 0 <= row < len(self.paths) for row in path)
            and path == list(self.paths[path[-1]])
        )

    def validate_path(self, path: Sequence[Any], *, allow_empty: bool = False) -> list[int]:
        """Return ``path`` as a list if it is one complete root path."""
        path = list(path)
        if not self.is_root_path(path, allow_empty=allow_empty):
            raise ValueError(self.spec.path_error)
        return path


def tree_depths(parents: Sequence[Any], spec: TopologySpec) -> list[int]:
    return list(TreeTopology(parents, spec).depths)


def tree_path_lists(parents: Sequence[Any], spec: TopologySpec) -> list[list[int]]:
    return TreeTopology(parents, spec).path_lists()


# Each executor's current acceptance law.  Row caps are 16 or 32; the
# multi-root (any negative parent) laws belong to kernel-level helpers that
# never see a scheduler tile directly.
LEGACY_TOPOLOGIES: Mapping[str, TopologySpec] = {
    "hybrid_serial": TopologySpec(
        16, rows_error="hybrid AST tile must contain 1..16 matching rows",
        root_error="hybrid AST parents must form one parent-first rooted tree"),
    "ling_packed": TopologySpec(
        32, rows_error="Ling tree requires 1..32 rows",
        root_error="Ling tree requires one parent-first root",
        path_error="Ling tree commit must follow one complete root path"),
    "nemotron_packed": TopologySpec(
        16, rows_error="Nemotron tree requires 1..16 rows",
        root_error="Nemotron tree requires one parent-first root",
        path_error="Nemotron tree commit must follow one complete root path"),
    "inkling_packed": TopologySpec(
        16, rows_error="Inkling trees require 1..16 rows",
        root_error="Inkling tree must have one parent-first root",
        path_error="Inkling commit must follow one complete root path"),
    "attention_tile": TopologySpec(
        32, type_rule=TYPE_COERCE_INT,
        rows_error="attention tree tile rows must match parents and fit 32 rows",
        root_error="attention tree tile must be rooted at its anchor row",
        order_error="tree parents must precede their children",
        path_error="attention tree publication needs one root-connected path"),
    "attention_depths": TopologySpec(
        None, 0, ROOT_ANY_NEGATIVE, TYPE_COERCE_INT,
        root_error="tree parents must precede their children"),
    "qwen_tree_gdn_paths": TopologySpec(
        None, 0, ROOT_ANY_NEGATIVE, TYPE_COERCE_INT,
        root_error="parents must come before their children"),
    "dflash_tree_paths": TopologySpec(
        None, 0, ROOT_ANY_NEGATIVE, TYPE_COERCE_INT,
        root_error="tree parents must precede their children"),
    "flash_tree_ancestry": TopologySpec(
        16, rows_error="Flash tree requires one parent-first root and 1..16 rows",
        root_error="Flash tree requires one parent-first root and 1..16 rows",
        path_error="Flash tree publication is not one root-connected path"),
    "qwen38_rooted_tree": TopologySpec(16),
    "glm_rooted_tree": TopologySpec(
        32, type_rule=TYPE_STRICT_INT_EXCEPT_ROOT,
        rows_error="GLM-5.3 AST requires a parent-ordered rooted tree of at most32 rows",
        root_error="GLM-5.3 AST requires a parent-ordered rooted tree of at most32 rows",
        path_error="GLM-5.3 AST commit requires a root-connected path"),
    "packed_attention_tile": TopologySpec(
        32, type_rule=TYPE_STRICT_INT_EXCEPT_ROOT,
        rows_error="packed tree needs uniform widths in 1..32",
        root_error="packed tree parents must form a rooted topological tree"),
    "q8_attention_tile": TopologySpec(
        32, type_rule=TYPE_STRICT_INT_EXCEPT_ROOT,
        rows_error="Q8 tree needs uniform tile widths in1..32",
        root_error="Q8 tree parents must form one rooted topological tree",
        path_error="Q8 tree commit needs a root-connected path"),
}


def legacy_topology(name: str) -> TopologySpec:
    return LEGACY_TOPOLOGIES[name]


def is_rooted_tree(parents: Sequence[Any], spec: TopologySpec) -> bool:
    try:
        validate_parents(parents, spec)
    except (TypeError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def publish_descriptors(prepared: Sequence[tuple[Any, dict]]) -> None:
    """Replace each live descriptor's fields with its prepared fields."""
    for live, fields in prepared:
        live.__dict__.clear()
        live.__dict__.update(fields)


class TreeLaneTransaction:
    """Open target-tree transaction over independent request lanes.

    Each lane (one request's tree) provides ``validate_path(path)``,
    ``prepare_commit(path) -> [(live_descriptor, fields), ...]`` and ``live``;
    ``release()`` is optional.  ``commit_paths`` validates every lane, prepares
    every lane, then publishes them all; once preparation starts the
    transaction closes (releasing lane material) in ``finally`` on success
    and on any failure.

    ``close_on_refused_path`` names what a refused path does.  ``False`` (the
    repository contract, pinned by the Ling/Inkling/attention/Flash tests): the
    refusal raises before anything is prepared and the transaction stays open
    for a valid path or an abort.  ``True`` (Nemotron's current behaviour):
    the refusal closes and releases the transaction.
    """

    executor_cached = False
    physical_packed = False
    protocol = "tree_lane_transaction_v1"
    count_error = "tree commit lane count mismatch or closed transaction"
    close_on_refused_path = False

    def __init__(self, trees, token, *, protocol=None, publish=publish_descriptors):
        self.trees = trees
        self.closed = False
        self.provider_protocol = protocol or type(self).protocol
        self.provider_revision_token = token
        self._publish = publish

    def commit_paths(self, paths):
        if self.closed or len(paths) != len(self.trees):
            raise RuntimeError(self.count_error)
        if not self.close_on_refused_path:
            paths = self._validated_paths(paths)
        try:
            if self.close_on_refused_path:
                paths = self._validated_paths(paths)
            prepared = []
            for tree, path in zip(self.trees, paths, strict=True):
                prepared.extend(tree.prepare_commit(path))
            self._publish(prepared)
            return [tree.live for tree in self.trees]
        finally:
            self._close()

    def _validated_paths(self, paths):
        # A single-use path must reach preparation exactly as it was validated.
        paths = [list(path) for path in paths]
        for tree, path in zip(self.trees, paths, strict=True):
            tree.validate_path(path)
        return paths

    def commit(self, *, accepted_lengths):
        return self.commit_paths([list(range(int(length))) for length in accepted_lengths])

    def abort(self):
        self._close()

    def _close(self):
        self.closed = True
        for tree in self.trees:
            release = getattr(tree, "release", None)
            if release is not None:
                release()


# ---------------------------------------------------------------------------
# Revision binding
# ---------------------------------------------------------------------------


_TECHNOLOGIES = Path(__file__).parent
TREE_VERIFY_SOURCE = Path(__file__)
# Every technology module (the package's own files and subpackages): the
# mixers' tree math, the shared dispatch, and this module.  Bound as a whole,
# so a new or edited technology module moves every tree revision.  Over-binding
# is deliberate; nothing here may be absent.
_FIRST = (TREE_VERIFY_SOURCE, _TECHNOLOGIES / "dispatch.py")
TECHNOLOGY_SOURCES = _FIRST + tuple(sorted(
    path for path in _TECHNOLOGIES.rglob("*.py")
    if "__pycache__" not in path.parts and path not in _FIRST))
TECHNOLOGY_SOURCE_PATHS = tuple(
    "src/mlx2/runtime/technologies/" + path.relative_to(_TECHNOLOGIES).as_posix()
    for path in TECHNOLOGY_SOURCES)


def source_stat(path: Path):
    """Metadata that moves on any write: device, inode, size, mtime, ctime."""
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def file_sha256(path: Path) -> str:
    """sha256 of a source."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_identity(sources: Sequence[Path], key: Callable[[Path], str]) -> tuple[tuple[str, str], ...]:
    names = [key(path) for path in sources]
    counts = Counter(names)
    collisions = [path for path, name in zip(sources, names) if counts[name] > 1]
    if collisions:
        # Receipts expose this manifest as a dictionary. Qualify colliding
        # basenames while preserving existing names for unambiguous sources.
        root = Path(os.path.commonpath([path.parent for path in collisions]))
        names = [path.relative_to(root).as_posix() if counts[name] > 1 else name
                 for path, name in zip(sources, names)]
    return tuple((name, file_sha256(path)) for name, path in zip(names, sources))


class SourceBoundRevision:
    """Mixin binding an executor to the exact bytes of its source manifest.

    State lives on the executor (``_sources``, ``source_identity``,
    ``_source_stats``) so tests and receipts read it directly.  The identity
    style is a class field:

    * ``"name"`` -- ``(file name, sha256)`` per source;
    * ``"relative"`` -- ``(path relative to _source_root, sha256)``;
    * ``"module"`` -- ``(("module", _source_module), ("sha256", h))`` for the
      first source, then ``(file name, sha256)`` for any further source.

    ``validate_revision`` re-hashes only when a source's (device, inode,
    size, mtime, ctime) moved, then refuses changed bytes and a source that
    moved again during validation.
    """

    _identity_style = "name"
    _source_root: Path | None = None
    _source_module: str | None = None
    _revision_error = "tree executor source changed"

    def _source_key(self, path: Path) -> str:
        if self._identity_style == "relative":
            return path.relative_to(self._source_root).as_posix()
        return path.name

    def _identity(self):
        if self._identity_style == "module":
            first, *rest = self._sources
            return (("module", self._source_module), ("sha256", file_sha256(first)),
                    *manifest_identity(rest, lambda path: path.name))
        return manifest_identity(self._sources, self._source_key)

    def _bind_sources(self, sources: Sequence[Path]) -> None:
        # A source listed twice (explicitly and via TECHNOLOGY_SOURCES) is bound once.
        self._sources = tuple(dict.fromkeys(Path(path) for path in sources))
        before = tuple(source_stat(path) for path in self._sources)
        self.source_identity = self._identity()
        self._source_stats = tuple(source_stat(path) for path in self._sources)
        if before != self._source_stats:
            raise RuntimeError(self._revision_error + " during binding")

    def validate_revision(self):
        stats = tuple(source_stat(path) for path in self._sources)
        if stats == self._source_stats:
            return
        if self._identity() != self.source_identity:
            raise RuntimeError(self._revision_error)
        if tuple(source_stat(path) for path in self._sources) != stats:
            raise RuntimeError(self._revision_error + " during validation")
        self._source_stats = stats


def revision_token(material: Any) -> str:
    """sha256 of ``repr(material)``, the packed executors' token law."""
    return hashlib.sha256(repr(material).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Serial reference versus packed provider
# ---------------------------------------------------------------------------

SERIAL = "serial"
PACKED = "packed"


def tree_executor_choice(policy: Mapping[str, Any], *, key: str = "ast_target_executor",
                         default: str = PACKED, explicit: str = SERIAL) -> str:
    """``"serial"`` or ``"packed"`` from an adapter policy.

    ``explicit`` is the value that must be spelled out to select it; every
    other value (including a missing key read as ``default``) selects the
    other executor.  Adapters keep their current key and default.
    """
    value = policy.get(key, default)
    if explicit == SERIAL:
        return SERIAL if value == SERIAL else PACKED
    if explicit == PACKED:
        return PACKED if value == PACKED else SERIAL
    raise ValueError(f"unknown explicit tree executor {explicit!r}")


def _executor_registry():
    """Serial reference is the registry reference; packed is a policy path.

    Counted as ``mlx2_technology_dispatch_total{technology="tree_verify",
    operation="executor", path="reference"|"packed"}``.  The packed path never
    refuses on its own: the adapter's policy is the choice.
    """
    from .dispatch import FastPath, KernelRegistry

    registry = KernelRegistry("tree_verify.executor", lambda serial, packed: serial())
    registry.register(FastPath(PACKED, lambda spec, **_: None,
                               lambda serial, packed: packed()))
    return registry


_EXECUTOR_REGISTRY = None


def build_tree_executor(choice: str, *, serial: Callable[[], Any], packed: Callable[[], Any],
                        rows: int | None = None, serial_max_rows: int | None = None,
                        serial_rows_error: str = "serial tree reference tile rows exceed its cap"):
    """Construct the chosen executor; refuse a serial choice above its row cap."""
    global _EXECUTOR_REGISTRY
    if choice not in (SERIAL, PACKED):
        raise ValueError(f"unknown tree executor choice {choice!r}")
    if (choice == SERIAL and serial_max_rows is not None and rows is not None
            and rows > serial_max_rows):
        raise ValueError(serial_rows_error)
    if _EXECUTOR_REGISTRY is None:
        _EXECUTOR_REGISTRY = _executor_registry()
    run, _receipt = _EXECUTOR_REGISTRY.select(
        None, policy="reference" if choice == SERIAL else PACKED)
    return run(serial, packed)


def check_serial_rows(choice: str, rows: int, *, serial_max_rows: int, error: str) -> None:
    if choice == SERIAL and rows > serial_max_rows:
        raise ValueError(error)


# ---------------------------------------------------------------------------
# Per-mixer ops
# ---------------------------------------------------------------------------


@runtime_checkable
class TreeMixerOps(Protocol):
    """One mixer's share of a tree verify.

    ``branch(parent_state)`` returns a private state a child row may advance;
    ``forward_rows(...)`` runs the mixer over the tile rows (the model's tree
    module owns this math); ``prepare_commit(...)`` builds the committed
    descriptor fields for one root path without touching the live cache; and
    ``publish(prepared)`` installs prepared fields.
    """

    mixer: str

    def branch(self, parent_state): ...

    def forward_rows(self, *args, **kwargs): ...

    def prepare_commit(self, *args, **kwargs): ...

    def publish(self, prepared) -> None: ...


KV_WINDOW_GATHER = "window_gather"
KV_WINDOW_GATHER_SKIP_CHAIN = "window_gather_skip_chain"
KV_RAW_ROWS_SKIP_CHAIN = "raw_rows_skip_chain"
KV_RAW_ROWS_SKIP_CHAIN_DTYPE_GUARD = "raw_rows_skip_chain_dtype_guard"


@dataclass(frozen=True)
class KVTreeOps:
    """Append-only attention KV: compact one root path behind the boundary.

    ``compaction`` names how the accepted rows are written to
    ``[start, start + len(path))``:

    * ``window_gather`` -- always gather from the private buffer's tile window;
    * ``window_gather_skip_chain`` -- gather from the private buffer only
      when the accepted rows do not occupy the leading tile positions;
    * ``raw_rows_skip_chain`` -- nothing for a chain prefix (rows already in
      place), else gather from the raw rows this round appended;
    * ``raw_rows_skip_chain_dtype_guard`` -- as above, but gather from the
      buffer window when the raw rows' dtype differs from the buffer's.
    """

    compaction: str = KV_WINDOW_GATHER
    mixer: str = "kv"

    def branch(self, parent_state):
        raise NotImplementedError("KV rows are lent append-only, never branched")

    def forward_rows(self, *args, **kwargs):
        raise NotImplementedError("tree attention math is owned by the model tree module")

    def prepare_commit(self, private, *, start, path, width, raw=None):
        path = list(path)
        keep = len(path)
        chain = path == list(range(keep))
        if self.compaction == KV_WINDOW_GATHER or (
            self.compaction == KV_WINDOW_GATHER_SKIP_CHAIN and not chain
        ):
            indices = mx.array(path, dtype=mx.int32)
            for field in ("keys", "values"):
                values = getattr(private, field)
                kept = mx.take(values[..., start:start + width, :], indices, axis=2)
                values[..., start:start + keep, :] = kept
                setattr(private, field, values)
        elif self.compaction in (KV_RAW_ROWS_SKIP_CHAIN, KV_RAW_ROWS_SKIP_CHAIN_DTYPE_GUARD):
            if not chain:
                indices = mx.array(path, dtype=mx.int32)
                for field, rows in zip(("keys", "values"), raw, strict=True):
                    values = getattr(private, field)
                    source = rows
                    if (self.compaction == KV_RAW_ROWS_SKIP_CHAIN_DTYPE_GUARD
                            and rows.dtype != values.dtype):
                        source = values[..., start:start + width, :]
                    values[..., start:start + keep, :] = mx.take(source, indices, axis=2)
                    setattr(private, field, values)
        elif self.compaction != KV_WINDOW_GATHER_SKIP_CHAIN:
            raise ValueError(f"unknown KV compaction {self.compaction!r}")
        private.offset = start + keep
        return private

    def publish(self, prepared):
        publish_descriptors(prepared)


@dataclass(frozen=True)
class RecurrentTerminalOps:
    """Recurrent planes (KDA / GDN / short-conv): commit the terminal node.

    The tree forward recorded every node's post-row planes; the committed
    state is the terminal row's, advanced by the path length.  ``planes``
    maps the recorded material to the cache's plane list for one row.
    """

    planes: Callable[[Any, int], list]
    mixer: str = "recurrent"

    def branch(self, parent_state):
        return parent_state

    def forward_rows(self, *args, **kwargs):
        raise NotImplementedError("recurrent tree math is owned by the model tree module")

    def prepare_commit(self, private, material, path):
        private.cache = self.planes(material, path[-1])
        private.advance(len(path))
        return private

    def publish(self, prepared):
        publish_descriptors(prepared)


def kda_terminal_planes(material, terminal):
    """``(conv q, conv k, conv v, states)`` recorded per row -> one row's planes."""
    qs, ks, vs, states = material
    return [qs[terminal:terminal + 1], ks[terminal:terminal + 1],
            vs[terminal:terminal + 1], states[terminal]]


def history_terminal_planes(count):
    """Per-plane row lists recorded per row -> one row's ``count`` planes."""
    def planes(material, terminal):
        return [material[index][terminal] for index in range(count)]
    return planes


@dataclass(frozen=True)
class MambaReplayOps:
    """Mamba-2: replay the accepted path's SSM from committed state.

    The mixer's own ``tree_commit_state(material, path, indices)`` replays the
    recurrence with the decode kernel; the conv tail is the path's last
    ``kernel - 1`` inputs over ``[committed history; tile rows]``.
    """

    mixer: str = "mamba"

    def branch(self, parent_state):
        raise NotImplementedError("Mamba branches by parent slots inside its tree kernel")

    def forward_rows(self, *args, **kwargs):
        raise NotImplementedError("Mamba tree math is owned by the model mixer")

    def prepare_commit(self, live, material, path, indices):
        import copy
        from collections import deque

        state = material["mixer"].tree_commit_state(material, path, indices)
        count = material["kernel"] - 1
        history = [*range(count), *(count + row for row in path)]
        tail = history[-count:] if count else []
        conv = mx.take(material["source"], mx.array(tail, dtype=mx.int32), axis=1)
        private = copy.copy(live)
        private.__dict__ = dict(live.__dict__)
        private.cache = [conv, state]
        private._rollbacks = deque()
        private.advance(len(path))
        return private

    def publish(self, prepared):
        publish_descriptors(prepared)


@dataclass(frozen=True)
class SnapshotBranchOps:
    """Serial reference: every node runs one-token decode on a private snapshot."""

    mixer: str = "snapshot"

    def branch(self, parent_state):
        from ..cow_cache import snapshot_prompt_cache_descriptors

        return snapshot_prompt_cache_descriptors(parent_state, require_committed=False)[0]

    def forward_rows(self, model, tokens, parents, live, layers):
        """``(logits, features, branches)``: one ordinary forward per row."""
        branches, logits, features = [], [], []
        for value, parent in zip(tokens, parents, strict=True):
            private = self.branch(live if parent < 0 else branches[parent])
            row_logits, row_features = model.forward_with_taps(
                mx.array([[int(value)]], dtype=mx.uint32), private, layers
            )
            branches.append(private)
            logits.append(row_logits)
            features.append(row_features)
        return mx.concatenate(logits, axis=1), mx.concatenate(features, axis=1), branches

    def prepare_commit(self, live, branches, terminal):
        selected = self.branch(branches[terminal])
        prepared = []
        for destination, source in zip(live, selected, strict=True):
            if type(destination) is not type(source):
                raise TypeError("hybrid tree cache type changed")
            prepared.append((destination, source.__dict__))
        return prepared

    def publish(self, prepared):
        publish_descriptors(prepared)


__all__ = [
    "KVTreeOps", "LEGACY_TOPOLOGIES", "MambaReplayOps", "PACKED", "RecurrentTerminalOps",
    "SERIAL", "SnapshotBranchOps", "SourceBoundRevision", "TopologySpec", "TreeLaneTransaction",
    "TreeMixerOps", "TreeTopology", "build_tree_executor", "check_serial_rows",
    "history_terminal_planes", "is_rooted_tree", "kda_terminal_planes", "legacy_topology",
    "manifest_identity", "publish_descriptors", "revision_token", "source_stat",
    "tree_depths", "tree_executor_choice", "tree_path_lists", "validate_parents",
]
