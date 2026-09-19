"""Side-effect-free logits-processor probes for provisional draft tokens."""

from __future__ import annotations

import copy
import enum
import types


_IMMUTABLE_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    range,
    slice,
    enum.Enum,
    types.CodeType,
)
_FUNCTION_TYPES = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.BuiltinMethodType,
)


def _preserve_by_identity(value):
    """Whether probe isolation must reuse this immutable/shared object."""
    return (
        isinstance(value, _IMMUTABLE_TYPES + _FUNCTION_TYPES + (types.ModuleType, type))
        # Extension functions (for example ``mlx.gc_func``) need not use a
        # standard Python function type, but a stateless callable without an
        # instance dictionary is still shared code rather than mutable state.
        or (callable(value) and not hasattr(value, "__dict__"))
    )


def _seed_preserved_references(value, memo, seen):
    """Keep modules/code/immutable values out of a nested ``deepcopy``."""
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if _preserve_by_identity(value):
        memo[identity] = value
        return
    if isinstance(value, dict):
        items = (*value.keys(), *value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = value
    else:
        state = getattr(value, "__dict__", None)
        items = state.values() if isinstance(state, dict) else ()
    for item in items:
        _seed_preserved_references(item, memo, seen)


def _copy_mutable_state(value):
    """Copy mutable probe state while sharing modules, code, and values."""
    if _preserve_by_identity(value):
        return value
    memo = {}
    _seed_preserved_references(value, memo, set())
    return copy.deepcopy(value, memo)


def _copy_function(function):
    """Clone a Python processor function, including mutable closure cells."""

    def make_cell(value):
        return (lambda: value).__closure__[0]

    closure = function.__closure__
    if closure is not None:
        closure = tuple(
            make_cell(_copy_mutable_state(cell.cell_contents)) for cell in closure
        )
    clone = types.FunctionType(
        function.__code__,
        function.__globals__,
        name=function.__name__,
        argdefs=_copy_mutable_state(function.__defaults__),
        closure=closure,
    )
    clone.__kwdefaults__ = _copy_mutable_state(function.__kwdefaults__)
    clone.__annotations__ = dict(function.__annotations__)
    clone.__dict__.update(_copy_mutable_state(function.__dict__))
    clone.__qualname__ = function.__qualname__
    clone.__module__ = function.__module__
    return clone


def isolated_logits_processor(processor):
    """Return a callable whose mutations cannot reach the target-path owner."""

    probe = getattr(processor, "probe", None)
    if callable(probe):
        # Stateless functions publish themselves here; stateful structured
        # processors publish a bound method that snapshots only lane state.
        return probe
    if isinstance(processor, types.FunctionType):
        return _copy_function(processor)
    return _copy_mutable_state(processor)


def probe_logits_processors(processors, tokens, logits):
    """Apply draft-side processors through isolated, disposable state."""

    value = logits
    for processor in processors:
        value = isolated_logits_processor(processor)(tokens, value)
    return value


__all__ = ["isolated_logits_processor", "probe_logits_processors"]
