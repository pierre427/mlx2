"""Explicit developer reload of resident-model tensor code.

This is local code execution, not a Python sandbox. Serving owns admission,
GPU synchronization, APCv2 deletion and failure quarantine. This module never
loads a model. Existing classes and Python functions keep their identities;
their bodies change, and kernel factories/caches are recreated.
"""

from __future__ import annotations

import ast
import dataclasses
import dis
import functools
import gc
import hashlib
import importlib.util
import inspect
import json
import re
import sys
import threading
import types
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _contains_identity(values, expected):
    # Keep the alias scan's own old/new loop variables out of closure cells:
    # get_referrers would otherwise mistake and rewrite its bookkeeping cell.
    for value in values:
        if value is expected:
            return True
    return False


class KernelReloadError(RuntimeError):
    code = "kernel_reload_refused"
    status = 409


_OWNER_LOCK = threading.RLock()
_OWNERS: dict[int, object] = {}
_CHANGED_OWNERS: set[int] = set()


def register_owner(owner: object) -> None:
    with _OWNER_LOCK:
        if _CHANGED_OWNERS and id(owner) not in _OWNERS:
            raise KernelReloadError("A kernel development engine owns modified process code")
        _OWNERS[id(owner)] = owner


def unregister_owner(owner: object) -> None:
    with _OWNER_LOCK:
        _OWNERS.pop(id(owner), None)
        # Source and constants outlive the model. A fresh engine must not inherit
        # developer edits under an ordinary qualification receipt.


def assert_exclusive_owner(owner: object) -> None:
    with _OWNER_LOCK:
        if set(_OWNERS) != {id(owner)}:
            raise KernelReloadError("Kernel reload requires the only loaded model engine in this process")


@contextmanager
def exclusive_owner(owner: object):
    with _OWNER_LOCK:
        assert_exclusive_owner(owner)
        # Even a failed source execution may have changed process globals.
        _CHANGED_OWNERS.add(id(owner))
        yield


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODULE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
_FORBIDDEN = {
    "apc_v2", "cache", "cow_cache", "model_residency", "io_ownership",
    "checkpoint_resources", "layer_stream", "import_env", "eval_watchdog",
    "paged_apc_index", "mlx_build", "kernel_reload", "serving", "server",
    "scheduler", "executor", "process_env", "http_security", "adapters",
    "kernel_reload_serving", "process_globals", "weight_space", "weight_space_metrics",
    "weight_program_observer", "weight_overlays", "layer_excision", "expert_masks",
}
_CACHE_NAME = re.compile(r"cache|compiled|kernel|probe|memo", re.IGNORECASE)
_HELPER = "_mlx2_kernel_reload_compile"
_METAL_HELPER = "_mlx2_kernel_reload_metal"


def _plain_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Kernel settings and arguments must be finite JSON values") from error


def _target(value: Any, *, call: bool = False) -> tuple[str, str]:
    if not isinstance(value, str) or value.count(":") != 1:
        raise ValueError("A target must be module:attribute, adapter:method or model:method")
    module, name = value.split(":")
    if not _IDENT.fullmatch(name) or name.startswith("__"):
        raise ValueError("A target attribute must be one non-dunder identifier")
    if module in {"adapter", "model", "layers"}:
        if not call or not name.startswith(("set_", "configure_", "enable_", "disable_")):
            raise ValueError("Resident object targets only permit explicit configuration methods")
    elif not _MODULE.fullmatch(module):
        raise ValueError("A target module must be a fully qualified Python module")
    return module, name


def _signature(fn):
    code = fn.__code__
    return (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS),
            code.co_varnames[:code.co_argcount + code.co_kwonlyargcount], code.co_freevars)


def _code_law(code):
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
            tuple(_code_law(x) if isinstance(x, types.CodeType) else x for x in code.co_consts))


def _metadata_law(value):
    """Structural dataclass metadata without comparing/evaluating tensors."""
    if value is dataclasses.MISSING:
        return ("missing",)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    origin = typing.get_origin(value)
    if origin is not None:
        return ("annotation", _metadata_law(origin), tuple(_metadata_law(x) for x in typing.get_args(value)))
    if type(value) in {str, int, float, bool, bytes, type(None)}:
        return (type(value).__name__, value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_metadata_law(x) for x in value))
    if isinstance(value, (dict, types.MappingProxyType)):
        return ("mapping", tuple(sorted((repr(key), _metadata_law(item)) for key, item in value.items())))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, tuple(sorted(repr(_metadata_law(item)) for item in value)))
    if type(value) is types.FunctionType:
        return ("function", _code_law(value.__code__), _metadata_law(value.__defaults__),
                _metadata_law(value.__kwdefaults__),
                tuple(_metadata_law(cell.cell_contents) for cell in value.__closure__ or ()))
    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return ("builtin", value.__module__, value.__qualname__)
    # Unknown user metadata must retain identity; do not assume two objects of
    # the same type mean the same constructor/default behavior.
    return ("identity", id(value))


def _dataclass_fields_law(fields):
    if not isinstance(fields, dict) or not all(isinstance(field, dataclasses.Field) for field in fields.values()):
        raise KernelReloadError("Malformed dataclass field metadata requires restart")
    attributes = ("name", "type", "default", "default_factory", "init", "kw_only",
                  "repr", "compare", "hash", "metadata", "_field_type")
    return tuple((name, tuple((attribute, _metadata_law(getattr(field, attribute, None)))
                             for attribute in attributes)) for name, field in fields.items())


def _dataclass_params_law(params):
    attributes = ("init", "repr", "eq", "order", "unsafe_hash", "frozen",
                  "match_args", "kw_only", "slots", "weakref_slot")
    return tuple((name, _metadata_law(getattr(params, name, dataclasses.MISSING))) for name in attributes)


def _python_function(value):
    # MLX's _gc_func proxies __class__ and passes isinstance(..., FunctionType)
    # despite having no writable Python code object.
    if type(value) is types.FunctionType:
        return value
    if isinstance(value, functools._lru_cache_wrapper):
        return value.__wrapped__
    return None


def _compiled(value):
    kind = type(value)
    return kind.__module__ == "mlx.core" and kind.__name__ == "_gc_func"


def _array(value):
    return type(value).__module__.startswith("mlx") and type(value).__name__ == "array"


def _contains_array(value, seen=None):
    if _array(value):
        return True
    if not isinstance(value, (dict, tuple, list, set)):
        return False
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    return any(_contains_array(x, seen) for x in (value.values() if isinstance(value, dict) else value))


def resident_weight_identity(adapter):
    """Object IDs and parameter leaves, without materializing device arrays."""
    if adapter is None:
        return None
    roots = tuple((name, id(getattr(adapter, name, None))) for name in
                  ("model", "draft_model", "draft", "weights", "weight_provider", "provider"))
    leaves = []
    seen = set()

    def walk(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + (str(key),))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))
        elif _array(value) or (hasattr(value, "shape") and hasattr(value, "dtype")):
            leaves.append((path, id(value), tuple(value.shape), str(value.dtype)))
        elif isinstance(value, (memoryview, bytearray)):
            leaves.append((path, id(value), (len(value),), type(value).__name__))
        elif (value is not None and not isinstance(value, (types.ModuleType, types.FunctionType, type))
              and hasattr(value, "__dict__")):
            for key, item in vars(value).items():
                if not key.startswith("_lock"):
                    walk(item, path + (key,))

    walk(adapter, ())
    return (id(adapter), roots, tuple(sorted(leaves)))


@dataclass(frozen=True)
class _Source:
    name: str
    path: Path
    sha256: str
    source: bytes


@dataclass(frozen=True)
class ReloadPlan:
    sources: tuple[_Source, ...]
    settings: tuple[dict, ...]
    calls: tuple[dict, ...]
    generation: int

    @property
    def modules(self):
        return tuple(item.name for item in self.sources)


_STRUCTURAL = {"__init__", "__new__", "__setattr__", "__delattr__", "__getattribute__", "__del__"}


_OTHER = "?"  # A binding that is not an MLX import (parameter, assignment, other module).


class _Scope:
    def __init__(self, parent, kind):
        self.parent, self.kind = parent, kind
        self.bindings, self.globals, self.nonlocals = {}, set(), set()

    def bind(self, name, target=_OTHER):
        self.bindings.setdefault(name, set()).add(target)

    def lookup(self, name):
        """Python's lexical resolution; class bodies are invisible to nested scopes."""
        scope, first = self, True
        while scope is not None:
            if name in scope.globals:
                while scope.parent is not None:
                    scope = scope.parent
                return scope.bindings.get(name)
            if name not in scope.nonlocals and (first or scope.kind != "class") and name in scope.bindings:
                return scope.bindings[name]
            scope, first = scope.parent, False
        return None


def _mlx_scopes(tree):
    """Per-scope bindings of MLX imports, including function-local ones."""
    scopes = {id(tree): _Scope(None, "module")}

    def arguments(scope, args):
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
            if arg is not None:
                scope.bind(arg.arg)

    def collect(scope, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            if not isinstance(node, ast.Lambda):
                scope.bind(node.name)
            outer = [*getattr(node, "decorator_list", ()), *getattr(node, "bases", ()), *getattr(node, "keywords", ())]
            if isinstance(node, ast.ClassDef):
                body = node.body
            else:
                outer += [node.args, *([node.returns] if getattr(node, "returns", None) else [])]
                body = [node.body] if isinstance(node, ast.Lambda) else node.body
            for child in outer:
                collect(scope, child)
            inner = scopes[id(node)] = _Scope(scope, "class" if isinstance(node, ast.ClassDef) else "function")
            if not isinstance(node, ast.ClassDef):
                arguments(inner, node.args)
            for child in body:
                collect(inner, child)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                local, target = (alias.asname, alias.name) if alias.asname else ((alias.name.split(".")[0],) * 2)
                scope.bind(local, target if target == "mlx" or target.startswith("mlx.") else _OTHER)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                target = f"{node.module}.{alias.name}" if node.module and not node.level else _OTHER
                scope.bind(alias.asname or alias.name, target if target.startswith("mlx.") else _OTHER)
        elif isinstance(node, ast.Global):
            scope.globals.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            scope.nonlocals.update(node.names)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            scope.bind(node.id)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            scope.bind(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            scope.bind(node.rest)
        for child in ast.iter_child_nodes(node):
            collect(scope, child)

    for child in tree.body:
        collect(scopes[id(tree)], child)
    # A global/nonlocal declaration rebinds the outer name from inside.
    module = scopes[id(tree)]
    for scope in tuple(scopes.values()):
        for name in scope.globals & scope.bindings.keys():
            module.bindings.setdefault(name, set()).update(scope.bindings.pop(name))
        for name in scope.nonlocals & scope.bindings.keys():
            owner = scope.parent
            while owner is not None and (owner.kind != "function" or name not in owner.bindings):
                owner = owner.parent
            if owner is not None:
                owner.bindings[name].update(scope.bindings.pop(name))
    return scopes


_REWRITES = types.MappingProxyType({"mlx.core.compile": _HELPER, "mlx.core.fast.metal_kernel": _METAL_HELPER})


class _CompileGlobals(ast.NodeTransformer):
    def __init__(self, tree):
        self.scopes = _mlx_scopes(tree)
        self.stack = [self.scopes[id(tree)]]
        self.ambiguous = []

    def _paths(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return set(), False
        bound = self.stack[-1].lookup(node.id) or {_OTHER}
        suffix = "".join("." + part for part in reversed(parts))
        return {target + suffix for target in bound if target != _OTHER}, len(bound) > 1

    def _rewrite(self, node):
        if not isinstance(node.ctx, ast.Load):
            return None
        paths, ambiguous = self._paths(node)
        if not paths.intersection(_REWRITES):
            return None
        if ambiguous:
            # One name bound to an MLX factory and to something else in the
            # same scope cannot be rewritten safely: refuse the reload.
            self.ambiguous.append(node.lineno)
            return node
        return ast.copy_location(ast.Name(id=_REWRITES[paths.pop()], ctx=ast.Load()), node)

    def visit_Attribute(self, node):
        return self._rewrite(node) or self.generic_visit(node)

    def visit_Name(self, node):
        return self._rewrite(node) or node

    def _visit_fields(self, node, fields):
        for field in fields:
            value = getattr(node, field, None)
            if isinstance(value, list):
                setattr(node, field, [self.visit(item) if isinstance(item, ast.AST) else item for item in value])
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))

    def _scoped(self, node, outer, inner):
        self._visit_fields(node, outer)
        self.stack.append(self.scopes[id(node)])
        try:
            self._visit_fields(node, inner)
        finally:
            self.stack.pop()
        return node

    def visit_FunctionDef(self, node):
        if node.name in _STRUCTURAL:
            return node
        return self._scoped(node, ("decorator_list", "args", "returns"), ("body",))

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        return self._scoped(node, ("args",), ("body",))

    def visit_ClassDef(self, node):
        return self._scoped(node, ("decorator_list", "bases", "keywords"), ("body",))


def _unrewritten_metal_kernel(tree):
    """A kernel factory the rewrite missed would reuse its old Metal library name."""
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef) and node.name in _STRUCTURAL:
            continue  # Constructors are proven unchanged and keep their old code.
        if ((isinstance(node, ast.Attribute) and node.attr == "metal_kernel" and isinstance(node.ctx, ast.Load))
                or (isinstance(node, ast.Name) and node.id == "metal_kernel" and isinstance(node.ctx, ast.Load))):
            return node.lineno
        pending.extend(ast.iter_child_nodes(node))
    return None


def _reload_tree(source, path, name):
    tree = ast.parse(source, filename=str(path))
    rewrite = _CompileGlobals(tree)
    tree = ast.fix_missing_locations(rewrite.visit(tree))
    line = rewrite.ambiguous[0] if rewrite.ambiguous else _unrewritten_metal_kernel(tree)
    if line is not None:
        raise KernelReloadError(f"Cannot namespace the Metal kernel factory at {name} line {line}; "
                                "call mx.fast.metal_kernel through an mlx.core import or restart")
    return tree


class KernelReloader:
    def __init__(self, source_root):
        self.source_root = Path(source_root).expanduser().resolve(strict=True)
        if not self.source_root.is_dir():
            raise ValueError("Kernel source root must be a directory")
        self.generation = 0
        self.last_receipt = None
        self._baseline = {}
        self.bind_loaded_modules()

    def bind_loaded_modules(self):
        """Call after model loading and before advertising developer readiness."""
        for name in list(sys.modules):
            try:
                _, path = self._module(name)
                self._baseline.setdefault(name, path.read_bytes())
            except (ValueError, OSError, KernelReloadError):
                pass

    def _module(self, name):
        if not isinstance(name, str) or not _MODULE.fullmatch(name):
            raise ValueError("Kernel modules must be fully qualified names")
        # Computational namespaces are model-neutral. Lifecycle modules are
        # deliberately not re-executed while their live owners exist.
        if any(part in _FORBIDDEN for part in name.split(".")):
            raise KernelReloadError(f"Not a reloadable tensor module: {name}")
        module = sys.modules.get(name)
        if not isinstance(module, types.ModuleType) or not getattr(module, "__file__", None):
            raise ValueError(f"Module must already be loaded: {name}")
        path = Path(module.__file__).resolve(strict=True)
        if path.suffix != ".py" or not path.is_relative_to(self.source_root):
            raise KernelReloadError(f"Module is outside the configured Python source root: {name}")
        return module, path

    def prepare(self, modules=(), settings=(), calls=(), adapter=None):
        if not isinstance(modules, (list, tuple)) or not all(isinstance(x, str) for x in modules):
            raise ValueError("modules must be a list of module names")
        if not isinstance(settings, (list, tuple)) or not isinstance(calls, (list, tuple)):
            raise ValueError("settings and calls must be lists")  # noqa: TRY004 - HTTP validation contract.
        if len(settings) > 128 or len(calls) > 128:
            raise ValueError("At most 128 settings and calls are allowed")
        if len(modules) != len(set(modules)) or len(modules) > 128:
            raise ValueError("modules must contain at most 128 unique names")
        if not (modules or settings or calls):
            raise ValueError("A kernel change needs modules, settings or calls")
        sources = []
        for name in modules:
            _, path = self._module(name)
            source = path.read_bytes()
            compile(source, str(path), "exec", dont_inherit=True)  # Never consult a pyc or inherit this module's future flags.
            _reload_tree(source, path, name)  # Refuse an un-namespaceable kernel factory before any drain.
            sources.append(_Source(name, path, hashlib.sha256(source).hexdigest(), source))
        normalized_settings = []
        for item in settings:
            if not isinstance(item, dict) or set(item) != {"target", "value"}:
                raise ValueError("Each setting needs exactly target and value")
            name, attribute = _target(item["target"])
            module, _ = self._module(name)
            if attribute not in vars(module):
                raise ValueError(f"Unknown module setting: {item['target']}")
            old = getattr(module, attribute)
            if callable(old) or isinstance(old, types.ModuleType) or _contains_array(old):
                raise KernelReloadError("Settings cannot replace code, modules or device arrays")
            tree = ast.parse(Path(module.__file__).read_bytes())
            for node in ast.walk(tree):
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"__init__", "__new__"}
                    and any(isinstance(child, ast.Name) and child.id == attribute for child in ast.walk(node))):
                    raise KernelReloadError(f"Setting is captured by a constructor; use a live setter: {item['target']}")
            normalized_settings.append({"target": item["target"], "value": _plain_json(item["value"])})
        normalized_calls = []
        for item in calls:
            if not isinstance(item, dict) or set(item) - {"target", "args", "kwargs"} or "target" not in item:
                raise ValueError("Each call needs target and optional args/kwargs")
            name, attribute = _target(item["target"], call=True)
            if not attribute.startswith(("set_", "configure_", "enable_", "disable_")):
                raise ValueError("Calls must name an explicit configuration setter")
            if name not in {"adapter", "model", "layers"}:
                module, _ = self._module(name)
                if not callable(getattr(module, attribute, None)):
                    raise ValueError(f"Unknown callable: {item['target']}")
            args, kwargs = _plain_json(item.get("args", [])), _plain_json(item.get("kwargs", {}))
            if not isinstance(args, list) or not isinstance(kwargs, dict):
                raise ValueError("Call args must be a list and kwargs an object")  # noqa: TRY004 - HTTP validation contract.
            if adapter is not None or name not in {"adapter", "model", "layers"}:
                for _, fn in self._call_targets(name, attribute, adapter):
                    try:
                        inspect.signature(fn).bind(*args, **kwargs)
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"Invalid configuration arguments for {item['target']}: {error}") from error
            normalized_calls.append({"target": item["target"], "args": args, "kwargs": kwargs})
        return ReloadPlan(tuple(sources), tuple(normalized_settings), tuple(normalized_calls), self.generation)

    def _call_targets(self, name, attribute, adapter):
        if name == "layers":
            model = getattr(adapter, "model", None)
            enumerate_modules = getattr(model, "named_modules", None)
            if not callable(enumerate_modules):
                raise ValueError("The resident model does not expose named_modules()")
            targets = [(str(path), getattr(module, attribute)) for path, module in enumerate_modules()
                       if callable(getattr(module, attribute, None))]
        else:
            owner = adapter if name == "adapter" else getattr(adapter, "model", None) if name == "model" else self._module(name)[0]
            fn = getattr(owner, attribute, None)
            targets = [(name, fn)] if callable(fn) else []
        if not targets:
            raise ValueError(f"No configuration method matches {name}:{attribute}")
        return targets

    def status(self):
        return {"source_root": str(self.source_root), "generation": self.generation,
                "last_receipt": self.last_receipt, "qualification": "unqualified"}

    def verify_sources(self, plan):
        """Raise before any code execution if a prepared source changed on disk."""
        try:
            if not isinstance(plan, ReloadPlan) or plan.generation != self.generation:
                raise KernelReloadError("Stale kernel reload plan")
            modules = []
            for source in plan.sources:
                live, path = self._module(source.name)
                if path != source.path or hashlib.sha256(path.read_bytes()).hexdigest() != source.sha256:
                    raise KernelReloadError(f"Source changed after preparation: {source.name}")
                modules.append((source, live, path))
        except Exception as error:
            error.code_executed = False  # Serving may return to admission: nothing ran.
            raise
        return modules

    def apply(self, plan, adapter=None):
        # Source hashes are rechecked on the worker immediately before code
        # execution, for every module before the first one executes.
        modules = self.verify_sources(plan)
        before_weights = resident_weight_identity(adapter)
        stages = []
        for source, live, path in modules:
            tree = _reload_tree(source.source, path, source.name)
            stage ={key: value for key, value in vars(live).items() if key.startswith("__")}

            def compile_live(fn=None, *args, _live=live, **kwargs):
                import mlx.core as mx
                if fn is None:
                    return lambda actual: compile_live(actual, *args, _live=_live, **kwargs)
                if type(fn) is types.FunctionType:
                    rebound = types.FunctionType(fn.__code__, vars(_live), fn.__name__, fn.__defaults__, fn.__closure__)
                    rebound.__kwdefaults__ = fn.__kwdefaults__
                    fn = rebound
                return mx.compile(fn, *args, **kwargs)

            stage[_HELPER] = compile_live

            def metal_live(*args, _namespace=f"mlx2_reload_{self.generation + 1}_{source.sha256[:12]}", **kwargs):
                import mlx.core as mx
                if args:
                    args = (str(args[0]) + "_" + _namespace, *args[1:])
                else:
                    kwargs["name"] = str(kwargs["name"]) + "_" + _namespace
                return mx.fast.metal_kernel(*args, **kwargs)

            stage[_METAL_HELPER] = metal_live
            exec(compile(tree, str(path), "exec", dont_inherit=True), stage)  # noqa: S102 - Explicit local developer code execution.
            stages.append((live, stage))
        patches, replacements, classes = [], {}, []
        # All class/layout and function-closure checks precede publishing bodies.
        for live, stage in stages:
            self._check_constructor_globals(live, stage)
            for name, old in list(vars(live).items()):
                if name.startswith("__") or name in {_HELPER, _METAL_HELPER}:
                    continue
                if name not in stage:
                    if getattr(old, "__module__", None) == live.__name__:
                        raise KernelReloadError(f"Removing live definitions requires restart: {live.__name__}:{name}")
                    continue
                new = stage[name]
                if isinstance(old, type) and old.__module__ == live.__name__:
                    self._class_plan(old, new, patches)
                    replacements[id(new)] = old
                    classes.append(f"{live.__name__}:{name}")
                elif _python_function(old) is not None and getattr(old, "__module__", None) == live.__name__:
                    self._function_plan(old, new, patches)
                    replacements[id(new)] = old
                elif _contains_array(old):
                    # Module-resident arrays may be weights or immutable tables.
                    # Preserve them; rebuilding their layout requires a restart.
                    baseline = self._baseline.get(live.__name__)
                    source = next(s.source for s in plan.sources if s.name == live.__name__)
                    if (not _contains_array(new) or baseline is None or
                            self._binding_law(baseline, name) != self._binding_law(source, name)):
                        raise KernelReloadError(f"Changing module-resident array layout requires restart: {live.__name__}:{name}")
                    replacements[id(new)] = old
        affected = set(plan.modules)
        affected.update(_target(item["target"])[0] for item in plan.settings)
        for item in plan.calls:
            name, attribute = _target(item["target"], call=True)
            affected.update(getattr(fn, "__module__", "") for _, fn in self._call_targets(name, attribute, adapter))
        caches = self._invalidate_external_caches(plan.modules, stages, affected)
        self._check_instance_compiled(adapter, plan.modules, stages)
        self._check_handle_frames(stages)
        stage_globals = {id(stage): vars(live) for live, stage in stages}
        # One memo keyed by id(): shared and cyclic references inside module
        # tables (dispatch dicts, class maps, defaults) translate to one object.
        memo, building = {}, set()

        def translate(value):
            # Staged classes/functions resolve to the preserved live identities.
            if id(value) in replacements:
                return replacements[id(value)]
            if id(value) in memo:
                return memo[id(value)]
            kind = type(value)
            if kind is dict:
                result = memo[id(value)] = {}
                result.update((translate(key), translate(item)) for key, item in value.items())
                return result
            if kind is list:
                result = memo[id(value)] = []
                result.extend(translate(item) for item in value)
                return result
            if kind in {tuple, set, frozenset}:
                if id(value) in building:
                    raise KernelReloadError("A cyclic immutable module table requires restart")
                building.add(id(value))
                result = memo[id(value)] = kind(translate(item) for item in value)
                building.discard(id(value))
                return result
            if kind is types.FunctionType and id(value.__globals__) in stage_globals:
                rebound = memo[id(value)] = types.FunctionType(
                    value.__code__, stage_globals[id(value.__globals__)], value.__name__, None, value.__closure__)
                rebound.__defaults__ = translate(value.__defaults__)
                rebound.__kwdefaults__ = translate(value.__kwdefaults__)
                return rebound
            return value

        # A staged table replacing a live dict is published into that dict, so
        # references to the staged table resolve to the live one.
        refill = {}
        for live, stage in stages:
            for name, new in stage.items():
                old = vars(live).get(name)
                if (not name.startswith("__") and id(new) not in replacements and type(new) is dict
                        and isinstance(old, dict) and old is not new and not _contains_array(old)):
                    memo[id(new)] = old
                    refill[id(new)] = new
        # Translate everything before mutating a live object: a refused table
        # must not leave a half-published module.
        defaults = [(translate(new.__defaults__), translate(new.__kwdefaults__)) for _, new in patches]
        refill = {key: [(translate(k), translate(v)) for k, v in new.items()] for key, new in refill.items()}
        published = {}
        for live, stage in stages:
            for name, new in stage.items():
                if name.startswith("__"):
                    continue
                value = replacements.get(id(new), new)
                if value is new and type(value) is types.FunctionType and value.__globals__ is stage:
                    value = translate(value)
                    value.__annotations__ = new.__annotations__
                elif value is new and type(value) in {dict, tuple, list, set, frozenset}:
                    value = translate(value)
                published[(id(live), name)] = value

        functions = []
        for (old, new), (new_defaults, new_kwdefaults) in zip(patches, defaults):
            old.__code__ = new.__code__
            old.__defaults__ = new_defaults
            old.__kwdefaults__ = new_kwdefaults
            old.__annotations__ = new.__annotations__
            old.__doc__ = new.__doc__
            functions.append(f"{old.__module__}:{old.__qualname__}")
        aliases = {}
        scalar_changes = {}
        for live, stage in stages:
            for name, new in stage.items():
                if name.startswith("__"):
                    continue
                old = vars(live).get(name)
                value = published[(id(live), name)]
                if isinstance(old, functools._lru_cache_wrapper):
                    old.cache_clear()
                    caches.append(f"{live.__name__}:{name}")
                elif isinstance(old, type) and old is value and old.__module__ == live.__name__:
                    for attribute, member in vars(old).items():
                        member = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
                        if isinstance(member, functools._lru_cache_wrapper):
                            member.cache_clear()
                            caches.append(f"{live.__name__}:{name}.{attribute}")
                elif id(new) in refill:
                    old.clear()
                    old.update(refill[id(new)])
                    caches.append(f"{live.__name__}:{name}")
                elif isinstance(old, dict) and isinstance(value, dict) and old is not value and not _contains_array(old):
                    old.clear()
                    old.update(value)
                    value = old
                    caches.append(f"{live.__name__}:{name}")
                if old is not None and old is not value and callable(old) and name not in {_HELPER, _METAL_HELPER}:
                    aliases[id(old)] = (old, value)
                if type(old) in {str, int, float, bool, tuple, type(None)} and old != value:
                    scalar_changes[(live.__name__, name)] = (old, value)
                vars(live)[name] = value
        self._rebind_aliases(aliases)
        for item in plan.settings:
            name, attribute = _target(item["target"])
            module, _ = self._module(name)
            scalar_changes[(name, attribute)] = (getattr(module, attribute), item["value"])
            setattr(module, attribute, item["value"])
        self._rebind_imported_constants(scalar_changes)
        call_receipts = []
        for item in plan.calls:
            name, attribute = _target(item["target"], call=True)
            targets = self._call_targets(name, attribute, adapter)
            target_modules = {getattr(fn, "__module__", "") for _, fn in targets}
            before_constants = {(module_name, key): value for module_name in target_modules
                                if isinstance(sys.modules.get(module_name), types.ModuleType)
                                for key, value in vars(sys.modules[module_name]).items()
                                if type(value) in {str, int, float, bool, tuple, type(None)}}
            for _, fn in targets:
                fn(*item["args"], **item["kwargs"])
            for (module_name, key), before in before_constants.items():
                after = vars(sys.modules[module_name]).get(key)
                if type(after) in {str, int, float, bool, tuple, type(None)} and before != after:
                    scalar_changes[(module_name, key)] = (before, after)
            self._rebind_imported_constants(scalar_changes)
            call_receipts.append({**item, "matched_paths": [path for path, _ in targets], "matched_count": len(targets)})
        if resident_weight_identity(adapter) != before_weights:
            error = KernelReloadError("Reload replaced resident model or parameter identities; engine must stay quarantined")
            error.weights_preserved = False
            raise error
        self.generation += 1
        self._baseline.update({s.name: s.source for s in plan.sources})
        self.last_receipt = {"status": "applied", "generation": self.generation,
            "modules": list(plan.modules), "source_sha256": {s.name: s.sha256 for s in plan.sources},
            "settings": list(plan.settings), "calls": call_receipts,
            "functions_patched": functions, "classes_preserved": classes,
            "caches_invalidated": sorted(set(caches)), "weights_preserved": True,
            "qualification": "unqualified"}
        return self.last_receipt

    @staticmethod
    def _check_constructor_globals(live, stage):
        pending = []
        for value in vars(live).values():
            if isinstance(value, type) and value.__module__ == live.__name__:
                pending.extend(vars(value).get(name) for name in ("__init__", "__new__"))
        seen = set()
        while pending:
            fn = pending.pop()
            if isinstance(fn, (staticmethod, classmethod)):
                fn = fn.__func__
            if type(fn) is not types.FunctionType or id(fn) in seen:
                continue
            seen.add(id(fn))
            pending.extend(cell.cell_contents for cell in fn.__closure__ or ()
                           if type(cell.cell_contents) is types.FunctionType)
            names = {instruction.argval for instruction in dis.get_instructions(fn)
                     if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}}
            for name in names:
                if name not in vars(live) or name not in stage:
                    continue
                old, new = vars(live)[name], stage[name]
                old_fn, new_fn = _python_function(old), _python_function(new)
                if old_fn is not None and old_fn.__module__ == live.__name__:
                    if new_fn is None or _code_law(old_fn.__code__) != _code_law(new_fn.__code__):
                        raise KernelReloadError(f"Changed constructor dependency requires restart: {live.__name__}:{name}")
                    pending.append(old_fn)
                elif isinstance(old, (str, int, float, bool, tuple, list, dict, set, frozenset, type(None))):
                    if type(old) is not type(new) or old != new:
                        raise KernelReloadError(f"Changed constructor global requires restart: {live.__name__}:{name}")

    @staticmethod
    def _function_plan(old, new, patches):
        old_fn, new_fn = _python_function(old), _python_function(new)
        if old_fn is None or new_fn is None or type(old) is not type(new) or _signature(old_fn) != _signature(new_fn):
            raise KernelReloadError(f"Changed callable structure requires restart: {getattr(old, '__qualname__', old)}")
        for name, left, right in zip(old_fn.__code__.co_freevars, old_fn.__closure__ or (), new_fn.__closure__ or ()):
            a, b = left.cell_contents, right.cell_contents
            if (isinstance(a, type) and isinstance(b, type) and (a.__module__, a.__qualname__) == (b.__module__, b.__qualname__)
                and (name == "__class__" or any(value is a for value in vars(sys.modules[a.__module__]).values())
                     or (name == "cls" and old_fn.__name__ in {"__setattr__", "__delattr__"}
                         and dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b)
                         and _code_law(old_fn.__code__) == _code_law(new_fn.__code__)
                         and _dataclass_fields_law(a.__dataclass_fields__) == _dataclass_fields_law(b.__dataclass_fields__)
                         and _dataclass_params_law(a.__dataclass_params__) == _dataclass_params_law(b.__dataclass_params__)))):
                continue
            if type(a) is types.FunctionType and type(b) is types.FunctionType:
                KernelReloader._function_plan(a, b, patches)
            elif a is not b and not (type(a) is type(b) and isinstance(a, (str, int, float, bool, tuple, list, dict, set, frozenset, type(None)))
                                     and _metadata_law(a) == _metadata_law(b)):
                raise KernelReloadError(f"Changed closure capture requires restart: {old_fn.__qualname__}:{name}")
        patches.append((old_fn, new_fn))

    @staticmethod
    def _binding_law(source, name):
        """Bind constant tables to their assignment and its module dependencies."""
        tree = ast.parse(source)
        definitions = {}
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        definitions[target.id] = node
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name] = node
        result, pending = {}, [name]
        while pending:
            key = pending.pop()
            if key in result or key not in definitions:
                continue
            node = definitions[key]
            result[key] = ast.dump(node, include_attributes=False)
            pending.extend(x.id for x in ast.walk(node) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load))
        return result or None

    @classmethod
    def _class_plan(cls, old, new, patches):
        if not isinstance(new, type) or type(old) is not type(new):
            raise KernelReloadError(f"Changed class kind requires restart: {old.__qualname__}")
        if tuple((x.__module__, x.__qualname__) for x in old.__bases__) != tuple((x.__module__, x.__qualname__) for x in new.__bases__):
            raise KernelReloadError(f"Changed class bases require restart: {old.__qualname__}")
        ignored = {"__doc__", "__dict__", "__weakref__", "__module__", "__firstlineno__", "__static_attributes__"}
        left, right = {k: v for k, v in vars(old).items() if k not in ignored}, {k: v for k, v in vars(new).items() if k not in ignored}
        if set(left) != set(right):
            raise KernelReloadError(f"Changed class layout requires restart: {old.__qualname__}")
        for key, before in left.items():
            after = right[key]
            if key == "__annotations__":
                if _metadata_law(before) != _metadata_law(after):
                    raise KernelReloadError(f"Changed class annotations require restart: {old.__qualname__}")
                continue
            if key == "__dataclass_fields__":
                if _dataclass_fields_law(before) != _dataclass_fields_law(after):
                    raise KernelReloadError(f"Changed dataclass fields require restart: {old.__qualname__}")
                continue
            if key == "__dataclass_params__":
                if _dataclass_params_law(before) != _dataclass_params_law(after):
                    raise KernelReloadError(f"Changed dataclass parameters require restart: {old.__qualname__}")
                continue
            if isinstance(before, (staticmethod, classmethod)):
                if type(before) is not type(after):
                    raise KernelReloadError(f"Changed descriptor: {old.__qualname__}.{key}")
                before, after = before.__func__, after.__func__
            if isinstance(before, property):
                if not isinstance(after, property):
                    raise KernelReloadError(f"Changed property: {key}")
                for a, b in zip((before.fget, before.fset, before.fdel), (after.fget, after.fset, after.fdel)):
                    if (a is None) != (b is None):
                        raise KernelReloadError(f"Changed property layout: {key}")
                    if a is not None:
                        cls._function_plan(a, b, patches)
            elif type(before) is types.FunctionType:
                if key in {"__init__", "__new__", "__setattr__", "__delattr__", "__getattribute__", "__del__"} and (
                    _code_law(before.__code__) != _code_law(after.__code__) or
                    before.__defaults__ != after.__defaults__ or before.__kwdefaults__ != after.__kwdefaults__):
                    raise KernelReloadError(f"Changed structural method requires reload: {old.__qualname__}.{key}")
                cls._function_plan(before, after, patches)
            elif isinstance(before, functools._lru_cache_wrapper):
                cls._function_plan(before, after, patches)
            elif _compiled(before) or (callable(before) and not isinstance(before, type)):
                raise KernelReloadError(f"Opaque compiled class method requires restart: {old.__qualname__}.{key}")
            elif type(before) is not type(after) or (isinstance(before, (str, int, float, bool, tuple, list, dict, set, frozenset, type(None))) and before != after):
                raise KernelReloadError(f"Changed class data requires restart: {old.__qualname__}.{key}")

    def _source_imports(self, module):
        path = getattr(module, "__file__", None)
        if not path:
            return []
        path = Path(path).resolve()
        if path.suffix != ".py" or not path.is_relative_to(self.source_root):
            return []
        try:
            tree = ast.parse(path.read_bytes())
        except (OSError, SyntaxError):
            return []
        result = []
        pending = list(tree.body)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if not isinstance(node, ast.ImportFrom):
                pending.extend(ast.iter_child_nodes(node))
                continue
            origin = node.module or ""
            if node.level:
                try:
                    origin = importlib.util.resolve_name("." * node.level + origin, module.__package__)
                except (ImportError, ValueError):
                    continue
            for alias in node.names:
                result.append((origin, alias.name, alias.asname or alias.name))
        return result

    def _rebind_imported_constants(self, changes):
        modules = [(module, self._source_imports(module)) for module in list(sys.modules.values())
                   if isinstance(module, types.ModuleType)]
        while True:
            propagated = False
            for module, imports in modules:
                for origin, imported, local in imports:
                    if (origin, imported) not in changes:
                        continue
                    before, after = changes[(origin, imported)]
                    current = vars(module).get(local)
                    if type(current) is not type(before) or current != before or before == after:
                        continue
                    alternatives = {(other_origin, other_imported) for other_origin, other_imported, other_local in imports
                                    if other_local == local}
                    if len(alternatives) > 1:
                        raise KernelReloadError(f"Ambiguous conditional constant import requires restart: {module.__name__}:{local}")
                    vars(module)[local] = after
                    changes[(module.__name__, local)] = (before, after)
                    propagated = True
            if not propagated:
                break

    def _invalidate_external_caches(self, selected, stages, affected):
        invalidated = []
        selected = set(selected)
        modules = {name: module for name, module in list(sys.modules.items()) if isinstance(module, types.ModuleType)}
        imports = {name: {origin for origin, _, _ in self._source_imports(module)}
                   for name, module in modules.items() if not hasattr(module, "__path__")}
        affected = set(affected)
        while True:
            previous = len(affected)
            for name, module in modules.items():
                if name in affected or hasattr(module, "__path__"):
                    continue
                dependencies = {getattr(value, "__name__", None) if isinstance(value, types.ModuleType)
                                else getattr(value, "__module__", None) for value in vars(module).values()}
                if (dependencies | imports.get(name, set())).intersection(affected):
                    affected.add(name)
            if len(affected) == previous:
                break
        selected_handles = {id(value) for live, _ in stages for value in vars(live).values() if _compiled(value)}
        for name in (affected - selected) & modules.keys():
            module = modules[name]
            for attribute, value in list(vars(module).items()):
                if _compiled(value) and id(value) not in selected_handles:
                    raise KernelReloadError(f"Include compiled caller module in reload: {name}")
                if isinstance(value, functools._lru_cache_wrapper) and value.__module__ == name:
                    value.cache_clear()
                    invalidated.append(f"{name}:{attribute}")
                elif isinstance(value, dict) and _CACHE_NAME.search(attribute) and not _contains_array(value):
                    value.clear()
                    invalidated.append(f"{name}:{attribute}")
        return invalidated

    @staticmethod
    def _check_handle_frames(stages):
        handles = {id(value) for live, _ in stages for value in vars(live).values() if _compiled(value)}
        if not handles:
            return
        for frame in sys._current_frames().values():
            while frame is not None:
                if frame.f_globals.get("__name__") != __name__:
                    for name, value in list(frame.f_locals.items()):
                        if id(value) in handles and frame.f_locals is not frame.f_globals:
                            raise KernelReloadError(f"Compiled kernel is retained in an active local variable ({name}); release it before reload")
                frame = frame.f_back

    @staticmethod
    def _check_instance_compiled(adapter, selected, stages):
        selected_handles = {id(value) for live, _ in stages for value in vars(live).values() if _compiled(value)}
        seen, pending = set(), [adapter]
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if _compiled(value) and id(value) not in selected_handles:
                raise KernelReloadError("Resident object holds an opaque compiled callable; remove its cache or restart before reload")
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
            elif value is not None and not isinstance(value, (types.ModuleType, types.FunctionType, type)) and hasattr(value, "__dict__"):
                pending.extend(vars(value).values())

    @staticmethod
    def _rebind_aliases(replacements):
        # Existing Python functions were patched in place. Opaque compiled/Metal
        # handles require rebinding. Refuse immutable captures instead of silently
        # leaving an old executable reachable through a default/closure.
        # Module dictionaries can be GC-untracked after a collection, so their
        # explicit inventory is authoritative; gc.get_referrers alone is not.
        for module in list(sys.modules.values()):
            if not isinstance(module, types.ModuleType):
                continue
            for name, value in list(vars(module).items()):
                replacement = replacements.get(id(value))
                if replacement is None or replacement[0] is not value:
                    continue
                if not callable(replacement[1]):
                    raise KernelReloadError("A lazy compiled kernel has an external live module alias; include its caller or restart")
                vars(module)[name] = replacement[1]
        # get_referrers also reports unreachable cycles. Collect those before
        # deciding that an immutable tuple is a live executable capture.
        gc.collect()
        internal_pairs = tuple(replacements.values())
        internal_pair_ids = {id(pair) for pair in internal_pairs}
        for old, new in internal_pairs:
            for ref in gc.get_referrers(old):
                if isinstance(ref, dict):
                    if not callable(new) and _contains_identity(ref.values(), old):
                        raise KernelReloadError("A lazy compiled kernel has an external live alias; include its caller or restart")
                    for key, value in list(ref.items()):
                        if value is old:
                            ref[key] = new
                elif isinstance(ref, list):
                    if not callable(new) and _contains_identity(ref, old):
                        raise KernelReloadError("A lazy compiled kernel is retained in a live list; remove the capture or restart")
                    for index, value in enumerate(ref):
                        if value is old:
                            ref[index] = new
                elif isinstance(ref, types.CellType):
                    if not callable(new):
                        raise KernelReloadError("A lazy compiled kernel is captured by a closure; remove the capture or restart")
                    if ref.cell_contents is old:
                        ref.cell_contents = new
                elif isinstance(ref, tuple):
                    # Exempt only our actual bookkeeping objects, never an
                    # external tuple that happens to contain the same pair.
                    if id(ref) in internal_pair_ids:
                        continue
                    raise KernelReloadError("Compiled kernel is captured in an immutable tuple; restart or remove that capture")
