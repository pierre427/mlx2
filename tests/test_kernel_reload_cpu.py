"""Resident-object and fresh-source contracts; no model or GPU is loaded."""

import importlib.util
import sys
import types

import pytest

from mlx2 import kernel_reload as reload


@pytest.fixture
def module_factory(tmp_path, monkeypatch):
    def create(source, name="mlx2.runtime.kernel_reload_cpu_fixture"):
        path = tmp_path / (name.replace(".", "_") + ".py")
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module, path, reload.KernelReloader(tmp_path)
    return create


def test_fresh_source_preserves_instances_functions_aliases_and_cached_factories(module_factory):
    source = '''from functools import lru_cache
OFFSET = 1
def value(x):
    return x + OFFSET
@lru_cache(None)
def factory(x):
    return value(x)
class Model:
    def __init__(self, weight):
        self.weight = weight
    def forward(self, x):
        return value(x)
'''
    module, path, reloader = module_factory(source)
    weight = object()
    model = module.Model(weight)
    alias, cached_alias, bound = module.value, module.factory, model.forward
    assert cached_alias(3) == 4
    path.write_text(source.replace("OFFSET = 1", "OFFSET = 7").replace("return value(x)\n", "return value(x) * 2\n"))
    receipt = reloader.apply(reloader.prepare([module.__name__]), types.SimpleNamespace(model=model))
    assert alias is module.value
    assert cached_alias is module.factory
    assert model.__class__ is module.Model
    assert model.weight is weight
    assert alias(3) == 10
    assert cached_alias(3) == bound(3) == 20
    assert receipt["weights_preserved"] is True


def test_no_pyc_reuse_for_same_size_edit(module_factory):
    module, path, reloader = module_factory("def kernel():\n    return 11\n")
    alias = module.kernel
    path.write_text("def kernel():\n    return 22\n")
    reloader.apply(reloader.prepare([module.__name__]))
    assert alias() == 22


def test_structural_edit_refused_before_functions_change(module_factory):
    source = "def kernel():\n    return 1\nclass Model:\n    def __init__(self):\n        self.weight = 3\n"
    module, path, reloader = module_factory(source)
    path.write_text(source.replace("return 1", "return 2").replace("self.weight = 3", "self.weight = 4"))
    with pytest.raises(reload.KernelReloadError, match="structural"):
        reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel() == 1


def test_changed_closure_refused(module_factory):
    source = "def wrap():\n    offset = 1\n    def kernel(x):\n        return x + offset\n    return kernel\nkernel = wrap()\n"
    module, path, reloader = module_factory(source)
    path.write_text(source.replace("offset = 1", "offset = 2"))
    with pytest.raises(reload.KernelReloadError, match="closure"):
        reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel(2) == 3


def test_class_super_closure_retains_live_class(module_factory):
    source = "class Base:\n    def value(self):\n        return 1\nclass Child(Base):\n    def value(self):\n        return super().value() + 1\n"
    module, path, reloader = module_factory(source)
    obj = module.Child()
    path.write_text(source.replace("return super().value() + 1", "return super().value() + 2"))
    reloader.apply(reloader.prepare([module.__name__]))
    assert obj.value() == 3


def test_settings_and_module_setters(module_factory):
    module, _, reloader = module_factory("MODE = 1\ndef set_mode(value):\n    global MODE\n    MODE = value\ndef kernel():\n    return MODE\n")
    reloader.apply(reloader.prepare(settings=[{"target": module.__name__ + ":MODE", "value": 2}]))
    assert module.kernel() == 2
    reloader.apply(reloader.prepare(calls=[{"target": module.__name__ + ":set_mode", "args": [3]}]))
    assert module.kernel() == 3


def test_constructor_latched_setting_refused(module_factory):
    module, _, reloader = module_factory("MODE = 1\nclass Model:\n    def __init__(self):\n        self.mode = MODE\n")
    with pytest.raises(reload.KernelReloadError, match="constructor"):
        reloader.prepare(settings=[{"target": module.__name__ + ":MODE", "value": 2}])


def test_layer_setters_validate_and_record_every_target(module_factory):
    _, _, reloader = module_factory("VALUE = 1\n")
    class Layer:
        def set_mode(self, value):
            self.mode = value
    layers = [Layer(), Layer()]
    model = types.SimpleNamespace(named_modules=lambda: [(str(i), layer) for i, layer in enumerate(layers)])
    adapter = types.SimpleNamespace(model=model)
    plan = reloader.prepare(calls=[{"target": "layers:set_mode", "args": [4]}], adapter=adapter)
    receipt = reloader.apply(plan, adapter)
    assert [layer.mode for layer in layers] == [4, 4]
    assert receipt["calls"][0]["matched_count"] == 2
    with pytest.raises(ValueError, match="Invalid configuration arguments"):
        reloader.prepare(calls=[{"target": "layers:set_mode", "args": []}], adapter=adapter)
    with pytest.raises(ValueError, match="No configuration method"):
        reloader.prepare(calls=[{"target": "model:set_unknown"}], adapter=adapter)


def test_replacing_resident_weights_fails_closed(module_factory):
    _, _, reloader = module_factory("VALUE = 1\n")
    class Array:
        shape = (3,)
        dtype = "float32"
    class Model:
        def __init__(self):
            self.weights = Array()
        def set_mode(self, value):
            self.weights = Array()
    adapter = types.SimpleNamespace(model=Model())
    with pytest.raises(reload.KernelReloadError, match="parameter identities"):
        reloader.apply(reloader.prepare(calls=[{"target": "model:set_mode", "args": [2]}], adapter=adapter), adapter)


def test_plan_source_hash_and_generation_are_exact(module_factory):
    module, path, reloader = module_factory("def kernel():\n    return 1\n")
    plan = reloader.prepare([module.__name__])
    path.write_text("def kernel():\n    return 2\n")
    with pytest.raises(reload.KernelReloadError, match="changed after"):
        reloader.apply(plan)
    fresh = reloader.prepare([module.__name__])
    reloader.apply(fresh)
    with pytest.raises(reload.KernelReloadError, match="Stale"):
        reloader.apply(fresh)


def test_owner_lifetime_and_sticky_modified_process(monkeypatch):
    monkeypatch.setattr(reload, "_OWNERS", {})
    monkeypatch.setattr(reload, "_CHANGED_OWNERS", set())
    first, second = object(), object()
    reload.register_owner(first)
    reload.register_owner(second)
    with pytest.raises(reload.KernelReloadError, match="only loaded"), reload.exclusive_owner(first):
        pass
    reload.unregister_owner(second)
    with reload.exclusive_owner(first):
        pass
    reload.unregister_owner(first)
    with pytest.raises(reload.KernelReloadError, match="modified process"):
        reload.register_owner(second)


def test_syntax_and_root_checks_happen_in_prepare(module_factory, tmp_path):
    module, path, reloader = module_factory("VALUE = 1\n")
    path.write_text("def broken(:\n")
    with pytest.raises(SyntaxError):
        reloader.prepare([module.__name__])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(reload.KernelReloadError, match="outside"):
        reload.KernelReloader(other).prepare([module.__name__])
    with pytest.raises(reload.KernelReloadError, match="tensor module"):
        reloader.prepare(["mlx2.serving"])


def test_constructor_global_edit_refused(module_factory):
    source = "MODE = 1\nclass Model:\n    def __init__(self):\n        self.mode = MODE\n"
    module, path, reloader = module_factory(source)
    path.write_text(source.replace("MODE = 1", "MODE = 2"))
    with pytest.raises(reload.KernelReloadError, match="constructor global"):
        reloader.apply(reloader.prepare([module.__name__]))
    assert module.MODE == 1


def test_compiled_variant_requires_recreating_its_trace(module_factory):
    # Constructing the lazy compiler wrapper performs no device operation.
    module, _, reloader = module_factory("import mlx.core as mx\nMODE = 1\n@mx.compile\ndef kernel(x):\n    return x + MODE\n")
    with pytest.raises(reload.KernelReloadError, match="Include compiled caller"):
        reloader.apply(reloader.prepare(settings=[{"target": module.__name__ + ":MODE", "value": 2}]))
    assert module.MODE == 1


def test_compiled_source_recreated_with_live_globals_and_global_alias(module_factory, monkeypatch):
    import gc
    source = "import mlx.core as mx\nMODE = 1\n@mx.compile\ndef kernel(x):\n    return x + MODE\n"
    module, path, reloader = module_factory(source)
    consumer = types.ModuleType("kernel_reload_compiled_consumer")
    consumer.kernel = module.kernel
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    old_id = id(module.kernel)
    path.write_text(source.replace("MODE = 1", "MODE = 2"))
    reloader.apply(reloader.prepare([module.__name__]))
    assert id(module.kernel) != old_id
    assert consumer.kernel is module.kernel
    fn = next(value for value in gc.get_referents(module.kernel) if isinstance(value, types.FunctionType))
    assert fn.__globals__ is vars(module)
    assert fn.__globals__["MODE"] == 2


def test_transitive_compiled_caller_refuses_source_reload(module_factory, monkeypatch):
    import mlx.core as mx
    module, path, reloader = module_factory("def kernel(x):\n    return x + 1\n")
    bridge = types.ModuleType("kernel_reload_bridge")
    bridge.source = module
    consumer = types.ModuleType("kernel_reload_transitive_consumer")
    consumer.bridge = bridge
    consumer.compiled = mx.compile(lambda x: x)
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    path.write_text("def kernel(x):\n    return x + 2\n")
    with pytest.raises(reload.KernelReloadError, match="kernel_reload_transitive_consumer"):
        reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel(1) == 2


def test_metal_factories_get_generation_namespace_without_touching_device(module_factory, monkeypatch):
    import mlx.core as mx
    names = []
    def make_kernel(*args, **kwargs):
        names.append(kwargs["name"])
        return lambda: kwargs["source"]
    monkeypatch.setattr(mx.fast, "metal_kernel", make_kernel)
    source = '''import mlx.core as mx
from functools import lru_cache
SOURCE = "old_body"
@lru_cache(None)
def kernel():
    return mx.fast.metal_kernel(name="test_kernel", input_names=[], output_names=[], source=SOURCE)
'''
    module, path, reloader = module_factory(source)
    assert module.kernel()() == "old_body"
    path.write_text(source.replace("old_body", "new_body"))
    reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel()() == "new_body"
    assert names[0] == "test_kernel"
    assert names[1].startswith("test_kernel_mlx2_reload_1_")
    reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel()() == "new_body"
    assert names[2].startswith("test_kernel_mlx2_reload_2_")


@pytest.mark.parametrize("prelude,body_import,call", [
    ("", "    import mlx.core as mx\n", "mx.fast.metal_kernel"),
    ("", "    import mlx.core\n", "mlx.core.fast.metal_kernel"),
    ("import mlx.core\n", "", "mlx.core.fast.metal_kernel"),
    ("", "    from mlx.core import fast\n", "fast.metal_kernel"),
    ("", "    from mlx.core.fast import metal_kernel as make\n", "make"),
])
def test_metal_factories_with_local_or_unaliased_imports_get_namespace(module_factory, monkeypatch, prelude, body_import, call):
    # Most kernel modules import MLX inside the factory; a missed rewrite
    # reuses the old Metal library name under an "applied" receipt.
    import mlx.core as mx
    names = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kwargs: names.append(kwargs["name"]) or (lambda: kwargs["source"]))
    source = (f'{prelude}from functools import lru_cache\nSOURCE = "old_body"\n@lru_cache(None)\ndef kernel():\n'
              f'{body_import}    return {call}(name="test_kernel", input_names=[], output_names=[], source=SOURCE)\n')
    module, path, reloader = module_factory(source)
    assert module.kernel()() == "old_body"
    path.write_text(source.replace("old_body", "new_body"))
    reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel()() == "new_body"
    assert names[0] == "test_kernel"
    assert names[1].startswith("test_kernel_mlx2_reload_1_")


def test_metal_aliases_resolve_per_scope(module_factory, monkeypatch):
    # The same local name means a Metal factory in one function, MLX compile
    # in another and an ordinary parameter in a third.
    import mlx.core as mx
    names = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kwargs: names.append(kwargs["name"]) or (lambda: kwargs["source"]))
    source = '''SOURCE = "old_body"
def kernel():
    from mlx.core.fast import metal_kernel as make
    return make(name="test_kernel", input_names=[], output_names=[], source=SOURCE)
def compiled():
    from mlx.core import compile as make
    return make
def apply(make, x):
    return make(x)
def shadow():
    from mlx.core.fast import metal_kernel as make
    def inner(make):
        return make(1)
    return inner(lambda x: x + 1)
'''
    module, path, reloader = module_factory(source)
    path.write_text(source.replace("old_body", "new_body"))
    reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel()() == "new_body"
    assert names[0].startswith("test_kernel_mlx2_reload_1_")
    assert module.apply(lambda x: x * 3, 2) == 6
    assert module.shadow() == 2
    assert module.compiled() is not mx.compile  # The reload helper binds live globals.


def test_metal_alias_ambiguous_within_one_scope_is_refused(module_factory):
    source = '''FAST = True
def kernel():
    if FAST:
        from mlx.core.fast import metal_kernel as make
    else:
        make = None
    return make(name="k", input_names=[], output_names=[], source="a")
'''
    module, _, reloader = module_factory(source)
    with pytest.raises(reload.KernelReloadError, match="Cannot namespace the Metal kernel factory"):
        reloader.prepare([module.__name__])


def test_unresolvable_metal_factory_is_refused_before_drain(module_factory):
    source = 'def kernel(backend):\n    return backend.fast.metal_kernel(name="k", input_names=[], output_names=[], source="a")\n'
    module, path, reloader = module_factory(source)
    path.write_text(source.replace('"a"', '"b"'))
    with pytest.raises(reload.KernelReloadError, match="Cannot namespace the Metal kernel factory"):
        reloader.prepare([module.__name__])
    assert reloader.generation == 0


def test_module_tables_keep_preserved_class_and_function_identities(module_factory):
    source = '''class Plain:
    pass
class Ring(Plain):
    pass
def helper():
    return 1
RING_CLASS = {Plain: Ring}
CANONICAL = {Ring: Plain}
RINGS = (Ring,)
ORDER = [Plain, Ring]
KINDS = {Ring}
DISPATCH = {"ring": helper}
def is_ring(plane):
    return type(plane) in CANONICAL
'''
    module, path, reloader = module_factory(source)
    plane, table = module.Ring(), module.RING_CLASS
    reloader.apply(reloader.prepare([module.__name__]))  # Unchanged source.
    assert module.is_ring(plane)
    assert module.RING_CLASS is table and table[module.Plain] is module.Ring
    assert module.RINGS == (type(plane),) and module.ORDER[1] is type(plane)
    assert module.KINDS == {type(plane)}
    path.write_text(source.replace("return 1", "return 2"))
    reloader.apply(reloader.prepare([module.__name__]))
    assert module.DISPATCH["ring"] is module.helper and module.DISPATCH["ring"]() == 2


def test_cyclic_and_shared_module_tables_translate_to_one_object(module_factory):
    source = '''class Ring:
    pass
TABLE = {Ring: 1}
TABLE["self"] = TABLE
ORDER = [Ring]
ORDER.append(ORDER)
SHARED = {"order": ORDER}
def lookup(key, table=TABLE):
    return table[key]
'''
    module, _, reloader = module_factory(source)
    ring, table = module.Ring, module.TABLE
    reloader.apply(reloader.prepare([module.__name__]))  # Unchanged source.
    assert module.Ring is ring and module.TABLE is table
    assert table["self"] is table and set(table) == {ring, "self"}
    assert module.ORDER[0] is ring and module.ORDER[1] is module.ORDER
    assert module.SHARED["order"] is module.ORDER
    assert module.lookup.__defaults__[0] is table and module.lookup(ring) == 1


def test_compiled_fast_local_capture_refused(module_factory):
    module, path, reloader = module_factory("import mlx.core as mx\n@mx.compile\ndef kernel(x):\n    return x + 1\n")
    captured = module.kernel
    path.write_text("import mlx.core as mx\n@mx.compile\ndef kernel(x):\n    return x + 2\n")
    with pytest.raises(reload.KernelReloadError, match="active local variable"):
        reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel is captured


def test_compiled_class_method_refused(module_factory):
    module, path, reloader = module_factory("import mlx.core as mx\nclass Model:\n    @mx.compile\n    def value(self, x):\n        return x + 1\n")
    path.write_text("import mlx.core as mx\nclass Model:\n    @mx.compile\n    def value(self, x):\n        return x + 2\n")
    with pytest.raises(reload.KernelReloadError, match="compiled class method"):
        reloader.apply(reloader.prepare([module.__name__]))


def test_scalar_from_import_aliases_follow_source_and_setting_changes(module_factory):
    source = "MODE = 1\ndef set_mode(value):\n    global MODE\n    MODE = value\n"
    module, path, reloader = module_factory(source)
    consumer, _, _ = module_factory(f"from {module.__name__} import MODE as SELECTED\ndef kernel():\n    return SELECTED\n", "mlx2.runtime.kernel_reload_cpu_consumer")
    path.write_text(source.replace("MODE = 1", "MODE = 2"))
    reloader.apply(reloader.prepare([module.__name__]))
    assert consumer.kernel() == 2
    reloader.apply(reloader.prepare(settings=[{"target": module.__name__ + ":MODE", "value": 3}]))
    assert consumer.kernel() == 3
    reloader.apply(reloader.prepare(calls=[{"target": module.__name__ + ":set_mode", "args": [4]}]))
    assert consumer.kernel() == 4


def test_package_containers_do_not_make_unrelated_compiled_module_a_consumer(module_factory, monkeypatch):
    import mlx.core as mx
    module, path, reloader = module_factory("def kernel(x):\n    return x + 1\n")
    package = types.ModuleType("kernel_reload_package")
    package.__path__ = []
    package.target = module
    consumer = types.ModuleType("kernel_reload_unrelated")
    consumer.package = package
    consumer.compiled = mx.compile(lambda x: x)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    path.write_text("def kernel(x):\n    return x + 2\n")
    reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel(1) == 3


def test_callable_defaults_bind_live_globals_after_reload(module_factory):
    source = "MODE = 1\ndef helper(x):\n    return x + MODE\ndef kernel(x, fn=helper):\n    return fn(x)\ndef inline(x, fn=lambda x: x + MODE):\n    return fn(x)\n"
    module, path, reloader = module_factory(source)
    path.write_text(source.replace("MODE = 1", "MODE = 2"))
    reloader.apply(reloader.prepare([module.__name__]))
    reloader.apply(reloader.prepare(settings=[{"target": module.__name__ + ":MODE", "value": 3}]))
    assert module.kernel(1) == module.inline(1) == 4
    assert module.kernel.__defaults__[0] is module.helper


def test_scalar_from_import_aliases_propagate_through_conditional_chains(module_factory):
    module, _, reloader = module_factory("MODE = False\n")
    bridge, _, _ = module_factory(f"try:\n    from {module.__name__} import MODE\nexcept ImportError:\n    MODE = None\n", "mlx2.runtime.kernel_reload_cpu_bridge")
    consumer, _, _ = module_factory(f"from {bridge.__name__} import MODE\ndef kernel():\n    return MODE\n", "mlx2.runtime.kernel_reload_cpu_consumer")
    reloader.apply(reloader.prepare(settings=[{"target": module.__name__ + ":MODE", "value": True}]))
    assert module.MODE is bridge.MODE is consumer.MODE is True
    assert consumer.kernel() is True


def test_compiled_global_alias_scan_does_not_depend_on_gc_tracking(module_factory, monkeypatch):
    import gc
    source = "import mlx.core as mx\n@mx.compile\ndef kernel(x):\n    return x + 1\n"
    module, path, reloader = module_factory(source)
    consumer = types.ModuleType("kernel_reload_gc_consumer")
    consumer.kernel = module.kernel
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    gc.collect()
    # GC need not report an untracked dictionary. Exercise that boundary
    # deterministically in addition to collecting the real wrapper graph.
    monkeypatch.setattr(gc, "get_referrers", lambda value: [])
    path.write_text(source.replace("x + 1", "x + 2"))
    reloader.apply(reloader.prepare([module.__name__]))
    assert consumer.kernel is module.kernel


@pytest.mark.parametrize("decorator", ["dataclass", "dataclass(frozen=True)", "dataclass(slots=True)", "dataclass(slots=True, frozen=True)"])
def test_unchanged_dataclass_args_preserve_metadata_and_live_model(module_factory, decorator):
    source = f'''from dataclasses import dataclass, field
@{decorator}
class Args:
    width: int = 4
    tags: list[str] = field(default_factory=list, metadata={{"description": "tags"}})
class Model:
    def __init__(self, args, weight):
        self.args = args
        self.weight = weight
    def forward(self, x):
        return x + self.args.width
'''
    module, path, reloader = module_factory(source)
    args = module.Args()
    model = module.Model(args, object())
    fields = module.Args.__dataclass_fields__
    width_field = fields["width"]
    path.write_text(source.replace("return x + self.args.width", "return x + self.args.width + 1"))
    reloader.apply(reloader.prepare([module.__name__]), types.SimpleNamespace(model=model))
    assert model.forward(2) == 7
    assert model.args is args
    assert type(args) is module.Args
    assert module.Args.__dataclass_fields__ is fields
    assert fields["width"] is width_field
    assert module.Args().width == 4


@pytest.mark.parametrize("edit", [
    ("width: int = 4", "width: int = 5"),
    ("width: int = 4", "width: str = '4'"),
    ("default_factory=list", "default_factory=dict"),
    ("description\": \"tags", "description\": \"different"),
    ("@dataclass", "@dataclass(frozen=True)"),
])
def test_changed_dataclass_structure_is_refused(module_factory, edit):
    source = '''from dataclasses import dataclass, field
@dataclass
class Args:
    width: int = 4
    tags: list[str] = field(default_factory=list, metadata={"description": "tags"})
def kernel(x):
    return x + 1
'''
    module, path, reloader = module_factory(source)
    path.write_text(source.replace(*edit).replace("x + 1", "x + 2"))
    with pytest.raises(reload.KernelReloadError, match="dataclass|structural|class (layout|data|annotations)"):
        reloader.apply(reloader.prepare([module.__name__]))
    assert module.kernel(1) == 2
