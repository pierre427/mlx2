"""L5-01 (sweep 2026-10-12): an inherited ``attention_topology()`` hook must
not read loader state the subclass's own loader never sets.

e3c19168e added revision-bound topology hooks to Flash-Next and Qwen3.6-35B.
Nemotron 3 Super / 3.5 Lightning (FlashNextAdapter) and Qwen3.5-122B
(Qwen3635BA3BAdapter) have their own loaders, so serving's unconditional
``apcv2_cache_layout_fingerprint(adapter)`` raised AttributeError at startup
for every route of those families.
"""

import importlib
import inspect
import pkgutil

import pytest

import mlx2.adapters as adapters_pkg
from mlx2.runtime.attention_topology import (
    FULL_KV,
    RECURRENT,
    apcv2_cache_layout_fingerprint,
)


def _adapter_classes():
    classes = {}
    for info in pkgutil.iter_modules(adapters_pkg.__path__):
        module = importlib.import_module(f"mlx2.adapters.{info.name}")
        for value in vars(module).values():
            if inspect.isclass(value) and value.__module__ == module.__name__:
                classes[f"{value.__module__}.{value.__qualname__}"] = value
    return classes


def _owner(cls, name):
    return next(klass for klass in cls.__mro__ if name in vars(klass))


def _loader_source(klass):
    """The constructor and ``_init*`` loaders this class defines itself."""
    return "".join(
        inspect.getsource(member)
        for name, member in vars(klass).items()
        if (name == "__init__" or name.startswith("_init")) and inspect.isfunction(member)
    )


def test_every_inherited_topology_hook_has_its_loader_state():
    checked = []
    for name, cls in sorted(_adapter_classes().items()):
        if not callable(getattr(cls, "attention_topology", None)):
            continue
        hook = _owner(cls, "attention_topology")
        init = _owner(cls, "__init__")
        if issubclass(hook, init):
            continue  # the hook is defined at or below the loader that runs
        hook_source = inspect.getsource(hook.attention_topology)
        if "_attention_topology_config" not in hook_source:
            continue
        checked.append(name)
        assert "_attention_topology_config" in _loader_source(init), (
            f"{name} inherits {hook.__name__}.attention_topology but its loader "
            f"{init.__name__} never sets _attention_topology_config"
        )
    # Guard the guard: the Qwen3.8 family inherits the hook with its loader.
    assert all("qwen35_122b" not in name and "nemotron" not in name for name in checked)


def _bare(cls, layout):
    adapter = object.__new__(cls)
    adapter.identity = {"fingerprint": "rev-1"}
    adapter.layout = layout
    return adapter


@pytest.mark.parametrize(
    "module,name",
    [
        ("mlx2.adapters.nemotron3_super", "Nemotron3SuperAdapter"),
        ("mlx2.adapters.nemotron35_lightning", "Nemotron35LightningAdapter"),
    ],
)
def test_nemotron_serving_namespace_constructs_without_flash_next_topology(module, name):
    module = importlib.import_module(module)
    cls = getattr(module, name)
    layout = getattr(module, "CACHE_LAYOUT", "nemotron-layout")
    # serving.py: APCv2 namespace composition at startup.  Nemotron-H's
    # mamba/attention/MoE pattern has no topology description yet, so it
    # keeps its legacy layout namespace instead of Flash-Next's QSA/GDN one.
    assert apcv2_cache_layout_fingerprint(_bare(cls, layout)) == layout


def _qwen35_122b_text_config():
    from mlx2.adapters.qwen35_122b import EXPECTED

    return {
        **EXPECTED,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(48)
        ],
        # The loader drops the embedded MTP layer before building the model.
        "mtp_num_hidden_layers": 0,
    }


def test_qwen35_122b_topology_describes_its_own_trunk():
    from mlx2.adapters.qwen35_122b import CACHE_LAYOUT, Qwen35122BA10BAdapter
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

    adapter = _bare(Qwen35122BA10BAdapter, CACHE_LAYOUT)
    adapter._attention_topology_config = _qwen35_122b_text_config()
    layout, fingerprint = apcv2_cache_layout_fingerprint(adapter)
    assert layout == CACHE_LAYOUT and len(fingerprint) == 64
    topology = adapter.attention_topology()
    assert len(topology.layers) == 48
    assert topology.layers.count(FULL_KV) == 12
    assert topology.layers.count(RECURRENT) == 36
    assert topology.auxiliary_layers == ()
    # 64 value / 16 key GDN heads: not the 35B's 32v16k recurrent state.
    sibling = _bare(Qwen3635BA3BAdapter, CACHE_LAYOUT)
    sibling._attention_topology_config = _qwen35_122b_text_config()
    assert topology.recurrent_layout != sibling.attention_topology().recurrent_layout
