from dataclasses import dataclass

import pytest

from mlx2.runtime.technologies.dispatch import (
    FastPath, KernelRegistry, counters_snapshot, refusal_code, reset_counters, spec_from_config,
)


@dataclass(frozen=True)
class _Spec:
    heads: int = 1
    eps: float = 1e-6


def test_spec_from_config_alternatives_and_overrides():
    spec = spec_from_config(_Spec, {"n_heads": 4, "eps": None}, {"heads": ("num_heads", "n_heads"), "eps": "eps"})
    assert spec == _Spec(heads=4, eps=1e-6)
    assert spec_from_config(_Spec, {}, {}, eps=0.5).eps == 0.5
    with pytest.raises(TypeError):
        spec_from_config(_Spec, {}, {}, bogus=1)


def test_registry_selects_admitting_path_and_fails_closed():
    reset_counters()
    reg = KernelRegistry("demo.op", reference=lambda: "ref")
    reg.register(FastPath("fast", admits=lambda spec, **s: None if s.get("n") == 1 else "n!=1", run=lambda: "fast"))
    fn, rec = reg.select(_Spec(), n=1)
    assert fn() == "fast" and rec.selected == "fast"
    fn, rec = reg.select(_Spec(), n=2)
    assert fn() == "ref" and rec.refusals == {"fast": "n!=1"}
    assert reg.select(_Spec(), policy="reference", n=1)[1].selected == "reference"
    with pytest.raises(RuntimeError):
        reg.select(_Spec(), policy="fast", n=2)
    snap = counters_snapshot()
    sel = {(r["path"]): r["count"] for r in snap["selections"]}
    assert sel == {"fast": 1, "reference": 2, "failed_closed": 1}
    assert {(r["code"], r["count"]) for r in snap["refusals"]} == {("n_1", 2)}
    reset_counters()


def test_refusal_code_and_operation_names():
    assert refusal_code("geometry: needs 32 heads") == "geometry"
    assert refusal_code("") == "unspecified"
    with pytest.raises(ValueError):
        KernelRegistry("NoDot", reference=lambda: None)


def test_prometheus_renders_technology_counters():
    from mlx2.prometheus import PrometheusBuilder, _add_technologies

    reset_counters()
    reg = KernelRegistry("demo.op", reference=lambda: "ref")
    reg.register(FastPath("fast", admits=lambda spec, **s: "geometry: no", run=lambda: "fast"))
    reg.select(_Spec())
    builder = PrometheusBuilder()
    _add_technologies(builder, {"technologies": counters_snapshot()})
    text = builder.render()
    assert 'mlx2_technology_dispatch_total{operation="op",path="reference",technology="demo"} 1' in text
    assert 'mlx2_technology_refusals_total{code="geometry",operation="op",path="fast",technology="demo"} 1' in text
    reset_counters()
