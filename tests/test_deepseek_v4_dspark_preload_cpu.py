"""All sidecar staging is materialized before serving admission; explicit CPU."""

from types import SimpleNamespace

import mlx.core as mx

mx.set_default_device(mx.cpu)
from test_deepseek_v4_forward_cpu import SmallWeights, config

from mlx2.runtime.drafters.deepseek_v4_dspark import DeepSeekV4DSparkBackbone
from mlx2.runtime.models.deepseek_v4 import HyperConnection, Rotary


def test_seven_hc_groups_and_native_projection_caches_preload_before_wire(monkeypatch):
    c = config((0, 0, 0))
    events = []

    class Provider(SmallWeights):
        def __init__(self, c):
            super().__init__(c)
            for name, value in list(self.data.items()):
                if ".hc_" in name or "_hc_" in name:
                    self.data[name] = value.astype(mx.float16)
            self._dense = self.data
            self._packed = {}
            self.tensors = {}
            self.promoted = {}

        def linear(self, name):
            events.append(("linear", name))
            self.promoted.setdefault(name, mx.ones((2, 2), dtype=mx.float32))
            return lambda x: x

        def native_fp32_parameters(self):
            return self.promoted

        def preload_native_linear_fp32(self):
            events.append(("promoted-preloaded", len(self.promoted)))

    w = Provider(c)
    b = object.__new__(DeepSeekV4DSparkBackbone)
    b.w = w
    b.config = SimpleNamespace(stages=3)
    b.final_prefix = "mtp.2."
    b.stages = [
        SimpleNamespace(
            attn_hc=HyperConnection(c, w, f"blk.{i}.hc_attn"),
            ffn_hc=HyperConnection(c, w, f"blk.{i}.hc_ffn"),
            rope=Rotary(2, 10000.0),
        )
        for i in range(3)
    ]
    b.head = HyperConnection(c, w, "output_hc", head=True)
    original = mx.eval
    captured = []

    def leaves(value):
        if isinstance(value, dict):
            return [x for v in value.values() for x in leaves(v)]
        if isinstance(value, (tuple, list)):
            return [x for v in value for x in leaves(v)]
        return [value]

    def evaluate(value):
        assert events[-1] == ("promoted-preloaded", 27)
        assert len(value["hc_static_fp32"]) == 7
        assert len(value["native_linear_fp32"]) == 27
        assert set(value["rotary_inv"]) == {"mtp.0", "mtp.1", "mtp.2"}
        captured.extend(leaves(value))
        return original(value)

    monkeypatch.setattr(mx, "eval", evaluate)
    assert b.preload() is b
    events.append(("wire", None))
    assert events.index(("promoted-preloaded", 27)) < events.index(("wire", None))
    hc = [b.head] + [v for stage in b.stages for v in (stage.attn_hc, stage.ffn_hc)]
    expected = {id(x) for h in hc for x in h.static_parameters().values()}
    assert expected <= {id(x) for x in captured}
    assert all(h.fn.dtype == mx.float16 and h.fn_fp32.dtype == mx.float32 for h in hc)
    assert b.parameters()["native_linear_fp32"] is w.promoted
    # Native promotions and static HC arrays reuse identity on another preload.
    before = {k: id(v) for k, v in w.promoted.items()}
    b.preload()
    assert before == {k: id(v) for k, v in w.promoted.items()}
