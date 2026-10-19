"""Host-only research harness gates; no model or GPU performance claims."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_flash_megakernel_ast.py"
SPEC = importlib.util.spec_from_file_location("flash_mega_probe_test", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_manifest_refuses_wrong_revision_before_reading(tmp_path):
    def command(*args, **kwargs):
        return "bad" if "rev-parse" in args else ""
    with pytest.raises(RuntimeError, match="pinned"):
        probe.donor_manifest(tmp_path, command)


def test_manifest_refuses_dirty_donor(tmp_path):
    def command(*args, **kwargs):
        return probe.DONOR_REVISION if "rev-parse" in args else " M kernel.py"
    with pytest.raises(RuntimeError, match="clean"):
        probe.donor_manifest(tmp_path, command)


def test_target_binding_is_deterministic_and_source_bound():
    donor = {"revision": "donor-a", "files": {"z.py": "3", "a.py": "1"}}
    first = probe.target_binding("mlx2-a", "model@one", "contract", donor)
    reordered = probe.target_binding(
        "mlx2-a", "model@one", "contract",
        {"revision": "donor-a", "files": {"a.py": "1", "z.py": "3"}},
    )
    assert first == reordered
    assert first["algorithm"] == "sha256"
    assert len(first["digest"]) == 64
    for changed in (
        probe.target_binding("mlx2-b", "model@one", "contract", donor),
        probe.target_binding("mlx2-a", "model@two", "contract", donor),
        probe.target_binding("mlx2-a", "model@one", "contract-2", donor),
        probe.target_binding("mlx2-a", "model@one", "contract",
                             {"revision": "donor-b", "files": donor["files"]}),
        probe.target_binding("mlx2-a", "model@one", "contract",
                             {"revision": "donor-a", "files": {"a.py": "2"}}),
    ):
        assert changed["digest"] != first["digest"]


def test_research_summary_keeps_approximate_state_boundaries():
    summary = json.loads((PATH.parents[1] /
        "qualification/runs/flash-megakernel-ast-20261019/summary.json").read_text())
    states = summary["states"]
    assert summary["status"] == "bounded_research_only"
    assert summary["ordinary_target_comparison"]["equivalent"] is False
    assert summary["device_oracle"]["numerical_oracle"] is False
    assert states == {
        "implemented": "research_harness",
        "qualified": False,
        "approximate_operation_qualified": False,
        "selected": False,
        "apcv2_publication": False,
        "observed_used": {
            "packing_and_state_gate": True,
            "layer_localization": True,
            "native_mtp_ast": False,
        },
    }


@pytest.mark.parametrize("key,value", [
    ("MLX_MK_EXP_FUSION", "none"),
    ("MLX_QWEN4_MEGAKERNEL_REQUIRE_PRIMITIVES", "0"),
    ("MLX_QWEN4_MEGAKERNEL_TUNE", "force"),
    ("MLX_QWEN4_MEGAKERNEL_GROUPS", "80"),
])
def test_overrides_fail_closed(monkeypatch, tmp_path, key, value):
    monkeypatch.setattr(probe.os, "environ", {key: value})
    with pytest.raises(RuntimeError):
        probe.configure(tmp_path)


def test_private_config(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.os, "environ", {})
    probe.configure(tmp_path)
    assert probe.os.environ["MLX_MK_EXP_WIDE_MOE_E2_ORDER"] == "serial"
    assert probe.os.environ["MLX_QWEN4_MEGAKERNEL_TUNE"] == "off"
    assert probe.os.environ["MLX_QWEN4_MEGAKERNEL_TUNE_CACHE"].startswith(str(tmp_path))


class HostTarget:
    def __init__(self, fault=None):
        self.committed = []
        self.pending = None
        self.fault = fault

    def launch(self, tokens):
        assert self.pending is None
        self.pending = list(tokens)
        output = list(tokens)
        if self.fault == "wide_logits" and len(tokens) > 1:
            output[-1] += 100
        return output, list(tokens), list(tokens), 0.01

    def commit(self):
        self.committed.extend(self.pending)
        if self.fault == "wide_state" and len(self.pending) > 1:
            self.committed[-1] += 100
        self.pending = None

    def rollback(self):
        self.pending = None
        if self.fault == "rollback":
            self.committed.append(-1)

    def snapshot(self):
        assert self.pending is None
        return {"position": len(self.committed), "tokens": list(self.committed)}

    def restore(self, snapshot):
        assert self.pending is None
        self.committed = list(snapshot["tokens"])

    def state(self):
        return {"tokens": tuple(self.committed)}


@pytest.fixture
def host_gate(monkeypatch):
    mx = SimpleNamespace(concatenate=lambda rows, axis: [value for row in rows for value in row])
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    for name in ("raw_array", "restore_array", "digest"):
        monkeypatch.setattr(probe, name, lambda value: value)


def test_every_prefix_and_following_step(host_gate):
    receipts = []
    rows = probe.state_gate(HostTarget(), list(range(40)), 2,
                            lambda values: receipts.append(len(values)), lambda: False)
    assert len(rows) == 6
    assert sum(len(row["prefixes"]) for row in rows) == 18
    assert all(row["logits_exact"] and row["features_exact"] for row in rows)
    assert all(not item["following_step_differences"] for row in rows for item in row["prefixes"])


@pytest.mark.parametrize("fault", ["wide_logits", "wide_state", "rollback"])
def test_counterexamples_stop_gate(host_gate, fault):
    with pytest.raises(RuntimeError, match="counterexample"):
        probe.state_gate(HostTarget(fault), list(range(40)), 1, lambda _: None, lambda: False)


def test_cooperative_cancel_before_launch(host_gate):
    target = HostTarget()
    with pytest.raises(InterruptedError):
        probe.state_gate(target, list(range(40)), 1, lambda _: None, lambda: True)
    assert target.committed == []
    assert target.pending is None


def test_no_cache_export_api():
    assert not hasattr(probe.ResearchTarget, "commit_to_caches")
    assert probe.differences({"a": 1}, {"a": 2, "b": 3}) == ["a", "b"]


def test_width_one_uses_single_token_contract():
    decoder = SimpleNamespace(step=lambda x, ple_embedding: ("single", x, ple_embedding),
                              step_slab=lambda x, ple_embeddings: ("slab", x, ple_embeddings))
    assert probe.launch_kernel(decoder, [[1]], [[2]]) == ("single", [1], [2])
    assert probe.launch_kernel(decoder, [[1], [2]], [[3], [4]]) == ("slab", [[1], [2]], [[3], [4]])


def test_split_projection_layout_and_rebinding():
    class Source:
        def __init__(self, root):
            self.root = root
        def _resolve(self, key):
            value = self.root
            for part in key.split("."):
                value = getattr(value, part)
            return value
        def rebind(self, key, parts):
            for part, value in parts.items():
                setattr(self._resolve(key), part, value)
    def projection(offset):
        return SimpleNamespace(weight=np.arange(24).reshape(2, 3, 4) + offset,
                               scales=np.ones((2, 3, 1)), biases=np.zeros((2, 3, 1)),
                               bits=4, group_size=64)
    gate, up = projection(0), projection(100)
    original_gate, original_up = gate.weight.copy(), up.weight.copy()
    source = probe.split_source_class(Source, SimpleNamespace(
        contiguous=np.ascontiguousarray, concatenate=np.concatenate))(
            SimpleNamespace(experts=SimpleNamespace(gate_proj=gate, up_proj=up)))
    fused = source._resolve("experts.gate_up_proj")
    np.testing.assert_array_equal(fused.weight[:, :3], original_gate)
    np.testing.assert_array_equal(fused.weight[:, 3:], original_up)
    packed = fused.weight.copy()
    gate._mlx2_expert_views = ((gate.weight, gate.scales, gate.biases), ())
    up._mlx2_expert_views = ((up.weight, up.scales, up.biases), ())
    source.rebind("experts.gate_up_proj", {"weight": packed})
    assert source.invalidated_expert_views == 2
    assert "_mlx2_expert_views" not in gate.__dict__
    assert "_mlx2_expert_views" not in up.__dict__
    assert np.shares_memory(gate.weight, packed) and np.shares_memory(up.weight, packed)
    np.testing.assert_array_equal(gate.weight, original_gate)
    np.testing.assert_array_equal(up.weight, original_up)
    assert source._resolve("experts.gate_up_proj") is fused
    with pytest.raises(AttributeError):
        source._resolve("experts.missing")


def test_split_projection_refuses_mixed_quantization():
    class Source:
        def __init__(self, root):
            self.root = root
        def _resolve(self, key):
            return getattr(self.root, key)
    root = SimpleNamespace(gate_proj=SimpleNamespace(weight=np.zeros((2, 3, 4)), bits=4, group_size=64),
                           up_proj=SimpleNamespace(weight=np.zeros((2, 3, 4)), bits=8, group_size=64))
    # Use a nested key so the exact suffix guard applies.
    class NestedSource(Source):
        def _resolve(self, key):
            return super()._resolve(key.removeprefix("experts."))
    source = probe.split_source_class(NestedSource, None)(root)
    with pytest.raises(ValueError, match="quantization mismatch"):
        source._resolve("experts.gate_up_proj")


@pytest.mark.parametrize("mode", ["plain", "k1", "k3", "ast"])
@pytest.mark.parametrize("reject", [None, 0, 1, 2])
def test_ast_composition_all_prefix_outcomes(monkeypatch, mode, reject):
    from mlx2.runtime.ast_planner import ExecutionProfile
    mx = SimpleNamespace(concatenate=np.concatenate)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    class Target(HostTarget):
        def __init__(self):
            super().__init__()
            self.decoder = SimpleNamespace(position=512)
        def launch(self, tokens):
            assert self.pending is None
            self.pending = list(tokens)
            return None, np.array(tokens)[None, :, None], [token + 1 for token in tokens], 0.01
        def commit(self):
            super().commit()
            self.decoder.position = 512 + len(self.committed)
    class Drafter:
        def draft_distributions(self, anchors, hidden, cache, depth, rngs, temperatures, *, context_tokens):
            assert hidden.shape[1] == len(context_tokens[0])
            assert context_tokens[0][-1] == anchors[0]
            tokens = list(range(anchors[0] + 1, anchors[0] + depth + 1))
            if reject is not None and reject < len(tokens):
                tokens[reject] += 100
            return [tokens], []
    profile = ExecutionProfile("source", "host", "fake", 1, (1, 2, 3), (2, 3, 4), 1,
                               (1, 1.3, 1.5, 1.7), 0, 0, 0.1, sequential_draft=True)
    target = Target()
    result = probe.ast_generate(target, Drafter(), [], np.zeros((1, 1, 1)), 0, 17, profile, mode)
    assert result["tokens"] == list(range(17))
    assert target.state() == {"tokens": tuple(range(17))}
    if mode == "ast":
        assert all(tile["plan"] for row in result["rounds"] for tile in row["tiles"])


def test_ast_cancellation_waits_for_closed_transaction(monkeypatch):
    from mlx2.runtime.ast_planner import ExecutionProfile
    mx = SimpleNamespace(concatenate=np.concatenate)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    class Target(HostTarget):
        def __init__(self):
            super().__init__()
            self.decoder = SimpleNamespace(position=512)
        def launch(self, tokens):
            assert self.pending is None
            self.pending = list(tokens)
            return None, np.array(tokens)[None, :, None], [token + 1 for token in tokens], 0.01
        def commit(self):
            super().commit()
            self.decoder.position = 512 + len(self.committed)

    class Drafter:
        def draft_distributions(self, anchors, hidden, cache, depth, rngs, temperatures, *, context_tokens):
            return [[anchors[0] + index + 1 for index in range(depth)]], []

    profile = ExecutionProfile("source", "host", "fake", 1, (1, 2, 3), (2, 3, 4), 1,
                               (1, 1.3, 1.5, 1.7), 0, 0, 0.1, sequential_draft=True)
    target = Target()
    with pytest.raises(InterruptedError, match="closing research transaction"):
        probe.ast_generate(
            target, Drafter(), [], np.zeros((1, 1, 1)), 0, 17, profile, "ast",
            lambda: bool(target.committed),
        )
    assert target.pending is None
    assert target.committed


def test_actual_native_mtp_cache_pairing_survives_rejected_prefix(monkeypatch):
    from mlx2.runtime.ast_planner import ExecutionProfile
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel

    def logsumexp(values):
        maximum = np.max(values)
        return np.array(maximum + np.log(np.exp(values - maximum).sum()))

    mx = SimpleNamespace(
        array=np.array,
        argpartition=np.argpartition,
        concatenate=np.concatenate,
        eval=lambda *values: None,
        float32=np.float32,
        int32=np.int32,
        logsumexp=logsumexp,
        uint32=np.uint32,
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    class HeadTarget:
        external_feature_source = "mtp_seed"
        external_feature_width = 1
        layers = []
        args = SimpleNamespace(
            vocab_size=64, hidden_size=1, num_key_value_heads=1,
            num_attention_heads=1, head_dim=1, intermediate_size=1,
            num_experts_per_tok=0,
        )
        mtp = SimpleNamespace(layers=(), named_modules=lambda: ())

        @staticmethod
        def forward_with_taps(*args, **kwargs):
            raise AssertionError("not called by the draft unit")

        @staticmethod
        def prefill_body(*args, **kwargs):
            raise AssertionError("not called by the draft unit")

        @staticmethod
        def make_mtp_cache():
            return [SimpleNamespace(state=[])]

        @staticmethod
        def mtp_step(hidden, tokens, cache):
            values = np.asarray(tokens, dtype=np.int64)
            cache[0].state.extend(values.reshape(-1).tolist())
            logits = np.full((*values.shape, 64), -100.0, dtype=np.float32)
            for column, token in enumerate(values[0]):
                logits[0, column, (int(token) + 1) % 64] = 0.0
            return logits, np.asarray(hidden, dtype=np.float32) + 1

    class NonlinearTarget(HostTarget):
        def __init__(self):
            super().__init__()
            self.decoder = SimpleNamespace(position=512)
        @staticmethod
        def next_token(token):
            return token + (2 if token == 2 else 1)
        def launch(self, tokens):
            assert self.pending is None
            self.pending = list(tokens)
            return (None, np.asarray(tokens)[None, :, None],
                    [self.next_token(token) for token in tokens], 0.01)
        def commit(self):
            super().commit()
            self.decoder.position = 512 + len(self.committed)

    profile = ExecutionProfile("source", "host", "fake", 1, (1, 2, 3), (2, 3, 4), 1,
                               (1, 1.3, 1.5, 1.7), 0, 0, 0.1, sequential_draft=True)
    expected_target = NonlinearTarget()
    expected = probe.ast_generate(
        expected_target, None, None, np.zeros((1, 1, 1)), 0, 12, profile, "plain")
    head = HeadTarget()
    drafter = NativeMTPDraftModel(head, num_draft=3, revision="bound-revision")
    head_cache = drafter.make_cache()
    actual_target = NonlinearTarget()
    actual = probe.ast_generate(
        actual_target, drafter, head_cache, np.zeros((1, 1, 1)), 0, 12, profile, "ast")
    assert actual["tokens"] == expected["tokens"]
    assert actual["final_state"] == expected["final_state"]
    assert 2 in actual["tokens"] and 3 not in actual["tokens"]
    assert head_cache[0].state
    assert actual_target.pending is None
