"""CPU contracts for the isolated TensorFold profile boundary."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.adapters.qwen38_tensorfold_owned import TensorfoldOwnedB1Profile
from mlx2.runtime.tensorfold_owned_worker import (
    artifact_identity,
    logical_cache_exact,
    mlx_lm_identity,
    profile_identity,
    replay_lane_serial,
    set_live_round_policy,
    source_identity,
)


TENSORFOLD_SOURCE_ENV = "MLX2_TEST_TENSORFOLD_OWNED_SOURCE"
MLX_LM_SOURCE_ENV = "MLX2_TEST_TENSORFOLD_OWNED_MLX_LM_SOURCE"


def _external_source(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"explicit external source {name} is unavailable")
    return Path(value).expanduser().resolve()


def _artifact(root: Path, *, indexed: bool):
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3_5"}')
    (root / "model.safetensors").write_bytes(b"weight-bytes")
    if indexed:
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"layer.weight": "model.safetensors"}
        }))
    return root


def test_artifact_identity_hashes_weight_bytes_and_rejects_traversal(tmp_path):
    target = _artifact(tmp_path / "target", indexed=True)
    first = artifact_identity(target)
    (target / "model.safetensors").write_bytes(b"other-bytes")
    assert artifact_identity(target)["content_digest"] != first["content_digest"]
    index = target / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "../escape.safetensors"}}))
    with pytest.raises(ValueError, match="invalid shard path"):
        artifact_identity(target)
    draft = _artifact(tmp_path / "draft", indexed=False)
    assert len(artifact_identity(draft)["content_digest"]) == 64


def test_profile_namespace_is_distinct_from_ordinary_and_changes_with_artifact(tmp_path):
    target = _artifact(tmp_path / "target", indexed=True)
    draft = _artifact(tmp_path / "draft", indexed=False)
    source = {"revision": "pinned", "source_digest": "a" * 64}
    mlx_lm = {"revision": "pinned", "source_digest": "c" * 64}
    first = profile_identity(source, artifact_identity(target), artifact_identity(draft), mlx_lm)
    assert first != profile_identity({**source, "source_digest": "b" * 64},
                                     artifact_identity(target), artifact_identity(draft), mlx_lm)
    assert first != profile_identity(source, artifact_identity(target), artifact_identity(draft),
                                     {**mlx_lm, "source_digest": "d" * 64})
    (draft / "model.safetensors").write_bytes(b"different")
    assert first != profile_identity(source, artifact_identity(target), artifact_identity(draft), mlx_lm)
    assert not first.startswith("qwen38-27b-hybrid-layer-segments-v1")


def test_owned_profile_refuses_other_target_topology_before_loading(tmp_path):
    target = _artifact(tmp_path / "other-target", indexed=False)
    with pytest.raises(ValueError, match="Qwen3.8 27B target topology"):
        TensorfoldOwnedB1Profile(tmp_path, target, tmp_path,
                                 mlx_lm_source=tmp_path, enabled=True)


def test_worker_process_lifecycle_and_opt_in():
    with pytest.raises(ValueError, match="explicit opt-in"):
        TensorfoldOwnedB1Profile(None, None, None)
    profile = TensorfoldOwnedB1Profile(None, None, None, enabled=True, fixture=True)
    assert not profile.apcv2_state_bridge_qualified
    assert profile.apcv2_namespace != profile.cache_layout
    receipt = profile.start(timeout=5)
    assert receipt["exact_width"] == 16
    opened = profile.call("open", prompt=[3, 5], timeout=5)
    assert opened["cache_layout"] == profile.cache_layout
    sid = opened["session"]
    assert profile.call("compare_round", session=sid, tokens=[5], parents=[-1], timeout=5)["exact"]
    assert profile.call("serial", session=sid, timeout=5)["token"] == 6
    assert profile.call("tree", session=sid, tokens=[6], parents=[-1], timeout=5)["tokens"] == [7]
    assert profile.call("close", session=sid, timeout=5)["closed"] == sid
    with pytest.raises(RuntimeError, match="KeyError"):
        profile.call("serial", session=sid, timeout=5)
    profile.close()
    assert profile.process is None


def test_live_ipc_is_explicit_unqualified_and_never_publishes_apcv2():
    profile = TensorfoldOwnedB1Profile(None, None, None, enabled=True, fixture=True)
    profile.start(timeout=5)
    try:
        first = profile.call("live_open", prompt=[3, 5], max_new_tokens=3, timeout=5)
        assert first["qualified"] is False
        assert first["apcv2_lookup"] is False and first["apcv2_store"] is False
        assert profile.call("live_step", timeout=5)["mode"] == "b1_tree_eligible"
        second = profile.call("live_open", prompt=[8, 9], max_new_tokens=3, timeout=5)
        shared = profile.call("live_step", timeout=5)
        assert shared["mode"] == "b2plus_shared_ordinary"
        assert set(shared["tokens"]) == {first["session"], second["session"]}
        profile.call("live_close", session=second["session"], timeout=5)
        assert profile.call("live_step", timeout=5)["mode"] == "b1_tree_eligible"
        profile.call("live_close", session=first["session"], timeout=5)
    finally:
        profile.close()


def test_live_round_policy_discards_uncommitted_tree_drafts_on_width_change():
    class Stream:
        def __init__(self, sid):
            self.stream_id = sid
            self.finished = False
            self.drafts = True

    class Engine:
        def __init__(self):
            self._next = {"a": ([1, 2, 3], [-1, 0, 0]), "b": [4, 5]}
            self._inflight = {}
            self.pipelined = False

    engine = Engine()
    a, b = Stream("a"), Stream("b")
    assert set_live_round_policy(engine, {"a": a}) == "b1_tree_eligible"
    assert a.drafts and "a" in engine._next
    assert set_live_round_policy(engine, {"a": a, "b": b}) == "b2plus_shared_ordinary"
    assert not a.drafts and not b.drafts and not engine._next
    b.finished = True
    assert set_live_round_policy(engine, {"a": a, "b": b}) == "b1_tree_eligible"
    assert a.drafts
    engine._next["a"] = ([1, 2], [-1, 0])
    engine._inflight["a"] = "already advanced target cache"
    b.finished = False
    with pytest.raises(RuntimeError, match="in-flight target forward"):
        set_live_round_policy(engine, {"a": a, "b": b})
    assert engine._next["a"] == ([1, 2], [-1, 0])
    assert a.drafts
    engine._inflight.clear()
    assert set_live_round_policy(engine, {"a": a, "b": b}) == "b2plus_shared_ordinary"
    assert not engine._next and not a.drafts
    b.finished = True
    a.force_ordinary = True
    assert set_live_round_policy(engine, {"a": a, "b": b}) == "b1_serial_ordinary"
    assert not a.drafts


def test_pinned_native_lane_engine_has_shared_round_and_cache_commit_contract():
    root = _external_source(TENSORFOLD_SOURCE_ENV)
    source_identity(root)
    engine = (root / "src/tensorfold/engine/lane_family.py").read_text()
    shared = (root / "src/tensorfold/engine/family_shared.py").read_text()
    family = (root / "src/tensorfold/families/qwen3_5/family.py").read_text()
    assert "if len(live) > 1 and self.family_streams:" in engine
    assert "self._family_round_streams(live)" in engine
    assert "kind, drafts, forced, parents = self._plan_window(stream)" in shared
    assert "model.hidden_rows(windows" in shared
    assert "model.keep_rows_streams(" in shared
    assert "def hidden_rows(" in family and "def keep_rows_streams(" in family
    assert "gpu_tokens" not in family


def test_live_serial_reference_uses_same_full_prompt_engine_prefill():
    calls = []

    class Stream:
        def __init__(self, sid, prompt, budget, *, sampling, drafts, retain):
            assert sampling is None and drafts is False and retain is False
            self.stream_id = sid
            self.prompt_ids = prompt
            self.max_new_tokens = budget
            self.emitted = []
            self.finished = False

    class Engine:
        def __init__(self, family, *, max_rows, max_draft):
            assert family is marker and max_rows == 1 and max_draft == 0
            self._live = []

        def add_stream(self, stream):
            calls.append(("prefill", list(stream.prompt_ids)))
            stream.emitted.append(98)
            self._live.append((stream, "reference-cache"))

        def step(self):
            calls.append(("serial_step",))
            self._live[0][0].emitted.append(81)

    marker = object()
    stream, cache = replay_lane_serial(marker, [44, 382, 991, 144], 2, Engine, Stream)
    assert stream.emitted == [98, 81] and cache == "reference-cache"
    assert calls == [("prefill", [44, 382, 991, 144]), ("serial_step",)]


def test_real_pinned_tensorfold_source_is_clean_and_bound():
    root = _external_source(TENSORFOLD_SOURCE_ENV)
    identity = source_identity(root)
    assert identity["revision"] == "71377a5373ed7b394f1b480ba2a6a3986b03af1c"
    assert len(identity["source_digest"]) == 64
    mlx_lm = mlx_lm_identity(_external_source(MLX_LM_SOURCE_ENV))
    assert mlx_lm["revision"] == "1104ced19ed98800bdaf4ebcdca14bbdeb597c23"


def test_exact_worker_subprocess_imports_pinned_full_loader_path():
    source = _external_source(TENSORFOLD_SOURCE_ENV)
    mlx_lm = _external_source(MLX_LM_SOURCE_ENV)
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "mlx2.runtime.tensorfold_owned_worker",
               "--import-smoke", "--source", str(source), "--mlx-lm-source", str(mlx_lm)]
    result = subprocess.run(command, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo / "src")},
                            capture_output=True, text=True, timeout=20, check=True)
    receipt = json.loads(result.stdout)
    assert receipt["mlx_lm"]["revision"] == "1104ced19ed98800bdaf4ebcdca14bbdeb597c23"
    assert receipt["tensorfold"]["revision"] == "71377a5373ed7b394f1b480ba2a6a3986b03af1c"


def test_logical_cache_ignores_rejected_kv_backing_but_checks_visible_and_gdn():
    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Array:
        def __init__(self, values):
            self.values = tuple(values)
            self.shape = (1, 1, len(values), 1)

    class MX:
        @staticmethod
        def array_equal(a, b):
            return Scalar(a.values == b.values)

    class KV:
        def __init__(self, values):
            self.keys = Array(values)
            self.values = Array(values)
            self.offset = 2

        @property
        def state(self):
            return self.keys, self.values

        def keys_and_values(self):
            return Array(self.keys.values[:self.offset]), Array(self.values.values[:self.offset])

    class GDN:
        def __init__(self, value):
            self.state = [[Array([value])], Array([value])]

    class DraftSlot:
        keys = None

        @property
        def state(self):
            return [["uncommitted draft context"]]

    left = [KV([1, 2, 0, 0]), GDN(7), DraftSlot()]
    right = [KV([1, 2, 9, 9]), GDN(7), DraftSlot()]
    assert left[0].state[0].values != right[0].state[0].values
    assert logical_cache_exact(MX, left, right)
    right[0].offset = 3
    assert not logical_cache_exact(MX, left, right)
    right[0].offset = 2
    right[1] = GDN(8)
    assert not logical_cache_exact(MX, left, right)
