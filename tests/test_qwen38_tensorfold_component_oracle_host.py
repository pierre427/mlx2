from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_oracle():
    path = ROOT / "scripts/qwen38_tensorfold_component_oracle.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_runner():
    path = ROOT / "scripts/qualify_qwen38_dflash_checkpoint_parity.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()


def candidate_source(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    source = tmp_path / "tensorfold"
    tracked = source / "src/tensorfold"
    tracked.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "host-test@example.invalid"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Host Test"], cwd=source, check=True
    )
    (tracked / "law.py").write_text("LAW = 'base'\n")
    subprocess.run(["git", "add", "src/tensorfold/law.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=source, check=True)
    parent = git(source, "rev-parse", "HEAD")
    (tracked / "law.py").write_text("LAW = 'candidate'\n")
    subprocess.run(["git", "add", "src/tensorfold/law.py"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "candidate"], cwd=source, check=True
    )
    return source, {
        "revision": git(source, "rev-parse", "HEAD"),
        "tree": git(source, "rev-parse", "HEAD^{tree}"),
        "parent_revision": parent,
    }


class FakeArray:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, _key):
        return self


class FakeProjection:
    def __init__(self, name):
        self.name = name

    def __call__(self, value):
        return FakeArray(f"{value.name}->{self.name}")


def fake_model():
    linear = SimpleNamespace(
        is_linear=True,
        linear_attn=SimpleNamespace(out_proj=FakeProjection("gdn.out")),
        mlp=SimpleNamespace(down_proj=FakeProjection("linear.mlp.down")),
    )
    attention = SimpleNamespace(
        is_linear=False,
        self_attn=SimpleNamespace(o_proj=FakeProjection("attn.out")),
        mlp=SimpleNamespace(down_proj=FakeProjection("attn.mlp.down")),
    )
    layers = [linear, SimpleNamespace(_layer=attention)]
    modules = [
        linear.linear_attn.out_proj,
        linear.mlp.down_proj,
        attention.self_attn.o_proj,
        attention.mlp.down_proj,
    ]
    return SimpleNamespace(
        layers=layers,
        named_modules=lambda: iter(
            (f"p{index}", item) for index, item in enumerate(modules)
        ),
    )


def test_policy_arms_are_explicit_and_do_not_mutate_input():
    module = load_oracle()
    source = {
        "min_rows": {"q4": 8, "q8": 8},
        "sources": {"min_rows.q4": "builtin", "min_rows.q8": "builtin"},
    }
    common = module.explicit_projection_policy(source, "common")
    crossover = module.explicit_projection_policy(source, "crossover")
    assert common["min_rows"] == {"q4": 1, "q8": 8}
    assert common["sources"]["min_rows.q4"] == "component-oracle:common"
    assert crossover == source
    assert source["min_rows"]["q4"] == 8
    assert module.explicit_execution_policy({}, "on") == {"fused_gdn": True}
    assert module.explicit_execution_policy({}, "off") == {"fused_gdn": False}
    with pytest.raises(ValueError, match="projection law"):
        module.explicit_projection_policy(source, "unknown")
    with pytest.raises(ValueError, match="fused GDN"):
        module.explicit_execution_policy({}, "unknown")


def test_projection_capture_is_instance_scoped_complete_and_restored():
    module = load_oracle()
    model = fake_model()
    original = FakeProjection.__call__
    capture = module.ProjectionCapture(model)
    targets = {value for value in capture.targets.values()}
    assert targets == {
        (0, "gdn_projection"),
        (0, "mlp_projection"),
        (1, "attention_projection"),
        (1, "mlp_projection"),
    }
    modules = [item for _, item in model.named_modules()]
    with capture:
        with capture.route("tensorfold"):
            for item in modules:
                item(FakeArray("same"))
        with capture.route("ordinary"):
            for item in modules:
                item(FakeArray("same"))
        with (
            pytest.raises(RuntimeError, match="already exists"),
            capture.route("ordinary"),
        ):
            pass
    assert FakeProjection.__call__ is original
    assert set(capture.records["tensorfold"]) == targets
    assert set(capture.records["ordinary"]) == targets


def test_component_report_names_first_projection_boundary():
    module = load_oracle()
    model = fake_model()
    capture = module.ProjectionCapture(model)
    modules = [item for _, item in model.named_modules()]
    with capture:
        with capture.route("tensorfold"):
            for item in modules:
                item(FakeArray("tree"))
        with capture.route("ordinary"):
            for item in modules:
                item(FakeArray("ordinary"))

    class FakeMx:
        @staticmethod
        def eval(*_values):
            return None

    def compare(_mx, left, right, *, atol, rtol):
        assert (atol, rtol) == (0.01, 0.02)
        equal = left.name == right.name
        return {"equal": equal, "close": equal}

    report = module.component_report(FakeMx, capture, compare, atol=0.01, rtol=0.02)
    assert report["complete"] is True
    assert report["missing"] == {"tensorfold": [], "ordinary": []}
    assert report["first_input_exact_divergence"] == {
        "layer": 0,
        "component": "gdn_projection",
    }
    assert report["first_output_tolerance_failure"] == {
        "layer": 0,
        "component": "gdn_projection",
    }


def test_fused_gdn_arm_requires_observed_engagement_without_fallback():
    module = load_oracle()
    before = {
        "enabled": True,
        "layers": 48,
        "decode_calls": 10,
        "batch_decode_calls": 2,
        "fallbacks": 3,
    }
    engaged = {
        **before,
        "decode_calls": 58,
    }
    fallback = {
        **before,
        "decode_calls": 57,
        "fallbacks": 4,
    }

    report = module.fused_gdn_engagement_report(before, engaged, "on")
    assert report["passed"] is True
    assert report["delta"] == {
        "decode_calls": 48,
        "batch_decode_calls": 0,
        "fallbacks": 0,
    }
    assert module.fused_gdn_engagement_report(before, fallback, "on")[
        "passed"
    ] is False


def test_unfused_gdn_arm_requires_disabled_zero_counter_delta():
    module = load_oracle()
    off = {
        "enabled": False,
        "layers": 48,
        "decode_calls": 0,
        "batch_decode_calls": 0,
        "fallbacks": 0,
    }
    assert module.fused_gdn_engagement_report(off, dict(off), "off")["passed"] is True
    wrongly_enabled = {**off, "enabled": True}
    assert module.fused_gdn_engagement_report(off, wrongly_enabled, "off")[
        "passed"
    ] is False


def test_native_runner_records_source_and_arm_identity():
    source = (ROOT / "scripts/qualify_qwen38_dflash_checkpoint_parity.py").read_text()
    for fragment in (
        '"--projection-law"',
        '"--fused-gdn"',
        '"--component-oracle"',
        'projection_capture.route("tensorfold")',
        'projection_capture.route("ordinary")',
        "finally:\n                tree_transaction.abort()",
        '"component_oracle_helper_sha256"',
        '"q4_min_rows"',
        '"tensorfold_binding"',
        '"component_oracle_identity_and_engagement"',
    ):
        assert fragment in source


def test_deepest_path_reference_replays_ordinary_one_token_decode():
    runner = load_runner()
    arrays = []
    evaluations = []

    class FakeMx:
        uint32 = "uint32"

        @classmethod
        def array(cls, value, *, dtype):
            arrays.append((value, dtype))
            return value

        @classmethod
        def eval(cls, *values):
            evaluations.append(values)

    calls = []

    class FakeModel:
        @staticmethod
        def forward_with_taps(token_ids, cache, layers):
            calls.append((token_ids, cache, layers))
            token = token_ids[0][0]
            return f"logits-{token}", f"taps-{token}"

    adapter = SimpleNamespace(model=FakeModel())
    cache, layers = object(), object()
    logits, taps = runner._serial_decode_path(
        FakeMx,
        adapter,
        cache,
        layers,
        [101, 102, 103, 104],
        [0, 1, 3],
    )

    assert arrays == [
        ([[101]], "uint32"),
        ([[102]], "uint32"),
        ([[104]], "uint32"),
    ]
    assert [call[0] for call in calls] == [[[101]], [[102]], [[104]]]
    assert all(call[1:] == (cache, layers) for call in calls)
    assert evaluations == [
        ("logits-101", "taps-101"),
        ("logits-102", "taps-102"),
        ("logits-104", "taps-104"),
    ]
    assert (logits, taps) == ("logits-104", "taps-104")


def test_deepest_path_reference_refuses_empty_path():
    runner = load_runner()

    class FakeMx:
        uint32 = "uint32"

    adapter = SimpleNamespace(model=SimpleNamespace())
    with pytest.raises(RuntimeError, match="empty"):
        runner._serial_decode_path(
            FakeMx, adapter, object(), object(), [], []
        )


def test_candidate_bind_requires_complete_full_identity():
    runner = load_runner()
    assert runner.qualification_candidate_spec(None, None, None) is None
    with pytest.raises(ValueError, match="requires revision, tree, and parent"):
        runner.qualification_candidate_spec("a" * 40, None, "b" * 40)
    with pytest.raises(ValueError, match="full lowercase Git OIDs"):
        runner.qualification_candidate_spec("a" * 7, "b" * 40, "c" * 40)


def test_candidate_bind_is_exact_process_local_and_receipted(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx2.runtime.qwen38_tensorfold", raising=False)
    runner = load_runner()
    source, candidate = candidate_source(tmp_path)
    fake = SimpleNamespace(EXPECTED_REVISION=runner.PRODUCTION_TENSORFOLD_REVISION)

    def validate_source(root):
        assert fake.EXPECTED_REVISION == candidate["revision"]
        return {
            "root": str(Path(root).resolve()),
            "revision": candidate["revision"],
            "tracked_source": "src/tensorfold",
            "tracked_diff": "",
        }

    fake.validate_source = validate_source
    receipt = runner.bind_tensorfold_source(
        source,
        candidate,
        source_module=fake,
    )

    assert fake.EXPECTED_REVISION == candidate["revision"]
    assert receipt["mode"] == "qualification_candidate"
    assert receipt["qualification_only"] is True
    assert receipt["production_revision_on_disk"] == (
        runner.PRODUCTION_TENSORFOLD_REVISION
    )
    assert receipt["candidate"] == candidate
    assert receipt["active"]["revision"] == candidate["revision"]
    assert receipt["active"]["tree"] == candidate["tree"]
    assert receipt["active"]["parent_revision"] == candidate["parent_revision"]


def test_candidate_bind_refuses_identity_mismatch_without_patching(
    tmp_path, monkeypatch
):
    monkeypatch.delitem(sys.modules, "mlx2.runtime.qwen38_tensorfold", raising=False)
    runner = load_runner()
    source, candidate = candidate_source(tmp_path)
    fake = SimpleNamespace(
        EXPECTED_REVISION=runner.PRODUCTION_TENSORFOLD_REVISION,
        validate_source=lambda _root: pytest.fail("validator must not run"),
    )
    wrong = {**candidate, "tree": "0" * 40}

    with pytest.raises(RuntimeError, match="identity"):
        runner.bind_tensorfold_source(source, wrong, source_module=fake)

    assert fake.EXPECTED_REVISION == runner.PRODUCTION_TENSORFOLD_REVISION


def test_candidate_bind_refuses_dirty_imported_source(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx2.runtime.qwen38_tensorfold", raising=False)
    runner = load_runner()
    source, candidate = candidate_source(tmp_path)
    (source / "src/tensorfold/untracked.py").write_text("DIRTY = True\n")
    fake = SimpleNamespace(
        EXPECTED_REVISION=runner.PRODUCTION_TENSORFOLD_REVISION,
        validate_source=lambda _root: pytest.fail("validator must not run"),
    )

    with pytest.raises(RuntimeError, match="differs from HEAD"):
        runner.bind_tensorfold_source(source, candidate, source_module=fake)

    assert fake.EXPECTED_REVISION == runner.PRODUCTION_TENSORFOLD_REVISION


def test_candidate_bind_refuses_after_tensorfold_runtime_import(tmp_path, monkeypatch):
    runner = load_runner()
    source, candidate = candidate_source(tmp_path)
    monkeypatch.setitem(sys.modules, "mlx2.runtime.qwen38_tensorfold", object())
    fake = SimpleNamespace(
        EXPECTED_REVISION=runner.PRODUCTION_TENSORFOLD_REVISION,
        validate_source=lambda _root: pytest.fail("validator must not run"),
    )

    with pytest.raises(RuntimeError, match="runtime imported before"):
        runner.bind_tensorfold_source(source, candidate, source_module=fake)

    assert fake.EXPECTED_REVISION == runner.PRODUCTION_TENSORFOLD_REVISION


@pytest.mark.parametrize(
    "environment,message",
    [
        ({}, "GPUQ_LEASE"),
        ({"GPUQ_LEASE": "lease-1"}, "GPUQ_SESSION"),
        (
            {"GPUQ_LEASE": "foreign", "GPUQ_SESSION": "session-1"},
            "GPUQ_LEASE",
        ),
        (
            {"GPUQ_LEASE": "lease-1", "GPUQ_SESSION": "foreign"},
            "GPUQ_SESSION",
        ),
    ],
)
def test_checkpoint_gpu_gate_refuses_missing_or_foreign_owner(
    tmp_path, environment, message
):
    runner = load_runner()
    owner = {"lease_id": "lease-1", "session": "session-1", "pid": 123}
    paths = (tmp_path / "shared.json", tmp_path / "temporary.json")
    for path in paths:
        path.write_text(json.dumps(owner))

    with pytest.raises(RuntimeError, match=message):
        runner.require_gpu_owners(environ=environment, paths=paths)


def test_checkpoint_gpu_gate_accepts_only_the_matching_pair(tmp_path):
    runner = load_runner()
    owner = {"lease_id": "lease-1", "session": "session-1", "pid": 123}
    paths = (tmp_path / "shared.json", tmp_path / "temporary.json")
    for path in paths:
        path.write_text(json.dumps(owner))

    assert runner.require_gpu_owners(
        environ={"GPUQ_LEASE": "lease-1", "GPUQ_SESSION": "session-1"},
        paths=paths,
    ) == owner
