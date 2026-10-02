"""Host-only tests for ``scripts/probe_native_mtp_head_d1.py``.

No MLX and no mlx2 runtime: ``mlx``, ``mlx_lm`` and ``mlx2`` imports are
blocked before the probe loads and checked absent around every test. No
model, weight, cache, Metal kernel or GPU is touched. The generator, the
``hybrid_speculative`` entry points, the host ops and the state digests are
HOST SUBSTITUTES that follow the runtime semantics read from source
(depth ``min(1, max_tokens - ntoks - 1)``, zero fast path without commit,
``consumed = min(emitted, accepted)``, lane stats arithmetic). Passing here
says the probe's orchestration and validators behave as designed on those
substitutes; it is not a native run, not acceptance evidence and not any
qualification.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest \\
      -p no:cacheprovider -o addopts= -q tests/test_native_mtp_head_probe_cpu.py
"""

import ast
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

BLOCKED = ("mlx", "mlx_lm", "mlx2")


class _BlockMLX:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked in host-only test: {name}")


def _loaded():
    return sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)


sys.meta_path.insert(0, _BlockMLX())
assert not _loaded(), _loaded()

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_native_mtp_head_d1.py"
_spec = importlib.util.spec_from_file_location("probe_native_mtp_head_d1_cpu", SCRIPT)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
ORACLE = P.load_oracle()


@pytest.fixture(autouse=True)
def _host_only():
    assert not _loaded()
    yield
    assert not _loaded(), _loaded()


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def complete(tag):
    return {"status": "complete", "sha256": _h(tag), "reason": None}


def metadata_unavailable(tag):
    return {"status": "metadata_unavailable", "sha256": None, "state_only_sha256": _h(tag),
            "reason": "no meta_state on substitute.Cache"}


# ------------------------------------------------------------ host substitutes

TARGET = [100 + i for i in range(400)]


class ZeroDepthFastUnavailable(RuntimeError):
    pass


class Tok:
    def __init__(self, token, from_draft):
        self.token, self.from_draft = token, from_draft


def fake_module():
    """Substitute for the three hybrid_speculative entry points."""
    module = types.SimpleNamespace()
    module.propose_batched_self_mtp = lambda model, state: state.propose()
    module.advance_batched_self_mtp_zero = lambda model, state: state.zero()

    def commit(state, proposal, *, emitted_counts, terminal):
        state.commit(proposal, emitted_counts, terminal)

    module.commit_batched_self_mtp = commit
    return module


class FakeState:
    """Substitute lane state: proposes from a scripted target and agreement."""

    def __init__(self, module, *, target, max_tokens, agree, knobs):
        self.module, self.target, self.max_tokens = module, target, max_tokens
        self.agree, self.k = agree, knobs
        self.ntoks, self.cycle, self.next_depth = 1, 0, None
        self.stats = {"cycles": 0, "draft_cycles": 0, "draft_proposed": 0, "draft_accepted": 0,
                      "bonus_tokens": 0, "plain_tokens": 1, "plain_cycles": 0,
                      "retrieval_cycles": 0, "retrieval_proposed": 0, "retrieval_accepted": 0}
        self._nesting = False

    def _proposal(self, depth, accepted, *, zero):
        i = self.cycle
        self.cycle += 1
        if self.k.get("raise_at") == i:
            raise RuntimeError("injected proposal failure")
        n = self.ntoks
        row = tuple([Tok(self.target[n + j], True) for j in range(accepted)]
                    + [Tok(self.target[n + accepted], False)])
        uids = self.k.get("uids", (7,))
        return types.SimpleNamespace(
            lane_uids=uids, draft_depths=(depth,) * len(uids),
            accepted_lengths=(accepted,) * len(uids), outputs=(row,) * len(uids),
            copy_spans=() if zero else (self.k.get("copy_spans", {}).get(i, 0),),
            copy_decisions=() if zero else (self.k.get("copy_decisions", {}).get(i, "off"),),
            relaxed_accepts=() if zero else (self.k.get("relaxed", {}).get(i, 0),),
            draft_features=self.k.get("features", {}).get(i, ()),
            zero_fast_path=zero, true_batched=False)

    def propose(self):
        depth = self.next_depth
        accepted = depth if depth and self.agree(self.cycle) else 0
        accepted = self.k.get("accepted", {}).get(self.cycle, accepted)
        return self._proposal(depth, accepted, zero=False)

    def zero(self):
        if self.k.get("zero_unavailable"):
            raise ZeroDepthFastUnavailable("substitute: no batched K=0 ABI")
        proposal = self._proposal(self.next_depth, 0, zero=True)
        self.ntoks += 1
        self.stats["cycles"] += 1
        self.stats["draft_cycles"] += 1
        self.stats["bonus_tokens"] += 1
        return proposal

    def commit(self, proposal, emitted_counts, terminal):
        if self.k.get("nested") and not self._nesting:
            # Segmented shape: the outer commit re-enters by module attribute.
            self._nesting = True
            try:
                self.module.commit_batched_self_mtp(
                    self, proposal, emitted_counts=emitted_counts, terminal=terminal)
            finally:
                self._nesting = False
            return
        count, accepted = emitted_counts[0], proposal.accepted_lengths[0]
        consumed = min(count, accepted)
        self.ntoks += count
        self.stats["cycles"] += 1
        self.stats["draft_cycles"] += 1
        self.stats["draft_proposed"] += proposal.draft_depths[0]
        self.stats["draft_accepted"] += consumed
        if count > accepted:
            self.stats["bonus_tokens"] += 1


class _Policy:
    def __init__(self, enabled=False):
        self.enabled = enabled


class FakeGen:
    """Substitute BatchGenerator, one lane; native arm calls the module per cycle."""

    def __init__(self, arm, module=None, *, max_tokens, target=TARGET, stops=(), agree=None,
                 lp=None, tag="", knobs=None):
        self.arm, self.module, self.max_tokens = arm, module, max_tokens
        self.target, self.stops, self.tag = target, set(stops), tag
        self.k = dict(knobs or {})
        self.lp = lp or (lambda pos: -0.25)
        self.self_mtp = dict(CONFIG) if arm == "native" else None
        self.adaptive_mtp_depth = None
        self.mtp_ordinary_handoff = self.mtp_admission = self.mtp_acceptance_logger = None
        self.fly_verification, self.copy_draft = _Policy(), _Policy()
        self.scheduler_stats = dict(self.k.get("scheduler", {}))
        self._generation_batch = types.SimpleNamespace(adaptive_depth_policy=None,
                                                       ordinary_handoff_policy=None)
        self.state = (FakeState(module, target=target, max_tokens=max_tokens,
                                agree=agree or (lambda i: i % 3 != 2), knobs=self.k)
                      if arm == "native" else None)
        self.count, self.uid, self.done, self.closed = 0, 41, False, False
        self.inserted = None

    def insert(self, prompts, **kwargs):
        if self.k.get("insert_raises"):
            raise RuntimeError("insert refused")
        self.inserted = (prompts, kwargs)
        return (self.uid,)

    def take_lane_failures(self):
        return list(self.k.get("lane_failures", ()))

    def close(self):
        self.closed = True
        if self.k.get("close_raises"):
            raise RuntimeError("close failed")

    def _respond(self, token, from_draft):
        pos = self.count
        self.count += 1
        reason = "length" if self.count >= self.max_tokens else None
        if token in self.stops:
            reason = "stop"
        flip = self.k.get("flip_from_draft_at")
        return types.SimpleNamespace(
            uid=self.uid, token=token, from_draft=(not from_draft) if flip == pos else from_draft,
            logprobs=self.k.get("logprob_factory", lambda p, t, v: ("row", p, t, v))(
                pos, token, self.lp(pos)), finish_reason=reason)

    def _finish(self, response):
        self.done = True
        response.prompt_cache = self.k.get("target_state", complete(f"target{self.tag}"))
        response.rng_draws = 0
        if self.arm == "native":
            response.mtp_state = self.k.get("sidecar", complete(f"sidecar{self.tag}"))
            stats = dict(self.state.stats)
            stats["total_emitted"] = (stats["retrieval_accepted"] + stats["draft_accepted"]
                                      + stats["bonus_tokens"] + stats["plain_tokens"])
            stats.update(self.k.get("stats_patch", {}))
            response.mtp_receipt = {"num_draft": 1, "observed_compute_widths": [1],
                                    "verification": "exact", "relaxed_accepts": 0,
                                    "adaptive_depth": None, "stats": stats,
                                    **self.k.get("receipt_patch", {})}
        elif "ordinary_sidecar" in self.k:
            response.mtp_state = self.k["ordinary_sidecar"]
        return response

    def next(self):
        if self.done:
            return None, []
        if self.k.get("foreign_patch_at") == self.count and self.module is not None:
            self.module.propose_batched_self_mtp = lambda model, state: state.propose()
        if self.count == 0 or self.arm == "ordinary":
            if self.arm == "ordinary" and self.k.get("ordinary_calls_mtp") and self.count == 2:
                self.module.propose_batched_self_mtp(None, types.SimpleNamespace(
                    propose=lambda: FakeState(self.module, target=self.target, max_tokens=8,
                                              agree=lambda i: True, knobs={})._proposal(
                                                  1, 1, zero=False)))
            response = self._respond(self.target[self.count], False)
            return None, [self._finish(response) if response.finish_reason else response]
        st = self.state
        planned = min(1, max(self.max_tokens - st.ntoks - 1, 0))
        st.next_depth = self.k.get("depth", lambda i, d: d)(st.cycle, planned)
        if st.next_depth == 0:
            try:
                proposal = self.module.advance_batched_self_mtp_zero(None, st)
            except ZeroDepthFastUnavailable:
                proposal = self.module.propose_batched_self_mtp(None, st)
        else:
            proposal = self.module.propose_batched_self_mtp(None, st)
        responses, emitted, terminal = [], 0, False
        for output in proposal.outputs[0]:
            emitted += 1
            response = self._respond(output.token, output.from_draft)
            responses.append(response)
            if response.finish_reason:
                terminal = True
                break
        if not proposal.zero_fast_path and not self.k.get("skip_commit"):
            self.module.commit_batched_self_mtp(st, proposal, emitted_counts=[emitted],
                                                terminal=[terminal])
        if self.k.get("extra_plain_at") == st.cycle - 1 and not terminal:
            # An untraced token, as a plain fallback would deliver it.
            responses.append(self._respond(self.target[self.count], False))
            st.ntoks += 1
        if terminal:
            self._finish(responses[-1])
        return None, responses


class Ops:
    """Substitute host ops: raw-row digest records from the substitute logprob tuple."""

    def __init__(self, *, incomplete_at=(), nan_at=(), posinf_at=()):
        self.t = 0.0
        self.incomplete_at, self.nan_at, self.posinf_at = incomplete_at, nan_at, posinf_at

    def now(self):
        self.t += 0.001
        return self.t

    def sync(self):
        pass

    def row(self, logprobs, token):
        _, pos, tok, value = logprobs
        digest = (metadata_unavailable(f"row-{pos}") if pos in self.incomplete_at
                  else complete(f"row-{pos}-{tok}-{value!r}"))
        return {"digest": digest, "nan": math.isnan(value) or pos in self.nan_at,
                "posinf": value == math.inf or pos in self.posinf_at, "chosen": value,
                "dtype": "substitute.bfloat16", "shape": [512]}

    def memory(self):
        return None

    def digest(self, obj):
        return dict(obj) if isinstance(obj, dict) else ORACLE.state_digest(None)


CONFIG = {"persistent": True, "num_draft": 1, "rate_gate": False, "prefill_step_size": 512,
          "segment_aware_live_tip": True, "segment_aware_cohort_size": 1}
PROMPT = [11, 12, 13, 14]


def one_run(arm, *, max_tokens=16, module=None, limit=None, ops=None, **kwargs):
    """Compose one run exactly as run_native does, over substitutes."""
    module = module or fake_module()
    gen = FakeGen(arm, module, max_tokens=max_tokens, **kwargs)
    observed, contract = P.generator_contract(gen, arm, CONFIG, environ={})
    record = P.drive_run(arm, gen, module, ops or Ops(), prompt=PROMPT, max_tokens=max_tokens,
                         rng=object(), trace_limit=limit or 2 * max_tokens + 8)
    record["contract"] = observed
    record["contract_refusals"] = contract + record.get("contract_refusals", [])
    return record, gen, module


def four(max_tokens=16, ordinary=None, native=None, native2=None, ordinary2=None):
    runs = []
    for arm, extra in zip(P.ARM_ORDER, (ordinary, native, native2 or native, ordinary2 or ordinary)):
        runs.append(one_run(arm, max_tokens=max_tokens, **(extra or {}))[0])
    return runs, P.decide(runs, max_tokens=max_tokens)


def native_eval(max_tokens=16, **kwargs):
    record = one_run("native", max_tokens=max_tokens, **kwargs)[0]
    return P.evaluate_run(record, max_tokens=max_tokens), record


def refused_with(problems, text):
    return any(text in p for p in problems)


# ------------------------------------------------------------ import / CLI

def test_module_import_is_stdlib_only_and_constants_are_fixed():
    assert P.LANES == 1 and P.NUM_DRAFT == 1
    assert P.ARM_ORDER == ("ordinary", "native", "native", "ordinary")
    assert P.MAX_PROMPT_TOKENS == 1024 and P.MAX_TOKENS_RANGE == (8, 128)
    assert ORACLE.STATE_ORACLE == P.ORACLE_VERSION
    assert Path(ORACLE.__file__).resolve() == ROOT / "scripts" / "paired_direct_ab.py"


def test_help_in_fresh_process_loads_no_mlx():
    code = (
        "import sys, runpy\n"
        "class B:\n"
        "    def find_spec(self, n, p=None, t=None):\n"
        "        if n.split('.')[0] in ('mlx', 'mlx_lm', 'mlx2'): raise ImportError(n)\n"
        "sys.meta_path.insert(0, B())\n"
        f"sys.argv = ['probe', '--help']\n"
        "try:\n"
        f"    runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    print('EXIT', e.code)\n"
        "print('LOADED', sorted(m for m in sys.modules if m.split('.')[0] in ('mlx','mlx_lm','mlx2')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                         timeout=60, cwd=ROOT).stdout
    assert "--i-own-gpu" in out and "--dry-run" in out and "EXIT 0" in out
    assert "LOADED []" in out
    assert "--lanes" not in out and "--num-draft" not in out


@pytest.mark.parametrize("argv", [
    [],
    ["--dry-run", "--i-own-gpu"],
    ["--dry-run", "--lanes", "2"],
    ["--dry-run", "--num-draft", "2"],
    ["--dry-run", "--max-tokens", "7"],
    ["--dry-run", "--max-tokens", "129"],
    ["--dry-run", "--prefill-step", "63"],
    ["--dry-run", "--prefill-step", "1025"],
    ["--dry-run", "--seed", "-1"],
    ["--dry-run", "--prompt", " "],
    ["--dry-run", "--prompt-format", "html"],
])
def test_cli_refuses_out_of_bounds(argv, tmp_path):
    with pytest.raises(SystemExit) as raised:
        P.resolve_args(P.build_parser(), [*argv, "--out", str(tmp_path / "r.json")])
    assert raised.value.code == 2


def test_cli_refuses_repository_and_existing_outputs(tmp_path):
    with pytest.raises(SystemExit):
        P.resolve_args(P.build_parser(), ["--dry-run", "--out", str(ROOT / "receipt.json")])
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit):
        P.resolve_args(P.build_parser(), ["--dry-run", "--out", str(existing)])


def make_artifact(root, *, mtp_layers=1, mtp_keys=True, shard_name="model-00001.safetensors"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5_moe", "text_config": {"mtp_num_hidden_layers": mtp_layers}}))
    weight_map = {"language_model.model.embed_tokens.weight": shard_name}
    if mtp_keys:
        weight_map["mtp.fc.weight"] = shard_name
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "tokenizer.json").write_text("{}")
    if not Path(shard_name).is_absolute() and ".." not in Path(shard_name).parts:
        (root / shard_name).write_bytes(b"weights-not-read")
    return root


def expected_fingerprint(root):
    """The adapters/qwen36_35b.py inspect_artifact recipe, restated."""
    digest = hashlib.sha256()
    for name in P.METADATA_FILES:
        if (root / name).is_file():
            digest.update(name.encode())
            digest.update((root / name).read_bytes())
    shards = sorted(set(json.loads((root / "model.safetensors.index.json").read_text())
                        ["weight_map"].values()))
    for name in shards:
        st = (root / name).stat()
        digest.update(json.dumps((name, st.st_size, st.st_mtime_ns)).encode())
    return digest.hexdigest()


def test_dry_run_writes_unexecuted_plan_without_mlx_or_native_body(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path / "artifact")
    monkeypatch.setattr(P, "run_native", lambda *a, **k: pytest.fail("native body ran"))
    out = tmp_path / "plan" / "plan.json"
    assert P.main(["--dry-run", "--model", str(artifact), "--out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert plan["executed"] is False and plan["plan_only"] is True
    assert plan["verdict"] == "not_run"
    assert plan["qualification"].startswith("none")
    assert plan["refusals_if_run"] == []
    assert plan["protocol"]["lanes"] == 1 and plan["protocol"]["num_draft"] == 1
    manifest = plan["identity"]["artifact_manifest"]
    assert manifest["fingerprint"] == expected_fingerprint(artifact)
    assert manifest["mtp_indexed_tensors"] == 1
    oracle_bytes = (ROOT / "scripts" / "paired_direct_ab.py").read_bytes()
    assert plan["identity"]["state_oracle"]["sha256"] == hashlib.sha256(oracle_bytes).hexdigest()
    with pytest.raises(SystemExit):  # never overwrites evidence
        P.main(["--dry-run", "--model", str(artifact), "--out", str(out)])


def clean_source():
    """A substitute source identity that passes the strict preflight."""
    return {"commit": "a" * 40, "status_known": True, "dirty": False, "status_porcelain": "",
            "files": {name: _h(name) for name in P.IDENTITY_FILES}}


def test_native_flag_reaches_mlx_only_in_native_body_and_writes_nothing(tmp_path, monkeypatch):
    # Only a clean (substituted) source identity passes preflight; the native
    # body's first MLX import is then blocked, so nothing is written.
    monkeypatch.setattr(P, "source_identity", clean_source)
    artifact = make_artifact(tmp_path / "artifact")
    out = tmp_path / "native.json"
    saved = list(sys.path)
    try:
        with pytest.raises(ImportError, match="blocked in host-only test: mlx"):
            P.main(["--i-own-gpu", "--model", str(artifact), "--out", str(out)])
    finally:
        sys.path[:] = saved
    assert not out.exists()


@pytest.mark.parametrize("case", ["dirty", "git_failure", "artifact"])
def test_preflight_refusal_writes_unexecuted_receipt_without_native_import(case, tmp_path,
                                                                          monkeypatch):
    artifact = make_artifact(tmp_path / "artifact")
    monkeypatch.setattr(P, "run_native", lambda *a, **k: pytest.fail("native body reached"))
    if case == "dirty":
        monkeypatch.setattr(P, "source_identity", lambda: {**clean_source(), "dirty": True})
        expected = "not clean at HEAD"
    elif case == "git_failure":
        def failing(*args, **kwargs):
            raise OSError("git unavailable")
        monkeypatch.setattr(P.subprocess, "run", failing)
        expected = "not a 40-hex revision"
    else:
        monkeypatch.setattr(P, "source_identity", clean_source)
        artifact = tmp_path / "missing-artifact"
        expected = "artifact manifest unreadable"
    out = tmp_path / "refused.json"
    assert P.main(["--i-own-gpu", "--model", str(artifact), "--out", str(out)]) == 2
    receipt = json.loads(out.read_text())
    assert receipt["executed"] is False and receipt["verdict"] == "refused"
    assert receipt["refused_at"].startswith("source/artifact preflight")
    assert receipt["runs"] == [] and receipt["qualification"].startswith("none")
    assert refused_with(receipt["refusals"], expected), receipt["refusals"]
    assert "source_before" in receipt["identity"]
    if case == "git_failure":
        assert refused_with(receipt["refusals"], "status of the identity files is unknown")
        assert receipt["identity"]["source_before"]["dirty"] is None


def test_real_worktree_source_identity_is_complete_and_strictly_judged():
    source = P.source_identity()
    assert set(source["files"]) == set(P.IDENTITY_FILES)
    assert all(P._is_sha256(v) for v in source["files"].values())
    assert P._is_commit(source["commit"]) and source["status_known"] is True
    assert source["dirty"] is bool(source["status_porcelain"])
    expected = ["identity files are not clean at HEAD (modified or untracked)"] if source["dirty"] else []
    assert P.source_identity_refusals(source) == expected


def test_source_identity_refusals():
    good = clean_source()
    assert P.source_identity_refusals(good) == []
    files = dict(good["files"])
    for patch, text in (
            ({"commit": None}, "not a 40-hex"), ({"commit": "a" * 39}, "not a 40-hex"),
            ({"commit": "A" * 40}, "not a 40-hex"),
            ({"status_known": False, "dirty": None}, "unknown"),
            ({"status_known": False}, "unknown"),
            ({"dirty": None}, "unknown"), ({"dirty": True}, "not clean"),
            ({"files": {k: v for k, v in files.items() if k != P.IDENTITY_FILES[0]}},
             "not exactly the required"),
            ({"files": {**files, "extra.py": _h("x")}}, "not exactly the required"),
            ({"files": {**files, P.IDENTITY_FILES[2]: None}}, "no sha256 for identity files")):
        assert refused_with(P.source_identity_refusals({**good, **patch}), text), patch
    assert P.source_identity_refusals(None) == ["source identity missing"]


def test_source_identity_records_git_failure_as_unknown(monkeypatch):
    def failing(*args, **kwargs):
        raise P.subprocess.CalledProcessError(128, args[0])
    monkeypatch.setattr(P.subprocess, "run", failing)
    source = P.source_identity()
    assert source["commit"] is None and source["status_known"] is False
    assert source["dirty"] is None and source["status_porcelain"] is None
    assert len(P.source_identity_refusals(source)) == 2


def test_source_identity_changes_during_the_run():
    before = clean_source()
    assert P.source_identity_changes(before, clean_source()) == []
    files = dict(before["files"])
    for after, text in (({**before, "commit": "b" * 40}, "commit changed"),
                        ({**before, "dirty": True}, "dirty changed"),
                        ({**before, "status_known": False}, "status_known changed"),
                        ({**before, "files": {**files, P.IDENTITY_FILES[3]: _h("edited")}},
                         f"{P.IDENTITY_FILES[3]} changed"),
                        ({**before, "files": {k: v for k, v in files.items()
                                              if k != P.IDENTITY_FILES[1]}},
                         f"{P.IDENTITY_FILES[1]} changed")):
        assert refused_with(P.source_identity_changes(before, after), text), text
    assert P.source_identity_changes(before, None)


def test_mlx_build_identity_refusals(tmp_path):
    good = {"version": "0.30.1.dev", "package": "0.30.1.dev", "path": str(tmp_path),
            "metallib_sha256": _h("metallib"), "device": "Device(gpu, 0)"}
    assert P.mlx_identity_refusals(good) == []
    for patch, text in (({"version": ""}, "no version"), ({"package": None}, "no package"),
                        ({"device": " "}, "no device"), ({"path": "relative/mlx"}, "core path"),
                        ({"path": str(tmp_path / "absent")}, "core path"),
                        ({"path": None}, "core path"), ({"metallib_sha256": None}, "metallib"),
                        ({"metallib_sha256": "ABC"}, "metallib")):
        assert refused_with(P.mlx_identity_refusals({**good, **patch}), text), patch
    assert P.mlx_identity_refusals(None) == ["MLX build identity missing"]


def test_module_path_refusals(tmp_path):
    inside = str(ROOT / "src" / "mlx2" / "runtime" / "generate.py")
    outside = tmp_path / "generate.py"
    outside.write_text("")
    assert P.module_path_refusals({"mlx2.runtime.generate": inside}) == []
    for file in (str(outside), None, "src/mlx2/runtime/generate.py",
                 str(ROOT / "src" / "mlx2" / "absent_module.py")):
        assert refused_with(P.module_path_refusals({"m": file}), "not a file under"), file


def _calls_in(node, name):
    return [n.lineno for n in ast.walk(node) if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == name]


def _function(name):
    tree = ast.parse(SCRIPT.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _guard_return_lines(func, variable):
    """Lines of ``if <variable>: return refused_receipt(...)`` guards."""
    return [n.body[0].lineno for n in ast.walk(func) if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name) and n.test.id == variable
            and isinstance(n.body[0], ast.Return) and _calls_in(n.body[0], "refused_receipt")]


def test_native_body_refuses_before_mlx2_import_and_model_construction():
    """Source-order proof over run_native (stdlib ast; nothing is imported or run)."""
    func = _function("run_native")
    imports = [(n.lineno, n.names[0].name if isinstance(n, ast.Import) else n.module)
               for n in ast.walk(func) if isinstance(n, (ast.Import, ast.ImportFrom))]
    mlx_core = min(line for line, name in imports if name == "mlx.core")
    mlx2_first = min(line for line, name in imports if name and name.split(".")[0] == "mlx2")
    (build_check,) = _calls_in(func, "mlx_identity_refusals")
    (build_guard,) = _guard_return_lines(func, "build_refusals")
    (path_check,) = _calls_in(func, "module_path_refusals")
    (path_guard,) = _guard_return_lines(func, "path_refusals")
    (adapter,) = [n.lineno for n in ast.walk(func) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "cls"]
    (resolve,) = _calls_in(func, "resolve_adapter")
    generators = _calls_in(func, "BatchGenerator")
    assert mlx_core < build_check < build_guard < mlx2_first
    assert mlx2_first < resolve < path_check < path_guard < adapter < min(generators)
    # After the bounded runs: source identity again, then the change check, then the verdict.
    after = max(_calls_in(func, "source_identity"))
    (change,) = _calls_in(func, "source_identity_changes")
    (verdict,) = _calls_in(func, "decide")
    assert max(_calls_in(func, "drive_run")) < after < change < verdict


def test_main_refuses_before_calling_the_native_body():
    func = _function("main")
    (pre,) = _calls_in(func, "preflight")
    (native,) = _calls_in(func, "run_native")
    guard = next(n for n in ast.walk(func) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == "refusals")
    assert pre < guard.lineno < native
    assert _calls_in(guard.body[0], "refused_receipt")
    assert native in _calls_in(guard.orelse[0], "run_native")


def test_manifest_reads_no_weight_bytes_and_binds_metadata(tmp_path):
    artifact = make_artifact(tmp_path / "artifact")
    shard = artifact / "model-00001.safetensors"
    first = P.artifact_manifest(artifact)
    shard.chmod(0)
    try:
        assert P.artifact_manifest(artifact)["fingerprint"] == first["fingerprint"]
    finally:
        shard.chmod(stat.S_IRUSR | stat.S_IWUSR)
    os.utime(shard, ns=(1, shard.stat().st_mtime_ns + 1000))
    touched = P.artifact_manifest(artifact)["fingerprint"]
    assert touched != first["fingerprint"]
    (artifact / "tokenizer.json").write_text('{"changed": true}')
    assert P.artifact_manifest(artifact)["fingerprint"] not in (first["fingerprint"], touched)
    assert P.manifest_refusals(P.artifact_manifest(artifact)) == []


def test_manifest_refuses_headless_or_escaping_artifacts(tmp_path):
    no_layers = P.artifact_manifest(make_artifact(tmp_path / "a", mtp_layers=0))
    assert refused_with(P.manifest_refusals(no_layers), "no MTP head")
    no_keys = P.artifact_manifest(make_artifact(tmp_path / "b", mtp_keys=False))
    assert refused_with(P.manifest_refusals(no_keys), "no MTP tensors")
    with pytest.raises(ValueError, match="within the artifact"):
        P.artifact_manifest(make_artifact(tmp_path / "c", shard_name="../escape.safetensors"))
    good = P.artifact_manifest(make_artifact(tmp_path / "d"))
    assert refused_with(P.manifest_refusals({**good, "fingerprint": "ABC"}), "not a sha256")
    assert P.manifest_refusals(None) == ["artifact manifest unavailable"]


# ------------------------------------------------------------ trace hygiene

def test_trace_restores_on_success_error_and_base_exception():
    module = fake_module()
    originals = {name: getattr(module, name) for name in P.TRACED}
    with P.ProposalTrace(module, limit=4) as trace:
        assert all(getattr(module, n) is not originals[n] for n in P.TRACED)
    assert trace.restored and all(getattr(module, n) is originals[n] for n in P.TRACED)
    with pytest.raises(KeyError), P.ProposalTrace(module, limit=4) as trace:
        raise KeyError("body failed")
    assert trace.restored and all(getattr(module, n) is originals[n] for n in P.TRACED)
    gen = FakeGen("native", module, max_tokens=16)
    with pytest.raises(KeyboardInterrupt):
        gen.take_lane_failures = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        P.drive_run("native", gen, module, Ops(), prompt=PROMPT, max_tokens=16, rng=None,
                    trace_limit=40)
    assert gen.closed and all(getattr(module, n) is originals[n] for n in P.TRACED)


def test_trace_refuses_a_foreign_patch_and_still_restores():
    record, _, _ = one_run("native", knobs={"foreign_patch_at": 3})
    assert record["trace"]["foreign_patch"] is True
    assert record["trace"]["restored"] is True
    _, problems = P.evaluate_run(record, max_tokens=16)
    assert refused_with(problems, "foreign patch")


def test_trace_is_bounded():
    record, _, _ = one_run("native", limit=2)
    assert len(record["trace"]["cycles"]) == 2 and record["trace"]["overflow"]
    _, problems = P.evaluate_run(record, max_tokens=16)
    assert refused_with(problems, "exceeded its bound")


def test_malformed_proposal_metadata_is_recorded_and_refused():
    module = types.SimpleNamespace(
        propose_batched_self_mtp=lambda m, s: object(),
        advance_batched_self_mtp_zero=lambda m, s: None,
        commit_batched_self_mtp=lambda *a, **k: None)
    with P.ProposalTrace(module, limit=4) as trace:
        module.propose_batched_self_mtp(None, None)
    assert "malformed" in trace.cycles[0]
    _, problems = P.validate_native_trace(trace.summary(), max_tokens=8, tokens=[1],
                                          from_draft=[False], stats=None, finish_reason="stop")
    assert refused_with(problems, "malformed proposal metadata")


def test_run_errors_close_generator_restore_hooks_and_refuse():
    for knobs, text in (({"raise_at": 2}, "injected proposal failure"),
                        ({"insert_raises": True}, "insert refused")):
        module = fake_module()
        originals = {name: getattr(module, name) for name in P.TRACED}
        record, gen, _ = one_run("native", module=module, knobs=knobs)
        assert gen.closed and record["closed"] is True
        assert record["trace"]["restored"] is True
        assert all(getattr(module, n) is originals[n] for n in P.TRACED)
        _, problems = P.evaluate_run(record, max_tokens=16)
        assert refused_with(problems, text)
    record, gen, _ = one_run("native", knobs={"close_raises": True})
    assert gen.closed and record["closed"] is False
    _, problems = P.evaluate_run(record, max_tokens=16)
    assert refused_with(problems, "close")


def test_lane_failures_are_refused():
    record, _, _ = one_run("native", knobs={"lane_failures": ["lane 41 dropped"]})
    _, problems = P.evaluate_run(record, max_tokens=16)
    assert refused_with(problems, "lane failure: lane 41 dropped")


# ------------------------------------------------------------ accounting

def test_clean_native_run_reconciles_raw_and_committed():
    (accounting, problems), record = native_eval(max_tokens=16)
    assert problems == []
    raw, committed = accounting["raw_head_d1"], accounting["committed_draft"]
    assert raw["proposals"] > 0 and raw["proposals"] == committed["proposed"]
    assert raw["accepted"] == committed["accepted"] == sum(record["from_draft"])
    stats = record["mtp_receipt"]["stats"]
    assert committed["accepted"] == stats["draft_accepted"]
    assert committed["proposed"] == stats["draft_proposed"]
    assert accounting["anchor_tokens"] == 1
    assert len(record["tokens"]) == 16 and record["finish_reason"] == "length"
    assert accounting["depth0"]["zero_fast_path_cycles"] == 1
    assert accounting["per_cycle"][-1]["depth"] == 0
    assert "Neither is the MTPLX paper D1 metric" in accounting["scope"]


def test_stop_on_accepted_draft_drops_only_the_bonus():
    stop = TARGET[3]  # the accepted draft of the second all-agree cycle
    (accounting, problems), record = native_eval(max_tokens=16, stops=[stop],
                                                 agree=lambda i: True)
    assert problems == []
    assert record["tokens"] == TARGET[:4] and record["finish_reason"] == "stop"
    assert accounting["raw_head_d1"] == {"proposals": 2, "accepted": 2, "rate": 1.0}
    assert accounting["committed_draft"]["accepted"] == 2
    clip = accounting["clipping"]
    assert clip["bonus_tokens_dropped"] == 1 and clip["outputs_not_delivered"] == 1
    assert clip["accepted_drafts_clipped"] == 0
    assert record["mtp_receipt"]["stats"]["bonus_tokens"] == 1


def test_stop_on_bonus_drops_nothing():
    (accounting, problems), _ = native_eval(max_tokens=16, stops=[TARGET[4]],
                                            agree=lambda i: True)
    assert problems == []
    assert accounting["clipping"]["bonus_tokens_dropped"] == 0
    assert accounting["clipping"]["outputs_not_delivered"] == 0


def test_raw_and_committed_rates_stay_separate_when_terminal_clips():
    trace = {"installed": True, "restored": True, "foreign_patch": False, "overflow": False,
             "errors": [], "nested_proposals": 0, "nested_commits": 0, "limit": 8,
             "cycles": [_cycle(1, 1, [5, 6]), _cycle(1, 0, [7])],
             "commits": [_commit(0, 2, False), _commit(1, 1, True)]}
    stats = _stats(proposed=2, accepted=1, bonus=2, tokens=4, cycles=2)
    accounting, problems = P.validate_native_trace(
        trace, max_tokens=16, tokens=[4, 5, 6, 7], from_draft=[False, True, False, False],
        stats=stats, finish_reason="stop")
    assert problems == []
    assert accounting["raw_head_d1"]["rate"] == 0.5
    assert accounting["committed_draft"]["rate"] == 0.5
    # A terminal commit that delivers nothing is outside the generator's contract.
    trace["commits"][1] = _commit(1, 0, True)
    _, problems = P.validate_native_trace(
        trace, max_tokens=16, tokens=[4, 5, 6], from_draft=[False, True, False],
        stats=stats, finish_reason="stop")
    assert refused_with(problems, "emitted 0 outside")


def _cycle(depth, accepted, tokens, *, lanes=(7,), zero=False, **extra):
    cycle = {"kind": "zero" if zero else "propose", "lane_uids": list(lanes),
             "draft_depths": [depth], "accepted_lengths": [accepted],
             "copy_spans": [] if zero else [0], "copy_decisions": [] if zero else ["off"],
             "relaxed_accepts": [] if zero else [0], "draft_feature_rows": 0,
             "zero_fast_path": zero, "true_batched": False, "output_tokens": [tokens],
             "output_from_draft": [[True] * accepted + [False]]}
    cycle.update(extra)
    return cycle


def _commit(cycle, count, terminal):
    return {"cycle": cycle, "emitted_counts": [count], "terminal": [terminal], "completed": True}


def _stats(*, proposed, accepted, bonus, tokens, cycles):
    return {"cycles": cycles, "draft_cycles": cycles, "draft_proposed": proposed,
            "draft_accepted": accepted, "bonus_tokens": bonus, "plain_tokens": 1,
            "plain_cycles": 0, "retrieval_cycles": 0, "retrieval_proposed": 0,
            "retrieval_accepted": 0, "total_emitted": tokens}


def _trace(cycles, commits, **extra):
    trace = {"installed": True, "restored": True, "foreign_patch": False, "overflow": False,
             "errors": [], "nested_proposals": 0, "nested_commits": 0, "limit": 8,
             "cycles": cycles, "commits": commits}
    trace.update(extra)
    return trace


def _structural(cycles, commits, tokens, flags, stats, finish="length", max_tokens=4, **extra):
    return P.validate_native_trace(_trace(cycles, commits, **extra), max_tokens=max_tokens,
                                   tokens=tokens, from_draft=flags, stats=stats,
                                   finish_reason=finish)[1]


def test_structural_trace_refusals():
    # A clean max_tokens=4 run: D1 accepted (2 tokens), then the depth-0 budget cycle.
    cycles = [_cycle(1, 1, [5, 6]), _cycle(0, 0, [7], zero=True)]
    commits = [_commit(0, 2, False)]
    tokens, flags = [4, 5, 6, 7], [False, True, False, False]
    stats = _stats(proposed=1, accepted=1, bonus=2, tokens=4, cycles=2)
    assert _structural(cycles, commits, tokens, flags, stats) == []
    cases = (
        (cycles, commits + [_commit(1, 1, True)], "zero fast path was committed"),
        ([cycles[0], _cycle(1, 0, [7], zero=True)], commits, "zero fast path at depth 1"),
        (cycles, [_commit(0, 2, True)], "cycle after the terminal"),
        (cycles, [_commit(5, 2, False)], "does not belong"),
        (cycles, [{**_commit(0, 2, False), "completed": False}], "did not complete"),
        (cycles, [_commit(0, 2, False), _commit(0, 2, False)], "committed twice"),
        (cycles, [{**_commit(0, 2, False), "terminal": ["no"]}], "malformed commit"),
        ([_cycle(1, 1, [5, 6], draft_depths=[True]), cycles[1]], commits, "not ints"),
        ([_cycle(1, 1, [5])] + cycles[1:], commits, "outputs for 1 accepted"),
        ([_cycle(1, 1, [5, 6], output_from_draft=[[False, False]]), cycles[1]], commits,
         "from_draft/tokens malformed"),
    )
    for case_cycles, case_commits, text in cases:
        assert refused_with(_structural(case_cycles, case_commits, tokens, flags, stats),
                            text), text
    for errors, refused in (([{"call": "zero", "error": "ZeroDepthFastUnavailable"}], False),
                            ([{"call": "propose", "error": "RuntimeError"}], True),
                            ([{"call": "commit", "error": "ValueError"}], True)):
        problems = _structural(cycles, commits, tokens, flags, stats, errors=errors)
        assert refused_with(problems, "traced call raised") is refused
    for flag, text in (({"installed": False}, "not installed"),
                       ({"restored": False}, "not restored"),
                       ({"nested_proposals": 1}, "nested proposal")):
        assert refused_with(_structural(cycles, commits, tokens, flags, stats, **flag), text)
    assert refused_with(_structural(cycles, commits, tokens, flags, stats, finish=None),
                        "finish reason None")
    assert refused_with(_structural(cycles, commits, tokens, flags, stats, max_tokens=5),
                        "off the fixed num_draft=1 schedule")


def test_zero_fast_unavailable_falls_back_to_a_reported_depth0_proposal():
    (accounting, problems), record = native_eval(max_tokens=16, knobs={"zero_unavailable": True})
    assert problems == []
    assert record["trace"]["errors"] == [{"call": "zero", "error": "ZeroDepthFastUnavailable"}]
    assert accounting["depth0"] == {**accounting["depth0"], "zero_fast_path_cycles": 0,
                                    "depth0_proposal_cycles": 1}


def test_segmented_nested_commit_is_counted_not_recorded():
    (accounting, problems), record = native_eval(max_tokens=16, knobs={"nested": True})
    assert problems == []
    assert record["trace"]["nested_commits"] == len(record["trace"]["commits"]) > 0
    assert accounting["nested_commits"] == record["trace"]["nested_commits"]


def test_all_rejected_drafts_are_engagement_not_refusal():
    (accounting, problems), _ = native_eval(max_tokens=16, agree=lambda i: False)
    assert problems == []
    assert accounting["raw_head_d1"]["accepted"] == 0
    assert accounting["raw_head_d1"]["rate"] == 0.0


@pytest.mark.parametrize("kwargs, text", [
    ({"stops": [TARGET[0]]}, "zero native-head D1 engagement"),
    ({"knobs": {"depth": lambda i, d: 0}}, "off the fixed num_draft=1 schedule"),
    ({"knobs": {"depth": lambda i, d: 2 if d else 0}}, "off the fixed num_draft=1 schedule"),
    ({"knobs": {"copy_spans": {1: 3}}}, "copy span"),
    ({"knobs": {"copy_decisions": {1: "hit"}}}, "copy decision"),
    ({"knobs": {"relaxed": {1: 1}}}, "relaxed accepts"),
    ({"knobs": {"features": {1: ((0.5,),)}}}, "confidence probe"),
    ({"knobs": {"uids": (7, 8)}}, "lane width [7, 8] is not exactly 1"),
    ({"knobs": {"accepted": {1: 2}}}, "outside 0..1"),
    ({"knobs": {"skip_commit": True}}, "never committed"),
    ({"knobs": {"extra_plain_at": 1}}, "outside the traced head path"),
    ({"knobs": {"flip_from_draft_at": 2}}, "from_draft flags disagree"),
    ({"knobs": {"stats_patch": {"draft_accepted": 99}}}, "draft_accepted=99"),
    ({"knobs": {"stats_patch": {"retrieval_accepted": 1}}}, "retrieval_accepted=1"),
    ({"knobs": {"stats_patch": {"plain_tokens": 2}}}, "plain_tokens=2"),
    ({"knobs": {"stats_patch": {"bonus_tokens": 0}}}, "bonus_tokens=0"),
    ({"knobs": {"stats_patch": {"cycles": 1}}}, "cycles=1"),
    ({"knobs": {"stats_patch": {"draft_proposed": "7"}}}, "draft_proposed missing or not an int"),
    ({"knobs": {"receipt_patch": {"observed_compute_widths": [1, 2]}}}, "compute widths"),
    ({"knobs": {"receipt_patch": {"verification": "fly"}}}, "not exact"),
    ({"knobs": {"receipt_patch": {"copy_draft": {}}}}, "copy draft state"),
    ({"knobs": {"receipt_patch": {"adaptive_depth": {"selected": True}}}}, "adaptive depth"),
    ({"knobs": {"receipt_patch": {"num_draft": 2}}}, "num_draft 2"),
    ({"knobs": {"scheduler": {"self_mtp_copy_rounds": 1}}}, "self_mtp_copy_rounds=1"),
    ({"knobs": {"scheduler": {"mtp_target_only_plain_fallbacks": 1}}}, "plain_fallbacks=1"),
    ({"knobs": {"sidecar": metadata_unavailable("s")}}, "native sidecar: status"),
    ({"knobs": {"sidecar": None}}, "native sidecar"),
    ({"knobs": {"target_state": {"status": "complete", "sha256": "ABC", "reason": None}}},
     "final target state"),
    ({"lp": lambda pos: float("nan") if pos == 3 else -0.25}, "contain NaN"),
    ({"lp": lambda pos: 0.5 if pos == 3 else -0.25}, "positive"),
])
def test_native_refusals(kwargs, text):
    (_, problems), _ = native_eval(max_tokens=16, **kwargs)
    assert refused_with(problems, text), problems


def test_ordinary_must_be_genuinely_ordinary():
    record, _, _ = one_run("ordinary")
    assert P.evaluate_run(record, max_tokens=16) == (None, [])
    assert record["final_native_sidecar"] is None
    assert record["trace"]["cycles"] == [] and record["trace"]["restored"]
    record, _, _ = one_run("ordinary", knobs={"ordinary_calls_mtp": True})
    assert refused_with(P.evaluate_run(record, max_tokens=16)[1], "made self-MTP calls")
    record, _, _ = one_run("ordinary", knobs={"ordinary_sidecar": complete("x")})
    assert refused_with(P.evaluate_run(record, max_tokens=16)[1], "delivered a draft sidecar")
    record, _, _ = one_run("ordinary", knobs={"target_state": metadata_unavailable("t")})
    assert refused_with(P.evaluate_run(record, max_tokens=16)[1], "final target state")


# ------------------------------------------------------------ contracts

def test_generator_contract_refuses_every_route_switch():
    gen = FakeGen("native", fake_module(), max_tokens=16)
    assert P.generator_contract(gen, "native", CONFIG, environ={})[1] == []
    for attr, value, text in (
            ("adaptive_mtp_depth", {"max_depth": 1}, "adaptive MTP depth"),
            ("mtp_ordinary_handoff", object(), "mtp_ordinary_handoff"),
            ("mtp_admission", lambda rows: {}, "mtp_admission"),
            ("mtp_acceptance_logger", object(), "mtp_acceptance_logger"),
            ("fly_verification", _Policy(True), "FLy"),
            ("copy_draft", _Policy(True), "copy draft"),
            ("copy_draft", None, "copy draft"),
            ("self_mtp", {**CONFIG, "num_draft": 2}, "differs from the adapter")):
        gen = FakeGen("native", fake_module(), max_tokens=16)
        setattr(gen, attr, value)
        assert refused_with(P.generator_contract(gen, "native", CONFIG, environ={})[1], text)
    gen = FakeGen("native", fake_module(), max_tokens=16)
    assert refused_with(P.generator_contract(
        gen, "native", CONFIG, environ={"MLX2_SELF_MTP_HOST_ACCEPT": "1"})[1], "host-accept")
    ordinary = FakeGen("ordinary", max_tokens=16)
    assert P.generator_contract(ordinary, "ordinary", CONFIG, environ={})[1] == []
    ordinary.self_mtp = dict(CONFIG)
    assert refused_with(P.generator_contract(ordinary, "ordinary", CONFIG, environ={})[1],
                        "ordinary generator has a self-MTP config")


def test_post_run_policies_are_refused():
    module = fake_module()
    gen = FakeGen("native", module, max_tokens=16)
    gen._generation_batch.adaptive_depth_policy = object()
    record = P.drive_run("native", gen, module, Ops(), prompt=PROMPT, max_tokens=16,
                         rng=None, trace_limit=40)
    assert refused_with(P.evaluate_run(record, max_tokens=16)[1], "adaptive depth or ordinary")


@pytest.mark.parametrize("patch, text", [
    ({"num_draft": 2}, "num_draft 2"), ({"num_draft": True}, "num_draft True"),
    ({"rate_gate": True}, "rate gate"), ({"persistent": False}, "not persistent"),
    ({"window_size": 4}, "window_size"), ({"speculation_router": "x"}, "speculation_router"),
    ({"backend": "external_draft"}, "not the native head"),
    ({"segment_aware_cohort_size": 2}, "wider than one lane"),
])
def test_config_refusals(patch, text):
    assert P.config_refusals(CONFIG) == []
    assert refused_with(P.config_refusals({**CONFIG, **patch}), text)


@pytest.mark.parametrize("logprobs, text", [
    ({"chosen": [-0.1]}, "cover every"),
    ({"chosen": [-0.1, "nan"]}, "non-finite"),
    ({"chosen": [-0.1, "inf"]}, "non-finite"),
    ({"chosen": [-0.1, "-inf"]}, "non-finite"),
    ({"chosen": [-0.1, 1]}, "non-float"),
    ({"chosen": [-0.1, 0.01]}, "positive"),
    ({"row_sha256": [_h("a"), "ABC"]}, "row digests"),
    ({"row_sha256": [_h("a"), None]}, "row digests"),
    ({"row_sha256": [_h("a")]}, "row digests"),
    ({"row_refusals": ["row 1: status 'metadata_unavailable' is not complete"]},
     "raw logprob row 1"),
    ({"row_refusals": None}, "row refusals missing"),
    ({"nan_rows": 1}, "NaN"),
    ({"posinf_rows": 1}, "+Inf"),
    ({"posinf_rows": None}, "+Inf"),
    (None, "missing"),
])
def test_logprob_refusals(logprobs, text):
    good = {"chosen": [-0.1, -0.2], "row_sha256": [_h("a"), _h("b")], "row_refusals": [],
            "nan_rows": 0, "posinf_rows": 0}
    assert P.logprob_refusals(good, 2) == []
    assert refused_with(P.logprob_refusals(None if logprobs is None else {**good, **logprobs}, 2),
                        text)


# ------------------------------------------------------------ raw logprob rows (_NativeOps)
# HOST SUBSTITUTES for mx / np / the oracle: they prove what _NativeOps hands to
# the digest and in which order, not anything about Metal or real MLX arrays.

class SubArray:
    def __init__(self, values, *, dtype="bfloat16", shape=None, bits=None, log=None):
        self.values, self.dtype = list(values), dtype
        self.shape = tuple(shape or (len(self.values),))
        self.bits = bits if bits is not None else repr(self.values).encode()
        self.log = log if log is not None else []

    def astype(self, dtype):
        self.log.append(("astype", dtype))
        return SubArray(self.values, dtype=dtype, log=self.log)


class SubHost:
    def __init__(self, values):
        self.values, self.size = values, len(values)

    def reshape(self, *shape):
        return self

    def __getitem__(self, index):
        return self.values[index]


class SubAny:
    def __init__(self, value):
        self.value = value

    def any(self):
        return self.value


SUB_NP = types.SimpleNamespace(
    array=lambda a: SubHost(list(a.values)),
    isnan=lambda h: SubAny(any(math.isnan(v) for v in h.values)),
    isposinf=lambda h: SubAny(any(v == math.inf for v in h.values)))
SUB_MX = types.SimpleNamespace(float32="substitute.float32", synchronize=lambda: None,
                               get_peak_memory=lambda: 0, get_active_memory=lambda: 0,
                               get_cache_memory=lambda: 0)


class SubOracle:
    """Records what it digests; digests (dtype, shape, raw bits) like the real oracle binds."""

    def __init__(self, log, status="complete"):
        self.log, self.status = log, status

    def state_digest(self, obj):
        if isinstance(obj, SubArray):
            self.log.append(("digest", obj))
            if self.status != "complete":
                return metadata_unavailable("row")
            return complete(f"{obj.dtype}|{obj.shape}|{obj.bits.hex()}")
        return dict(obj) if isinstance(obj, dict) else ORACLE.state_digest(None)


def test_native_ops_digest_the_original_array_before_any_cast():
    log = []
    ops = P._NativeOps(SUB_MX, SUB_NP, SubOracle(log))
    array = SubArray([-1.0, -0.5, -float("inf")], log=log)
    row = ops.row(array, 1)
    assert log[0][0] == "digest" and log[0][1] is array
    assert log[1] == ("astype", "substitute.float32")
    assert row["chosen"] == -0.5 and row["nan"] is False and row["posinf"] is False
    assert row["dtype"] == "bfloat16" and row["shape"] == [3]
    assert ORACLE.snapshot_refusal(row["digest"]) is None


def test_native_ops_row_digest_keeps_dtype_shape_and_bits_apart():
    ops = P._NativeOps(SUB_MX, SUB_NP, SubOracle([]))
    values = [-1.0, -0.5]
    base = ops.row(SubArray(values), 0)["digest"]["sha256"]
    for other in (SubArray(values, dtype="float16"), SubArray(values, shape=(1, 2)),
                  SubArray(values, bits=b"\x00\x01")):
        assert ops.row(other, 0)["digest"]["sha256"] != base
        assert ops.row(other, 0)["chosen"] == -1.0  # same host value, different raw row


def _row_factory(unselected):
    def make(pos, token, value):
        values = [-7.0] * 512
        values[token] = value
        values[(token + 1) % 512] = unselected
        return SubArray(values)
    return make


@pytest.mark.parametrize("unselected, chosen, status, text", [
    (-float("inf"), -0.25, "complete", None),
    (float("nan"), -0.25, "complete", "contain NaN"),
    (float("inf"), -0.25, "complete", "contain +Inf"),
    (-7.0, -float("inf"), "complete", "non-finite"),
    (-7.0, -0.25, "metadata_unavailable", "raw logprob row 0: status"),
])
def test_native_ops_rows_end_to_end_on_substitutes(unselected, chosen, status, text):
    ops = P._NativeOps(SUB_MX, SUB_NP, SubOracle([], status=status))
    record, _, _ = one_run("native", ops=ops,
                           lp=lambda pos: chosen if pos == 2 else -0.25,
                           knobs={"logprob_factory": _row_factory(unselected)})
    problems = P.evaluate_run(record, max_tokens=16)[1]
    if text is None:
        assert problems == []
        assert all(P._is_sha256(sha) for sha in record["logprobs"]["row_sha256"])
        assert record["logprobs"]["row_layouts"] == [{"dtype": "bfloat16", "shape": [512]}]
    else:
        assert refused_with(problems, text), problems


def test_substitute_rows_refuse_incomplete_nan_and_posinf_in_drive_run():
    for ops, text in ((Ops(incomplete_at=(3,)), "raw logprob row 3"),
                      (Ops(nan_at=(4,)), "contain NaN"), (Ops(posinf_at=(5,)), "contain +Inf")):
        record, _, _ = one_run("native", ops=ops)
        if ops.incomplete_at:
            assert record["logprobs"]["row_sha256"][3] is None
        assert refused_with(P.evaluate_run(record, max_tokens=16)[1], text)


def test_receipt_protocol_names_the_raw_row_oracle(tmp_path):
    args = P.resolve_args(P.build_parser(), ["--dry-run", "--out", str(tmp_path / "plan.json")])
    oracle = P.protocol_section(args)["raw_row_oracle"]
    assert "before any cast" in oracle["function"] and "dtype, shape" in oracle["binds"]
    assert "-Inf" in oracle["allows"] and "+Inf" in oracle["refuses"]


# ------------------------------------------------------------ comparisons / verdict

def test_abba_parity_on_identical_substitutes():
    runs, decision = four(ordinary={"tag": ""}, native={"tag": ""})
    assert decision["refusals"] == [] and decision["differing_components"] == []
    assert decision["verdict"] == "parity"
    assert set(decision["comparisons"]) == {"ordinary_repeat", "native_repeat",
                                            "ordinary_vs_native_first", "ordinary_vs_native_second"}
    assert "native_sidecar" in decision["comparisons"]["native_repeat"]
    assert "native_sidecar" not in decision["comparisons"]["ordinary_vs_native_first"]
    assert decision["acceptance"]["native_repeat_trace_equal"] is True
    assert [r["arm"] for r in runs] == list(P.ARM_ORDER)
    assert runs[1]["final_native_sidecar"]["status"] == "complete"
    assert runs[0]["final_native_sidecar"] is None


def test_token_divergence_is_a_counterexample_with_its_index():
    diverged = TARGET[:6] + [999] + TARGET[7:]
    _, decision = four(ordinary={"tag": ""}, native={"tag": "", "target": diverged})
    assert decision["refusals"] == []
    assert decision["verdict"] == "counterexample"
    first = decision["comparisons"]["ordinary_vs_native_first"]
    assert first["tokens"] == "differs" and first["first_token_difference"] == 6
    assert decision["comparisons"]["native_repeat"]["tokens"] == "equal"


def test_logprob_only_divergence_is_a_counterexample_component():
    _, decision = four(ordinary={"tag": ""},
                       native={"tag": "", "lp": lambda pos: -0.2501 if pos == 5 else -0.25})
    assert decision["verdict"] == "counterexample"
    first = decision["comparisons"]["ordinary_vs_native_first"]
    assert first["tokens"] == "equal" and first["logprob_rows"] == "differs"
    assert first["first_logprob_row_difference"] == 5
    assert first["chosen_logprob_max_abs_diff_on_equal_tokens"] == pytest.approx(1e-4)
    assert "ordinary_vs_native_first.logprob_rows" in decision["differing_components"]


def test_target_state_and_sidecar_differences_are_counterexamples():
    _, decision = four(ordinary={"tag": "-o"}, native={"tag": "-n"})
    assert decision["verdict"] == "counterexample"
    assert decision["comparisons"]["ordinary_vs_native_first"]["target_state"] == "differs"
    assert decision["comparisons"]["ordinary_repeat"]["target_state"] == "equal"
    _, decision = four(ordinary={"tag": ""}, native={"tag": ""},
                       native2={"tag": "", "knobs": {"sidecar": complete("other")}})
    assert decision["differing_components"] == ["native_repeat.native_sidecar"]


def test_refusal_wins_over_parity_and_keeps_mismatches():
    _, decision = four(ordinary={"tag": ""},
                       native={"tag": "", "knobs": {"sidecar": metadata_unavailable("s")}})
    assert decision["verdict"] == "refused"
    assert decision["comparisons"]["native_repeat"]["native_sidecar"] == "unavailable"
    diverged = TARGET[:6] + [999] + TARGET[7:]
    _, decision = four(ordinary={"tag": ""},
                       native={"tag": "", "target": diverged, "knobs": {"copy_spans": {1: 2}}})
    assert decision["verdict"] == "refused"
    assert "ordinary_vs_native_first.tokens" in decision["differing_components"]


def test_incomplete_abba_is_refused_and_never_parity():
    runs, _ = four(ordinary={"tag": ""}, native={"tag": ""})
    decision = P.decide(runs[:3], max_tokens=16)
    assert decision["verdict"] == "refused" and decision["comparisons"] == {}
    decision = P.decide([], max_tokens=16, identity_refusals=["adapter fingerprint differs"])
    assert decision["verdict"] == "refused"
    assert decision["refusals"][0] == "adapter fingerprint differs"


def test_no_output_claims_qualification_or_selection():
    _, decision = four(ordinary={"tag": ""}, native={"tag": ""})
    text = json.dumps(decision)
    assert decision["verdict"] in ("parity", "counterexample", "refused")
    for word in ('"qualified"', '"selected"', '"observed-used"', '"pass"'):
        assert word not in text
    for item in ("MTPLX paper D1", "acceptance improvement", "qualification"):
        assert any(item in claim for claim in P.NOT_CLAIMED)


def test_receipt_json_is_strict(tmp_path):
    runs, decision = four(ordinary={"tag": ""},
                          native={"tag": "", "lp": lambda pos: float("nan") if pos == 2 else -0.25})
    out = tmp_path / "strict.json"
    P.write_receipt(out, P._jsonable({"decision": decision, "runs": runs}))
    loaded = json.loads(out.read_text())
    assert loaded["runs"][1]["logprobs"]["chosen"][2] == "nan"
    assert loaded["decision"]["verdict"] == "refused"


def test_long_lane_list_keeps_refusal_reason():
    (_, problems), _ = native_eval(max_tokens=16, knobs={"uids": tuple(range(100))})
    message = next(p for p in problems if "lane width" in p)
    assert "is not exactly 1" in message
    assert len(message) <= 200
