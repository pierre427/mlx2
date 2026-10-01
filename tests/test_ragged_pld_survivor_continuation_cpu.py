"""Host-only tests: removal-arm survivor continuations in ``run_all``.

``scripts/qualify_ragged_pld.py``'s ``run_all`` is EXECUTED over a host
substitute of ``Driver`` that keeps the real ``Driver.continuation`` and
``Driver.check_geometry`` and replaces only the native ``__init__`` and
``run``; ``preflight`` and the post-run identity are host substitutes too.
Lane records carry valid strict v2 digests, so the unchanged ``compare``
oracle decides every verdict. The removal arm must continue every lane that
actually survived, never resume the lane actually removed (its record keeps
the existing non-empty prefix-only scope), and refuse qualification when a
survivor's or the reference's continuation is missing, incomplete or
differs.

No mlx, mlx_lm or mlx2 module may load: imports of them are blocked at
``builtins.__import__`` and on ``sys.meta_path`` before the driver is
imported, and every test checks that none is present. Nothing here runs a
model, cache, generator, tensor or GPU. These are host fake checks: they
are not native batching, rollback or performance qualification, and the
``--i-own-the-gpu`` flag below is the CLI's caller assertion, not GPU
ownership enforcement.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -p no:cacheprovider \
      -o addopts= -q tests/test_ragged_pld_survivor_continuation_cpu.py
"""

import builtins
import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NATIVE = frozenset({"mlx", "mlx_lm", "mlx2"})
_REAL_IMPORT = builtins.__import__


def _native_loaded():
    return sorted(name for name in sys.modules if name.split(".")[0] in NATIVE)


class _NativeBlocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in NATIVE:
            raise ImportError(f"native import blocked in a host test: {name}")


def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] in NATIVE:
        raise ImportError(f"native import blocked at builtins in a host test: {name}")
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


assert not _native_loaded(), _native_loaded()
sys.meta_path.insert(0, _NativeBlocker())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
builtins.__import__ = _blocked_import
try:
    from scripts import paired_direct_ab as P
    from scripts import qualify_ragged_pld as Q
finally:
    builtins.__import__ = _REAL_IMPORT

UNAVAILABLE = P.state_digest(None)  # the real output; returns before any native import
_REAL_STATE_DIGEST = P.state_digest
REAL_DRIVER = Q.Driver
ROWS = 2
CONTINUATION = 2
REMOVE_AFTER = 1
WIDTHS = [pytest.param(2, 0, id="B2-remove0"), pytest.param(4, 2, id="B4-remove2")]
KINDS = ("tokens", "bits", "incomplete", "raises", "failures", "empty", "absent")


@pytest.fixture(autouse=True)
def host_only(monkeypatch):
    def guarded(obj):
        if obj is not None:
            raise AssertionError("state_digest over a real object must not run in host tests")
        return _REAL_STATE_DIGEST(None)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setattr(P, "state_digest", guarded)
    assert not _native_loaded()
    yield
    assert not _native_loaded(), _native_loaded()


def digest(char):
    return {"status": "complete", "sha256": char * 64, "reason": None}


def prompt(i):
    return [10 * (i + 1) + k for k in range(4 + i)]  # unequal prompt lengths


def cap(i):
    return 2 + i  # unequal output caps


def reference_tokens(i):
    return [200 + 10 * i + k for k in range(cap(i))]


ROUNDS_ZERO = {"rounds": 1, "proposed": 0, "accepted": 0, "rejected_rounds": 0,
               "partial_accept_rounds": 0, "rollback_then_append": False}
ROUNDS_ROLLBACK = {"rounds": 3, "proposed": 6, "accepted": 3, "rejected_rounds": 1,
                   "partial_accept_rounds": 1, "rollback_then_append": True}


class HostDriver(REAL_DRIVER):
    """The real Driver minus its native ``__init__`` and ``run``.

    ``continuation`` and ``check_geometry`` are the real methods. ``run``
    returns strict v2 lane records whose ``_final`` carries a host cache
    tagged with its arm and lane, so a resume of any cache is observable.
    """

    def __init__(self, args, scenario):
        width = args.lanes
        self.args, self.mx, self.stops = args, None, ()
        self.identity = {"model": "host substitute (no model)"}
        self.native_identity = {"mlx": {"host": "substituted"}, "modules_before_arms": None,
                                "adapter_identity": None}
        self.workload_receipt, self.lane_policies = None, None
        self.prompts = [prompt(i) for i in range(width)]
        self.caps = [cap(i) for i in range(width)]
        self.followup = [5, 6, 7]
        self.scenario = scenario
        self.events, self.issued, self.arm = [], [], None

    def assert_released(self, where, *, except_arm=None):
        """No lane response or cache of a finished arm (or continuation) is still held."""
        alive = [(arm, i) for arm, i, record in self.issued if arm != except_arm and "_final" in record]
        assert not alive, f"{where}: final responses still held for {alive}"

    def run(self, arm, lanes, *, caches=None, all_tokens=None, caps=None, remove=None, prompts=None):
        if caches is not None:
            return self._resume(arm, lanes, caches, all_tokens, caps, prompts)
        if arm != self.arm:
            self.assert_released(f"before arm {arm}")
            self.arm = arm
        self.events.append(("run", arm, tuple(lanes), remove))
        if self.scenario.arm_error == arm:
            raise self.scenario.arm_error_type(f"host arm failure in {arm}")
        width = len(lanes)
        removed_lane = self.scenario.removed if arm == "pld_removal" else None
        assert removed_lane is None or (remove is not None and removed_lane in lanes)
        out = {}
        for i in lanes:
            tokens = list(reference_tokens(i))
            finish, final = "length", SimpleNamespace(prompt_cache=SimpleNamespace(arm=arm, lane=i))
            if i == removed_lane:
                # Removed between polls: no finish response, so no final cache.
                tokens, finish, final = self.scenario.removed_prefix(tokens), None, None
            record = {
                "tokens": tokens, "finish_reason": finish,
                "execution_widths": ("not reported by the ordinary route" if arm.startswith("ordinary")
                                     else [width - 1, width] if arm == "pld_removal" else [width]),
                **Q.lane_row_evidence([digest(c) for c in "ab"[: min(ROWS, len(tokens))]], tokens, ROWS),
                "covered_tokens": None if final is None else len(prompt(i)) + len(tokens) - 1,
                "final_state": dict(UNAVAILABLE) if final is None else digest("e"),
                "boundary": None,
                "rounds": dict(ROUNDS_ZERO if i == 0 else ROUNDS_ROLLBACK),
                "_final": final,
            }
            self.issued.append((arm, i, record))
            out[i] = record
        stats = ({"pld_proposed": 6, "pld_rollbacks": 1, "pld_batched_max_width": width}
                 if arm.startswith("pld") else {})
        removed = None if removed_lane is None else {"lane": removed_lane, "after_tokens": len(out[removed_lane]["tokens"])}
        return out, stats, [], removed

    def _resume(self, arm, lanes, caches, all_tokens, caps, prompts):
        """A host continuation from one lane's final cache (no tensors)."""
        assert arm == "ordinary_b1" and len(lanes) == len(caches) == 1
        lane, cache = lanes[0], caches[0]
        assert cache.lane == lane and cache.arm == self.arm, (cache, lane, self.arm)
        assert not (cache.arm == "pld_removal" and lane == self.scenario.removed), "removed lane resumed"
        self.events.append(("resume", cache.arm, lane))
        source = [r for a, i, r in self.issued if a == cache.arm and i == lane][-1]
        full = prompt(lane) + source["tokens"]
        covered = source["covered_tokens"]
        assert all_tokens == [full[:covered]] and prompts == [full[covered:] + self.followup]
        assert caps == [self.args.continuation_tokens]
        kind = self.scenario.continuation.get((cache.arm, lane))
        if kind == "raises":
            raise RuntimeError("host continuation failure")
        tokens = [70 + lane + k for k in range(self.args.continuation_tokens)]
        final_state = digest("c")
        if kind == "tokens":
            tokens[0] += 1000
        elif kind == "bits":
            final_state = digest("d")
        elif kind == "incomplete":
            final_state = dict(UNAVAILABLE)
        elif kind == "empty":
            tokens = []
        # Not added to ``issued``: Driver.continuation keeps this dict local, so
        # it is released when that call returns, whichever path it takes.
        record = {"tokens": tokens, "final_state": final_state,
                  "_final": SimpleNamespace(prompt_cache=SimpleNamespace(arm="continuation", lane=lane))}
        return {lane: record}, {}, ["host lane failure"] if kind == "failures" else [], None

    def continuation(self, lane, record):
        self.events.append(("continuation", self.arm, lane))
        hook = self.scenario.continuation_hook
        if hook is not None:
            replaced = hook(self, lane, record)
            if replaced is not NotImplemented:
                return replaced
        return super().continuation(lane, record)


def native_args(width, remove_lane, *extra):
    """The default CLI protocol's real-run arguments (caller GPU assertion only; nothing is loaded)."""
    return Q.resolve_args(Q.build_parser(), [
        "--i-own-the-gpu", "--model", "/nonexistent/host-model", "--out", "/nonexistent/host-out.json",
        "--lanes", str(width), "--remove-lane", str(remove_lane), "--remove-after", str(REMOVE_AFTER),
        "--logprob-rows", str(ROWS), "--continuation-tokens", str(CONTINUATION), *extra])


@pytest.fixture
def host(monkeypatch):
    state = SimpleNamespace(driver=None, post_run=0)
    gate = {"enforced": True, "required_files": [], "source": {"files": {}}, "artifact": {"family": "host"},
            "workload": None, "workload_raw": None, "refusals": []}
    monkeypatch.setattr(Q, "preflight", lambda args: copy.deepcopy(gate))

    def post_run(args, gate_, driver):
        driver.assert_released("post-run identity")
        state.post_run += 1
        return {"host": "post-run identity substituted"}, []

    monkeypatch.setattr(Q, "checked_post_run_identity", post_run)

    def go(width, removed, *, remove_lane=None, extra=(), continuation=None, removed_prefix=None,
           continuation_hook=None, arm_error=None, arm_error_type=RuntimeError):
        scenario = SimpleNamespace(
            removed=removed, continuation=dict(continuation or {}), continuation_hook=continuation_hook,
            removed_prefix=removed_prefix or (lambda tokens: tokens[:REMOVE_AFTER]),
            arm_error=arm_error, arm_error_type=arm_error_type)

        def make(args, gate_):
            assert gate_["refusals"] == [] and gate_["enforced"]
            state.driver = HostDriver(args, scenario)
            return state.driver

        monkeypatch.setattr(Q, "Driver", make)
        args = native_args(width, removed if remove_lane is None else remove_lane, *extra)
        return Q.run_all(args), state.driver

    go.state = state
    return go


def removal(record):
    [found] = [c for c in record["comparisons"] if c["arm"] == "pld_removal"]
    assert found["reference"] == Q.PRIMARY_REFERENCE
    return found


def others(record):
    return [c for c in record["comparisons"] if c["arm"] != "pld_removal"]


def survivors(width, removed):
    return [i for i in range(width) if i != removed]


def calls(driver, kind, arm):
    return [event[2] for event in driver.events if event[0] == kind and event[1] == arm]


# ---- blocking ----

def test_native_imports_are_blocked_at_builtins_and_meta_path():
    assert any(isinstance(f, _NativeBlocker) for f in sys.meta_path)
    assert builtins.__import__ is _blocked_import
    for name in ("mlx.core", "mlx_lm", "mlx2.runtime.pld"):
        with pytest.raises(ImportError, match="blocked at builtins"):
            __import__(name)
        with pytest.raises(ImportError, match="blocked in a host test"):
            importlib.import_module(name)
    assert _native_loaded() == []


# ---- clean scenarios: every survivor continued, the removed lane never ----

@pytest.mark.parametrize("width,removed", WIDTHS)
def test_survivors_are_continued_and_the_removed_lane_never(host, width, removed):
    record, driver = host(width, removed)
    assert record["verdict"] == "pass", (record["refusals"], record["coverage"], record["comparisons"])
    assert record["removed"] == {"lane": removed, "after_tokens": REMOVE_AFTER}
    assert calls(driver, "continuation", "pld_removal") == survivors(width, removed)
    assert calls(driver, "resume", "pld_removal") == survivors(width, removed)
    for arm in Q.ARMS[:-1]:
        assert calls(driver, "continuation", arm) == list(range(width))
        assert calls(driver, "resume", arm) == list(range(width))
    lanes = record["results"]["pld_removal"]
    assert "continuation" not in lanes[str(removed)]
    for i in survivors(width, removed):
        # Exactly what the real Driver.continuation builds from the host resume.
        assert lanes[str(i)]["continuation"] == {"status": "complete", "tokens": [70 + i, 71 + i],
                                                 "final_state": digest("c")}
    cmp = removal(record)
    assert cmp["tokens_exact"] and cmp["bits_exact"] and not cmp["incomparable"], cmp
    assert all(c["bits_exact"] for c in record["comparisons"]), record["comparisons"]


@pytest.mark.parametrize("width,removed", WIDTHS)
def test_arm_order_continuations_follow_each_arm_and_finals_are_released(host, width, removed):
    record, driver = host(width, removed)
    expected = [("run", "ordinary_b1", (i,), None) for i in range(width)]
    for arm in Q.ARMS:
        if arm != "ordinary_b1":
            expected.append(("run", arm, tuple(range(width)),
                             (removed, REMOVE_AFTER) if arm == "pld_removal" else None))
        for i in range(width):
            if arm == "pld_removal" and i == removed:
                continue
            expected += [("continuation", arm, i), ("resume", arm, i)]
    assert driver.events == expected
    # HostDriver.run asserted at each new arm that no earlier arm's final was
    # held; post-run identity asserted the same for every arm and continuation.
    assert host.state.post_run == 1
    assert all("_final" not in r for _a, _i, r in driver.issued)
    assert not any(k.startswith("_") for res in record["results"].values() for r in res.values() for k in r)


# ---- the actual removal is the only source of the removed lane ----

@pytest.mark.parametrize("width", [2, 4])
def test_missing_actual_removal_discards_no_lane_and_keeps_coverage_refusal(host, width):
    record, driver = host(width, None, remove_lane=1)
    assert record["removed"] is None
    assert record["coverage"]["closed_boundary_removal"] is False
    assert record["verdict"] == "coverage_refused"
    # args.remove_lane is not substituted: lane 1 survived and is continued and compared in full.
    assert calls(driver, "continuation", "pld_removal") == list(range(width))
    assert calls(driver, "resume", "pld_removal") == list(range(width))
    assert record["results"]["pld_removal"]["1"]["continuation"]["status"] == "complete"
    assert removal(record)["bits_exact"]


def test_missing_actual_removal_still_compares_the_requested_lane_strictly(host):
    record, _ = host(2, None, remove_lane=0, continuation={("pld_removal", 0): "tokens"})
    cmp = removal(record)
    assert cmp["token_differences"] == ["lane 0: continuation tokens differ"], cmp
    assert record["verdict"] == "counterexample"


def test_removed_lane_comes_from_the_actual_removal_not_the_argument(host):
    # Host-only inconsistency (the real run removes the requested lane): the
    # orchestration follows what run reported, never --remove-lane.
    record, driver = host(4, 3, remove_lane=1)
    assert record["removed"]["lane"] == 3
    assert calls(driver, "continuation", "pld_removal") == [0, 1, 2]
    assert "continuation" not in record["results"]["pld_removal"]["3"]


# ---- removed lane keeps the existing non-empty prefix-only scope ----

@pytest.mark.parametrize("width,removed", WIDTHS)
def test_removed_lane_needs_only_its_prefix(host, width, removed):
    record, _ = host(width, removed)
    lane = record["results"]["pld_removal"][str(removed)]
    assert lane["final_state"] == UNAVAILABLE and lane["covered_tokens"] is None
    assert lane["finish_reason"] is None and "continuation" not in lane
    assert removal(record)["bits_exact"]


@pytest.mark.parametrize("width,removed", WIDTHS)
def test_removed_lane_prefix_difference_is_a_counterexample(host, width, removed):
    record, driver = host(width, removed, removed_prefix=lambda t: [t[0] + 1])
    assert removal(record)["token_differences"] == [f"lane {removed}: removed-lane prefix differs"]
    assert record["verdict"] == "counterexample"
    assert calls(driver, "continuation", "pld_removal") == survivors(width, removed)


@pytest.mark.parametrize("width,removed", WIDTHS)
def test_empty_removed_prefix_is_incomparable(host, width, removed):
    record, _ = host(width, removed, removed_prefix=lambda t: [])
    assert removal(record)["incomparable"] == [f"lane {removed}: removed-lane prefix is empty"]
    assert record["verdict"] == "token_exact_bits_diverge"


# ---- survivor and reference continuations refuse qualification ----

def _absent(target):
    def hook(driver, lane, record):
        final = record.get("_final")
        if final is not None and (final.prompt_cache.arm, lane) == target:
            record.pop("_final")
            return None  # an absent continuation record
        return NotImplemented
    return hook


def _expected(kind, lane, side):
    if kind == "tokens":
        return "token_differences", f"lane {lane}: continuation tokens differ", "counterexample"
    if kind == "bits":
        return "bit_differences", f"lane {lane}: continuation state bits differ", "token_exact_bits_diverge"
    reason = {"incomplete": "continuation final state: status 'unavailable'",
              "raises": "RuntimeError: host continuation failure", "failures": "host lane failure",
              "empty": "continuation final state: no continuation tokens", "absent": "absent"}[kind]
    return "incomparable", f"lane {lane}: continuation unavailable ({side}: ", reason


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("source", ["pld_removal", "ordinary_b1"])
@pytest.mark.parametrize("width,removed", WIDTHS)
def test_survivor_or_reference_continuation_defect_refuses(host, width, removed, source, kind):
    lane = survivors(width, removed)[-1]
    target = (source, lane)
    if kind == "absent":
        record, driver = host(width, removed, continuation_hook=_absent(target))
    else:
        record, driver = host(width, removed, continuation={target: kind})
    assert record["verdict"] != "pass"
    field, needle, extra = _expected(kind, lane, source)
    cmp = removal(record)
    assert not cmp["bits_exact"]
    hits = [item for item in cmp[field] if item.startswith(needle)]
    assert len(hits) == 1, cmp
    others_found = [item for f in ("token_differences", "bit_differences", "incomparable")
                    for item in cmp[f] if item not in hits]
    assert others_found == [], cmp
    if field == "incomparable":
        assert extra in hits[0], hits
        assert record["verdict"] == "token_exact_bits_diverge"
    else:
        assert record["verdict"] == extra
    if source == "pld_removal":
        # Only the removal comparison could see this survivor's continuation.
        assert all(c["bits_exact"] for c in others(record)), others(record)
    else:
        assert all(not c["bits_exact"] for c in others(record) if c["arm"] != "ordinary_b1"
                   and c["reference"] == Q.PRIMARY_REFERENCE)
    stored = record["results"][source][str(lane)].get("continuation")
    if kind == "absent":
        assert stored is None
    elif kind not in ("tokens", "bits"):
        assert stored["status"] == "unavailable"  # never manufactured as complete
    assert calls(driver, "continuation", "pld_removal") == survivors(width, removed)


@pytest.mark.parametrize("width,removed", WIDTHS)
def test_disabled_continuation_is_unavailable_and_refuses_exactly(host, width, removed):
    record, driver = host(width, removed, extra=("--continuation-tokens", "0"))
    assert record["protocol"]["continuation_tokens"] == 0
    assert [e for e in driver.events if e[0] == "resume"] == []
    assert calls(driver, "continuation", "pld_removal") == survivors(width, removed)
    lanes = record["results"]["pld_removal"]
    assert "continuation" not in lanes[str(removed)]
    for i in survivors(width, removed):
        assert lanes[str(i)]["continuation"] == {"status": "unavailable",
                                                 "reason": "disabled (--continuation-tokens 0)"}
    cmp = removal(record)
    assert cmp["tokens_exact"] and not cmp["bits_exact"]
    assert [item.split(" (")[0] for item in cmp["incomparable"]] == [
        f"lane {i}: continuation unavailable" for i in survivors(width, removed)]
    assert record["verdict"] == "token_exact_bits_diverge"
    assert all("_final" not in r for _a, _i, r in driver.issued)


# ---- unrelated errors are not masked ----

@pytest.mark.parametrize("error", [KeyError, TypeError])
def test_an_unrelated_continuation_error_propagates(host, error):
    def hook(driver, lane, record):
        if driver.arm == "pld_removal":
            raise error("unrelated host defect")
        return NotImplemented

    with pytest.raises(error, match="unrelated host defect"):
        host(2, 0, continuation_hook=hook)
    assert host.state.post_run == 0  # no receipt was assembled


def test_a_removal_arm_error_propagates(host):
    with pytest.raises(ValueError, match="host arm failure in pld_removal"):
        host(4, 2, arm_error="pld_removal", arm_error_type=ValueError)
    assert host.state.post_run == 0


# ---- unchanged surfaces ----

def test_default_cli_protocol_is_unchanged():
    args = Q.build_parser().parse_args(["--out", "/nonexistent/host-out.json"])
    assert (args.logprob_rows, args.continuation_tokens, args.remove_lane, args.remove_after) == (16, 4, 0, 4)
    assert Q.ARMS == ("ordinary_b1", "ordinary_bN", "pld_per_lane", "pld_batched", "pld_removal")
    assert Q.PRIMARY_REFERENCE == "ordinary_b1"
    assert Q.SCHEMA == "mlx2.direct-model.ragged-pld.v2"


def test_host_driver_replaces_only_the_native_surface():
    # check_geometry, generator and the continuation body stay the real ones;
    # HostDriver.continuation only logs, then delegates unless a test hook answers.
    overridden = {k for k in HostDriver.__dict__
                  if k in REAL_DRIVER.__dict__ and (k == "__init__" or not k.startswith("__"))}
    assert overridden == {"__init__", "run", "continuation"}
    assert HostDriver.check_geometry is REAL_DRIVER.check_geometry
