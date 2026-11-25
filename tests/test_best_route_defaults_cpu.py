"""Per-artifact best-route defaults: rule, resolver, auto-pairing, receipts."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2 import best_route_defaults as brd
from mlx2.adapters import drafter_pairing
from mlx2.adapters.qwen38_27b import QWEN38_27B, QWEN38_27B_ORDINARY, Qwen3827BAdapter
from mlx2.adapters.registry import AdapterResolution
from mlx2.server import (
    RouteSelection,
    apply_ast_invocation,
    apply_best_route_default,
    resolve_route_selection,
    serving_engine_kwargs,
)

ROOT = Path(__file__).resolve().parents[1]
FP = "ab" * 32
GOOD = "cd" * 32
OTHER = "ef" * 32
EVIDENCE = {"files": ["qualification/runs/best-route-defaults-20261021/summary.json"],
            "source_revision": "7ad53cc86d49f5ae509dbb919ded235b6db6fa55"}


# ------------------------------------------------------------------ rule --

def _cells(**ratios):
    base = {cell: 100.0 for cell in brd.rule_cells()}
    return base, {cell: 100.0 * ratios.get(cell.replace("/", "_"), 1.0) for cell in brd.rule_cells()}


def test_rule_geomean_and_floor_on_synthetic_cells():
    base, fast = _cells(code_B1=4.0, chat_B1=2.0, long_B1=2.0, copy_B1=4.0)
    score = brd.route_score(base, fast)
    assert score["qualifies"]
    assert math.isclose(score["geomean"], (4 * 2 * 2 * 4) ** (1 / 8), rel_tol=1e-4)
    assert score["min_cell"] == 1.0

    # One cell below 0.85x plain decode disqualifies despite a big mean.
    _, floor = _cells(code_B1=4.0, copy_B1=4.0, chat_B4=0.84)
    assert not brd.route_score(base, floor)["qualifies"]
    _, at_floor = _cells(code_B1=4.0, copy_B1=4.0, chat_B4=0.85)
    assert brd.route_score(base, at_floor)["qualifies"]

    # Geometric mean below 1.05 disqualifies even with every cell >= 0.85.
    _, flat = _cells(code_B1=1.2, chat_B1=0.9, chat_B4=0.9)
    flat_score = brd.route_score(base, flat)
    assert flat_score["min_cell"] >= 0.85 and flat_score["geomean"] < 1.05
    assert not flat_score["qualifies"]


def test_rule_picks_highest_qualifying_geomean_else_keeps_current_default():
    base, big = _cells(code_B1=3.0, copy_B1=3.0)
    _, small = _cells(code_B1=1.5)
    verdict = brd.rule_verdict(base, {"ast": big, "default": small}, "default")
    assert verdict["winner"] == "ast" and verdict["decision"] == "selected"

    _, losing = _cells(code_B1=3.0, chat_B4=0.5)
    verdict = brd.rule_verdict(base, {"ast": losing}, "default")
    assert verdict["winner"] == "default" and verdict["decision"] == "kept_current_default"

    missing = dict(big)
    del missing["long/B4"]
    verdict = brd.rule_verdict(base, {"ast": missing}, "default")
    assert verdict["winner"] == "default"
    assert verdict["scores"]["ast"]["missing"] == ["long/B4"]


@pytest.mark.parametrize("value", [-100.0, 0, True, float("inf"), float("nan"), "100"])
@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_rule_refuses_invalid_throughput_measurements(value, arm):
    baseline = dict.fromkeys(brd.rule_cells(), 100.0)
    candidate = dict.fromkeys(brd.rule_cells(), 200.0)
    (baseline if arm == "baseline" else candidate)["code/B1"] = value
    with pytest.raises(ValueError, match="throughput must be finite and positive"):
        brd.route_score(baseline, candidate)


def test_rule_cannot_qualify_two_negative_measurements():
    baseline = dict.fromkeys(brd.rule_cells(), -100.0)
    candidate = dict.fromkeys(brd.rule_cells(), -200.0)
    with pytest.raises(ValueError, match="throughput must be finite and positive"):
        brd.route_score(baseline, candidate)


def test_rule_chooses_fastest_before_rounding_receipt_scores():
    baseline = dict.fromkeys(brd.rule_cells(), 100.0)
    candidates = {
        "slower": dict.fromkeys(brd.rule_cells(), 110.001),
        "faster": dict.fromkeys(brd.rule_cells(), 110.002),
    }
    verdict = brd.rule_verdict(baseline, candidates, "ordinary")
    assert verdict["scores"]["slower"]["geomean"] == verdict["scores"]["faster"]["geomean"]
    assert verdict["winner"] == "faster"


# ---------------------------------------------------------------- fixtures --

def _args(**overrides):
    values = {
        "model": "/m/target", "ast": False, "ast_drafter": None, "ast_topology": None,
        "ast_graft_pld": False, "ast_tiles": None, "ast_tile_rows": None, "draft_model": [],
        "external_draft": False, "native_mtp": False, "prompt_lookup": False, "ordinary": False,
        "execution_policy": None, "adaptive_mtp_depth": False, "mtp_acceptance_log": None,
        "approximate_kv": None, "no_auto_route": False, "drafter_root": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolution(*, mtp=True, fingerprint=FP):
    descriptor = QWEN38_27B if mtp else QWEN38_27B_ORDINARY
    return AdapterResolution(Qwen3827BAdapter, descriptor,
                             {"identity": {"fingerprint": fingerprint}, "has_mtp": mtp})


def _table(entry):
    table = {"schema": brd.SCHEMA, "entries": [entry], "_sha256": "0" * 64}
    for item in table["entries"]:
        brd.validate_entry(item)
    return table


def _entry(default, fallback=(), status="selected_unqualified"):
    entry = {"checkpoint": f"Fixture-27B@sha256:{FP}", "fingerprint": FP,
             "adapter": "Qwen3827BAdapter", "status": status, "default": default,
             "reason": "fixture", "evidence": EVIDENCE}
    if fallback:
        entry["fallback"] = list(fallback)
    return entry


DFLASH = {"name": "ast_dflash2", "kind": "ast",
          "drafter": {"architectures": ["DFlash2DraftModel"], "identities": [GOOD]}}


def _drafter_dir(root: Path, name: str, arch="DFlash2DraftModel") -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"architectures": [arch], "model_type": "x"}))
    (path / "model.safetensors").write_bytes(b"\0" * 16)
    return path


@pytest.fixture
def fake_pairing(monkeypatch):
    """The target adapter's inspector, emulated: identity by directory name."""
    calls = []

    def pair(target, draft, cache=None, *, target_revision=True):
        calls.append(draft["path"])
        assert target_revision is False, "auto-pairing never hashes the target twice"
        name = Path(draft["path"]).name
        if name.startswith("badgeom"):
            raise ValueError("DFlash2 target hidden_size mismatch")
        identity = GOOD if name.startswith("good") else OTHER
        return {"draft": f"{name}@sha256:{identity}", "identity": identity,
                "draft_path": draft["path"], "draft_kind": "DFlash2DraftModel",
                "pins": {"draft_revision": identity}}

    monkeypatch.setattr(drafter_pairing, "pair_drafter", pair)
    return calls


# --------------------------------------------------------------- resolver --

def test_resolver_picks_the_table_entry_and_marks_it_unqualified():
    args = _args()
    record = apply_best_route_default(args, _resolution(), table=_table(_entry({"kind": "native_mtp"})))
    assert args.native_mtp and record["applied"]
    assert record["status"] == "selected_unqualified"
    assert record["qualification"] == "unqualified"
    assert record["evidence"] == EVIDENCE and record["match"] == "checkpoint"
    selection = resolve_route_selection(args, None, _resolution())
    assert selection.route == "native_mtp"


def test_unmatched_fingerprint_keeps_todays_default():
    args = _args()
    record = apply_best_route_default(args, _resolution(fingerprint="12" * 32),
                                      table=_table(_entry({"kind": "native_mtp"})))
    assert record["status"] == "no_entry" and not record["applied"]
    assert not args.native_mtp and not args.ast


def test_exact_artifact_policy_defaults_reach_resolution_and_receipts():
    defaults = {"tensorfold_prefill": True, "tensorfold_prefill_backend": "native",
                "moe_fused_gate_up": True}
    args = _args()
    table = _table(_entry({"kind": "ordinary", "execution_policy": defaults}))
    record = apply_best_route_default(args, _resolution(), table=table)
    policy = brd.finish_best_route_default(args, {"moe_fused_gate_up": False, "other": None})
    assert policy == {**defaults, "moe_fused_gate_up": False, "other": None}
    assert resolve_route_selection(args, policy, _resolution()).route == "ordinary"
    assert brd.receipt_view(record)["execution_policy"] == defaults
    assert record["qualification"] == "unqualified"
    unmatched = _args()
    apply_best_route_default(unmatched, _resolution(fingerprint=OTHER), table=table)
    assert brd.finish_best_route_default(unmatched, None) is None
    assert not unmatched.ordinary


@pytest.mark.parametrize("policy", [None, [], "native"])
def test_default_policy_requires_an_object(policy):
    with pytest.raises(ValueError, match="execution policy must be an object"):
        _table(_entry({"kind": "ordinary", "execution_policy": policy}))


def test_default_policy_cannot_leak_into_a_family_entry():
    entry = _entry({"kind": "ordinary", "execution_policy": {"tensorfold_prefill": True}})
    del entry["fingerprint"]
    entry["family"] = entry.pop("adapter")
    with pytest.raises(ValueError, match="exact artifact fingerprint"):
        brd.validate_entry(entry)


def test_tiel_approved_no_flag_profile_is_exactly_the_gpu_tested_policy():
    from mlx2.server import build_parser, resolve_execution_policy_defaults

    checkpoint = "AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP8-MTP"
    fingerprint = "e50c224a715030005f931df75c2c6d0b4ca183ca5c583b09c9c2d6e9dc6aae5f"
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter, descriptor_for

    resolution = AdapterResolution(Qwen3635BA3BAdapter, descriptor_for(has_mtp=True),
                                   {"identity": {"fingerprint": fingerprint}, "has_mtp": True})
    args = build_parser().parse_args(["--model", f"/m/{checkpoint}"])
    record = apply_best_route_default(args, resolution)
    policy = brd.finish_best_route_default(args, None)
    route = resolve_route_selection(args, policy, resolution)
    resolved = resolve_execution_policy_defaults(policy, route, resolution, max_lanes=4)
    tested_path = ROOT / "qualification/policies/tiel-coder-optimized.json"
    if not tested_path.is_file():
        pytest.skip("private qualification policy not present (public mirror)")
    tested = json.loads(tested_path.read_text())
    assert record["checkpoint"] == f"{checkpoint}@sha256:{fingerprint}"
    assert record["match"] == "checkpoint" and record["qualification"] == "unqualified"
    assert route.route == "ordinary" and policy == tested
    assert all(resolved[key] == value for key, value in tested.items())


@pytest.mark.parametrize("flags", [["--lane-matmul", "exact"], ["--int8-prefill", "mlp"],
                                   ["--microscaling-prefill", "experts"]])
def test_tiel_policy_defaults_yield_to_an_explicit_projection_owner(flags):
    from mlx2.server import build_parser
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter, descriptor_for

    fingerprint = "e50c224a715030005f931df75c2c6d0b4ca183ca5c583b09c9c2d6e9dc6aae5f"
    resolution = AdapterResolution(Qwen3635BA3BAdapter, descriptor_for(has_mtp=True),
                                   {"identity": {"fingerprint": fingerprint}, "has_mtp": True})
    args = build_parser().parse_args(["--model", "/m/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP8-MTP", *flags])
    record = apply_best_route_default(args, resolution)
    assert record["applied"]
    assert brd.finish_best_route_default(args, None) is None
    assert record["execution_policy_defaults_skipped"] == [flags[0][2:].replace("-", "_")]
    view = brd.receipt_view(record)
    assert "execution_policy" not in view
    assert view["execution_policy_defaults_skipped"] == record["execution_policy_defaults_skipped"]


@pytest.mark.parametrize("override", [
    {"ordinary": True}, {"native_mtp": True}, {"prompt_lookup": True}, {"ast": True},
    {"external_draft": True}, {"draft_model": ["/m/d"]}, {"execution_policy": Path("/p.json")},
    {"adaptive_mtp_depth": True}, {"approximate_kv": "{}"},
])
def test_explicit_route_options_always_win(override):
    args = _args(**override)
    before = dict(vars(args))
    record = apply_best_route_default(args, _resolution(), table=_table(_entry(DFLASH)))
    assert record["status"] == "explicit_route" and not record["applied"]
    assert vars(args) == {**before, "route_default_selection": record}


def test_mtp_head_flag_disables_the_auto_default():
    # --mtp-head names the route (native MTP on the attached head); the
    # measured no-flag default must not replace it.
    args = _args(mtp_head="/m/head")
    before = dict(vars(args))
    record = apply_best_route_default(args, _resolution(), table=_table(_entry(DFLASH)))
    assert record["status"] == "explicit_route" and not record["applied"]
    assert record["explicit_options"] == ["mtp_head"]
    assert vars(args) == {**before, "route_default_selection": record}
    assert not args.ast and not args.native_mtp


def test_no_auto_route_opts_out():
    args = _args(no_auto_route=True)
    record = apply_best_route_default(args, _resolution(), table=_table(_entry({"kind": "native_mtp"})))
    assert record["status"] == "opted_out" and not args.native_mtp


def test_auto_expert_streaming_keeps_todays_route():
    args = _args()
    record = apply_best_route_default(args, _resolution(), table=_table(_entry({"kind": "native_mtp"})),
                                      streaming={"selected": True})
    assert record["status"] == "not_applicable" and not record["applied"]
    assert not args.native_mtp
    # Not selected: the entry applies as usual.
    args = _args()
    record = apply_best_route_default(args, _resolution(), table=_table(_entry({"kind": "native_mtp"})),
                                      streaming={"selected": False})
    assert record["applied"] and args.native_mtp


def test_kept_and_pending_entries_change_nothing_but_say_why():
    for status in ("kept_current_default", "pending"):
        args = _args()
        entry = _entry({"kind": "adapter_default"}, status=status)
        record = apply_best_route_default(args, _resolution(), table=_table(entry))
        assert record["status"] == status and not record["applied"]
        assert record["reason"] == "fixture" and not args.ast and not args.native_mtp


def test_family_entry_applies_when_no_checkpoint_entry():
    entry = {"family": "Qwen3827BAdapter", "status": "kept_current_default",
             "default": {"kind": "adapter_default"}, "reason": "family", "evidence": EVIDENCE}
    record = apply_best_route_default(_args(), _resolution(fingerprint="12" * 32), table=_table(entry))
    assert record["match"] == "family" and record["status"] == "kept_current_default"


# ------------------------------------------------------------ auto-pairing --

def test_verified_drafter_is_paired_and_pinned(tmp_path, fake_pairing):
    _drafter_dir(tmp_path, "badgeom-dflash2")
    good = _drafter_dir(tmp_path, "good-dflash2")
    _drafter_dir(tmp_path, "other-arch", arch="Qwen3DSparkModel")
    args = _args(drafter_root=[tmp_path])
    record = apply_best_route_default(args, _resolution(), table=_table(_entry(DFLASH)))
    assert record["status"] == "selected_unqualified"
    assert args.ast and args.ast_drafter == str(good)
    assert record["drafter"]["identity"] == f"good-dflash2@sha256:{GOOD}"
    assert any("hidden_size" in item["reason"] for item in record["route"]["refused_drafters"])
    assert all("other-arch" not in path for path in fake_pairing)
    # The same translation an explicit --ast --ast-drafter gets, plus the pin.
    from mlx2.best_route_defaults import finish_best_route_default

    policy = apply_ast_invocation(args, None, _resolution())
    policy = finish_best_route_default(args, policy)
    assert policy["draft_model"] == str(good)
    assert policy["draft_revision"] == GOOD
    assert policy["batch_size_route"] == "tree15_b1_b4_chain_b5plus_v1"
    assert record["route"]["ast_route"]["tree_tiles"] is True
    selection = resolve_route_selection(args, policy, _resolution())
    assert selection.route == "external_draft"


def test_wrong_identity_or_geometry_is_refused_and_next_route_used(tmp_path, fake_pairing):
    _drafter_dir(tmp_path, "badgeom-dflash2")
    _drafter_dir(tmp_path, "stale-dflash2")          # verifies, but not the evidence drafter
    args = _args(drafter_root=[tmp_path])
    entry = _entry(DFLASH, fallback=[{"name": "native_mtp", "kind": "native_mtp"}])
    record = apply_best_route_default(args, _resolution(), table=_table(entry))
    assert record["status"] == "failed_over_unqualified"
    assert record["default"] == "native_mtp" and args.native_mtp
    assert not args.ast and args.ast_drafter is None
    reasons = " ".join(item["reason"] for item in record["failover"])
    assert "no verified DFlash2DraftModel drafter" in reasons
    assert record["drafter"] is None


def test_no_drafter_found_falls_back_to_the_adapter_default(tmp_path, fake_pairing):
    args = _args(drafter_root=[tmp_path / "empty"])
    record = apply_best_route_default(args, _resolution(), table=_table(_entry(DFLASH)))
    assert record["status"] == "failed_over_to_adapter_default" and not record["applied"]
    assert not args.ast and not args.native_mtp and fake_pairing == []
    selection = resolve_route_selection(args, None, _resolution())
    assert selection == RouteSelection("native_mtp", "adapter_default")


def test_mtp_spec_fails_over_without_a_head():
    entry = _entry({"name": "ast_mtp", "kind": "ast", "drafter": "mtp"},
                   fallback=[{"kind": "ordinary"}])
    args = _args()
    record = apply_best_route_default(args, _resolution(mtp=False), table=_table(entry))
    assert record["status"] == "failed_over_unqualified" and args.ordinary and not args.ast


def test_config_digest_prefilter_skips_hashing(tmp_path, monkeypatch):
    _drafter_dir(tmp_path, "good-dflash2")
    monkeypatch.setattr(drafter_pairing, "pair_drafter",
                        lambda *a, **k: pytest.fail("a non-evidence config must not be hashed"))
    spec = dict(DFLASH, drafter=dict(DFLASH["drafter"], config_sha256=["00" * 32]))
    record = apply_best_route_default(_args(drafter_root=[tmp_path]), _resolution(),
                                      table=_table(_entry(spec)))
    assert record["status"] == "failed_over_to_adapter_default"
    assert "config.json differs" in record["failover"][0]["refused_drafters"][0]["reason"]


def test_discovery_reuses_the_inventory_classifier(tmp_path):
    _drafter_dir(tmp_path, "a-dflash2")
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"],
                                                    "model_type": "qwen3_5"}))
    (target / "model.safetensors").write_bytes(b"\0")
    incomplete = tmp_path / "half-dflash2"
    incomplete.mkdir()
    (incomplete / "config.json").write_text(json.dumps({"architectures": ["DFlash2DraftModel"]}))
    rows = drafter_pairing.discover_drafters([tmp_path], include_hf=False)
    assert [Path(row["path"]).name for row in rows] == ["a-dflash2"]


# --------------------------------------------------------------- receipts --

def test_receipts_and_engine_arguments_carry_the_selection(tmp_path, fake_pairing):
    _drafter_dir(tmp_path, "good-dflash2")
    args = _args(drafter_root=[tmp_path])
    record = apply_best_route_default(args, _resolution(), table=_table(_entry(DFLASH)))
    view = brd.receipt_view(record)
    assert view["status"] == "selected_unqualified" and view["qualification"] == "unqualified"
    assert view["evidence"]["source_revision"] == EVIDENCE["source_revision"]
    assert view["drafter"]["identity"].endswith(GOOD) and view["reason"] == "fixture"
    from mlx2.serving import ServingEngine

    ServingEngine.validate_arguments(
        "/m/target", route_selection_source="best_route_default", route_default_selection=record)
    with pytest.raises(ValueError, match="route default selection"):
        ServingEngine.validate_arguments("/m/target", route_default_selection=["bad"])


def test_request_receipt_and_status_include_the_selection_view():
    source = (ROOT / "src/mlx2/serving.py").read_text()
    tree = ast.parse(source)
    receipt = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict) and all(
        name in [k.value for k in node.keys if isinstance(k, ast.Constant)]
        for name in ("route", "route_receipt", "route_selection_source", "request_controls")))
    splats = [ast.unparse(v) for k, v in zip(receipt.keys, receipt.values) if k is None]
    assert any("receipt_view(self.route_default_selection)" in text for text in splats)
    assert 'settings["route_default_selection"] = deepcopy(self.route_default_selection)' in source
    from mlx2.qualification import PROVENANCE_ONLY_SETTINGS

    assert "route_default_selection" in PROVENANCE_ONLY_SETTINGS


def test_server_kwargs_forward_the_record():
    from mlx2.server import build_parser

    args = build_parser().parse_args(["--model", "/m"])
    args.route_default_selection = {"status": "no_entry"}
    kwargs = serving_engine_kwargs(args, None, native_mtp=False, approximate_kv=None,
                                   max_request_bytes=1)
    assert kwargs["route_default_selection"] == {"status": "no_entry"}


def test_cli_parses_opt_out_and_roots():
    from mlx2.server import build_parser

    args = build_parser().parse_args(
        ["--model", "/m", "--no-auto-route", "--drafter-root", "/d1", "--drafter-root", "/d2"])
    assert args.no_auto_route and args.drafter_root == [Path("/d1"), Path("/d2")]


# ------------------------------------------------------------------ table --

def test_shipped_table_is_valid_and_cites_committed_evidence():
    table = brd.load_table()
    assert table["rule"]["min_geomean"] == brd.RULE["min_geomean"] == 1.05
    assert table["rule"]["min_cell"] == brd.RULE["min_cell"] == 0.85
    cited = [path for entry in table["entries"] for path in entry["evidence"]["files"]]
    # The public code mirror ships no run evidence or research notes: when
    # none of the cited private evidence is present, check only the rest.
    private = ("qualification/runs/", "qualification/policies/", "docs/research/")
    evidence_present = any((ROOT / path).exists() for path in cited if path.startswith(private))
    for path in cited:
        if evidence_present or not path.startswith(private):
            assert (ROOT / path).exists(), path
    for entry in table["entries"]:
        if entry["status"] == "selected_unqualified":
            assert entry["default"]["kind"] != "adapter_default"
            if entry["evidence"].get("decision_row"):
                # Imported from a decision row: another lane's winner, or an
                # operator decision overriding the rule, which must say so.
                if "measured_scores" in entry["evidence"]:
                    assert "Operator decision:" in entry["reason"]
                continue
            score = entry["evidence"]["scores"][entry["default"]["name"]]
            assert score["qualifies"] and score["geomean"] >= 1.05 and score["min_cell"] >= 0.85
            drafter = entry["default"].get("drafter")
            if isinstance(drafter, dict):
                assert drafter["identities"] and drafter["architectures"]


# ------------------------------------------------- other lanes' decision rows --

def test_lane_decision_rows_import_by_convention(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_brd", ROOT / "scripts/build_best_route_defaults.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    header = ["model", "checkpoint", "scope", "feature_or_flag", "implemented", "qualified", "selected",
              "observed_used", "decision", "reason", "evidence", "source_revision", "artifact_profile"]

    def row(checkpoint, reason, evidence="qualification/runs/x/summary.json", selected="yes_default"):
        values = {"model": "m", "checkpoint": checkpoint, "feature_or_flag": "best_route_default",
                  "selected": selected, "reason": reason, "evidence": evidence, "source_revision": "a" * 40}
        return "\t".join(values.get(name, "") for name in header)

    tsv = tmp_path / "decisions.tsv"
    tsv.write_text("\n".join([
        "\t".join(header),
        row(f"Flint@sha256:{FP}", "won the rule. Route: prompt_lookup adapter=FlintAdapter"),
        row(f"Own@sha256:{GOOD}", "Route: ast", evidence="src/mlx2/best_route_defaults.json"),
        row(f"Vague@sha256:{OTHER}", "EAGLE-4 wins"),
        row(f"NotSel@sha256:{OTHER}", "Route: ordinary adapter=X", selected="no"),
        row(f"Packed@sha256:{GOOD}", 'Operator decision: Route: ordinary adapter=X '
            'Execution-policy: {"tensorfold_prefill": true}'),
        row(f"Broken@sha256:{OTHER}", "Route: ordinary adapter=X Execution-policy: {bad"),
    ]) + "\n")
    monkeypatch.setattr(build, "DECISIONS", tsv)
    entries, skipped = build.lane_rows()
    assert [e["checkpoint"] for e in entries] == [f"Flint@sha256:{FP}", f"Packed@sha256:{GOOD}"]
    assert entries[0]["default"] == {"name": "prompt_lookup", "kind": "prompt_lookup"}
    assert entries[0]["adapter"] == "FlintAdapter" and entries[0]["evidence"]["decision_row"]
    assert entries[1]["default"]["execution_policy"] == {"tensorfold_prefill": True}
    assert [item["checkpoint"] for item in skipped] == [f"Vague@sha256:{OTHER}", f"Broken@sha256:{OTHER}"]


def test_tiel_shipped_default_matches_the_durable_decision_row():
    import importlib.util

    if not (ROOT / "docs/research/model-feature-decisions.tsv").is_file():
        pytest.skip("private decision rows not present (public mirror)")

    spec = importlib.util.spec_from_file_location("build_brd", ROOT / "scripts/build_best_route_defaults.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    entries, _ = build.lane_rows()
    fingerprint = "e50c224a715030005f931df75c2c6d0b4ca183ca5c583b09c9c2d6e9dc6aae5f"
    imported = next(entry for entry in entries if entry["fingerprint"] == fingerprint)
    assert brd.find_entry(brd.load_table(), "Qwen3635BA3BAdapter", fingerprint) == imported


def test_operator_decision_row_overrides_a_measured_entry():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_brd", ROOT / "scripts/build_best_route_defaults.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    measured = {"checkpoint": f"M@sha256:{FP}", "fingerprint": FP, "adapter": "MAdapter",
                "status": "kept_current_default", "default": {"kind": "adapter_default"},
                "reason": "rule kept the default",
                "evidence": {"files": ["qualification/runs/m.json"], "source_revision": "b" * 40,
                             "scores": {"default": 1}}}

    def lane(reason):
        return {"checkpoint": f"M@sha256:{FP}", "fingerprint": FP, "adapter": "MAdapter",
                "status": "selected_unqualified", "default": {"name": "ordinary", "kind": "ordinary"},
                "reason": f"decision row from another lane: {reason}",
                "evidence": {"files": ["docs/research/model-feature-decisions.tsv"],
                             "source_revision": "a" * 40, "decision_row": True}}

    skipped = []
    kept = build.merge_lane_entries([measured], [lane("Route: ordinary adapter=MAdapter")], skipped)
    assert kept == [measured] and skipped[0]["why"].startswith("measured by this campaign")
    skipped = []
    merged = build.merge_lane_entries(
        [measured], [lane("Operator decision: plain decode. Route: ordinary adapter=MAdapter")], skipped)
    assert not skipped and merged[0]["default"]["kind"] == "ordinary"
    assert "qualification/runs/m.json" in merged[0]["evidence"]["files"]
    assert merged[0]["evidence"]["measured_scores"] == {"default": 1}


def test_hy4_assembled_pack_no_flag_start_selects_embedded_dfly2():
    """Pierre 2026-11-20: the assembled Hy4-flash MLX pack serves its built-in
    DFly2 drafter by default (--ast selects the embedded native head); an
    explicit route flag keeps the operator's route."""
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter, descriptor_for
    from mlx2.server import build_parser

    checkpoint = "Hy4-flash-MLX-4.6bit-mlx2"
    fingerprint = "f2a8fa8d2b7f22150ee5bd988c5acc1814d8494aa82ee01aff40a7efc13aad93"
    resolution = AdapterResolution(HYV4FlashAdapter, descriptor_for(vision=True),
                                   {"identity": {"fingerprint": fingerprint}, "has_mtp": True,
                                    "has_dfly": True})
    args = build_parser().parse_args(["--model", f"/m/{checkpoint}"])
    record = apply_best_route_default(args, resolution)
    assert record["checkpoint"] == f"{checkpoint}@sha256:{fingerprint}"
    assert record["match"] == "checkpoint" and record["qualification"] == "unqualified"
    assert args.ast is True and getattr(args, "ast_drafter", None) is None
    from mlx2.server import apply_ast_invocation

    policy = apply_ast_invocation(args, None, resolution)
    policy = brd.finish_best_route_default(args, policy)
    route = resolve_route_selection(args, policy, resolution)
    assert route.route == "external_draft"
    # The artifact's own head: the Hy4 adapter maps the model path to DFly2.
    assert policy["draft_model"] in (f"/m/{checkpoint}", "mtp", "embedded_dfly")
    explicit = build_parser().parse_args(["--model", f"/m/{checkpoint}", "--ordinary"])
    assert apply_best_route_default(explicit, resolution).get("match") != "checkpoint"
    assert explicit.ordinary is True and not explicit.ast


# --------------------------------------- execution policy vs route choice --

def _hy4_resolution():
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter, descriptor_for

    fingerprint = "f2a8fa8d2b7f22150ee5bd988c5acc1814d8494aa82ee01aff40a7efc13aad93"
    return AdapterResolution(HYV4FlashAdapter, descriptor_for(vision=True),
                             {"identity": {"fingerprint": fingerprint}, "has_mtp": True,
                              "has_dfly": True})


def _hy4_args(tmp_path, policy):
    from mlx2.server import build_parser

    path = tmp_path / "p.json"
    path.write_text(json.dumps(policy))
    return build_parser().parse_args(
        ["--model", "/m/Hy4-flash-MLX-4.6bit-mlx2", "--execution-policy", str(path)])


HY4_NEUTRAL = {"engram_table_format", "engram_precision", "engram_cache_rows", "engram_hash",
               "vision", "moe_fused_gate_up", "amoeba_det_tiles", "prefill_step_cap",
               "sliding_attention_blocks", "sparse_prefill_attention",
               "hc_seam_fused", "mla_fused_input_proj", "engram_fused_proj", "indexer_fused_scores",
               "engram_gate_rounding", "moe_expert_read_once",
               "prefill_head_trim", "prefill_tail_skip", "moe_sorted_weighted_sum",
               # Model-level installs on both routes: union prefill (audit H4), exact 16-bit planes (H5).
               "sparse_prefill_union_rows", "resident_bf16", "prefill_step_fixed"}


@pytest.mark.parametrize("policy", [
    {"engram_table_format": "iq4_nl"},
    # --det-tile merges this key: the calculator keeps the DFly2 default.
    {"engram_table_format": "iq4_nl", "amoeba_det_tiles": "/m/tiles/calculator/manifest.json"},
    {"engram_hash": "device", "engram_precision": "vllm", "engram_cache_rows": 4096,
     "vision": False, "moe_fused_gate_up": True},
    # The opt-outs of the 2026-11-21 Hy4 defaults (calculator, band, step cap).
    {"amoeba_det_tiles": False, "sliding_attention_blocks": 0, "prefill_step_cap": 0},
    {},
])
def test_route_neutral_policy_keeps_the_best_route_default(tmp_path, policy):
    """A policy of only adapter-declared route-neutral keys keeps the Hy4
    pack's embedded DFly2 default (a 2026-11-21 GSM8K receipt with only
    ``engram_table_format`` served ordinary decode, verify_rounds 0)."""
    resolution = _hy4_resolution()
    for passed in (True, False):  # the parsed policy, or the file read here
        args = _hy4_args(tmp_path, policy)
        kwargs = {"policy": dict(policy)} if passed else {}
        record = apply_best_route_default(args, resolution, **kwargs)
        assert record["applied"] and record["status"] == "selected_unqualified"
        assert args.ast is True
    merged = apply_ast_invocation(args, dict(policy), resolution)
    merged = brd.finish_best_route_default(args, merged)
    route = resolve_route_selection(args, merged, resolution)
    assert route.route == "external_draft"
    assert {k: merged[k] for k in policy} == policy


def test_hy4_neutral_keys_parse_identically_on_ordinary_and_dfly2():
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter, _parse_policy

    assert set(HYV4FlashAdapter.route_neutral_policy_keys) == HY4_NEUTRAL
    neutral = {"engram_table_format": "iq4_nl", "engram_precision": "vllm",
               "engram_cache_rows": 4096, "engram_hash": "device", "vision": False,
               "moe_fused_gate_up": True, "amoeba_det_tiles": "/m/tiles/calculator/manifest.json",
               "sparse_prefill_attention": True, "sparse_prefill_union_rows": 64, "resident_bf16": True}
    ordinary, external, kind = _parse_policy(dict(neutral), "/m/Hy4")
    assert ordinary == neutral and not external and kind is None
    dfly, external, kind = _parse_policy({**neutral, "draft_model": "embedded_dfly"}, "/m/Hy4")
    assert dfly == neutral and kind == "dfly"
    pld, _, kind = _parse_policy({**neutral, "draft_model": "prompt_lookup"}, "/m/Hy4")
    assert pld == neutral and kind == "prompt_lookup"


@pytest.mark.parametrize("policy,key", [
    ({"engram_table_format": "iq4_nl", "draft_model": "embedded_dfly"}, "draft_model"),
    ({"draft_models": [{"id": "d", "path": "/m/d"}]}, "draft_models"),
    ({"engram_table_format": "iq4_nl", "num_draft": 3}, "num_draft"),
    ({"ast_target_executor": "serial"}, "ast_target_executor"),
    ({"moe_nax": "gather"}, "moe_nax"),  # refused with a draft_model
    ({"latent_kv": "fp8"}, "latent_kv"),  # another cache shape than was measured
    ({"mtp_head": "/m/head"}, "mtp_head"),
    ({"adaptive_mtp_depth": True}, "adaptive_mtp_depth"),
    ({"moe_expert_streaming": {"enabled": True}}, "moe_expert_streaming"),
    ({"dense_weight_streaming": {"enabled": True}}, "dense_weight_streaming"),
    ({"some_future_key": 1}, "some_future_key"),  # unknown: fail safe
])
def test_non_neutral_policy_keeps_todays_resolution_and_says_why(tmp_path, policy, key, caplog):
    args = _hy4_args(tmp_path, policy)
    before = dict(vars(args))
    with caplog.at_level("INFO", logger="mlx2.server"):
        record = apply_best_route_default(args, _hy4_resolution(), policy=dict(policy))
    assert record["status"] == "explicit_route" and not record["applied"]
    assert f"execution_policy:{key}" in record["explicit_options"]
    assert "explicit route choice" in record["reason"] and key in record["reason"]
    assert vars(args) == {**before, "route_default_selection": record}
    view = brd.receipt_view(record)
    assert view["reason"] == record["reason"]
    assert view["explicit_options"] == record["explicit_options"]
    assert any("best-route default not applied (explicit_route)" in r.getMessage()
               and key in r.getMessage() for r in caplog.records)


def test_streaming_and_route_keys_are_never_neutral_even_if_declared():
    class Declares:
        route_neutral_policy_keys = frozenset({"moe_expert_streaming", "dense_weight_streaming",
                                               "draft_model", "adaptive_mtp_depth", "vision"})

    assert brd.route_neutral_policy_keys(Declares) == {"vision"}
    assert brd.route_choosing_policy_keys(
        {"moe_expert_streaming": {}, "dense_weight_streaming": {}, "vision": True}, Declares
    ) == ["dense_weight_streaming", "moe_expert_streaming"]


def test_undeclared_adapter_keeps_the_old_rule_for_any_policy():
    # Regression (review of f6305e393): Qwen3.8 27B refuses target-only
    # numerics such as fp32_head_logits on its external/AST route, so a
    # policy of them must not pull in the AST default.  Undeclared adapters
    # treat every policy, even an empty one, as an explicit choice.
    assert getattr(Qwen3827BAdapter, "route_neutral_policy_keys", None) is None
    table = _table(_entry(DFLASH))
    for policy, expected in (
        ({"fp32_head_logits": True}, ["execution_policy:fp32_head_logits"]),
        ({"gdn_core": "x", "invariant_prefill": True},
         ["execution_policy:gdn_core", "execution_policy:invariant_prefill"]),
        ({"adaptive_mtp_depth": True}, ["execution_policy:adaptive_mtp_depth"]),
        ({"moe_expert_streaming": {"enabled": True}}, ["execution_policy:moe_expert_streaming"]),
        ({}, ["execution_policy:*"]),
    ):
        args = _args(execution_policy=Path("/p.json"))
        record = apply_best_route_default(args, _resolution(), table=table, policy=policy)
        assert record["status"] == "explicit_route", policy
        assert record["explicit_options"] == expected
        assert not args.ast and not args.native_mtp


def test_adaptive_mtp_depth_policy_keeps_native_mtp(tmp_path):
    # Regression: the server's adaptive_mtp_depth check needs the native-MTP
    # route; a best-route default must not replace it.
    from mlx2.server import build_parser

    path = tmp_path / "p.json"
    path.write_text(json.dumps({"adaptive_mtp_depth": True}))
    args = build_parser().parse_args(["--model", "/m/target", "--execution-policy", str(path)])
    record = apply_best_route_default(args, _resolution(), table=_table(_entry(DFLASH)),
                                      policy={"adaptive_mtp_depth": True})
    assert record["status"] == "explicit_route" and not args.ast
    route = resolve_route_selection(args, {"adaptive_mtp_depth": True}, _resolution())
    assert route.route == "native_mtp"


def test_explicit_streaming_policy_suppresses_the_default_on_a_declared_adapter(tmp_path):
    # Regression: an operator-chosen streaming setup is not auto streaming
    # (streaming_selected stays false), so the policy itself must suppress.
    for policy in ({"moe_expert_streaming": {"enabled": True}, "vision": True},
                   {"dense_weight_streaming": {"enabled": True}}):
        args = _hy4_args(tmp_path, policy)
        record = apply_best_route_default(args, _hy4_resolution(), policy=policy,
                                          streaming=None)
        assert record["status"] == "explicit_route" and not args.ast


def test_skip_reasons_are_logged(caplog):
    with caplog.at_level("INFO", logger="mlx2.server"):
        apply_best_route_default(_args(no_auto_route=True), _resolution(),
                                 table=_table(_entry(DFLASH)))
        apply_best_route_default(_args(), _resolution(), table=_table(_entry(DFLASH)),
                                 streaming={"selected": True})
    messages = [r.getMessage() for r in caplog.records]
    assert any("not applied (opted_out): --no-auto-route" in m for m in messages)
    assert any("not applied (not_applicable): automatic expert streaming" in m for m in messages)


def test_flag_merged_policy_keys_are_not_judged_without_execution_policy():
    # --det-tile merges amoeba_det_tiles into the policy; without an operator
    # --execution-policy that is not a policy route choice (as before).
    args = _args()
    record = apply_best_route_default(args, _resolution(), table=_table(_entry({"kind": "native_mtp"})),
                                      policy={"amoeba_det_tiles": "/m/manifest.json"})
    assert record["applied"] and args.native_mtp is True


def test_unreadable_policy_file_is_conservatively_a_route_choice(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    for path in (bad, tmp_path / "missing.json"):
        args = _args(execution_policy=path)
        record = apply_best_route_default(args, _resolution(), table=_table(_entry(DFLASH)))
        assert record["status"] == "explicit_route"
        assert record["explicit_options"] == ["execution_policy"]


def test_draft_policy_refusal_names_the_flags_to_pass():
    from mlx2.server import build_parser

    args = build_parser().parse_args(["--model", "/m/target"])
    with pytest.raises(ValueError) as error:
        resolve_route_selection(args, {"draft_model": "/m/d"}, _resolution())
    message = str(error.value)
    assert "requires --external-draft" in message
    assert "pass --external-draft" in message and "--ast" in message
    assert "--ast-drafter" in message


# ------------------------------------------- width-keyed (Maxwell, 2026-11-24) --

MAXWELL_FP = "6618b91785140c136ba618a910e4e1a439f2ef9f61622a1d3140bb0683a0a294"


def _maxwell_resolution():
    from mlx2.adapters.maxwell_moe import MaxwellMoeAdapter, descriptor_for
    return AdapterResolution(MaxwellMoeAdapter, descriptor_for(has_mtp=True),
                             {"identity": {"fingerprint": MAXWELL_FP}, "has_mtp": True})


@pytest.mark.parametrize("lanes", [1, 4])
def test_maxwell_single_lane_default_is_keyed_on_serving_width(lanes, monkeypatch):
    monkeypatch.delenv("MLX2_MAXWELL_INDEX_KERNEL", raising=False)
    monkeypatch.delenv("MLX2_MAXWELL_SPARSE_PREFILL", raising=False)
    # Pierre 2026-11-24: the candidate (native MTP d2 + index kernel + masked
    # prefill, no fused QKV) is the no-flag default only at --max-lanes 1.
    args = _args(max_lanes=lanes)
    record = apply_best_route_default(args, _maxwell_resolution(), table=brd.load_table())
    policy = brd.finish_best_route_default(args, None)
    assert record["qualification"] == "unqualified"
    if lanes == 1:
        assert record["applied"]
        assert record["status"] == "selected_unqualified" and record["default"] == "native_mtp"
        assert record["route"]["max_lanes"] == 1
        assert args.native_mtp
        assert policy == {"num_draft": 2, "index_kernel": True, "sparse_prefill": "masked"}
        assert resolve_route_selection(args, policy, _maxwell_resolution()).route == "native_mtp"
    else:
        assert record["status"] == "width_keyed_adapter_default" and not record["applied"]
        assert "--max-lanes 4 exceeds" in record["reason"]
        assert "--max-lanes 4 exceeds" in brd.receipt_view(record)["reason"]
        assert not args.native_mtp and policy is None
        selection = resolve_route_selection(args, policy, _maxwell_resolution())
        assert selection.route == "ordinary" and selection.source == "adapter_default"


def test_maxwell_single_lane_default_yields_to_an_explicit_route():
    args = _args(max_lanes=1, ordinary=True)
    record = apply_best_route_default(args, _maxwell_resolution(), table=brd.load_table())
    assert record["status"] == "explicit_route" and not record.get("applied")


def test_width_keyed_spec_validation():
    entry = _entry({"kind": "native_mtp", "max_lanes": 0})
    with pytest.raises(ValueError, match="max_lanes"):
        brd.validate_entry(entry)
    with pytest.raises(ValueError, match="width-keyed"):
        brd.validate_entry(_entry({"kind": "native_mtp", "max_lanes": 1},
                                  fallback=[{"kind": "adapter_default", "max_lanes": 1}]))


def test_maxwell_policy_levers_apply_unless_the_operator_set_the_env(monkeypatch):
    from mlx2.runtime.models import maxwell_moe as M
    monkeypatch.delenv("MLX2_MAXWELL_INDEX_KERNEL", raising=False)
    monkeypatch.setenv("MLX2_MAXWELL_SPARSE_PREFILL", "union")
    monkeypatch.setattr(M, "INDEX_KERNEL", False)
    monkeypatch.setattr(M, "SPARSE_PREFILL", "union")
    from mlx2.adapters.maxwell_moe import MaxwellMoeAdapter
    from mlx2.serving import adapter_numerical_contract

    first, applied = M.resolve_policy_levers(index_kernel=True, sparse_prefill="masked")
    assert applied == {"index_kernel": "policy", "sparse_prefill": "environment"}
    assert first == {"index_kernel": True, "sparse_prefill": "union"}
    second, applied = M.resolve_policy_levers(index_kernel=False)
    assert second["index_kernel"] is False and applied == {"index_kernel": "policy"}
    assert M.maxwell_stats(attention_levers=first)["levers"]["index_kernel"] is True
    adapter = object.__new__(MaxwellMoeAdapter)
    adapter.attention_levers = first
    assert adapter_numerical_contract(adapter) == {"maxwell_attention": first}
    adapter.attention_levers = second
    assert adapter_numerical_contract(adapter) is None
    with pytest.raises(ValueError):
        M.resolve_policy_levers(sparse_prefill="dense")


def test_maxwell_width_key_boundary_and_unknown_width():
    for lanes, applies in ((1, True), (2, False), (None, False)):
        args = _args(max_lanes=lanes)
        record = apply_best_route_default(args, _maxwell_resolution(), table=brd.load_table())
        assert bool(args.native_mtp) is applies, lanes
        assert record["status"] == ("selected_unqualified" if applies else "width_keyed_adapter_default")


def test_maxwell_env_owned_lever_is_dropped_from_the_default_and_the_receipt(monkeypatch):
    monkeypatch.setenv("MLX2_MAXWELL_SPARSE_PREFILL", "union")
    monkeypatch.setenv("MLX2_MAXWELL_INDEX_KERNEL", "")          # empty: not operator intent
    args = _args(max_lanes=1)
    record = apply_best_route_default(args, _maxwell_resolution(), table=brd.load_table())
    policy = brd.finish_best_route_default(args, None)
    assert policy == {"num_draft": 2, "index_kernel": True}
    view = brd.receipt_view(record)
    assert view["execution_policy"] == {"num_draft": 2, "index_kernel": True}
    assert view["execution_policy_env_kept"] == {"sparse_prefill": "MLX2_MAXWELL_SPARSE_PREFILL"}


def test_builder_parses_max_lanes_and_emits_the_adapter_fallback(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("brd_builder", ROOT / "scripts/build_best_route_defaults.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    header = (ROOT / "docs/research/model-feature-decisions.tsv").read_text().splitlines()[0]
    row = ["maxwell_moe:base", f"Fixture@sha256:{MAXWELL_FP}", "feature", "best_route_default", "yes", "no",
           "yes_default", "no", "x",
           'Operator decision: fixture. Route: native_mtp adapter=MaxwellMoeAdapter Max-lanes: 1 '
           'Execution-policy: {"num_draft": 2}', "qualification/runs/x", "abc", "bf16"]
    path = tmp_path / "decisions.tsv"
    path.write_text(header + "\n" + "\t".join(row) + "\n")
    monkeypatch.setattr(builder, "DECISIONS", path)
    entries, skipped = builder.lane_rows()
    assert not skipped and entries[0]["default"]["max_lanes"] == 1
    assert entries[0]["fallback"] == [{"name": "adapter_default", "kind": "adapter_default"}]
    assert "only at `--max-lanes` <= 1" in builder.route_text(entries[0]["default"])
