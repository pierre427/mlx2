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
    for entry in table["entries"]:
        for path in entry["evidence"]["files"]:
            assert (ROOT / path).exists(), path
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
    ]) + "\n")
    monkeypatch.setattr(build, "DECISIONS", tsv)
    entries, skipped = build.lane_rows()
    assert [e["checkpoint"] for e in entries] == [f"Flint@sha256:{FP}"]
    assert entries[0]["default"] == {"name": "prompt_lookup", "kind": "prompt_lookup"}
    assert entries[0]["adapter"] == "FlintAdapter" and entries[0]["evidence"]["decision_row"]
    assert [item["checkpoint"] for item in skipped] == [f"Vague@sha256:{OTHER}"]


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
