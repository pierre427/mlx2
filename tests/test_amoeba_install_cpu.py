"""CPU tests of `mlx2 amoeba install` (no model load): settings, the joint memory plan, the smoke-test label, the topo
stage on bundle masks, the report's verdict rule against controls, and the CLI wiring."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mlx2.experimental.amoeba import install as I
from mlx2.experimental.amoeba import plan as P
from mlx2.experimental.amoeba import state as ST


def info(weights_gb=1.4, L=28, bits=None):
    return {"path": "/models/toy", "checkpoint": "toy@00000000", "fingerprint": "0" * 64, "layers": L,
            "weight_bytes": int(weights_gb * 2 ** 30), "quantization": {"bits": bits} if bits else None}


def test_settings_profiles_and_overrides():
    full, quick = I.settings("full", {}), I.settings("quick", {})
    assert full["norm_budget"] == 0.25 and full["drift_cap"] == 0.05 and full["data"]["n_test"] == 100
    assert quick["epochs"] == 1 and quick["data"]["n_test"] == 12 and quick["norm_budget"] == 0.25
    s = I.settings("full", {"epochs": "2", "data.n_test": "200", "lr.bridge": "5e-4"})
    assert s["epochs"] == 2 and s["data"]["n_test"] == 200 and s["lr"]["bridge"] == pytest.approx(5e-4)


def test_settings_refuse_unknown_keys():
    for bad in ({"epoch": "2"}, {"foo.bar": "1"}, {"data.n_tests": "5"}, {"teacher": "full"}):
        with pytest.raises(SystemExit):
            I.settings("full", bad)


def test_params_from_safetensors_headers(tmp_path):
    import struct
    hdr = {"a.weight": {"dtype": "U32", "shape": [64, 8], "data_offsets": [0, 0]},       # 4-bit: 64 x 64 params
           "a.scales": {"dtype": "BF16", "shape": [64, 1], "data_offsets": [0, 0]},
           "a.biases": {"dtype": "BF16", "shape": [64, 1], "data_offsets": [0, 0]},
           "emb.weight": {"dtype": "BF16", "shape": [10, 64], "data_offsets": [0, 0]}}
    raw = json.dumps(hdr).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    n = I.params_estimate({"path": str(tmp_path), "weight_bytes": 10 ** 9, "quantization": {"bits": 4}})
    assert n == 64 * 64 + 10 * 64


def test_joint_plan_counts_logprob_store(monkeypatch):
    monkeypatch.setattr(P, "physical_gb", lambda: 137.0)
    s = I.settings("full", {})
    h = {**info(28.0, bits=8), "vocab": 262144}
    jp = I.joint_plan(info(33.8, bits=8), h, None, "", cpu=False, s=s, routes=("tile",))
    assert jp["logprob_store_gb"] > 5 and jp["need_gb"] > 33.8 + 28.0 + jp["logprob_store_gb"]


def test_joint_plan_sums_both_models(monkeypatch):
    monkeypatch.setattr(P, "physical_gb", lambda: 137.0)
    ok = I.joint_plan(info(33.8), info(28.0), None, "", cpu=False)
    assert ok["fits"] and ok["need_gb"] < 80 and "route:side" in ok["processes"]
    big = I.joint_plan(info(75.3), info(30.0), None, "", cpu=False)
    assert not big["fits"]
    with pytest.raises(SystemExit):
        I.joint_plan(info(1), info(1), 96, "", cpu=False)


def test_label_marks_small_models_as_smoke_tests():
    assert I.label_for(info(8.0), info(1.2)).startswith("smoke test")      # 4B donor / 0.6B host (bf16)
    assert I.label_for(info(33.8, bits=8), info(28.0, bits=8)) == ""        # 31B / 26B at 8 bits


class _B:
    def __init__(self, out, bundle, layers=6):
        self.out = out
        self.info = {"layers": layers}
        self.rec = {"install": {"bundle": str(bundle), "settings": I.settings("quick", {})}}

    def stage_dir(self, s):
        d = self.out / s
        d.mkdir(parents=True, exist_ok=True)
        return d


def _bundle(tmp_path, L=8, W=16):
    b = tmp_path / "bundle"
    (b / "core").mkdir(parents=True)
    core = np.zeros((L, W), np.uint8)
    core[:, :4] = 1
    box = np.zeros((L, W), np.uint8)
    box[2, 4:10] = 1
    box[5, 4:12] = 1
    box[6, 4:6] = 1
    box[0, 4:15] = 1           # layer 0 and the last layer never carry a bridge
    np.savez(b / "core/core.npz", core=core)
    np.savez(b / "box_mask.npz", box=box)
    return b


def test_topo_picks_interior_box_layers_and_warps(tmp_path):
    bundle = _bundle(tmp_path)
    outs, summ = I.run_topo(_B(tmp_path / "out", bundle, layers=6), None)
    topo = json.loads((tmp_path / "out/topo/topo.json").read_text())
    assert topo["box_layers"] == [2, 5]                      # 5 (8 units) and 2 (6): 6 is too close to 5
    assert topo["warp"] == {"2": 1, "5": 4}                  # round(l * 5 / 7)
    assert "relative depth" in topo["warp_method"]


@pytest.mark.parametrize("missing", [None, "control", "row", "drift"])
def test_report_verdict_needs_every_control(tmp_path, missing):
    out = tmp_path / "out"
    rd = out / "route_tile/run"
    rd.mkdir(parents=True)
    texts = [f"t{i}" for i in range(40)]
    rows = []
    rng = np.random.default_rng(0)
    for t in texts:
        h = 1.0 + rng.normal(0, 0.01)
        rows += [{"arm": "tile_box", "probe_id": t, "nats_per_byte": h - 0.1},
                 {"arm": "tile_init", "probe_id": t, "nats_per_byte": h},
                 {"arm": "tile_rand", "probe_id": t, "nats_per_byte": h - 0.1 + rng.normal(0, 0.01)}]
    if missing == "row":
        rows.pop()
    (rd / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = {"route": "tile", "teacher": "box", "teacher_gap": 0.2, "teacher_gap_ci95": [0.1, 0.3],
           "arms": {"tile_box": {"delta_vs_host": 0.1, "delta_vs_host_ci95": [0.09, 0.11], "gap_closed": 0.5,
                                 "gap_closed_ci95": [0.4, 0.6], "english_drift_test": 0.01,
                                 "controls": ["host", "tile_init", "tile_rand"]},
                    "tile_init": {}, "tile_rand": {}}}
    if missing == "control":
        del res["arms"]["tile_rand"]
    if missing == "drift":
        res["arms"]["tile_box"]["english_drift_test"] = None
    (rd / "result.json").write_text(json.dumps(res))
    for f in ("pair/pair.json", "topo/topo.json"):
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).write_text(json.dumps({"topic": "sci.biology"}))

    (out / "route_side").mkdir(parents=True)                   # a failed route: recorded, the report still runs
    (out / "route_side/result.json").write_text(json.dumps({"route": "side", "failed": "child rc=1"}))

    partial = out / "route_side/run"
    partial.mkdir()
    # Older children wrote result.json after the first arm, before side_rand ran.
    side = {"side_box": {**res["arms"]["tile_box"], "controls": ["host", "side_rand"]}}
    (partial / "result.json").write_text(json.dumps({**res, "arms": side}))
    (partial / "rows.jsonl").write_text("")

    class B(_B):
        def __init__(self):
            super().__init__(out, tmp_path)
            self.rec.update(label="", donor_inspection={"checkpoint": "d@1", "artifact": "d", "fingerprint": "1"},
                            joint_plan={"processes": {}}, bundle_quality={"verdict": "pass", "profile": "full"})
            self.rec["install"]["routes"] = ["side", "tile"]
            self.checkpoint = "h@2"
            self.info.update(artifact="h", fingerprint="2")
            self.cfg = {"profile": "quick"}

    class C:
        commit = {"commit": "x"}
    I.run_report(B(), C())
    vs = {v["route"]: v for v in map(json.loads, (out / "report/verdicts.jsonl").read_text().splitlines())}
    assert vs["side"]["verdict"] == "failed" and "rc=1" in vs["side"]["error"]
    assert not (out / "report/side").exists()
    v = vs["tile"]
    if missing:
        assert v["verdict"] == "failed" and not (out / "report/tile").exists()
        return
    assert v["beats"]["host"] and v["beats"]["tile_init"] and not v["beats"]["tile_rand"]
    assert v["verdict"] == "helps, not specific"             # no better than the same-count random teacher
    assert v["bundle"]["verdict"] == "pass"
    m = json.loads((out / "report/manifest.json").read_text())
    assert m["schema"] == I.INSTALL_SCHEMA and "verdicts.jsonl" in m["files"]
    assert m["host_checkpoint"] == "h@2" and m["donor_checkpoint"] == "d@1"


def test_label_carries_bundle_quality():
    man = {"donor_quality": {"verdict": "miss"}, "config": {"profile": "quick"}}
    lab = I.label_for(info(33.8, bits=8), info(28.0, bits=8), man)
    assert "quick-profile" in lab and "missed" in lab and "smoke test" not in lab


def test_params_count_six_bit_from_scales(tmp_path):
    import struct
    # 6-bit, group 64: 64 rows x 128 columns -> 128 * 6 / 32 = 24 packed columns, 2 scale groups
    hdr = {"a.weight": {"dtype": "U32", "shape": [64, 24], "data_offsets": [0, 0]},
           "a.scales": {"dtype": "BF16", "shape": [64, 2], "data_offsets": [0, 0]}}
    raw = json.dumps(hdr).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    n = I.params_estimate({"path": str(tmp_path), "weight_bytes": 1, "quantization": {"bits": 4, "group_size": 64}})
    assert n == 64 * 128                                       # right even though the global bits say 4


def test_install_kind_is_wired():
    assert set(ST.KINDS["install"]) == set(I.INSTALL_STAGES)
    assert set(I.INSTALL_RUNNERS) == set(I.INSTALL_STAGES)
    from mlx2.experimental.amoeba import cli
    assert I.MODEL_STAGES <= cli.MODEL_STAGES
    a = cli.parse(["install", "--box", "B", "--donor", "D", "--host", "H", "--out", "O", "--routes", "side,tile"])
    assert a.cmd == "install" and a.routes == "side,tile" and a.teacher == "box"
    assert cli.parse(["route_bridge", "--out", "O"]).cmd == "route_bridge"
