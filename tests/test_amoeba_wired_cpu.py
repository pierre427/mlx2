"""CPU tests of the weight wiring in research/amoeba/carve/ops.py (2026-11-19 19:19 FN battery stop: footprint 79 GiB,
resident 46 GiB, the compressor +27 GiB in 3 min, i.e. macOS compressing the battery's own model buffers). The sizing
is pure; wire_weights is driven with a fake mlx.core and fake kernel counters (no Metal, no GPU)."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.amoeba.carve import memwatch as MW  # noqa: E402
from research.amoeba.carve import ops as O  # noqa: E402

GIB = 2**30


def test_plan_sizes_from_active_and_caps():
    # FN on this machine: ~69.5 GiB active after load, iogpu.wired_limit_mb 114688 (112 GiB), recommended 112 GiB
    p = O.plan_wired_limit("auto", 69.5, 114688, 112.0, memory_gb=88)
    assert p["wired"] and p["limit_gib"] == 79.5 and p["weights_fit"] and not p["capped"]
    assert p["limit_bytes"] == int(79.5 * GIB) and p["caps"]["iogpu_wired_limit_gib"] == 112.0
    assert O.plan_wired_limit(80, 69.5, 114688, 112.0)["limit_gib"] == 80.0
    # capped 8 GiB under the smaller of iogpu.wired_limit_mb and the recommended working set
    p = O.plan_wired_limit(200, 69.5, 114688, 112.0)
    assert p["limit_gib"] == 104.0 and p["capped"]
    assert O.plan_wired_limit(200, 10, 20480, 112.0)["limit_gib"] == 12.0          # iogpu 20 GiB
    assert O.plan_wired_limit(200, 10, 0, 28.0)["limit_gib"] == 20.0                # iogpu unset (0): recommended
    assert O.plan_wired_limit("auto", 69.5, 114688, 112.0, memory_gb=76)["limit_gib"] == 76.0   # --memory-gb
    # under the weights: MLX wires only up to the limit, the rest stays pageable; said so
    p = O.plan_wired_limit(40, 69.5, 114688, 112.0)
    assert p["wired"] and not p["weights_fit"]
    for off in (0, "0", "off", None, ""):
        assert O.plan_wired_limit(off, 69.5, 114688, 112.0)["wired"] is False
    assert O.plan_wired_limit("auto", 1, 4096, 4.0)["wired"] is False               # cap below zero
    assert O.plan_wired_limit("auto", 1, 114688, None)["wired"] is False            # no Metal
    with pytest.raises(ValueError):
        O.plan_wired_limit("lots", 1, 0, 28.0)
    with pytest.raises(ValueError):
        O.plan_wired_limit(-1, 1, 0, 28.0)
    for bad in ("nan", "inf", "-inf"):
        with pytest.raises(ValueError):
            O.plan_wired_limit(bad, 1, 0, 28.0)
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        O.wired_arg("nan")
    assert O.wired_arg("auto") == "auto" and O.wired_arg("80") == "80"


def test_prepare_wiring_sets_the_refresh_env(monkeypatch):
    monkeypatch.setattr(O, "WIRED", {})
    monkeypatch.delenv("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", raising=False)
    monkeypatch.delenv("AMOEBA_RESIDENCY_REFRESH_MS", raising=False)
    assert O.prepare_wiring("0") == {"refresh_ms": None}
    assert "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS" not in __import__("os").environ
    assert O.prepare_wiring("auto") == {"refresh_ms": "1000"}
    monkeypatch.setenv("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "250")      # an existing value wins
    assert O.prepare_wiring("auto") == {"refresh_ms": "250"}
    monkeypatch.delenv("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS")
    monkeypatch.setenv("AMOEBA_RESIDENCY_REFRESH_MS", "0")
    assert O.prepare_wiring("auto") == {"refresh_ms": None}
    with pytest.raises(ValueError):
        O.prepare_wiring("lots")                       # a bad flag fails before the model loads


def test_wired_default(monkeypatch):
    monkeypatch.delenv("AMOEBA_WIRED_GB", raising=False)
    assert O.wired_default() == "auto" and O.wired_default("0") == "0"
    monkeypatch.setenv("AMOEBA_WIRED_GB", " 80 ")
    assert O.wired_default() == "80"


class FakeMx:
    """mlx.core as wire_weights uses it; wiring 'happens' at the first eval after set_wired_limit."""

    def __init__(self, active_gib, rec_gib, wired_box, wire_on_eval=True):
        self.limit, self.calls, self.box, self.wire_on_eval = 0, [], wired_box, wire_on_eval
        self.active = int(active_gib * GIB)
        self.metal = types.SimpleNamespace(is_available=lambda: True)
        self.rec = int(rec_gib * GIB)

    def device_info(self):
        return {"max_recommended_working_set_size": self.rec}

    def get_active_memory(self):
        return self.active

    def set_wired_limit(self, n):
        self.calls.append(("set_wired_limit", n))
        prev, self.limit = self.limit, n
        return prev

    def synchronize(self, *a):
        pass

    def clear_cache(self):
        self.calls.append(("clear_cache",))

    def zeros(self, shape):
        return 0

    def eval(self, *a):
        if self.wire_on_eval and self.limit:
            self.box[0] = self.box[1] + min(self.active, self.limit) / GIB


def _fake(monkeypatch, active_gib, wire_on_eval=True):
    box = [6.0, 6.0]                                   # machine wired GiB: now, before wiring
    mx = FakeMx(active_gib, 112.0, box, wire_on_eval)
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(MW, "vm_pages", lambda: {"wire_count": int(box[0] * GIB / MW._PAGE)})
    monkeypatch.setattr(MW, "_sysctl_int", lambda name, d: 114688 if name == "iogpu.wired_limit_mb" else d)
    exits = []
    monkeypatch.setattr("atexit.register", lambda f: exits.append(f))
    monkeypatch.setattr(O, "WIRED", {})
    monkeypatch.setattr(O, "WATCH", {})
    monkeypatch.setattr(O.time, "sleep", lambda s: None)
    return mx, exits


def test_wire_weights_sets_verifies_records_and_restores(monkeypatch):
    mx, exits = _fake(monkeypatch, 69.5)
    logs = []
    rec = O.wire_weights("auto", logs.append, memory_gb=88)
    assert mx.calls == [("clear_cache",), ("set_wired_limit", int(79.5 * GIB))]   # the cache freed first
    assert rec["verified"] and rec["machine_wired_delta_gib"] == 69.5 and rec["previous_wired_limit_bytes"] == 0
    assert rec["iogpu_wired_limit_mb"] == 114688 and O.WATCH["wired"] is rec and O.WATCH["wired_at_wire_gb"] == 75.5
    assert "verified" in logs[-1] and "NOT" not in logs[-1]
    assert not rec["refresh_prepared_before_metal"] and rec["refresh_ms"] is None   # prepare_wiring never ran
    assert O.wire_weights("auto", logs.append) is rec and len(mx.calls) == 2        # idempotent
    assert len(exits) == 1
    exits[0]()                                                                       # atexit: the previous limit
    assert mx.calls[-1] == ("set_wired_limit", 0)


def test_wire_weights_not_verified_when_nothing_wires(monkeypatch):
    mx, _ = _fake(monkeypatch, 69.5, wire_on_eval=False)
    t = [0.0]
    monkeypatch.setattr(O.time, "time", lambda: t.__setitem__(0, t[0] + 5.0) or t[0])
    logs = []
    rec = O.wire_weights(80, logs.append, settle_s=30)
    assert rec["wired"] and not rec["verified"] and rec["machine_wired_delta_gib"] == 0.0
    assert "NOT verified" in logs[-1]


def test_wire_weights_off_sets_nothing(monkeypatch):
    mx, exits = _fake(monkeypatch, 69.5)
    rec = O.wire_weights("0", lambda m: None)
    assert rec["wired"] is False and mx.calls == [] and exits == []


def test_refresh_recorded_only_when_prepared_first(monkeypatch):
    monkeypatch.delenv("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", raising=False)
    monkeypatch.delenv("AMOEBA_RESIDENCY_REFRESH_MS", raising=False)
    mx, _ = _fake(monkeypatch, 69.5)
    O.prepare_wiring("auto")
    rec = O.wire_weights("auto", lambda m: None)
    assert rec["refresh_prepared_before_metal"] and rec["refresh_ms"] == "1000" and rec["refresh_env_now"] == "1000"


def test_battery_refuses_a_bad_wired_value_at_parse(monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("battery_parse", ROOT / "scripts/amoeba_box_battery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "p", "--out", "o", "--held", "a=b",
                                      "--wired-gb", "nan"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 2 and "--wired-gb" in capsys.readouterr().err


def test_carve_cli_wires_after_the_host_and_records(tmp_path, monkeypatch):
    """The carve CLI: off by default (its own env, AMOEBA_CARVE_WIRED_GB; the battery's AMOEBA_WIRED_GB does not
    reach it), --cpu never wires, a given host (host_factory: loaded before main) skips the refresh env."""
    pytest.importorskip("mlx.core")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_amoeba_carve_cpu import Tok, toy
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve.host import Host
    calls = []
    monkeypatch.setattr(O, "prepare_wiring", lambda spec: calls.append(("prepare", spec)) or {})
    monkeypatch.setattr(O, "setup_mlx", lambda *a, **k: None)     # no process-wide MLX limits inside pytest
    monkeypatch.setattr(O, "wire_weights", lambda spec, log, memory_gb=None: calls.append(("wire", spec)) or
                        ({"wired": True, "limit_gib": 1.0} if spec not in ("0",) else {"wired": False}))
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    base = ["--model", "toy", "--smoke", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect"]
    monkeypatch.setenv("AMOEBA_WIRED_GB", "80")
    monkeypatch.delenv("AMOEBA_CARVE_WIRED_GB", raising=False)
    D.main(base + ["--cpu", "--out", str(tmp_path / "a")], host_factory=fac)
    assert calls == [("wire", "0")]                       # off by default, the battery's env ignored, no prepare
    assert "wired" not in __import__("json").loads((tmp_path / "a" / "carve.json").read_text())
    calls.clear()
    D.main(base + ["--cpu", "--wired-gb", "80", "--out", str(tmp_path / "b")], host_factory=fac)
    assert calls == [("wire", "0")]                       # --cpu never wires
    calls.clear()
    monkeypatch.setenv("AMOEBA_CARVE_WIRED_GB", "auto")
    D.main(base + ["--out", str(tmp_path / "c")], host_factory=fac)
    assert calls == [("wire", "auto")]
    assert __import__("json").loads((tmp_path / "c" / "carve.json").read_text())["wired"]["limit_gib"] == 1.0
