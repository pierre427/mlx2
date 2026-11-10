#!/usr/bin/env python3
"""Tier cost model for DS4.1 Source routed-expert fetches (CPU research; no runtime changes).

Every number carries a provenance tag:
  measured-capture  derived from the 2026-11-08 routing-capture run (counters + per-layer timestamps;
                    see calibration.json), i.e. the M5 serving with 16 read workers, no new benchmark
  measured-probe    light probe run for this study (ping / 1 GiB TCP / F_NOCACHE preads on M3/M2)
  measured-prior    measured in earlier lab work (Kimi K3.1 split README, 2026-11-07)
  assumed           not measured; swept in the simulations

Fetch model (per expert, 18,800,640 B): a tier is a serial pipe whose aggregate service time per expert
is  lat_ms + bytes / bw  (aggregate bandwidth of all readers; per-request latency overlaps across the
reader pool, so it is charged once per expert at the small effective value).  After the bytes land a
parse/copy/upload step costs  parse_ms  on a host worker that overlaps compute ("low"), or, with
parse="high", additionally steals  gpu_steal_ms  from the compute timeline for EVERY loaded expert
(demand or prefetch) -- the K3.1 lesson that MLX parse/copy, not disk, dominated expert I/O.
"""
import copy

from common import EXPERT_BYTES, OUT, atomic_json, provenance

# ---- probes (2026-11-08, this study) ----
PROBES = {
    "m3_tb_bridge_rtt_ms": {"value": 0.354, "how": "ping -c40 over the Thunderbolt Bridge link, min/avg/max 0.327/0.354/0.426"},
    "m3_tb_tcp_GBps": {"value": 1.01, "how": "1 GiB single TCP stream M5->M3 over bridge0 (python socket, nc sink)"},
    "m2_rtt_ms": {"value": 0.651, "how": "ping -c40 over a USB Ethernet adapter link (mtu 1500)"},
    "m2_tcp_GBps": {"value": 0.0366, "how": "1 GiB single TCP stream M5->M2 over en8: 36.6 MB/s -> 514 ms per expert; the M2 is NOT on the TB bridge"},
    "m3_disk_GBps_1thr": {"value": 3.58, "how": "F_NOCACHE 18.8 MB preads of an existing safetensors, 1 thread (M3 disk is 100% full: 2.6 GiB free, cannot hold experts)"},
    "m3_disk_GBps_4thr": {"value": 6.62, "how": "same, 4 threads"},
    "m2_disk_GBps_1thr": {"value": 3.12, "how": "F_NOCACHE 18.8 MB preads, 1 thread (M2 has 386 GiB free)"},
    "m2_disk_GBps_4thr": {"value": 5.52, "how": "same, 4 threads"},
    "m3_state": {"value": "busy", "how": "Sage server + mlx2 fabric running, load 1.4, swap 8.0/8.5 GiB used"},
    "m2_state": {"value": "idle-ish", "how": "mlx2 fabric running, load 2.2, 81% memory free, swap 3.6/4 GiB"},
}

# ---- default cost table (ms, GB/s) ----
DEFAULT = {
    "compute_ms_per_layer": {"value": 12.0, "src": "measured-capture (derived)",
                             "note": "capture per-layer delta with 0 misses = 17.2 ms incl. capture sync; 1.55 tok/s "
                                     "uninstrumented serving implies ~12 ms; swept 6/10/12/15"},
    "router_frac": {"value": 0.5, "src": "assumed", "note": "fraction of a layer before its router decision"},
    "tiers": {
        "ram":  {"lat_ms": 0.0, "bw_GBps": None, "src": "-"},
        "nvme": {"lat_ms": 0.0, "bw_GBps": 10.5, "src": "measured-capture",
                 "note": "regression slope 1.80 ms per extra demand miss in a layer (16 workers) = 10.5 GB/s "
                         "effective incl. read+copy; K3.1 prior: 10-13 GB/s with 8 readers"},
        "t7":   {"lat_ms": 1.0, "bw_GBps": 0.045, "src": "measured-prior",
                 "note": "T7 exFAT/FSKit, 96% full, shared: 35-50 MB/s (K3.1 README); nominal ~0.9 GB/s swept"},
        "m3ram": {"lat_ms": 0.354, "bw_GBps": 1.01, "src": "measured-probe",
                  "note": "byte tier over TB bridge: 18.6 ms/expert + RTT; busy-node RTT 7-25 ms (prior)"},
        "m2ram": {"lat_ms": 0.651, "bw_GBps": 0.0366, "src": "measured-probe", "note": "not on TB: 514 ms/expert"},
    },
    "parse": {"mode": "low", "low": {"gpu_steal_ms": 0.0}, "high": {"gpu_steal_ms": 0.8},
              "src": "assumed", "note": "high = 0.8 ms of compute timeline per loaded expert (mx.array copy + eval "
                                         "on the GPU queue); demand-path copy is already inside the 1.8 ms slope"},
    "remote": {
        "m3": {"rtt_ms_idle": 0.354, "rtt_ms_busy": 15.0, "compute_factor": 3.5, "expert_capacity": int(26e9 // EXPERT_BYTES),
               "src": "rtt measured-probe (idle) / measured-prior 7-25 ms (busy); compute_factor assumed "
                      "(M3 Pro 150 GB/s vs M5 Max ~550 GB/s, bandwidth-bound decode)"},
        "m2": {"rtt_ms_idle": 0.651, "rtt_ms_busy": 15.0, "compute_factor": 5.0, "expert_capacity": int(16e9 // EXPERT_BYTES),
               "link_GBps": 0.0366,
               "src": "rtt/link measured-probe; compute_factor assumed (M2 100 GB/s)"},
        "hidden_bytes": 4 * 5120 * 2,
        "hidden_src": "assumed: 4 hyper-connection streams x 5120 x bf16 cross a stage boundary",
    },
    "nvme_capacity_experts": {"value": 15360, "src": "assumed default (everything on NVMe); M5 internal has ~98 GiB "
                                                     "free today, so the T7-subset case is the realistic one"},
}

SWEEP = {
    "ram_gb": [40, 60, 75, 90],
    "compute_ms_per_layer": [6, 10, 15],
    "parse": ["low", "high"],
    "t7_subset": [False, True],
    "t7_bw_GBps": [0.045, 0.4],
    "remote": ["idle", "busy"],
}


def expert_ms(tier, cfg=DEFAULT):
    t = cfg["tiers"][tier]
    if t["bw_GBps"] is None:
        return 0.0
    return t["lat_ms"] + EXPERT_BYTES / (t["bw_GBps"] * 1e9) * 1e3


def make(**over):
    """Copy of DEFAULT with flat overrides: compute, parse, t7_bw, nvme_bw, rtt_busy, m3_factor..."""
    c = copy.deepcopy(DEFAULT)
    if "compute" in over:
        c["compute_ms_per_layer"]["value"] = float(over["compute"])
    if "parse" in over:
        c["parse"]["mode"] = over["parse"]
    if "t7_bw" in over:
        c["tiers"]["t7"]["bw_GBps"] = float(over["t7_bw"])
    if "nvme_bw" in over:
        c["tiers"]["nvme"]["bw_GBps"] = float(over["nvme_bw"])
    if "router_frac" in over:
        c["router_frac"]["value"] = float(over["router_frac"])
    if "m3_factor" in over:
        c["remote"]["m3"]["compute_factor"] = float(over["m3_factor"])
    if "m2_factor" in over:
        c["remote"]["m2"]["compute_factor"] = float(over["m2_factor"])
    return c


def main():
    rows = {k: {"ms_per_expert": round(expert_ms(k), 3), **v} for k, v in DEFAULT["tiers"].items()}
    out = provenance(sweep=SWEEP)
    out.update({"probes": PROBES, "default": DEFAULT, "ms_per_expert": rows})
    atomic_json(f"{OUT}/costs.json", out)
    for k, v in rows.items():
        print(f"{k:6s} {v['ms_per_expert']:8.2f} ms/expert  [{v['src']}]")


if __name__ == "__main__":
    main()
