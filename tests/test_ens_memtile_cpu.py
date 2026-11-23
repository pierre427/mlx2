"""H41 memory tile toy (research/ens/memtile.py): world invariants, injection ceiling, bus-only writes. CPU only."""
from __future__ import annotations

import os
import random

os.environ.setdefault("ENS_DEVICE", "cpu")

import mlx.core as mx  # noqa: E402

from research.ens import families as F  # noqa: E402
from research.ens import memtile as MT  # noqa: E402
from research.ens import vocab as V  # noqa: E402
from research.ens.model import Core, CoreConfig  # noqa: E402


def test_world_links_have_unique_sources_and_ends_in_z_domain():
    rng = random.Random(0)
    for k in (1, 2, 3):
        chains = MT.make_world(rng, k, 16)
        srcs = [a for c in chains for a, _ in MT.links(c)]
        assert len(srcs) == len(set(srcs))
        assert all(V.V0 <= c[-1] < V.V0 + 16 and len(c) == k + 1 for c in chains)
        assert len({c[-1] for c in chains}) == len(chains)          # distinct ends: no cross-chain credit
        sessions, used = MT.make_sessions(rng, chains, 24, 0.5, 32, 4)
        facts = {}
        for ses in sessions:
            assert len(ses) <= 32
            for a, b in MT._facts(ses):
                facts.setdefault(a, set()).add(b)
        for c in chains:                                     # every chain link is stored and never shadowed
            for a, b in MT.links(c):
                assert facts[a] == {b}


def test_keyspace_projections_are_projections():
    mx.random.seed(0)
    core = Core(CoreConfig(d=64, layers=4, heads=4, ffn=128, bus_width=16))
    H = mx.random.normal((200, 64))
    ks = MT.KeySpace(core, "early", H, "endpoint", r=4)
    for P in (ks.P_end, ks.P_orth, ks.P_anch):
        assert mx.allclose(P @ P, P, atol=1e-4).item()
    assert abs(float(mx.trace(ks.P_end).item()) - 16) < 1e-3         # rank = bus width, not d


def test_symbolic_recall_end_to_end(tmp_path):
    """With an untrained core, the toy runs end to end and the symbolic arm retrieves exactly. (CORE-001 holds by
    construction: the tile writes only a bus message at the core's trained site.)"""
    from research.ens import common as C
    mx.random.seed(1)
    cfg = CoreConfig(d=64, layers=4, heads=4, ffn=128, bus_width=64)
    core = Core(cfg)
    d = tmp_path / "core"
    ckpt = C.save_core(core, C.fresh_out(d), "test-core")
    import json
    (d / "receipt.json").write_text(json.dumps({"checkpoint": ckpt, "config": cfg.to_json(), "site": "early",
                                                "families": F.registry_record()}))
    r = MT.run(MT_args(d, tmp_path / "out"))
    sym = [s for s in r["summary"] if s["arm"] == "symbolic"]
    assert sym and all(s["retrieval_used"] == 1.0 and s["retrieval_unused"] == 1.0 for s in sym)


def MT_args(core_dir, out):
    import argparse
    return argparse.Namespace(core=str(core_dir), family="program.ADD.h1", key_site="ingress", inject_site="early",
                              hops=[1, 2], worlds=1, sessions=24, distract=4, p_used=0.5, rank=8, learn=False,
                              learn_steps=10, seed=0, out=str(out))
