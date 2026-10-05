"""Default-dry-run, bounded native atomic Qwen3 request spot gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import signal
import urllib.request
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    "native/paged_kv/arena.cpp",
    "native/paged_kv/arena.h",
    "native/paged_kv/binding.cpp",
    "src/mlx2/runtime/paged_native_atomic_owner.py",
    "src/mlx2/runtime/qwen3_paged_native_backend.py",
    "src/mlx2/adapters/qwen3_paged_candidate.py",
    "src/mlx2/runtime/paged_kv_token.py",
    "src/mlx2/runtime/paged_attention_native.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in SOURCE}


def binary_identity() -> dict[str, str]:
    spec = importlib.util.find_spec("_paged_kv_native")
    if spec is None or spec.origin is None:
        raise RuntimeError("native binary is unavailable")
    path = Path(spec.origin).resolve()
    return {"path": str(path), "sha256": digest(path)}


def preflight(lease_id: str, session: str, frozen: Path) -> None:
    expected = json.loads(frozen.read_text())
    if (expected["source_sha256"] != source_hashes() or
            expected["native_binary"] != binary_identity()):
        raise RuntimeError("source or native binary drifted from CPU preflight")
    for path in (Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
                 Path("/tmp/gpu.lock/owner.json")):
        owner = json.loads(path.read_text())
        if owner.get("lease_id") != lease_id or owner.get("session") != session:
            raise RuntimeError(f"GPU lock ownership mismatch: {path}")
    with urllib.request.urlopen("http://127.0.0.1:8600/health", timeout=2) as response:
        music = json.load(response)
    if music.get("busy") is not False or music.get("loaded") is not False:
        raise RuntimeError("Music3 is busy or loaded")


def cell() -> dict:
    import mlx.core as mx
    from mlx2.adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
    from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
    from mlx2.runtime.paged_pack_scheduler import PagedPackDecision, ReservedRows
    from mlx2.runtime.paged_request_transaction import STATE_PLANES, CandidateRequest, execute_paged_request
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

    if not mx.metal.is_available():
        raise RuntimeError("M5 GPU required")
    mx.set_default_device(mx.gpu)

    class SyntheticQwen3:
        args = SimpleNamespace(model_type="qwen3", num_experts=0, rope_scaling=None,
                               head_dim=128, num_attention_heads=4, num_key_value_heads=2)
        layers = (object(), object())

        def paged_embed(self, tokens):
            return mx.array(tokens, dtype=mx.float16) / 16

        def paged_project(self, layer, hidden, counts, offsets):
            rows = hidden.shape[0]
            base = hidden[:, None, None]
            return (mx.broadcast_to(base / 8, (rows, 4, 128)),
                    mx.broadcast_to((base + layer) / 4, (rows, 2, 128)),
                    mx.broadcast_to((base + layer + 1) / 2, (rows, 2, 128)))

        def paged_finish_layer(self, layer, hidden, attended):
            return hidden + mx.mean(attended, axis=(1, 2))

        def paged_logits(self, hidden):
            return hidden

    profile = TokenKVProfile(2, 128, "float16")
    pool = PagedKVPool(16)
    native = NativeWriteBackend(pool.capacity * profile.page_bytes,
                                mx.default_stream(mx.gpu), permit_candidate=True)
    writer = PagedKVWriteOwner(pool, native, page_bytes=profile.page_bytes,
                               permit_candidate=True)
    backend = NativeQwen3PagedBackend(writer, permit_candidate=True)
    model = Qwen3PackedCandidate(SyntheticQwen3(), backend)
    public = tuple(PagedKVTokenOwner(writer, profile, permit_candidate=True) for _ in range(2))
    owner = NativeAtomicRequestOwner("b3047184-synthetic", public,
                                     {p: ("base",) for p in STATE_PLANES if p != "kv"},
                                     enabled=True)
    rounds = []
    for generation, proposed, accepted in ((1, 63, 63), (2, 3, 1)):
        before = owner.snapshot()
        decision = PagedPackDecision(True, "spot", (ReservedRows(1, "verify", proposed, 2, 1000.0),),
                                     None, 0, 1.0, "synthetic-spot")
        request = CandidateRequest(1, "b3047184-synthetic", proposed, STATE_PLANES)
        intermediate = {}

        def run(branch):
            assert all(a is not b for a, b in zip(branch.layers, owner._public.layers))
            tokens = tuple(range(before.offset + 1, before.offset + proposed + 1))
            output, route = model.forward((PackedLane(tokens, branch.layers),),
                                          permit_candidate=True, atomic_branch=branch)
            mx.eval(output)
            assert branch._proved == {0, 1} and route["simulated_or_backend_reads"] == 2
            assert owner.snapshot() == before
            intermediate["proposed_tables"] = [list(map(str, x.accepted_handles())) for x in branch.layers]
            for plane in STATE_PLANES[1:]:
                branch.stage(plane, [f"g{generation}:{plane}:{i}" for i in range(proposed)])
            return accepted

        receipt = execute_paged_request(decision, request, owner, run, permit_candidate=True)
        after = owner.snapshot()
        assert receipt.published and receipt.accepted_rows == accepted
        assert after.generation == generation and after.offset == before.offset + accepted
        assert all(len(rows) == 1 + after.offset for _, rows in after.companions)
        assert writer.ledger.pending_count == 0 and not writer.pending_epochs
        if generation == 2:
            assert all(len(table) == 1 for table in after.layer_tables)
            assert all(len(table) == 2 for table in intermediate["proposed_tables"])
            assert all(str(table[0]) != intermediate["proposed_tables"][i][1]
                       for i, table in enumerate(after.layer_tables))
        rounds.append({"generation": generation, "proposed": proposed, "accepted": accepted,
                       "offset": after.offset, "layer_tables": [list(map(str, t)) for t in after.layer_tables],
                       "proposed_tables": intermediate["proposed_tables"],
                       "terminal_layer_proofs": 2, "request_receipt": receipt.__dict__})
    mx.synchronize(native.stream)
    for state in (*owner._retired, owner._public):
        for layer in state.layers:
            layer.close()
    pool.retire(writer.ledger.completed_epoch)
    assert pool.free_count == pool.capacity
    import _paged_kv_native as binary
    return {"schema": "mlx2.varlen-atomic-qwen3-m5-spot.v1", "gpu_executed": True,
            "source_revision": "b3047184", "source_sha256": source_hashes(),
            "native_binary": str(Path(binary.__file__).resolve()),
            "native_binary_sha256": digest(Path(binary.__file__)),
            "rounds": rounds, "all_pages_retired": True, "pool_pages": pool.capacity,
            "qualification": False, "serving_selected": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--lease-id")
    parser.add_argument("--session")
    args = parser.parse_args()
    if args.execute_gpu:
        signal.alarm(290)
        if not args.preflight or not args.lease_id or not args.session:
            parser.error("GPU execution requires preflight, lease ID, and session")
        try:
            preflight(args.lease_id, args.session, args.preflight)
            result = cell()
        except BaseException as error:
            if args.receipt:
                args.receipt.parent.mkdir(parents=True, exist_ok=True)
                args.receipt.write_text(json.dumps({
                    "schema": "mlx2.varlen-atomic-qwen3-m5-spot.v1",
                    "gpu_executed": None, "success": False,
                    "source_revision": "b3047184", "source_sha256": source_hashes(),
                    "native_binary": binary_identity(),
                    "error": f"{type(error).__name__}: {error}",
                }, indent=2) + "\n")
            raise
    else:
        result = {"schema": "mlx2.varlen-atomic-qwen3-m5-spot.v1", "gpu_executed": False,
                  "source_revision": "b3047184", "source_sha256": source_hashes(),
                  "native_binary": binary_identity(),
                  "planned": "2 layers; 63/63 full then 1/3 partial over page boundary; 16 pages",
                  "qualification": False, "serving_selected": False}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
