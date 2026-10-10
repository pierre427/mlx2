"""Run bounded synthetic executor state tests on an explicitly owned GPU."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Synthetic executor state smoke; requires owned GPU locks."
)
parser.add_argument("--i-own-the-gpu", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if not args.i_own_the_gpu:
    parser.error("explicit GPU ownership required")
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "tests")]
import mlx.core as mx
import pytest
import test_speculative_decode_first_cpu as suite

mx.set_default_device(mx.gpu)
passed = []
for kind in ("pld", "external"):
    for maximum in (1, 17):
        suite.test_single_lane_tokens_rng_and_terminal_state_match_reference(
            kind, maximum
        )
        passed.append(f"{kind}:terminal-{maximum}")
    suite.test_uneven_lane_completion_and_cancel_leave_peer_outputs_and_state_exact(
        kind
    )
    passed.append(f"{kind}:uneven-completion-cancel")
    for fn in (
        suite.test_publishes_before_prefill_and_rechecks_cancelled_candidates,
        suite.test_new_arrival_does_not_join_a_suspended_prefill_snapshot,
    ):
        with pytest.MonkeyPatch.context() as patch:
            fn(kind, patch)
        passed.append(f"{kind}:{fn.__name__}")
    for kill in (False, True):
        with pytest.MonkeyPatch.context() as patch:
            suite.test_pending_phase_kill_switch_and_budget(kind, kill, patch)
        passed.append(f"{kind}:kill-{kill}")
with pytest.MonkeyPatch.context() as patch:
    suite.test_external_one_token_drain_stays_one_per_lane_per_poll(patch)
passed.append("external:one-token-drain")
suite.test_external_terminal_boundary_from_resumed_prefill_is_not_lost()
passed.append("external:terminal-resumed-boundary")
suite.test_external_packed_prefill_respects_shared_padded_token_budget()
passed.append("external:packed-budget")
with pytest.MonkeyPatch.context() as patch:
    suite.test_external_memory_stall_survives_the_publication_boundary(patch)
passed.append("external:memory-stall")
mx.synchronize()
proof = {
    "schema": "mlx2.architecture-metal-smoke.v1",
    "device": str(mx.default_device()),
    "scope": "synthetic tiny hybrid PLD and DFlash2 models; no artifact route qualification",
    "passed": passed,
    "source_sha256": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            root / "src/mlx2/runtime/round_phases.py",
            root / "src/mlx2/runtime/pld.py",
            root / "src/mlx2/runtime/external_speculative.py",
            root / "tests/test_speculative_decode_first_cpu.py",
        ]
    },
}
args.output.write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof, indent=2))
