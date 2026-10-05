#!/usr/bin/env python3
"""Bounded weighted-model N20 ragged verification parity gate.

This gate uses three frozen SPoMIN inputs from one domain, the source-bound
Qwen3.8 27B adapter and the native N20 arena.  The controlled arm constructs
proposal rows from an independent ordinary target oracle.  The prompt-lookup
arm selects the real deterministic proposal provider and derives the stock
ordinary reference from the provider's exact rows and acceptance decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_SECONDS = 120
MAX_RSS = 48 << 30
FAILURE_ROOTS = []


def save(path, value):
    from varlen_hybrid_serving_smoke import save as atomic_save

    atomic_save(path, value)


def plan(proposal_source):
    return {
        "schema": "mlx2.native-n20-weighted-ragged-gate.v1",
        "status": "planned",
        "gpu_executed": False,
        "implemented": True,
        "qualified": False,
        "selected_by_default": False,
        "performance_claim": False,
        "route_observed_used": False,
        "prompt_lookup_source_observed_used": False,
        "proposal_source_requested": proposal_source,
        "target_reference": "stock_merged_ordinary_dynamic_shrink",
        "hard_seconds": MAX_SECONDS,
        "max_rss_bytes": MAX_RSS,
    }


def natural_reference(model, references, anchors, proposals, mx):
    """Run the stock merged target under the proposal's actual shrink law."""
    from mlx2.runtime.generate import _merge_caches

    if (
        type(proposals) is not tuple
        or len(proposals) != len(anchors)
        or any(type(row) is not tuple or not row or row[0] != anchor
               for row, anchor in zip(proposals, anchors))
    ):
        raise ValueError("natural proposal rows must begin with every target anchor")
    reference_batch = _merge_caches([list(row) for row in references])
    target_logits = [[] for _ in proposals]
    target_tokens = [[] for _ in proposals]
    target_boundaries = [None for _ in proposals]
    active = tuple(range(len(proposals)))
    next_tokens = list(anchors)
    round_widths = []
    step = 0
    while active:
        logits, sampled = advance_reference_round(
            model, reference_batch,
            tuple(next_tokens[index] for index in active), mx)
        round_widths.append(len(active))
        following = []
        for row, (index, token) in enumerate(zip(active, sampled)):
            target_logits[index].append(logits[row])
            target_tokens[index].append(token)
            next_tokens[index] = token
            if (step + 1 < len(proposals[index]) and
                    token == proposals[index][step + 1]):
                following.append(index)
        following = tuple(following)
        for lane, boundary in retire_reference_lanes(
                reference_batch, active, following, mx).items():
            target_boundaries[lane] = boundary
        active = following
        step += 1
    if any(boundary is None for boundary in target_boundaries):
        raise RuntimeError("ordinary natural-proposal reference left a lane live")
    return (
        tuple(tuple(rows) for rows in target_logits),
        tuple(tuple(tokens) for tokens in target_tokens),
        tuple(target_boundaries),
        tuple(round_widths),
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_metric(actual, expected, mx):
    from varlen_hybrid_b1_numeric_gate import tensor_metrics

    return tensor_metrics(actual, expected, mx)


def reference_batch_size(cache):
    """Return the host-visible row count for ordinary recurrent or KV caches."""
    batch = getattr(cache, "batch_size", None)
    if isinstance(batch, int):
        return batch
    offset = getattr(cache, "offset", None)
    shape = getattr(offset, "shape", ())
    if len(shape) == 1:
        return int(shape[0])
    return None


def advance_reference_round(model, caches, tokens, mx):
    """Advance one stock merged ordinary batch at the native round's width.

    ``mixed_forward`` is not this control: it shares projections but runs GDN
    separately for every segment.  The N20 candidate runs one stock batched
    GDN call, so the parity oracle must use the ordinary model's actual merged
    batch and shrink that batch between rounds.
    """
    if (
        type(caches) is not list
        or not caches
        or type(tokens) is not tuple
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
        or any(reference_batch_size(cache) != len(tokens) for cache in caches)
    ):
        raise ValueError("one token per row of a stock merged batch is required")
    hidden = model.model(mx.array([[token] for token in tokens]), cache=caches)
    if hidden.shape[:2] != (len(tokens), 1):
        raise RuntimeError("ordinary ragged batch reference geometry differs")
    logits = model.logits(hidden)[:, 0, :]
    roots = tuple(
        value
        for cache in caches
        for value in cache.state
        if value is not None
    )
    mx.eval(logits, *roots)
    if not bool(mx.all(mx.isfinite(logits)).item()):
        raise ValueError("ordinary ragged target reference is nonfinite")
    return logits, tuple(int(token.item()) for token in mx.argmax(logits, axis=-1))


def retire_reference_lanes(caches, active, following, mx):
    """Snapshot retiring rows, then filter the stock batch for the next round."""
    if (
        type(caches) is not list
        or not caches
        or type(active) is not tuple
        or not active
        or type(following) is not tuple
        or any(type(index) is not int or index < 0 for index in (*active, *following))
        or len(set(active)) != len(active)
        or len(set(following)) != len(following)
        or any(index not in active for index in following)
        or any(reference_batch_size(cache) != len(active) for cache in caches)
    ):
        raise ValueError("a distinct active batch and ordered surviving subset are required")
    following_set = set(following)
    retired = {}
    for row, lane in enumerate(active):
        if lane not in following_set:
            retired[lane] = tuple(cache.extract(row) for cache in caches)
    keep = [row for row, lane in enumerate(active) if lane in following_set]
    if following:
        for cache in caches:
            cache.filter(keep)
        roots = tuple(
            value for cache in caches for value in cache.state if value is not None
        )
        mx.eval(*roots)
    return retired


def controlled_proposals(anchors, target_tokens, vocab_size):
    """Build full/partial/reject rows without depending on draft luck."""
    if (
        type(anchors) is not tuple
        or len(anchors) != 3
        or any(type(token) is not int or token < 0 for token in anchors)
        or type(target_tokens) is not tuple
        or tuple(map(len, target_tokens)) != (3, 2, 1)
        or any(
            type(token) is not int or token < 0
            for lane in target_tokens
            for token in lane
        )
        or type(vocab_size) is not int
        or vocab_size < 2
        or any(token >= vocab_size for token in (*anchors, *sum(target_tokens, ())))
    ):
        raise ValueError("three bounded target paths with depths 3,2,1 are required")

    def wrong(token):
        return (token + 1) % vocab_size

    return (
        (anchors[0], target_tokens[0][0], target_tokens[0][1]),
        (anchors[1], target_tokens[1][0], wrong(target_tokens[1][1])),
        (anchors[2], wrong(target_tokens[2][0]), wrong(wrong(target_tokens[2][0]))),
    )


def compare_public_state(owners, candidate, references, offsets, mx):
    from varlen_hybrid_fa_boundary_probe import _export_logical_native_kv

    recurrent = []
    kv = []
    public = []
    with ExitStack() as stack:
        views = tuple(stack.enter_context(owner.snapshot()) for owner in owners)
        for lane, (view, caches, offset) in enumerate(zip(views, references, offsets)):
            checkpoint = dict(view.companions)["gdn"][0]
            public.append(
                {
                    "lane": lane,
                    "offset": view.offset,
                    "generation": view.generation,
                    "gdn_offset": checkpoint.offset,
                    "gdn_generation": checkpoint.generation,
                }
            )
            if (
                view.offset != offset
                or checkpoint.offset != offset
                or view.generation != 1
                or checkpoint.generation != 1
            ):
                raise RuntimeError("weighted ragged public boundary differs")
            for ordinal, layer in enumerate(candidate.layer_map.recurrent):
                for slot in (0, 1):
                    recurrent.append(
                        {
                            "lane": lane,
                            "layer": layer,
                            "slot": slot,
                            **tensor_metric(
                                checkpoint.caches[ordinal].cache[slot],
                                caches[layer].cache[slot],
                                mx,
                            ),
                        }
                    )
            for ordinal, layer in enumerate(candidate.layer_map.full_attention):
                keys, values = _export_logical_native_kv(view.layer_owners[ordinal], mx)
                ref_keys, ref_values = caches[layer].keys_and_values()
                for plane, actual, expected in (
                    ("key", keys, ref_keys[0]),
                    ("value", values, ref_values[0]),
                ):
                    kv.append(
                        {
                            "lane": lane,
                            "layer": layer,
                            "plane": plane,
                            **tensor_metric(actual, expected, mx),
                        }
                    )
    return {
        "public": public,
        "recurrent": recurrent,
        "kv": kv,
        "passed": all(item["finite"] and item["exact"] for item in (*recurrent, *kv)),
    }


def execute(args, result):
    from varlen_pack_price_bench import _gpuq_owner

    result["gpuq_owner"] = _gpuq_owner()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.source_commit or dirty:
        raise RuntimeError("weighted ragged gate requires exact clean source")
    if not args.native.is_file() or sha256(args.native) != args.native_sha256:
        raise RuntimeError("weighted ragged gate native identity differs")

    sys.path[:0] = [str(args.native.parent), str(ROOT / "src")]
    from spomin_400case_native_suite import validate_inputs
    from varlen_hybrid_packed_prefill_n2_geometry_gate import (
        make_profile,
        stock_mixed_reference,
    )

    from mlx2.runtime.paged_price_identity import cached_live_price_identity

    identity = cached_live_price_identity(
        args.artifact_manifest,
        args.mlx_wheel,
        args.native,
        adapter_artifact_root=args.model.resolve(),
    )
    inputs = validate_inputs(json.loads(args.inputs.read_text()))
    rows = tuple(inputs["rows"][:3])
    if len({row["domain"] for row in rows}) != 1:
        raise ValueError("weighted ragged gate requires one frozen domain")
    token_rows = tuple(tuple(row["prompt_token_ids"]) for row in rows)
    source_ids = tuple(row["case_id"] for row in rows)
    counts = tuple(map(len, token_rows))
    result.update(
        source_commit=head,
        native_path=str(args.native),
        native_sha256=args.native_sha256,
        identity=identity,
        inputs_sha256=inputs["inputs_sha256"],
        source_input_ids=list(source_ids),
        prompt_lengths=list(counts),
    )
    proposed = make_profile(identity, inputs)
    if args.prepare_profile:
        save(args.profile, proposed)
        result["status"] = "profile_prepared"
        return

    os.environ.update(proposed["required_environment"])
    os.environ.update(
        MLX2_PAGED_N20_RAGGED_PROMPT_LOOKUP=(
            "1" if args.proposal_source == "prompt_lookup" else "0"),
        MLX2_PAGED_N20_RAGGED_DEPTH="2",
        MLX2_PAGED_N20_RAGGED_NGRAM_MIN=str(args.prompt_ngram_min),
        MLX2_PAGED_N20_RAGGED_NGRAM_MAX=str(args.prompt_ngram_max),
    )
    from mlx2.runtime.hybrid_packed_prefill_n import (
        create_cold_packed_hybrid_n,
        load_profile,
    )

    profile = load_profile(
        args.profile,
        live_identity=identity,
        source_input_ids=source_ids,
        counts=counts,
        tokens=token_rows,
        environment_values=os.environ,
    )
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, configure_environment

    configure_environment()
    import mlx.core as mx

    from mlx2.runtime import qwen35_paged_graph_factory as resources_module
    from mlx2.runtime.generate import StopSequenceMatcher
    from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation
    from mlx2.runtime.paged_native_graph_n import (
        _prompt_lookup_token_rows,
        run_native_graph_n,
        run_native_ragged_verify,
    )

    adapter = candidate = None
    owners = ()
    initial_charge = resources_module._CHARGED
    result.update(status="running", gpu_executed=True)
    try:
        result["phase"] = "load_model"
        began = time.perf_counter()
        adapter = Qwen3827BAdapter(str(args.model), require_mtp=False)
        result["model_load_seconds"] = time.perf_counter() - began
        loaded_identity = cached_live_price_identity(
            args.artifact_manifest,
            args.mlx_wheel,
            args.native,
            adapter_artifact_root=Path(adapter.identity["path"]).resolve(),
        )
        if loaded_identity != identity:
            raise RuntimeError("weighted ragged loaded identity differs")
        model = getattr(adapter.model, "language_model", adapter.model)

        result["phase"] = "ordinary_mixed_prefill_reference"
        references, initial_logits = stock_mixed_reference(model, token_rows, mx)

        requests = tuple(
            (lane, adapter.identity["fingerprint"], tokens, 8)
            for lane, tokens in enumerate(token_rows)
        )
        boundaries = []

        def phase_boundary(event):
            if (
                event.get("public_state_published") is not False
                or event.get("materialized") is not True
                or event.get("native_terminals_drained") is not True
            ):
                raise RuntimeError("weighted ragged private phase lacks terminal proof")
            boundaries.append(dict(event))

        result["phase"] = "native_packed_prefill"
        owners, candidate, boots = create_cold_packed_hybrid_n(
            adapter,
            requests,
            profile=profile,
            live_identity=identity,
            source_input_ids=source_ids,
            permit_candidate=True,
            phase_boundary=phase_boundary,
        )
        if len(boundaries) != 64:
            raise RuntimeError("weighted ragged prefill lacks all 64 layer phases")
        result["private_phase_count"] = len(boundaries)
        result["packed_prefill_receipt"] = candidate._packed_prefill_receipt
        bootstrap = []
        for lane, boot in enumerate(boots):
            bootstrap.append(tensor_metric(boot.logits[0], initial_logits[lane], mx))
        result["bootstrap_logits"] = bootstrap
        if not all(item["finite"] and item["exact"] for item in bootstrap):
            raise RuntimeError("weighted ragged bootstrap logits differ")

        sampler_calls = []

        def sampler(logprobs):
            sampled = mx.argmax(logprobs, axis=-1)
            sampler_calls.append(sampled)
            return sampled

        continuations = tuple(
            NativeQwen3Continuation(
                uid=lane,
                revision=adapter.identity["fingerprint"],
                prompt_tokens=tokens,
                first_logits=boot.logits[0],
                owner=owner,
                candidate=candidate,
                maximum=8,
                sampler=sampler,
                processors=[],
                matcher=StopSequenceMatcher([]),
            )
            for lane, (tokens, boot, owner) in enumerate(zip(token_rows, boots, owners))
        )

        result["phase"] = "native_priming"
        priming = run_native_graph_n(continuations)
        anchors = tuple(response.token for response in priming)
        if anchors != tuple(int(mx.argmax(logits).item()) for logits in initial_logits):
            raise RuntimeError("weighted ragged priming token parity differs")

        if args.proposal_source == "controlled":
            from mlx2.runtime.generate import _merge_caches

            target_logits = [[], [], []]
            target_tokens = [[], [], []]
            target_boundaries = [None, None, None]
            next_tokens = list(anchors)
            reference_batch = _merge_caches([list(row) for row in references])
            rounds = (
                ((0, 1, 2), (0, 1)),
                ((0, 1), (0,)),
                ((0,), ()),
            )
            for active, following in rounds:
                logits, sampled = advance_reference_round(
                    model, reference_batch,
                    tuple(next_tokens[index] for index in active), mx)
                for row, (index, token) in enumerate(zip(active, sampled)):
                    target_logits[index].append(logits[row])
                    target_tokens[index].append(token)
                    next_tokens[index] = token
                for lane, boundary in retire_reference_lanes(
                        reference_batch, active, following, mx).items():
                    target_boundaries[lane] = boundary
            if any(boundary is None for boundary in target_boundaries):
                raise RuntimeError("ordinary controlled reference left a lane live")
            target_logits = tuple(tuple(rows) for rows in target_logits)
            target_tokens = tuple(tuple(tokens) for tokens in target_tokens)
            target_boundaries = tuple(target_boundaries)
            proposals = controlled_proposals(
                anchors, target_tokens, candidate.args.vocab_size)
            proposal_receipts = tuple({
                "source": "controlled_ordinary_target_oracle",
                "proposed": 2,
                "proposal_distribution": "deterministic_point_mass",
            } for _ in rows)
            expected_round_widths = (3, 2, 1)
        else:
            proposals, proposal_receipts = _prompt_lookup_token_rows(
                continuations, candidate)
            if not any(len(row) > 1 for row in proposals):
                raise RuntimeError("real prompt-lookup source produced no K>0 proposal")
            (target_logits, target_tokens, target_boundaries,
             expected_round_widths) = natural_reference(
                model, references, anchors, proposals, mx)
        executed = tuple(map(len, target_tokens))
        result.update(
            declared_query_lengths=list(map(len, proposals)),
            proposal_rows=[list(row) for row in proposals],
            proposal_receipts=[dict(receipt) for receipt in proposal_receipts],
            expected_executed_query_lengths=list(executed),
            expected_round_widths=list(expected_round_widths),
        )

        captured = {}
        original = candidate.forward_staged_ragged

        def capture(*call_args, **call_kwargs):
            call_kwargs["collect_logits"] = True
            logits, proof = original(*call_args, **call_kwargs)
            captured.update(logits=logits, proof=proof)
            return logits, proof

        candidate.forward_staged_ragged = capture
        result["phase"] = "native_weighted_ragged_verify"
        try:
            if args.proposal_source == "prompt_lookup":
                responses = run_native_graph_n(continuations)
            else:
                responses = run_native_ragged_verify(
                    continuations, proposals, proposal_receipts=proposal_receipts)
        finally:
            candidate.forward_staged_ragged = original

        proof = captured["proof"]
        if tuple(proof["executed_query_lengths"]) != executed or tuple(
            proof["round_widths"]
        ) != expected_round_widths:
            raise RuntimeError("weighted ragged shrink geometry differs")
        layer_count = candidate.native_layer_count
        rounds = len(expected_round_widths)
        singleton_rounds = sum(width == 1 for width in expected_round_widths)
        expected_counters = {
            "grouped_n20_write_count": layer_count * rounds,
            "grouped_n20_row_count": layer_count * sum(expected_round_widths),
            "q1_stock_long_n20_partial_dispatch_count": layer_count * rounds,
            "q1_stock_long_n20_reduce_dispatch_count": layer_count * rounds,
            "q1_scalar_dispatch_count": 0,
            "q1_stock_long_n20_singleton_partial_dispatch_count": (
                layer_count * singleton_rounds),
            "q1_stock_long_n20_singleton_reduce_dispatch_count": (
                layer_count * singleton_rounds),
        }
        result.update(
            executed_query_lengths=list(proof["executed_query_lengths"]),
            round_widths=list(proof["round_widths"]),
            physical_counters=proof["physical_counters"],
        )
        if proof["physical_counters"] != expected_counters:
            raise RuntimeError("weighted ragged physical counters differ")

        logits_parity = []
        for lane, (actual_rows, expected_rows) in enumerate(
            zip(captured["logits"], target_logits)
        ):
            actual = actual_rows
            expected = mx.stack(expected_rows)
            logits_parity.append({"lane": lane, **tensor_metric(actual, expected, mx)})
        result["logits_parity"] = logits_parity
        if not all(item["finite"] and item["exact"] for item in logits_parity):
            raise RuntimeError("weighted ragged target logits differ")

        grouped = {lane: [] for lane in range(3)}
        for response in responses:
            grouped[response.uid].append(response)
        response_tokens = tuple(
            tuple(item.token for item in grouped[lane]) for lane in range(3)
        )
        response_from_draft = tuple(
            tuple(bool(item.from_draft) for item in grouped[lane]) for lane in range(3)
        )
        expected_from_draft = tuple(
            tuple(
                step + 1 < len(proposals[lane]) and
                token == proposals[lane][step + 1]
                for step, token in enumerate(target_tokens[lane])
            )
            for lane in range(3)
        )
        accepted_depths = tuple(sum(values) for values in expected_from_draft)
        result.update(
            target_tokens=[list(tokens) for tokens in target_tokens],
            response_tokens=[list(tokens) for tokens in response_tokens],
            response_from_draft=[list(values) for values in response_from_draft],
        )
        if (response_tokens != target_tokens or
                response_from_draft != expected_from_draft):
            raise RuntimeError("weighted ragged response acceptance differs")
        for lane, expected_accepted in enumerate(accepted_depths):
            if any(
                item.mtp_receipt.get("draft_accepted") != expected_accepted
                or item.mtp_receipt.get("verifier_executed_rows") != executed[lane]
                or item.mtp_receipt.get("verifier_round_widths") != list(
                    expected_round_widths)
                or item.mtp_receipt.get("native_n20_ragged_observed_used") != bool(
                    len(proposals[lane]) > 1)
                or item.mtp_receipt.get("draft_source") != proposal_receipts[lane].get(
                    "source")
                for item in grouped[lane]
            ):
                raise RuntimeError("weighted ragged route receipt differs")

        offsets = tuple(count + rows for count, rows in zip(counts, executed))
        state = compare_public_state(
            owners, candidate, target_boundaries, offsets, mx
        )
        result["state_parity"] = state
        if not state["passed"]:
            raise RuntimeError("weighted ragged published state differs")
        result.update(
            status="passed",
            route_observed_used=True,
            prompt_lookup_source_observed_used=(
                args.proposal_source == "prompt_lookup" and
                any(len(row) > 1 for row in proposals)),
            observed_route_scope=(
                "weighted adapter, native prefill, online ragged logits loop, "
                "sampler, selected KV/GDN publication and response lifecycle; "
                f"{args.proposal_source} proposals, no HTTP"
            ),
            proposal_source=args.proposal_source,
            anchors=list(anchors),
            draft_accepted=list(accepted_depths),
            sampler_calls=len(sampler_calls),
            pending_epochs=len(candidate.backend.writer.pending_epochs),
            pending_leases=candidate.backend.writer.ledger.pending_count,
            peak_mlx_bytes=int(mx.get_peak_memory()),
            active_mlx_bytes=int(mx.get_active_memory()),
            cache_mlx_bytes=int(mx.get_cache_memory()),
        )
    finally:
        primary_error = sys.exc_info()[1]
        if primary_error is not None:
            result["primary_error"] = repr(primary_error)
            result["primary_traceback"] = traceback.format_exc()
        try:
            if candidate is not None:
                for owner in owners:
                    owner.close()
                deadline = time.monotonic() + 5
                while (
                    not candidate._serving_resources.closed
                    and time.monotonic() < deadline
                ):
                    candidate._serving_resources.reap()
                    resources_module.reap_hybrid_admission_orphans()
                    time.sleep(0.005)
                writer = candidate.backend.writer
                result["cleanup"] = {
                    "pending_epochs": len(writer.pending_epochs),
                    "pending_leases": writer.ledger.pending_count,
                    "allocated_pages": writer.pool.allocated_count,
                    "owners_retired": all(owner.fully_retired for owner in owners),
                    "resources_closed": candidate._serving_resources.closed,
                    "global_charge_restored": resources_module._CHARGED
                    == initial_charge,
                }
                if (
                    writer.pending_epochs
                    or writer.ledger.pending_count
                    or writer.pool.allocated_count
                    or not result["cleanup"]["owners_retired"]
                    or not result["cleanup"]["resources_closed"]
                    or not result["cleanup"]["global_charge_restored"]
                ):
                    FAILURE_ROOTS.append((adapter, candidate, owners))
                    raise RuntimeError("weighted ragged gate retains native state")
            if adapter is not None:
                adapter.close()
        except BaseException as cleanup_error:
            result["cleanup_error"] = repr(cleanup_error)
            result["cleanup_traceback"] = traceback.format_exc()
            FAILURE_ROOTS.append((adapter, candidate, owners))
            if primary_error is None:
                raise


def supervise(args, result):
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    process = subprocess.Popen(command, start_new_session=True)
    began = time.monotonic()
    peak = 0
    try:
        while process.poll() is None:
            if time.monotonic() - began > MAX_SECONDS:
                raise TimeoutError("weighted ragged gate exceeded hard deadline")
            try:
                rss = (
                    int(
                        subprocess.check_output(
                            ["ps", "-o", "rss=", "-p", str(process.pid)], text=True
                        ).strip()
                        or "0"
                    )
                    * 1024
                )
            except (subprocess.CalledProcessError, ValueError):
                rss = 0
            peak = max(peak, rss)
            if rss > MAX_RSS:
                raise MemoryError("weighted ragged gate exceeded RSS ceiling")
            time.sleep(0.2)
        if args.output.is_file():
            result = json.loads(args.output.read_text())
            result.update(
                supervisor_peak_rss_bytes=peak,
                supervisor_seconds=time.monotonic() - began,
            )
            save(args.output, result)
        return process.returncode
    except BaseException as error:  # noqa: BLE001 - supervisor must kill on any worker failure
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        result.update(
            status="failed",
            error=repr(error),
            worker_killed=True,
            supervisor_peak_rss_bytes=peak,
        )
        save(args.output, result)
        return 1


def main():
    from varlen_hybrid_serving_smoke import MANIFEST, MODEL, WHEEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(MODEL))
    parser.add_argument("--artifact-manifest", type=Path, default=Path(MANIFEST))
    parser.add_argument("--mlx-wheel", type=Path, default=Path(WHEEL))
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--proposal-source", choices=("controlled", "prompt_lookup"),
        default="controlled")
    parser.add_argument("--prompt-ngram-min", type=int, default=1)
    parser.add_argument("--prompt-ngram-max", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-profile", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.prompt_ngram_min <= args.prompt_ngram_max <= 16):
        parser.error("prompt ngram bounds must satisfy 1 <= min <= max <= 16")
    result = plan(args.proposal_source)
    if not args.worker:
        return supervise(args, result)
    code = 0
    try:
        execute(args, result)
    except BaseException as error:  # noqa: BLE001 - receipt must preserve every failure
        result.update(
            status="failed",
            error=repr(error),
            traceback=traceback.format_exc(),
            error_phase=result.get("phase", "unknown"),
            retained_failure_roots=len(FAILURE_ROOTS),
        )
        code = 1
    finally:
        save(args.output, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
