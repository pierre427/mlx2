#!/usr/bin/env python3
# ruff: noqa: EXE001
"""Bounded synthetic hybrid/parallel-draft numerical validation on M3 Metal.

The operator owns the GPU lease and locks. This script never acquires them.
Dry-run imports no tensor libraries. Synthetic random tensors exercise runtime
contracts, not trained checkpoint qualification or a performance benchmark.
Each cell records comparison errors, tolerances and actual device assertions.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib
import json
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mlx2.parallel-hybrids-metal-validation.v1"
FAMILIES = ("mamba2", "gdn", "qsa")
KINDS = ("xpress", "lilicorr")
REFERENCES = (
    "test_nemotron_external_taps_cpu",
    "test_qwen38_dflash2_cpu",
    "test_batched_mtp",
    "test_lilicorr_cpu",
    "test_lilicorr_training",
    "test_dpara_cpu",
    "test_standard_xpress_serving_cpu",
    "test_lilicorr_serving_cpu",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--deadline-seconds", type=int, default=900)
    result.add_argument("--expected-chip", choices=("M3",), default="M3")
    result.add_argument("--i-own-the-gpu", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def cell_plan():
    cells = [f"lilicorr_head_w{width}_s{slots}" for width in (4, 6) for slots in (1, 3)]
    cells.append("lilicorr_cpu_training_and_gpu_refusal")
    cells.extend(
        f"mamba2_native_s{length}_{dtype}"
        for length in (1, 3)
        for dtype in ("float32", "bfloat16")
    )
    cells.extend(
        f"gdn_native_d{width}_{dtype}"
        for width in (32, 64)
        for dtype in ("float32", "bfloat16")
    )
    for family in FAMILIES:
        cells.extend(f"{family}_b1_keep{keep}" for keep in (1, 2, 3))
        cells.extend(
            (
                f"{family}_b2_uniform",
                f"{family}_b2_ragged",
                f"{family}_abort_recover_retry",
            )
        )
        cells.extend(f"{family}_{kind}_mixed_adaptive" for kind in KINDS)
    cells.extend(f"{kind}_rotating_draft_apcv2_resume" for kind in KINDS)
    cells.extend(f"dpara_d{depth}_c{context}" for depth in (2, 3) for context in (0, 2))
    cells.extend(
        f"qsa_{kind}_{source}_composition"
        for kind in KINDS
        for source in ("prompt_lookup", "native_mtp")
    )
    cells.append("lilicorr_committed_runtime_feedback")
    return cells


def preflight(args):
    if not 1 <= args.deadline_seconds <= 900:
        raise ValueError("deadline-seconds must be in [1,900]")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError(
            "Metal validation requires --i-own-the-gpu under operator ownership"
        )
    return {
        "schema": SCHEMA,
        "cells_planned": cell_plan(),
        "deadline_seconds": args.deadline_seconds,
        "expected_chip": args.expected_chip,
        "will_execute": not args.dry_run,
        "synthetic_weights": True,
        "trained_checkpoint_qualification": False,
        "performance_claim": False,
        "gpu_training": False,
        "reference_modules": list(REFERENCES),
        "qualified": False,
        "selected": False,
    }


def source_identity():
    files = [Path(__file__).resolve()]
    files += sorted(
        path
        for path in (ROOT / "src").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    # Reference modules can import other fixtures; hash the complete test
    # source tree so those indirect imports are covered too.
    files += sorted((ROOT / "tests").rglob("*.py"))
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"root": str(ROOT), "sha256": digest, "files_sha256": hashes}


class Validation:
    def __init__(self, report):
        import mlx.core as mx
        import numpy as np

        self.mx, self.np, self.report = mx, np, report
        self.current = None
        if not mx.metal.is_available():
            raise RuntimeError("Metal backend unavailable")
        info = mx.metal.device_info()
        if report["expected_chip"].lower() not in json.dumps(info).lower():
            raise RuntimeError(
                f"expected {report['expected_chip']} Metal device, found {info}"
            )
        report["metal_device_info"] = info
        # Import reference helpers first. Several test modules set the default
        # CPU device at import; no test functions or fixtures are executed.
        sys.path.insert(0, str(ROOT / "tests"))
        self.refs = {name: importlib.import_module(name) for name in REFERENCES}
        mx.set_default_device(mx.gpu)
        self.require_gpu()
        from mlx2.runtime import external_speculative
        from mlx2.runtime.models import qwen4_exp

        if (
            not Path(external_speculative.__file__)
            .resolve()
            .is_relative_to(ROOT / "src")
        ):
            raise RuntimeError("executor import is outside hashed source")
        # Exercise existing real pooled-summary producer, not a synthetic
        # stand-in for the QSA ledger. These are opt-in math/cache settings.
        qwen4_exp._QSA_POOLED_KEY_CACHE = True
        qwen4_exp._QSA_APC_SUMMARIES = True
        report["device"] = "Metal GPU"
        report["cells"] = []

    def require_gpu(self):
        if self.mx.default_device() != self.mx.gpu:
            raise RuntimeError("Metal cell unexpectedly changed default device")

    @contextmanager
    def cpu(self):
        self.mx.set_default_device(self.mx.cpu)
        try:
            with self.mx.stream(self.mx.cpu):
                yield
        finally:
            self.mx.set_default_device(self.mx.gpu)
        self.require_gpu()

    def host(self, value):
        # MLX's buffer export on this M3 build dereferences a null pointer for
        # empty lattice edges (one-slot LiLiCoRR has no adjacent-slot edges).
        # Preserve their exact shape without requesting a buffer export.
        if value.size == 0:
            return self.np.empty(value.shape, dtype=self.np.float32)
        value = value.astype(self.mx.float32)
        self.mx.eval(value)
        return self.np.asarray(value)

    def compare(self, name, actual, expected, *, atol=5e-4, rtol=5e-4):
        a = (
            self.host(actual)
            if isinstance(actual, self.mx.array)
            else self.np.asarray(actual)
        )
        b = (
            self.host(expected)
            if isinstance(expected, self.mx.array)
            else self.np.asarray(expected)
        )
        if a.shape != b.shape:
            raise AssertionError(f"{name}: shape {a.shape} != {b.shape}")
        error = self.np.abs(a.astype(float) - b.astype(float))
        maximum = float(error.max()) if error.size else 0.0
        relative = (
            float((error / self.np.maximum(self.np.abs(b), 1e-12)).max())
            if error.size
            else 0.0
        )
        passed = bool(
            self.np.isfinite(a).all()
            and self.np.isfinite(b).all()
            and self.np.all(error <= atol + rtol * self.np.abs(b))
        )
        self.current["comparisons"].append(
            {
                "name": name,
                "shape": list(a.shape),
                "max_abs_error": maximum,
                "max_relative_error": relative,
                "atol": atol,
                "rtol": rtol,
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(
                f"{name}: max abs {maximum}, relative {relative}, tolerance {atol}+{rtol}*abs(expected)"
            )

    def cell(self, name, fn):
        self.require_gpu()
        record = {
            "name": name,
            "device": "Metal GPU",
            "synthetic_weights": True,
            "comparisons": [],
        }
        self.current = record
        self.report["cells"].append(record)
        started = time.monotonic()
        try:
            fn()
            self.require_gpu()
            record["passed"] = True
        except TimeoutError:
            raise
        except Exception as error:  # noqa: BLE001 - persist independent cell failures
            record["passed"] = False
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
            self.mx.set_default_device(self.mx.gpu)
        finally:
            record["seconds_diagnostic"] = time.monotonic() - started
            out = self.report.get("progress_path")
            if out:
                Path(out).write_text(
                    json.dumps(self.report, indent=2, sort_keys=True) + "\n"
                )
            print(
                json.dumps(
                    {
                        "cell": name,
                        "passed": record.get("passed"),
                        "error": record.get("error"),
                    }
                ),
                flush=True,
            )

    def target(self, family):
        self.require_gpu()
        if family == "mamba2":
            model = self.refs["test_nemotron_external_taps_cpu"].tiny_model()
            captures, width, vocab, count = [0, 1, 4], 8, 16, 5
        elif family == "gdn":
            model = self.refs["test_qwen38_dflash2_cpu"].tiny_target()
            captures, width, vocab, count = [1, 6], 32, 128, 8
        else:
            model = self.refs["test_batched_mtp"]._tiny_qwen4_model()
            captures, width, vocab, count = [0, 1], 32, 64, 2
        model.eval()
        self.mx.eval(model.parameters())
        self.require_gpu()
        return model, (captures, width, vocab, count)

    def native_ssm(self, length, dtype):
        from mlx2.runtime.models import ssm

        mx = self.mx
        kind = getattr(mx, dtype)
        rng = self.np.random.default_rng(871)
        shapes = (
            (2, length, 4, 8),
            (4,),
            (2, length, 2, 64),
            (2, length, 2, 64),
            (4,),
            (2, length, 4),
            (4,),
            (2, 4, 8, 64),
        )
        host = [
            rng.normal(size=shape).astype(self.np.float32) * 0.1 for shape in shapes
        ]
        values = [mx.array(value).astype(kind) for value in host]
        mx.eval(values)
        field = "ssm_update_kernel" if length == 1 else "ssm_update_seq_kernel"
        original = getattr(ssm, field)
        calls = []

        def tracked(*args, **kwargs):
            calls.append(field)
            return original(*args, **kwargs)

        setattr(ssm, field, tracked)
        try:
            actual = ssm.ssm_update(*values)
            mx.eval(actual)
        finally:
            setattr(ssm, field, original)
        if calls != [field]:
            raise AssertionError("supported unmasked native SSM kernel did not execute")
        with self.cpu():
            expected = ssm.ssm_attn(*values)
            mx.eval(expected)
            expected = [self.host(value) for value in expected]
        tolerance = 8e-3 if dtype == "bfloat16" else 5e-5
        for index, (value, reference) in enumerate(zip(actual, expected)):
            self.compare(
                f"native_output{index}",
                value,
                reference,
                atol=tolerance,
                rtol=tolerance,
            )
        self.current.update(native_kernel_calls=calls, state_dim=64, dtype=dtype)

    def native_gdn(self, width, dtype):
        from mlx2.runtime.models import gated_delta as gdn

        mx = self.mx
        kind = getattr(mx, dtype)
        rng = self.np.random.default_rng(884)
        shapes = (
            (2, 3, 2, width),
            (2, 3, 2, width),
            (2, 3, 4, 16),
            (2, 3, 4),
            (2, 3, 4),
            (2, 4, 16, width),
        )
        values = [
            mx.array(rng.normal(size=shape).astype(self.np.float32) * 0.05)
            for shape in shapes
        ]
        values[:3] = [value.astype(kind) for value in values[:3]]
        values[3] = mx.exp(-mx.abs(values[3]))
        values[4] = mx.sigmoid(values[4])
        mx.eval(values)
        original = gdn._gated_delta_kernel
        calls = []

        def tracked(*args, **kwargs):
            calls.append("generic_gated_delta_kernel")
            return original(*args, **kwargs)

        gdn._gated_delta_kernel = tracked
        try:
            actual = gdn._gated_delta_kernel_impl(*values, allow_packed=False)
            mx.eval(actual)
        finally:
            gdn._gated_delta_kernel = original
        if not calls:
            raise AssertionError("supported native GDN kernel did not execute")
        with self.cpu():
            expected = gdn.gated_delta_ops(*values)
            mx.eval(expected)
            expected = [self.host(value) for value in expected]
        tolerance = 8e-3 if dtype == "bfloat16" else 5e-5
        for index, (value, reference) in enumerate(zip(actual, expected)):
            self.compare(
                f"native_output{index}",
                value,
                reference,
                atol=tolerance,
                rtol=tolerance,
            )
        self.current.update(native_kernel_calls=calls, key_dim=width, dtype=dtype)

    def draft(self, model, geometry, kind, windows=None, lilicorr_topk=2):
        from mlx2.adapters.lilicorr import LiLiCorrConfig
        from mlx2.adapters.xpress import XPressConfig
        from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel
        from mlx2.runtime.drafters.xpress import XPressDraftModel

        captures, width, vocab, count = geometry
        common = {
            "hidden_size": width,
            "intermediate_size": width * 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": width // 2,
            "vocab_size": vocab,
            "mask_token_id": vocab - 1,
            "num_target_layers": count,
            "target_layer_ids": captures,
            "block_size": 4,
            "layer_types": ["full_attention"],
        }
        if kind == "xpress":
            draft = XPressDraftModel(
                XPressConfig(
                    **common, xpress_rank=4, xpress_mlp_hidden=8, xpress_num_passes=2
                ),
                draft_attention_windows=windows,
            )
        else:
            draft = LiLiCorrDraftModel(
                LiLiCorrConfig(
                    **common,
                    lilicorr_hidden_size=8,
                    lilicorr_candidate_topk=lilicorr_topk,
                    lilicorr_num_layers=2,
                    lilicorr_num_heads=2,
                    lilicorr_mlp_ratio=1.5,
                    lilicorr_factor_dim=3,
                ),
                draft_attention_windows=windows,
            )
        draft.bind(model)
        self.mx.eval(draft.parameters())
        return draft

    def state(self, actual, expected, name):
        from mlx2.runtime.models.cache import ArraysCache
        from mlx2.runtime.models.qwen4_exp import QSAKVCache

        if len(actual) != len(expected):
            raise AssertionError("cache topology mismatch")
        for layer, (a, b) in enumerate(zip(actual, expected)):
            label = f"{name}.layer{layer}"
            if isinstance(a, ArraysCache):
                if a.speculating or a._rollbacks:
                    raise AssertionError(
                        "committed recurrent cache retains live rollback records"
                    )
                for slot, (av, bv) in enumerate(zip(a.cache, b.cache)):
                    if av is None or bv is None:
                        if av is not bv:
                            raise AssertionError("recurrent None state mismatch")
                    else:
                        self.compare(f"{label}.recurrent_slot{slot}", av, bv)
            else:
                if a.offset != b.offset:
                    raise AssertionError(f"{label}: offset {a.offset} != {b.offset}")
                for field in ("keys", "values"):
                    self.compare(
                        f"{label}.{field}",
                        getattr(a, field)[:, :, : a.offset],
                        getattr(b, field)[:, :, : b.offset],
                    )
                if isinstance(a, QSAKVCache):
                    if a.index_keys.shape[1] != a.offset or a._mtp_share_topk:
                        raise AssertionError("QSA raw ledger/cycle boundary mismatch")
                    self.compare(f"{label}.raw_index", a.index_keys, b.index_keys)
                    if a._qsa_pooled_keys is not None or b._qsa_pooled_keys is not None:
                        self.compare(
                            f"{label}.pooled_keys",
                            a._qsa_pooled_keys,
                            b._qsa_pooled_keys,
                        )
                        if (
                            a._qsa_summary_identity["complete_blocks"]
                            != a.offset // a._qsa_pooled_ratio
                        ):
                            raise AssertionError(
                                "QSA pooled coverage exceeds committed boundary"
                            )

    def prefix(self, family, lengths, kept, abort=False):
        from mlx2.runtime.cow_cache import (
            restore_recovery_descriptors,
            snapshot_recovery_descriptors,
        )
        from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

        model, geometry = self.target(family)
        prompts = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ][: len(lengths)]
        rows, references, snapshots = [], [], []
        for prompt in prompts:
            row, reference = model.make_cache(), model.make_cache()
            self.mx.eval(model(self.mx.array([prompt]), cache=row))
            self.mx.eval(model(self.mx.array([prompt]), cache=reference))
            rows.append(row)
            references.append(reference)
            snapshots.append(snapshot_recovery_descriptors(row))
        inputs = self.mx.array([[5, 6, 7], [7, 6, 5]][: len(rows)])
        txn = HybridVerifyRows(rows).begin(lengths)
        logits, taps = model.forward_with_taps(inputs, txn.caches, geometry[0])
        self.mx.eval(logits, taps, [c.state for c in txn.caches])
        if abort:
            txn.abort()
            rows = [
                restore_recovery_descriptors(snapshot, sidecar, borrowed)[0]
                for snapshot, sidecar, borrowed in snapshots
            ]
            for index, (row, ref) in enumerate(zip(rows, references)):
                self.state(row, ref, f"recovery.row{index}")
            inputs = self.mx.array([[9, 8, 6], [8, 9, 6]][: len(rows)])
            txn = HybridVerifyRows(rows).begin(lengths)
            logits, taps = model.forward_with_taps(inputs, txn.caches, geometry[0])
            self.mx.eval(logits, taps)
        txn.commit(kept)
        for index, count in enumerate(kept):
            expected, expected_taps = model.forward_with_taps(
                inputs[index : index + 1, :count], references[index], geometry[0]
            )
            self.compare(
                f"row{index}.kept_logits", logits[index : index + 1, :count], expected
            )
            self.compare(
                f"row{index}.kept_taps", taps[index : index + 1, :count], expected_taps
            )
            self.state(rows[index], references[index], f"commit.row{index}")
            # A subsequent token observes recurrent/attention state, making
            # hidden rejected suffixes detectable beyond the current logits.
            actual_next = model(self.mx.array([[3]]), cache=rows[index])
            expected_next = model(self.mx.array([[3]]), cache=references[index])
            self.compare(f"row{index}.next_logits", actual_next, expected_next)
        self.current.update(
            family=family,
            verified_lengths=lengths,
            kept_prefixes=kept,
            abort_recovery_retry=abort,
        )

    def head(self, width, slots):
        ref = self.refs["test_lilicorr_cpu"]
        args = ref.config(lilicorr_hidden_size=width)
        head = ref.initialized_head(args)
        rng = self.np.random.default_rng(9)
        host_inputs = [
            rng.normal(size=shape).astype(self.np.float32)
            for shape in ((2, slots, 2, 4), (2, slots, 2), (2, slots, 4), (2, 4))
        ]
        host_inputs[1] = self.np.log(self.np.array([0.6, 0.3], self.np.float32))[
            None, None
        ] + self.np.zeros((2, slots, 2), self.np.float32)
        valid = self.np.array([True, False])
        actual = head.score(
            *[self.mx.array(value) for value in host_inputs], self.mx.array(valid)
        )
        self.mx.eval(actual)
        expected = ref.oracle(head, args, *host_inputs, valid)
        with self.cpu():
            cpu = head.score(
                *[self.mx.array(value) for value in host_inputs], self.mx.array(valid)
            )
            self.mx.eval(cpu)
            cpu = [self.host(value) for value in cpu]
        for index, (value, baseline, oracle) in enumerate(zip(actual, cpu, expected)):
            self.compare(
                f"score{index}.cpu_vs_metal", value, baseline, atol=3e-5, rtol=3e-5
            )
            self.compare(
                f"score{index}.numpy_oracle", value, oracle, atol=3e-5, rtol=3e-5
            )

    def training(self):
        from mlx2.runtime.lilicorr_training import train_shadow

        reference = self.refs["test_lilicorr_training"]
        head = self.refs["test_lilicorr_cpu"].initialized_head()
        buffer = reference.buffer()
        buffer.add(**reference.example())
        try:
            train_shadow(head, buffer, steps=1)
        except ValueError as error:
            if "explicit CPU" not in str(error):
                raise
        else:
            raise AssertionError("GPU training was not refused")
        from mlx.utils import tree_flatten

        before = {
            name: self.host(value).copy()
            for name, value in tree_flatten(head.parameters())
        }
        with self.cpu():
            result = train_shadow(head, buffer, steps=25, learning_rate=0.01)
        if result.final_loss >= result.initial_loss:
            raise AssertionError("CPU shadow loss did not decrease")
        for name, value in tree_flatten(head.parameters()):
            self.compare(
                f"source_head_unchanged.{name}", value, before[name], atol=0, rtol=0
            )
        self.current.update(
            training_device="CPU",
            gpu_training_refused=True,
            gpu_training_executed=False,
            initial_loss=result.initial_loss,
            final_loss=result.final_loss,
            labels="synthetic explicitly declared teacher examples",
        )

    def ordinary(self, model, prompt, count):
        cache, inputs, out = model.make_cache(), list(prompt), []
        for _ in range(count):
            logits = model(self.mx.array([inputs]), cache=cache)
            token = int(self.mx.argmax(logits[0, -1]).item())
            out.append(token)
            inputs = [token]
        return out

    def drain(self, batch):
        outputs, ends = {}, {}
        for lane in list(batch.lanes.values()):
            while lane.remaining:
                batch._prefill(lane)
        for _ in range(160):
            _, responses = batch.next()
            failures = batch.take_lane_failures()
            if failures:
                raise RuntimeError(f"lane failures: {failures}")
            for response in responses:
                outputs.setdefault(response.uid, []).append(response.token)
                if response.finish_reason:
                    ends[response.uid] = response
            if not batch.lanes:
                return outputs, ends
        raise RuntimeError("bounded scheduler polling exhausted")

    def adaptive(self, family, kind):
        import mlx2.runtime.external_speculative as external
        from mlx2.runtime.sample_utils import LaneRNG

        model, geometry = self.target(family)
        draft = self.draft(model, geometry, kind)
        original = draft.draft_distributions

        def propose(*args, **kwargs):
            result = original(*args, **kwargs)
            draft.adaptive_confidence_features = [
                [40 if history[0] == 1 else -40] * len(tokens)
                for history, tokens in zip(kwargs["processor_histories"], result[0])
            ]
            return result

        draft.draft_distributions = propose
        policy = {
            "verification_costs": [0.5, 1, 1.4],
            "verification_costs_by_cohort": {"2": [1, 2, 10], "4": [2, 4, 100]},
            "mode": "per_request",
            "min_observations": 1,
        }
        batch = external.ExternalDraftBatchGenerator(
            model,
            draft_model=draft,
            binding=f"synthetic-{family}-{kind}",
            num_draft=2,
            prefill_step_size=4,
            completion_batch_size=4,
            ready_drain="all",
            adaptive_verification=policy,
        )
        batch.acceptance_estimator.observe([0, 0], 0, rejected=True)
        batch.acceptance_estimator.rounds = 1
        prompts = [[1, 2, 3], [1, 2], [2, 3, 4, 5], [2, 3]]
        ids = batch.insert(
            prompts,
            max_tokens=[8] * 4,
            lane_rngs=[LaneRNG(60 + i) for i in range(4)],
            sampling_configs=[{"sampling_temp": 0}, {"sampling_temp": 0.8}] * 2,
        )
        for lane in batch.lanes.values():
            while lane.remaining:
                batch._prefill(lane)
        law = {"rows": 0, "sampled_rows": 0}
        verify = external.verify_proposals

        def checked(tokens, proposals, targets, rng, **kwargs):
            for token, q, p in zip(tokens, proposals, targets):
                q, p = self.np.asarray(q), self.np.asarray(p)
                if q[token] != 1 or self.np.count_nonzero(q) != 1:
                    raise AssertionError(
                        "parallel proposal law is not an exact point mass"
                    )
                if (
                    not self.np.isfinite(p).all()
                    or p.min() < 0
                    or abs(p.sum() - 1) > 1e-6
                ):
                    raise AssertionError("invalid transformed target probability law")
                residual = p.copy()
                residual[token] = 0
                self.compare(
                    "acceptance_residual_identity",
                    residual + p[token] * q,
                    p,
                    atol=1e-12,
                    rtol=1e-12,
                )
                law["rows"] += 1
                law["sampled_rows"] += int(self.np.count_nonzero(p) > 1)
            return verify(tokens, proposals, targets, rng, **kwargs)

        external.verify_proposals = checked
        try:
            batch._round(list(batch.lanes.values()))
            for index, lane in enumerate(batch.lanes.values()):
                fresh = model.make_cache()
                self.mx.eval(model(self.mx.array([lane.history]), cache=fresh))
                self.state(lane.cache, fresh, f"adaptive.row{index}")
                batch._sidecar(lane).validate(batch.binding, len(lane.history))
            outputs, ends = self.drain(batch)
        finally:
            external.verify_proposals = verify
            draft.draft_distributions = original
        for index in (0, 2):
            expected = self.ordinary(model, prompts[index], 8)
            if outputs[ids[index]] != expected:
                raise AssertionError(f"greedy row{index} differs from ordinary decode")
        if (
            batch.scheduler_stats["target_max_width"] < 2
            or batch.scheduler_stats["external_adaptive_verification_groups"] < 2
            or not law["sampled_rows"]
        ):
            raise AssertionError(
                "batched grouped adaptive or mixed sampling was not engaged"
            )
        if batch.scheduler_stats["draft_fallbacks"]:
            raise AssertionError("ordinary fallback occurred")
        self.current.update(
            stats=dict(batch.scheduler_stats),
            token_ids=outputs,
            law_checks=law,
            receipts={uid: end.speculative_receipt for uid, end in ends.items()},
            confidence_source="synthetic forced high/low inputs for grouping contract; not predictor quality",
        )

    def apc(self, kind):
        from mlx2.runtime.apc_v2 import APCKey, APCv2
        from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
        from mlx2.runtime.models.cache import RotatingKVCache

        model, _ = self.refs["test_standard_xpress_serving_cpu"].tiny()
        draft = self.draft(model, ([0, 2], 8, 9, 3), kind, windows=[2])
        binding = f"synthetic-window2-{kind}"

        def engine():
            return ExternalDraftBatchGenerator(
                model,
                draft_model=draft,
                binding=binding,
                num_draft=2,
                prefill_step_size=3,
                ready_drain="all",
            )

        batch = engine()
        prompt = [1, 2, 3, 4, 5, 6]
        uid = batch.insert([prompt], max_tokens=[8])[0]
        outputs, ends = self.drain(batch)
        if outputs[uid] != self.ordinary(model, prompt, 8):
            raise AssertionError("cold windowed greedy differs from ordinary")
        end = ends[uid]
        cache = APCv2(max_size=1, layout_name=binding)
        key = APCKey(
            "synthetic-qwen3", revision=binding, cache_layout_fingerprint=binding
        )
        hit = None
        try:
            cache.store(
                key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar
            )
            hit = cache.lookup(key, end.all_tokens + [end.token])
            if not hit.hit or hit.hit_kind != "external_draft_sidecar":
                raise AssertionError("paired APCv2 hit missing")
            resumed = engine()
            rid = resumed.insert(
                [[end.token]],
                max_tokens=[8],
                caches=[hit.cache],
                all_tokens=[end.all_tokens],
                cache_states=[hit.sidecar],
            )[0]
            lane = resumed.lanes[rid]
            if not all(
                isinstance(entry, RotatingKVCache) for entry in lane.draft_cache
            ):
                raise AssertionError("resumed draft cache is not rotating")
            actual, final = self.drain(resumed)
            expected = self.ordinary(model, end.all_tokens + [end.token], 8)
            if (
                actual[rid] != expected
                or resumed.scheduler_stats["paired_cache_resumes"] != 1
            ):
                raise AssertionError("paired windowed resume parity/engagement failed")
            final[rid].cache_sidecar.validate(binding, len(final[rid].all_tokens))
            self.current.update(
                cold_tokens=outputs[uid],
                warm_tokens=actual[rid],
                ordinary_tokens=expected,
                stats=dict(resumed.scheduler_stats),
                receipt=final[rid].speculative_receipt,
                draft_attention_windows=[2],
                apcv2_hit=True,
            )
        finally:
            if hit is not None and hit.hit:
                hit.cache.close()
            cache.clear(release_memory=False)

    def composition(self, kind, source):
        from mlx2.adapters.proposal_sources import native_mtp_source
        from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
        from mlx2.runtime.proposal_composition import ComposedDraftModel
        from mlx2.runtime.sample_utils import LaneRNG

        model, geometry = self.target("qsa")
        backend = self.draft(model, geometry, kind, windows=[4])
        options = {
            "prompt_lookup": source == "prompt_lookup",
            "native_mtp": source == "native_mtp",
            "ngram_min": 1,
            "ngram_max": 1,
        }
        draft = ComposedDraftModel(
            backend, options, native_mtp_source=native_mtp_source(model)
        )
        policy = {
            "verification_costs": [0.5, 1, 1.4],
            "verification_costs_by_cohort": {"2": [1, 2, 10], "4": [2, 4, 100]},
            "mode": "per_request",
            "min_observations": 1,
        }
        batch = ExternalDraftBatchGenerator(
            model,
            draft_model=draft,
            binding=f"synthetic-qsa-{kind}-{source}-composition",
            num_draft=2,
            prefill_step_size=16,
            completion_batch_size=4,
            ready_drain="all",
            adaptive_verification=policy,
        )
        batch.acceptance_estimator.observe([0, 0], 0, rejected=True)
        batch.acceptance_estimator.rounds = 1
        prompts = (
            [list(range(64)) * 2 + [1, 2], list(range(63, -1, -1)) * 2 + [2, 3]] * 2
            if source == "prompt_lookup"
            else [[1, 2, 3], [2, 3, 4], [4, 5], [5, 6, 7]]
        )
        ids = batch.insert(
            prompts,
            max_tokens=[8] * 4,
            lane_rngs=[LaneRNG(490 + index) for index in range(4)],
            sampling_configs=[{"sampling_temp": 0}, {"sampling_temp": 0.8}] * 2,
        )
        outputs, ends = self.drain(batch)
        for index in (0, 2):
            if outputs[ids[index]] != self.ordinary(model, prompts[index], 8):
                raise AssertionError("composed greedy output differs from ordinary")
        if (
            not draft.composition_stats[source]
            or batch.scheduler_stats["target_max_width"] < 2
        ):
            raise AssertionError(
                "requested proposal source/batched verification did not execute"
            )
        if batch.scheduler_stats["draft_fallbacks"]:
            raise AssertionError("composed route used ordinary fallback")
        for end in ends.values():
            end.cache_sidecar.validate(batch.binding, len(end.all_tokens))
        self.current.update(
            stats=dict(batch.scheduler_stats),
            tokens=outputs,
            composition_diagnostic=dict(draft.composition_stats),
            receipts={uid: end.speculative_receipt for uid, end in ends.items()},
            source=source,
            native_mtp_private_state=True,
            cost_tables="synthetic contract inputs, not measurements",
            trained_companion=False,
        )

    def feedback(self):
        import tempfile

        from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
        from mlx2.runtime.lilicorr_feedback import LiLiCorrFeedbackManager

        model, _ = self.refs["test_standard_xpress_serving_cpu"].tiny()
        draft = self.draft(
            model, ([0, 2], 8, 9, 3), "lilicorr", windows=[4], lilicorr_topk=8
        )
        binding = hashlib.sha256(b"synthetic-metal-lilicorr-feedback").hexdigest()
        with tempfile.TemporaryDirectory(prefix="mlx2-feedback-metal-") as directory:
            manager = LiLiCorrFeedbackManager(
                draft,
                {
                    "directory": directory,
                    "min_examples": 1,
                    "train_every": 1,
                    "steps": 5,
                },
                target_revision="a" * 64,
                draft_revision="b" * 64,
                binding=binding,
            )
            draft.feedback_manager = manager
            from mlx.utils import tree_flatten

            before = {
                name: self.host(value).copy()
                for name, value in tree_flatten(draft.lilicorr.parameters())
            }
            try:
                batch = ExternalDraftBatchGenerator(
                    model,
                    draft_model=draft,
                    binding=binding,
                    num_draft=2,
                    prefill_step_size=3,
                    completion_batch_size=2,
                    ready_drain="all",
                )
                prompts = [[1, 2, 3], [2, 3, 4]]
                ids = batch.insert(prompts, max_tokens=[16, 16])
                actual, ends = self.drain(batch)
                for index, uid in enumerate(ids):
                    if actual[uid] != self.ordinary(model, prompts[index], 16):
                        raise AssertionError(
                            "feedback-enabled output differs from ordinary"
                        )
                deadline = time.monotonic() + 20
                while manager.child is not None and time.monotonic() < deadline:
                    time.sleep(0.05)
                    manager._poll()
                if (
                    not manager.stats["committed"]
                    or not manager.stats["training_completed"]
                ):
                    raise AssertionError(
                        f"actual verified feedback/training did not execute: {manager.receipt()}"
                    )
                self.require_gpu()
                for name, value in tree_flatten(draft.lilicorr.parameters()):
                    self.compare(
                        "live_head_unchanged." + name,
                        value,
                        before[name],
                        atol=0,
                        rtol=0,
                    )
                self.current.update(
                    feedback=manager.receipt(),
                    stats=dict(batch.scheduler_stats),
                    receipts={
                        uid: end.speculative_receipt for uid, end in ends.items()
                    },
                    synthetic_labels=False,
                    synthetic_target_weights=True,
                )
            finally:
                manager.close()

    def dpara(self, depth, length):
        from mlx2.runtime.dpara import DParaVerification
        from mlx2.runtime.speculative_sampling import RequestRNG, softmax

        ref = self.refs["test_dpara_cpu"]
        model, context, spine = ref.tiny(d=depth, context_length=length)
        ticket = model.prepare(spine, context)
        self.mx.eval(ticket.payload.hidden, ticket.payload.logits)
        with self.cpu():
            cpu = model.prepare(spine, context)
            self.mx.eval(cpu.payload.hidden, cpu.payload.logits)
            baseline = [self.host(cpu.payload.hidden), self.host(cpu.payload.logits)]
            cpu.discard()
        self.compare(
            "all_branch_hidden.cpu_vs_metal",
            ticket.payload.hidden,
            baseline[0],
            atol=3e-5,
            rtol=3e-5,
        )
        self.compare(
            "all_branch_logits.cpu_vs_metal",
            ticket.payload.logits,
            baseline[1],
            atol=3e-5,
            rtol=3e-5,
        )
        for accepted in range(depth + 1):
            hidden, logits = ref.separate_prefix_oracle(model, context, spine, accepted)
            self.compare(
                f"branch{accepted}.hidden.prefix_oracle",
                ticket.payload.hidden[accepted],
                hidden,
                atol=3e-5,
                rtol=3e-5,
            )
            self.compare(
                f"branch{accepted}.logits.prefix_oracle",
                ticket.payload.logits[accepted],
                logits,
                atol=3e-5,
                rtol=3e-5,
            )
        ticket.discard()
        before = [self.host(value).copy() for pair in context.layers for value in pair]
        verified_features = self.mx.random.normal(
            (1, depth + 1, 16), key=self.mx.random.key(121)
        )
        for accepted in range(depth + 1):
            for bonus in (0, 5):
                fresh = model.prepare(spine, context)
                base = self.host(fresh.payload.logits[accepted])
                outcome = DParaVerification(
                    context.binding, accepted, bonus, verified_features
                )
                result = model.resolve(
                    fresh, outcome, temperature=0.8, rng=RequestRNG(99)
                )
                previous = bonus
                for position, (token, law) in enumerate(
                    zip(result.draft_tokens, result.proposal_probabilities)
                ):
                    expected = softmax(
                        base[position]
                        + self.host(model.markov_head(self.mx.array(previous))),
                        0.8,
                    )
                    self.compare(
                        f"r{accepted}.bonus{bonus}.q{position}",
                        law,
                        expected,
                        atol=2e-7,
                        rtol=2e-7,
                    )
                    previous = token
                projected = model._project_features(
                    verified_features[:, : accepted + 1], context.length
                )
                for layer, (old, new, published) in enumerate(
                    zip(context.layers, projected, result.context.layers)
                ):
                    for axis in range(2):
                        self.compare(
                            f"r{accepted}.bonus{bonus}.context{layer}.{axis}",
                            published[axis],
                            self.mx.concatenate((old[axis], new[axis]), axis=2),
                            atol=3e-5,
                            rtol=3e-5,
                        )
                try:
                    model.resolve(fresh, outcome)
                except ValueError:
                    pass
                else:
                    raise AssertionError("resolved branch handle was reusable")
        for index, value in enumerate(
            value for pair in context.layers for value in pair
        ):
            self.compare(
                f"immutable_context{index}", value, before[index], atol=0, rtol=0
            )
        stale = model.prepare(spine, context)
        stale.discard()
        try:
            model.resolve(
                stale, DParaVerification(context.binding, 0, 0, verified_features)
            )
        except ValueError:
            pass
        else:
            raise AssertionError("discarded branch published state")
        self.current.update(
            draft_length=depth,
            context_length=length,
            trained_m_dflash=False,
            hardware_overlap_observed=False,
        )

    def run(self):
        for width in (4, 6):
            for slots in (1, 3):
                self.cell(
                    f"lilicorr_head_w{width}_s{slots}",
                    lambda width=width, slots=slots: self.head(width, slots),
                )
        self.cell("lilicorr_cpu_training_and_gpu_refusal", self.training)
        for length in (1, 3):
            for dtype in ("float32", "bfloat16"):
                self.cell(
                    f"mamba2_native_s{length}_{dtype}",
                    lambda length=length, dtype=dtype: self.native_ssm(length, dtype),
                )
        for width in (32, 64):
            for dtype in ("float32", "bfloat16"):
                self.cell(
                    f"gdn_native_d{width}_{dtype}",
                    lambda width=width, dtype=dtype: self.native_gdn(width, dtype),
                )
        for family in FAMILIES:
            for keep in (1, 2, 3):
                self.cell(
                    f"{family}_b1_keep{keep}",
                    lambda family=family, keep=keep: self.prefix(family, [3], [keep]),
                )
            self.cell(
                f"{family}_b2_uniform",
                lambda family=family: self.prefix(family, [3, 3], [1, 2]),
            )
            self.cell(
                f"{family}_b2_ragged",
                lambda family=family: self.prefix(family, [3, 2], [1, 1]),
            )
            self.cell(
                f"{family}_abort_recover_retry",
                lambda family=family: self.prefix(family, [3, 2], [1, 2], abort=True),
            )
            for kind in KINDS:
                self.cell(
                    f"{family}_{kind}_mixed_adaptive",
                    lambda family=family, kind=kind: self.adaptive(family, kind),
                )
        for kind in KINDS:
            self.cell(
                f"{kind}_rotating_draft_apcv2_resume", lambda kind=kind: self.apc(kind)
            )
        for depth in (2, 3):
            for context in (0, 2):
                self.cell(
                    f"dpara_d{depth}_c{context}",
                    lambda depth=depth, context=context: self.dpara(depth, context),
                )
        for kind in KINDS:
            for source in ("prompt_lookup", "native_mtp"):
                self.cell(
                    f"qsa_{kind}_{source}_composition",
                    lambda kind=kind, source=source: self.composition(kind, source),
                )
        self.cell("lilicorr_committed_runtime_feedback", self.feedback)


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        report = preflight(args)
    except ValueError as error:
        arguments.error(str(error))
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    report["started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report["progress_path"] = str(args.out)
    report["passed"] = False
    faulthandler.enable()

    def deadline(_signal, _frame):
        raise TimeoutError(f"validation exceeded {args.deadline_seconds} seconds")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        report["source_identity"] = source_identity()
        Validation(report).run()
        report["passed"] = all(cell["passed"] for cell in report["cells"])
        final_identity = source_identity()
        report["source_unchanged"] = (
            final_identity["sha256"] == report["source_identity"]["sha256"]
        )
        if not report["source_unchanged"]:
            report["passed"] = False
            report["error"] = "source changed during validation"
    except Exception as error:  # noqa: BLE001 - always persist failed execution evidence
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    report["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "out": str(args.out),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
