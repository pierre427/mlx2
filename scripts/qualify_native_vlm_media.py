#!/usr/bin/env python3
"""Candidate source-parity and ordinary-serving probes for native mlx-vlm VLMs."""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import inspect
import io
import json
import os
import tempfile
import traceback
import wave
from pathlib import Path

FAMILIES = {
    "gemma3n": ("image", "video", "audio"),
    "gemma4": ("image", "video"),
    "minicpmo": ("image", "audio"),
}
SETUP = {
    "qualification_mode": True,
    "mtp": False,
    "max_lanes": 2,
    "max_inflight": 4,
    "max_context": 8192,
    "prefill_step": 256,
    "cache_bytes": 2 << 30,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _harness_identity(family):
    root = Path(__file__).resolve().parents[1]
    return {
        "name": "scripts/qualify_native_vlm_media.py",
        "sha256": sha(Path(__file__).read_bytes()),
        "evaluator_sha256": sha(
            (root / "src/mlx2/media_qualification.py").read_bytes()
        ),
        "ownership_sha256": sha(
            (root / "scripts/qualification_gpu_ownership.py").read_bytes()
        ),
        "adapter_batching_sha256": sha(
            (root / "src/mlx2/adapters/multimodal.py").read_bytes()
        ),
        "adapter_contract_sha256": sha(
            (
                root
                / (
                    "src/mlx2/adapters/gemma4.py"
                    if family == "gemma4"
                    else "src/mlx2/adapters/mlx_vlm.py"
                )
            ).read_bytes()
        ),
    }


def _fixture(kind, changed=False):
    if kind == "image":
        from PIL import Image

        stream = io.BytesIO()
        Image.new("RGB", (32, 32), (220, 35, 80) if changed else (30, 100, 220)).save(
            stream, format="PNG"
        )
        data, mime = stream.getvalue(), "image/png"
    elif kind == "audio":
        import numpy as np

        x = np.arange(6400, dtype=np.float32)
        signal = np.sin(x * (0.07 if changed else 0.11)) * 0.08
        stream = io.BytesIO()
        with wave.open(stream, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes((signal * 32767).astype("<i2").tobytes())
        data, mime = stream.getvalue(), "audio/wav"
    elif kind == "video":
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="mlx2-vlm-qual-") as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 32)
            )
            if not writer.isOpened():
                raise RuntimeError("video writer unavailable")
            try:
                for index in range(17):
                    level = (
                        ((index * 37 + 40) % 240)
                        if changed
                        else ((index * 29 + 20) % 240)
                    )
                    writer.write(np.full((32, 32, 3), level, dtype=np.uint8))
            finally:
                writer.release()
            data, mime = path.read_bytes(), "video/mp4"
    else:
        raise ValueError(f"unsupported modality: {kind}")
    return {
        "data": data,
        "sha256": sha(data),
        "bytes": len(data),
        "uri": f"data:{mime};base64,{base64.b64encode(data).decode()}",
    }


def _request(
    kind,
    fixture,
    lead="Inspect this input.",
    tail="Describe the media briefly.",
    repeat_images=1,
):
    if kind == "image":
        parts = [
            {"type": "input_image", "image_url": fixture["uri"]}
            for _ in range(repeat_images)
        ]
    elif kind == "video":
        parts = [{"type": "input_video", "video_url": fixture["uri"]}]
    else:
        parts = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": fixture["uri"].split(",", 1)[1],
                    "format": "wav",
                },
            }
        ]
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": lead},
                    *parts,
                    {"type": "text", "text": tail},
                ],
            }
        ],
        "max_tokens": 8,
        "temperature": 0,
    }


def _adapter(family, path):
    if family == "gemma3n":
        from mlx2.adapters.mlx_vlm import Gemma3nAdapter

        return Gemma3nAdapter(path)
    if family == "minicpmo":
        from mlx2.adapters.mlx_vlm import MiniCPMOAdapter

        return MiniCPMOAdapter(path)
    from mlx2.adapters.gemma4 import (
        Gemma4A4BAdapter,
        Gemma431BAdapter,
        inspect_gemma4_artifact,
    )

    cls = (
        Gemma4A4BAdapter
        if inspect_gemma4_artifact(path)["variant"] == "26b-a4b"
        else Gemma431BAdapter
    )
    return cls(path)


def _row(mx, source, route):
    if tuple(source.shape) != tuple(route.shape):
        raise AssertionError("source/route logit shapes differ")
    mx.eval(source, route)
    diff = float(
        mx.max(mx.abs(source.astype(mx.float32) - route.astype(mx.float32))).item()
    )
    return {
        "shape": list(source.shape),
        "max_abs": diff,
        "argmax_match": int(mx.argmax(source[0, -1]).item())
        == int(mx.argmax(route[0, -1]).item()),
    }


def _array_leaves(value):
    if hasattr(value, "shape") and hasattr(value, "astype"):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _array_leaves(item)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _array_leaves(item)]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return [leaf for item in attributes.values() for leaf in _array_leaves(item)]
    return []


def _feature_parity(mx, reference, route):
    reference_arrays = _array_leaves(reference)
    route_arrays = _array_leaves(route)
    if not reference_arrays or len(reference_arrays) != len(route_arrays):
        return {
            "passed": False,
            "tensor_count": 0,
            "max_abs": None,
            "shapes_match": False,
        }
    if any(
        tuple(left.shape) != tuple(right.shape)
        for left, right in zip(reference_arrays, route_arrays)
    ):
        return {
            "passed": False,
            "tensor_count": len(reference_arrays),
            "max_abs": None,
            "shapes_match": False,
        }
    rows = []
    for left, right in zip(reference_arrays, route_arrays):
        mx.eval(left, right)
        delta = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
        finite = mx.all(mx.isfinite(left)) & mx.all(mx.isfinite(right))
        mx.eval(delta, finite)
        rows.append({"max_abs": float(delta.item()), "finite": bool(finite.item())})
    maximum = max(row["max_abs"] for row in rows)
    return {
        "passed": all(row["finite"] and row["max_abs"] <= 1e-4 for row in rows),
        "tensor_count": len(rows),
        "max_abs": maximum,
        "shapes_match": True,
        "rows": rows,
    }


def _enable_batching_observer(adapter, family):
    name, limit = {
        "gemma3n": (
            "_mlx2_gemma3n_vision_batching",
            adapter.video_policy.frame_batch_size,
        ),
        "minicpmo": (
            "_mlx2_minicpmo_vision_batching",
            adapter.media_policy.vision_batch_size,
        ),
    }.get(family, (None, None))
    if name is None:
        return None
    value = {
        "schema": "mlx2.encoder-batching-observation.v1",
        "policy_limit": limit,
        "enabled": True,
        "calls": [],
    }
    setattr(adapter.model._model, name, value)
    return value


def _batching_observation(adapter, family):
    model = adapter.model._model
    name = {
        "gemma3n": "_mlx2_gemma3n_vision_batching",
        "minicpmo": "_mlx2_minicpmo_vision_batching",
    }.get(family)
    if name is None:
        return None
    value = getattr(model, name, None)
    return json.loads(json.dumps(value)) if isinstance(value, dict) else None


def _ordinary_bound_method(source, name):
    """Bind the class descriptor without discarding its descriptor kind."""
    descriptor = inspect.getattr_static(type(source), name, None)
    if descriptor is None:
        return None
    getter = getattr(type(descriptor), "__get__", None)
    return getter(descriptor, source, type(source)) if getter else descriptor


def _parity_caches(source, routed, cache_factory=None):
    """Build independent reference and route caches through the runtime policy."""
    if cache_factory is None:
        from mlx2.runtime.models.cache import make_prompt_cache

        cache_factory = make_prompt_cache
    return cache_factory(source), cache_factory(routed)


def _parity(adapter, prepared, family):
    import mlx.core as mx

    source = adapter.model._model
    tokens = list(prepared["_mlx2_prompt_tokens"])
    ids = mx.array([tokens], dtype=mx.int32)
    sc, rc = _parity_caches(source, adapter.model)
    kwargs = dict(prepared.get("_mlx2_prefill_inputs") or {})
    ordinary_kwargs = {
        key: value for key, value in kwargs.items() if not key.startswith("_mlx2_")
    }
    patched = {}
    method_names = ["get_input_embeddings"]
    if family == "gemma3n":
        method_names.append("get_image_features")
    elif family == "minicpmo":
        method_names.append("get_vision_embedding")
    for name in method_names:
        current = getattr(source, name, None)
        ordinary = _ordinary_bound_method(source, name)
        if callable(current) and callable(ordinary):
            patched[name] = current
            setattr(source, name, ordinary)
    try:
        if family == "gemma3n":
            features = source.get_input_embeddings(ids, **ordinary_kwargs)
            ref = source.language_model(
                ids,
                inputs_embeds=features.inputs_embeds,
                per_layer_inputs=features.per_layer_inputs,
                cache=sc,
            )
        else:
            ref = source(ids, cache=sc, **ordinary_kwargs)
    finally:
        for name, method in patched.items():
            setattr(source, name, method)
    feature_parity = None
    feature_method = {
        "gemma3n": "get_image_features",
        "minicpmo": "get_vision_embedding",
    }.get(family)
    if feature_method and kwargs.get("pixel_values") is not None:
        route_method = getattr(source, feature_method)
        ordinary_method = _ordinary_bound_method(source, feature_method)
        pixels = kwargs["pixel_values"]
        if feature_method == "get_image_features":
            args = (pixels, source.vision_tower, source.config, source.embed_vision)
        else:
            args = (pixels, kwargs.get("tgt_sizes"))
        reference_features = ordinary_method(*args)
        mx.eval(reference_features)
        route_features = route_method(*args)
        feature_parity = _feature_parity(mx, reference_features, route_features)
    forward = getattr(adapter.model, "prefill_forward", adapter.model)
    got = forward(
        ids, cache=rc, **adapter.validate_prefill_inputs(prepared, tokens, dict(kwargs))
    )
    rows = [_row(mx, getattr(ref, "logits", ref), got)]
    token = int(mx.argmax(getattr(ref, "logits", ref)[0, -1]).item())
    decode_reference = source.language_model if family == "gemma3n" else source
    for _ in range(4):
        one = mx.array([[token]], dtype=mx.int32)
        ref = decode_reference(one, cache=sc)
        got = adapter.model(one, cache=rc)
        logits = getattr(ref, "logits", ref)
        rows.append(_row(mx, logits, got))
        token = int(mx.argmax(logits[0, -1]).item())
    from mlx2.adapters.vlm_runtime import _package_root, verify_contract

    contract = verify_contract(family, _package_root())
    return {
        "rows": rows,
        "feature_parity": feature_parity,
        "reference_revision": contract["reference_revision"],
        "reference_source_sha256": contract["source_sha256"],
    }


def _media_start(adapter, tokens):
    config = adapter.identity.get("config") or {}
    ids = set()
    for key in ("image_token_id", "video_token_id", "audio_token_id"):
        if type(config.get(key)) is int:
            ids.add(config[key])
    tokenizer = getattr(adapter.processor, "tokenizer", None)
    for key in ("image_token_id", "video_token_id", "audio_token_id"):
        val = getattr(tokenizer, key, None)
        if type(val) is int:
            ids.add(val)
    pos = [i for i, t in enumerate(tokens) if t in ids]
    if not pos:
        raise AssertionError("processor emitted no recognized media token")
    return min(pos)


def _drain(job):
    content = []
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise RuntimeError(str(event))
        delta = event.get("delta")
        if isinstance(delta, dict):
            piece = delta.get("content")
            if piece is not None:
                if not isinstance(piece, str):
                    raise TypeError("serving delta content must be text")
                content.append(piece)
        elif isinstance(delta, str):
            content.append(delta)
        elif delta is not None:
            raise TypeError("serving delta must be text or a content mapping")
        if "finish_reason" in event:
            return {
                "finish_reason": event["finish_reason"],
                "text": "".join(content),
                "receipt": event.get("receipt"),
                "request_id": getattr(job, "id", None),
            }


def _clear_apc_for_isolated_baseline(engine):
    operation = getattr(engine, "_exclusive_adapter_operation", None)
    if not callable(operation):
        raise RuntimeError("serving engine lacks an exclusive APC reset boundary")  # noqa: TRY004 - operational prerequisite
    operation(
        "qualification_baseline_apc_clear",
        lambda _adapter: engine.apc.clear(),
    )


def run_family(family, model_path):
    if family not in FAMILIES:
        raise ValueError(f"unsupported native VLM family: {family}")
    path = Path(model_path).resolve()
    adapter = _adapter(family, path)
    direct = {}
    fixtures = {}
    try:
        plain = adapter.prepare_multimodal_request(
            {"messages": [{"role": "user", "content": "Reply with exactly VLM_READY"}]}
        )
        plain["_mlx2_prompt_tokens"] = adapter.prompt_tokens(plain)
        plain["_mlx2_prefill_inputs"] = {}
        direct["ordinary_text_parity"] = _parity(adapter, plain, family)
        repeated_images = (
            adapter.media_policy.vision_batch_size + 1 if family == "minicpmo" else 1
        )
        for kind in FAMILIES[family]:
            original, changed = _fixture(kind), _fixture(kind, True)
            fixtures[kind] = (original, changed)
            prep = adapter.prepare_multimodal_request(
                _request(kind, original, repeat_images=repeated_images)
            )
            tokens = prep.get("_mlx2_prompt_tokens")
            end = prep.get("_mlx2_media_token_end")
            fingerprint = prep.get("_mlx2_media_fingerprint")
            if (
                not isinstance(tokens, list)
                or type(end) is not int
                or not 0 < end < len(tokens)
                or not fingerprint
            ):
                raise AssertionError(f"{kind} processor boundary/fingerprint missing")
            changed_prep = adapter.prepare_multimodal_request(
                _request(kind, changed, repeat_images=repeated_images)
            )
            ct = changed_prep.get("_mlx2_prompt_tokens")
            ce = changed_prep.get("_mlx2_media_token_end")
            if (
                not isinstance(ct, list)
                or type(ce) is not int
                or not 0 < ce < len(ct)
                or not changed_prep.get("_mlx2_media_fingerprint")
            ):
                raise AssertionError(
                    f"{kind} changed-media boundary/fingerprint missing"
                )
            if (family == "gemma3n" and kind == "video") or (
                family == "minicpmo" and kind == "image"
            ):
                _enable_batching_observer(adapter, family)
            direct[kind] = {
                "fixture": {
                    "sha256": original["sha256"],
                    "payload_base64": base64.b64encode(original["data"]).decode(),
                    "bytes": original["bytes"],
                    "changed_sha256": changed["sha256"],
                    "changed_payload_base64": base64.b64encode(
                        changed["data"]
                    ).decode(),
                    "prompt_tokens": len(tokens),
                    "media_token_end": end,
                    "media_token_start": _media_start(adapter, tokens),
                    "changed_prompt_tokens": len(ct),
                    "changed_media_token_end": ce,
                    "changed_media_token_start": _media_start(adapter, ct),
                    "media_fingerprint": str(fingerprint),
                    "changed_media_fingerprint": str(
                        changed_prep["_mlx2_media_fingerprint"]
                    ),
                },
                "parity": _parity(adapter, prep, family),
                "encoder_batching": _batching_observation(adapter, family),
            }
        artifact = adapter.identity["fingerprint"]
        runtime = adapter.mlx_vlm_runtime
        revision = runtime["reference_revision"]
        source_sha = runtime["source_sha256"]
        provenance = adapter.mlx_vlm_build
        defaults = adapter.sampling_defaults.as_dict()
    finally:
        adapter.close()
    del adapter
    gc.collect()
    import mlx.core as mx

    mx.clear_cache()
    from mlx2.serving import ServingEngine

    engine = ServingEngine(str(path), **SETUP)
    try:
        if not engine.ready.wait(180) or engine.error:
            raise RuntimeError(f"engine load failed: {engine.error}")
        if engine.snapshot.get("artifact") != artifact:
            raise AssertionError("direct/serving artifact identities differ")
        serving_runtime = engine.snapshot.get("runtime")
        if (engine.snapshot.get("dependency_builds") or {}).get(
            "mlx_vlm"
        ) != provenance:
            raise AssertionError("installed source dependency identity differs")
        ordinary = _drain(
            engine.submit({"prompt": "Reply with exactly VLM_READY", "max_tokens": 8})
        )
        batch_prompts = ("Two plus two is", "Three plus three is")

        def batch_request(prompt):
            return {"prompt": prompt, "temperature": 0, "max_tokens": 8}

        def batch_prompt_sha(prompt):
            return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        batch_reference = []
        for prompt in batch_prompts:
            _clear_apc_for_isolated_baseline(engine)
            row = _drain(engine.submit(batch_request(prompt)))
            batch_reference.append(
                {
                    **row,
                    "prompt": prompt,
                    "prompt_sha256": batch_prompt_sha(prompt),
                }
            )
        _clear_apc_for_isolated_baseline(engine)
        jobs = [
            (prompt, engine.submit(batch_request(prompt))) for prompt in batch_prompts
        ]
        batch_rows = []
        for prompt, job in jobs:
            batch_rows.append(
                {
                    **_drain(job),
                    "prompt": prompt,
                    "prompt_sha256": batch_prompt_sha(prompt),
                }
            )
        batch_status = engine.status()
        batch = {
            "reference_rows": batch_reference,
            "rows": batch_rows,
            "peak_observed_width": (batch_status.get("counts") or {}).get(
                "peak_observed_width", 0
            ),
        }
        arms = {}
        apc_before = dict(engine.apc.apc_stats)
        serving_batching = _enable_batching_observer(engine.adapter, family)
        for kind in FAMILIES[family]:
            original, changed = fixtures[kind]
            body = _request(kind, original, repeat_images=repeated_images)
            arms[kind] = {
                name: _drain(engine.submit(req))
                for name, req in (
                    ("cold", body),
                    ("warm", body),
                    ("changed", _request(kind, changed, repeat_images=repeated_images)),
                    ("return_original", body),
                )
            }
        apc_after = dict(engine.apc.apc_stats)
        apc_delta = {
            key: apc_after.get(key, 0) - apc_before.get(key, 0)
            for key in set(apc_before) | set(apc_after)
        }
        settings = engine.snapshot.get("settings")
        if not isinstance(settings, dict) or not isinstance(serving_runtime, dict):
            raise TypeError("serving runtime/settings identity missing")
        producer_sha = sha(Path(__file__).read_bytes())
        return {
            "schema": "mlx2.media-serving-qualification.v1",
            "family": family,
            "model_type": family,
            "qualification_harness": _harness_identity(family),
            "producer_sha256": producer_sha,
            "source_revision": revision,
            "source_runtime": runtime,
            "mlx_vlm_runtime": runtime,
            "reference_revision": revision,
            "reference_source_sha256": source_sha,
            "artifact": artifact,
            "artifact_sha256": artifact,
            "runtime": serving_runtime,
            "serving_runtime": serving_runtime,
            "settings": settings,
            "sampling_defaults": defaults,
            "ordinary_reference": ordinary,
            "batch": batch,
            "serving_encoder_batching": serving_batching,
            "direct": direct,
            "arms": arms,
            "apc_stats_delta": apc_delta,
            "checks": {},
            "passed": False,
            "status": "candidate_evidence",
        }
    finally:
        engine.close()


def evaluate_native_report(report, **expected):
    from mlx2.media_qualification import evaluate_native_vlm_traces

    return evaluate_native_vlm_traces(report, **expected)


def _prove_lease(generation, *, owner_lock, task_id):
    from qualification_gpu_ownership import require_qualification_lease

    return require_qualification_lease(
        task_id=task_id,
        cpg_owner_lock=owner_lock,
        generation=generation,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=tuple(FAMILIES))
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--cpg-owner-lock", required=True, type=Path)
    parser.add_argument("--cpg-task", required=True)
    args = parser.parse_args(argv)
    try:
        owner = _prove_lease(
            args.generation, owner_lock=args.cpg_owner_lock, task_id=args.cpg_task
        )
        if not args.model_path.is_dir():
            raise FileNotFoundError(f"model artifact missing: {args.model_path}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        result = run_family(args.family, str(args.model_path))
        result["gpu_lease_owner"] = owner
        from mlx2.media_qualification import (
            NATIVE_VLM_MEDIA_CHECKS,
            evaluate_native_vlm_report,
        )

        checks = evaluate_native_vlm_report(
            result,
            expected_family=args.family,
            expected_harness=_harness_identity(args.family),
        )
        result["checks"] = {
            name: {"passed": value, "evidence": "recomputed_from_native_trace"}
            for name, value in sorted(checks.items())
        }
        result["passed"] = set(checks) == set(
            NATIVE_VLM_MEDIA_CHECKS[args.family]
        ) and all(checks.values())
        result["status"] = (
            "candidate_checks_passed" if result["passed"] else "candidate_checks_failed"
        )
        print(json.dumps(result, indent=2, default=str), flush=True)
        return 0 if result["passed"] else 1
    except Exception as exc:  # noqa: BLE001 - CLI must emit a failure report
        print(
            json.dumps(
                {
                    "family": args.family,
                    "passed": False,
                    "status": "producer_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                },
                indent=2,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
