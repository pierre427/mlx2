# Clef decision serving

The multi-family service overview is in
[DECISION-MODELS.md](DECISION-MODELS.md). This page records Clef-specific
artifact and joint-head details.

`mlx2-decisions` is a standalone, prefill-only decision service. It does not
register Clef as a causal LM, enter the generation scheduler, or introduce a
second prefix cache. The initial implementation supports prepared Clef-Flash
9B and Clef 27B MLX artifacts through `POST /v1/systemone`.

```bash
mlx2-decisions --model /path/to/clef-flash-MLX-8bit \
  --served-model-name clef-flash \
  --qualification /path/to/clef-qualification.json
```

```json
{
  "model": "clef-flash",
  "state": {"message": "I was charged twice. Refund me now."},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What does the customer want?",
      "criteria": {
        "refund": "wants money back",
        "track": "asks where an order is"
      }
    },
    "angry": {
      "type": "noul",
      "instructions": "Is the customer angry?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["can wait", "today", "right now"]
    }
  }
}
```

Every successful response includes `mlx2`, a route receipt distinguishing
implementation, qualification, selection, and observed execution. Capability
refusals include the same receipt with `observed_used: false`.

## Artifact contract

The artifact must have:

- root `model_type: "clef"`;
- the published Qwen3.5 9B or Qwen3.8 27B text geometry;
- the published `head_config` geometry;
- `model.safetensors.index.json` containing `language_model.*`, `head.*`, and
  optionally `vision_tower.*` tensors;
- a local tokenizer.

Unknown tensor namespaces, MTP heads, escaped shard paths, unsupported Qwen
geometry, missing trained LM heads, or incomplete head metadata fail before
tensor loading. Vision
tensors are deliberately discarded before materialization in this slice.

## Current status

| State | Value |
|---|---|
| Implemented | Text-only joint-schema prompt, head, loader and HTTP route |
| Qualified | No on this public source; the private-main source-bound run qualified Clef-Flash 9B revision `8f9beb2e63474d547c70810b41b468d3b4a84932` |
| Selected | Only when the separate `mlx2-decisions` process is launched |
| Observed-used | Per-process counter and per-response route receipt |
| Real-model smoke | Passed on pinned Clef-Flash 9B MLX 8-bit artifact, 2026-10-08 |
| Full promotion | Historical private-main run passed 13/13 gates with exact source-derived parity and validated Prometheus metrics; public-safe summary only in `qualification/runs/decision-model-full-qualification-20261009/` |

Images, video, generation, APCv2 reuse and multi-model residency are not
implemented. The package and registry are intentionally
family-neutral so later JEV and candidate-head implementations can reuse the
request, transport, lifecycle and receipt contracts without sharing Clef's
tensor math.
