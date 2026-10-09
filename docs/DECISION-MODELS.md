# Decision-model serving

`mlx2-decisions` is a standalone, prefill-only service for typed classifiers.
It does not register these artifacts as causal LMs, enter the generation
scheduler, or introduce a prefix-cache engine. One process owns one prepared
artifact and exposes `POST /v1/systemone`.

```bash
mlx2-decisions --model /path/to/prepared-artifact \
  --served-model-name local-decider
```

A qualified route also supplies an exact qualification receipt created from
that checkout:

```bash
PYTHONPATH="$PWD/src" python -m mlx2.decisions.server \
  --model /path/to/models--nativ-community--clef-flash-MLX-8bit/snapshots/8f9beb2e63474d547c70810b41b468d3b4a84932 \
  --served-model-name clef_flash_9b_mlx_8bit \
  --max-connections 2 \
  --max-request-bytes 4194304 \
  --qualification /path/to/source-bound-qualification-receipt.json
```

Run that command only from the frozen checkout whose `src/mlx2` hash matches
the receipt. Qualification receipts are deliberately not included in this
public projection because its source identity differs from the qualified
private-main source. Any source-changing merge or edit requires a new
qualification run. The model path must likewise resolve to the exact recorded
Hub snapshot revision.

The service validates the receipt against the approved qualification producer,
installed mlx2 runtime source hash, Python/MLX/tokenizer runtime, MLX binding,
native dylibs and Metal library,
artifact revision
and fingerprint, served name, capabilities, request-size bound, and connection
bound. Any mismatch fails startup; omitting the receipt keeps the same model
servable but truthfully reports `unqualified`.

The following exact artifacts are implemented and can be selected by the
standalone service. This public source is unqualified until it receives its own
qualification run and matching receipts:

| Family | Supported artifact | Scoring path | Initial context |
|---|---|---|---:|
| Clef | Clef-Flash 9B MLX 8-bit, revision `8f9beb2e...` | joint schema head | 16,384 |
| Decision 2.0 | Lux 9B full package, revision `e0cd1389...` | candidate/query head | 16,384 |
| pplx-decider | v1 27B, revision `b01a5cba...` | 255-row trained readout | 32,768 |
| JEV | JEV 9B MLX 8-bit, revision `c0eafb30...` | calibrated verbalizers | 16,384 |

On 2026-10-09 the private-main source identity
`07178885cc7bd8f52fe7a93682528c32a0e9953f8ff7172b3bedb89b1aeadf0b`
passed 13/13 production gates for each family (52/52 total). It verified exact
prompt/head/post-processing parity against source-derived references over the
same runtime trunk and weights, matching token counts, restart determinism, context
boundaries, fail-closed behavior, route receipts, qualified startup, a real
qualified request, and Prometheus reconciliation. Receipts, raw metrics, GPU
lock records, and full evidence remain source-bound and private. A public-safe
aggregate and bounded Prometheus scrapes are in
[`qualification/runs/decision-model-full-qualification-20261009/`](../qualification/runs/decision-model-full-qualification-20261009/README.md).
That historical result does not qualify this public projection. It is not a
performance comparison and does not imply that a persistent endpoint was
deployed.

The service selects a family from strict artifact metadata, never from a
model-name substring. Decision 2.0, pplx-decider, and JEV share the Qwen trunk
owner and service lifecycle, while their trained prompt and tensor math remain
in separate modules under `mlx2.decisions.candidates`.

## Request

```json
{
  "model": "local-decider",
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
  },
  "truncate": true
}
```

The trained family calibration is always used, so request temperature must be
`1`. Images, video, generation, and unknown artifact layouts fail closed.
JEV score questions require exactly six levels; Decision 2.0 and pplx-decider
accept two to ten. The shared request cap is narrowed further where a trained
family head requires it.
`truncate` defaults to `true`; candidate-family truncation is an mlx2 service
extension rather than reference behavior. Set it to `false` to fail closed on
an overlong state. Truncation also fails closed if fitting the fixed prompt
would remove the entire nonempty state. A request may contain at most 64
questions, 255 choice options per question, and 10 score levels before stricter
family-specific checks. Candidate-family
requests also have an aggregate prompt budget of eight times the model's
per-question context, preventing a large question matrix from multiplying
prefill work without bound.

The HTTP listener applies absolute header and body deadlines, including the
period before the first byte arrives. Malformed or non-finite JSON, excessive
nesting, invalid Unicode, embedded media parts, and unsupported tokenizer
integrity states are refused before model execution. Rendered state is capped
at 1 MiB, and reserved `<|...|>` model-control tokens are refused rather than
being interpreted as prompt structure.

Successful inference and errors emitted after decision-application dispatch
carry an `mlx2` receipt with separate `implemented`, `qualified`, `selected`,
and `observed_used` fields. A decision validation refusal has
`observed_used: false`. Successful inference sets it to `true` and increments
the process counters. Host, Origin, or API-key rejection occurs in the shared
HTTP security layer before decision dispatch and therefore has no route receipt.

## Prometheus telemetry

`GET /metrics` publishes the same Prometheus text format used by the main mlx2
server. It has bounded family, variant, outcome, route, method, and status-class
labels; request IDs, prompts, served aliases, and full artifact fingerprints
are never labels. The decision surface exports:

- `mlx2_decision_route_info` and `mlx2_decision_qualified`;
- successful, refused, and failed request counters;
- cumulative input-token counters and histograms;
- serialized decision-dispatch latency histograms;
- current and peak in-flight work;
- model-load duration, process CPU time, process start time, and peak resident
  memory;
- bounded HTTP request counters and latency histograms, including the
  `systemone` route.

Metrics are host-side observations. Scraping does not synchronize MLX device
work or make destructive counter reads. Qualification verifies that scrape
counters reconcile with status and route receipts; it does not turn those
operational statistics into a controlled performance claim.

## Artifact boundaries

- Decision 2.0 currently accepts only the full Lux 9B package with its local
  Qwen3.5 backbone and exact ten-tensor F32 shared head. LoRA packages must be
  merged first; calibrated package variants are refused until implemented.
- pplx-decider requires the official bare 27B Qwen3.8 backbone,
  `decision_config.json`, and the exact `255 x 5120` readout. Vision tensors
  may be present but are discarded before materialization.
- JEV requires a prepared MLX text artifact whose merged weights and
  calibration are represented by `model_type: jev_text`. The ordinary LM head
  is retained only as JEV's trained verbalizer readout.
- Clef's separate joint-schema contract is detailed in
  [CLEF-DECISIONS.md](CLEF-DECISIONS.md).

Hugging Face cache snapshots bind receipts to the detected 40-hex revision and
the declared Hub blob filename identity of every weight shard. Blob payloads
are not re-hashed at startup, so this is explicitly labelled
`hub-blob-identity`, not a locally verified content hash. Non-Hub artifact
fingerprints cover metadata, lexical shard paths, and sizes and are labelled
`layout-metadata`; they are not represented as full content hashes.

CPU contracts and pinned-tokenizer checks are not model qualification. A route
remains unqualified until its real artifact loads and its default questions
pass functional oracles plus a source-derived prompt/scoring comparison. That
comparison executes pinned upstream Clef/JEV code where practical and checked
translations for Decision 2.0/pplx over the same frozen backbone and weights.
It validates prompt construction, family head math, and answer formatting; it
is not an independently loaded trunk, cannot by itself detect a shared trunk or
weight-mapping defect, and is not a performance claim.
