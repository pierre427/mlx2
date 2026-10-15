`continuation_pool` selects up to 15 complete continuation sequences in the
external draft route. The limit counts whole sequences. Fifteen paths of depth
15 may require 225 target token nodes; no 15-node budget is substituted.

```json
{
  "draft_model": "/absolute/path/to/pinned/drafter",
  "num_draft": 3,
  "continuation_pool": {"limit": 15}
}
```

The standard Qwen3 adapter also accepts a selectable resident portfolio. A
single entry and a multi-entry list use the same contract; the latter enables
`continuation_pool` automatically:

```json
{
  "draft_models": [
    {"id": "xpress-main", "path": "/models/xpress", "tile_rows": 4},
    {"id": "lilicorr-main", "path": "/models/lilicorr", "lane_cost": 1}
  ],
  "num_draft": 4,
  "continuation_pool": {
    "limit": 15,
    "admission": {
      "max_active_providers": 5,
      "lane_slots": 5,
      "tile_rows": 75
    }
  }
}
```

Every artifact keeps its adapter-owned tensor math, cache slice and revision.
The host compares complete paths only. It admits work from lagged committed
acceptance per reserved tile row, subject to provider/lane/tile budgets; the
current proposal's private score cannot admit itself. Identical paths are
verified once while retaining every contributing provider. The processed
target law remains the output authority.

The default inventory includes the bound external head and exact prompt
lookup. It also includes native MTP when an actual target head remains loaded
and supplies its native protocol. An explicit `sources` list restricts the
inventory to resident portfolio IDs, `external`, `prompt_lookup` or
`native_mtp`; an unavailable request fails. A known mechanism label does not
load a checkpoint or fabricate a provider. DFlash2 is a complete-path tree
provider. A FOLD-configured DFlash2 instance is reported separately as
`dflash2_fold`, because FOLD changes context retention, not the proposal law.
DSpark is supported through an adapter-owned provider declaring
`continuation_protocol = "dspark"` and implementing
`propose_complete_paths(context, limit)`. The bridge supplies that method a
frozen `ContinuationContext` whose draft cache is a private deep copy. The
existing LFM2.5-VL mlx-vlm artifact is still an offline unqualified candidate
because no adapter currently supplies that explicit protocol and its cache
transaction is not yet an mlx2 serving cache contract. DPara likewise remains
unavailable without a resident compatible provider.

XPress paths come from the existing final Jacobi refined block. LiLiCoRR paths
come from its existing conditional correlation lattice, with the actual
preceding candidate column retained at every depth. Backbone unary logits are
not added to correlation logits. Native MTP uses the loaded target's public
teacher-forced backbone/head protocol and the actual current anchor. Each beam
branch owns a private head-cache copy. Prompt lookup enumerates distinct exact
matching continuations from the committed request prefix and current anchor.

Provider beam log scores rank candidates within their own source. They are
labelled proxies and are never combined into a joint proposal probability.
The pool's source critic learns from earlier committed target observations:
bounded process-wide in-memory global mechanism priors, revision-bound model/source observations, and hashed
session residuals with shrinkage. Raw cold logits from different heads are not
treated as comparable confidence. Costs require explicit compatible inputs;
source ranking alone is not a performance measurement.

Effective refinement passes, per-layer draft windows, proposal depth and
normalized adaptive policy are bound before constructing pool sessions.
Different configurations have distinct source entries at model and session
ranking tiers; the broad mechanism prior can still transfer across them.
Artifact revisions remain separate from these execution settings.

The existing external backbone advances its committed feature cache once.
Additional model paths recompute from private copies of the cache before that
advance, using the same pending target features. This adds work and currently
claims no saved backbone compute. Target verification draws from the processed
target law first, follows matching path prefixes, and publishes only the
selected accepted state. Request, round, source, tokenizer, target, session
and causal-context bindings travel with each selection. Failed selections
must be discarded; critic and LiLiCoRR training feedback are published only
after the whole outer request round commits.

Prompt lookup is fallback-only when a model drafter is admitted. Providers
with enough committed evidence and low expected accepted depth are parked and
periodically re-probed. PLD then supplies repeated spans when available. If it
has no match, one bounded resident re-probe prevents the content latch from
turning ordinary prose into a route failure. Adapters that declare prompt
lookup now select its cost-aware latch by default; explicit ordinary decode is
still the reference path.

`lilicorr_feedback` is a separate explicit policy requiring a resident
correlator head. The adapter starts its manager with the final route binding
and closes its owned worker on adapter shutdown. A bounded capture from the
public backbone supplies the teacher-labelled lattice; private provider
recomputations do not replace it. The manager trains an isolated CPU clone and
exports an unqualified, unselected head artifact. Capture tickets authenticate
immutable feature bytes and are one-shot; expired tickets cannot re-enter the
buffer. Each manager owns an isolated artifact namespace. Failed export, spawn,
timeout and malformed-result paths retain bounded artifacts and clear owned
workers. Both parent and worker check a conservative training memory estimate
before training allocation (`max_training_bytes`, default 512 MiB, maximum
2 GiB, also capped at half currently available memory). This is a launch guard,
not an allocator-enforced memory ceiling. It does not change live
weights or APCv2 revisions.

CPU provider evidence uses tiny random models: XPress beam versus exhaustive
refined-block enumeration, LiLiCoRR complete lattice paths, native QSA MTP
branches, multiple copy matches, B2 cache immutability, source admission and
revision refusal, and final-binding feedback lifetime. This is implemented
protocol evidence, not trained hybrid checkpoint or GPU qualification.


For example, a route with an actual resident correlator can enable:

```json
{
  "continuation_pool": {"limit": 15},
  "lilicorr_feedback": {
    "directory": "/absolute/path/to/shadow-heads",
    "max_examples": 32,
    "max_bytes": 16777216,
    "min_examples": 8,
    "train_every": 16,
    "steps": 25,
    "timeout_seconds": 120,
    "max_artifacts": 4,
    "max_training_bytes": 536870912
  }
}
```

These policy fields supplement a valid `draft_model` and `num_draft` policy.
A head-only shadow output cannot be selected as a full serving companion.


Adaptive verification uses `stable-truncated-prefix-dedup-v2`. All complete
continuations remain ranked and available for reached-prefix feedback. After
width/depth admission, identical truncated paths share one physical branch;
receipts carry both semantic counts and physical-to-semantic mappings. Native
standard decoding enables this candidate only for selected BF16 row-exact
verification, and steering disables it. Ordinary, FP32/FP16, hybrid and unknown
math keep literal rows unless an explicit target equivalence contract is given.
Costs use the unique physical width; missing cost coverage keeps the full
conservative shape. This reduction is not a throughput claim.

An ordinary final token retains a read-only snapshot of already committed
ranking evidence. It cannot add labels or replay a feedback ticket. Speculative
rounds still refresh the snapshot only after every request commits.
