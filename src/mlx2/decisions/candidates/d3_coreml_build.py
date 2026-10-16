"""Phase-1 feasibility: the D3 edge decision trunk as a Core ML mlprogram.

The D3 text backbone (pinned ``vllm-sr/d3-edge``) is written directly with
the Core ML MIL builder from the verified BF16 safetensors bytes and run with
compute units ``CPU_ONLY`` or ``CPU_AND_NE`` only (the GPU is never used).
Graph contract, per fixed sequence bucket ``S``:

* inputs ``embeddings`` fp16 ``[1, S, 1024]`` (the host gathers the 248K-row
  embedding; BF16 rows are exact in fp16) and ``mask`` int32 ``[1, S]``
  (1 = real token, right padding only; pad rows carry a real, finite
  embedding so no NaN can arise from a zero RMSNorm row);
* full-attention layers are non-causal; pads are masked as keys with an
  fp16-safe additive ``-1e4``; GDN layers keep their causal scan, so right
  pads never reach a real position;
* GDN prefill runs in-graph in the chunkwise-parallel WY form (matmul-only
  intra-chunk algebra, exact cancellation-free decay sums, a sequential pass
  over ``S / chunk`` chunks);
* the readout position is the last real token (D3 ``pooling: last``),
  selected in-graph as ``mask[t] - mask[t + 1]``; outputs are the final
  normed hidden state ``[1, 1024]`` and the 255 trained readout logits.

The host applies the per-question option subset, temperature and softmax in
float32 exactly as ``D3Engine._predict_locked`` does.

This module is evidence tooling for a feasibility probe, not a serving
route. Compute-plan counts are anticipated placement, never observed
execution. Importing it imports no Core ML, MLX or NumPy.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
from pathlib import Path

SOURCE_REVISION = "cab71e6ad3461b8086d51ddb572c656babd63b0b"
BACKBONE_FILE = "model-00001-of-00001.safetensors"
BACKBONE_SHA256 = "91bb8618450a7f628f1432bbced76bc19ee7346c93d191a228fdd33c25a57a9d"
READOUT_FILE = "readout.safetensors"
READOUT_SHA256 = "de33d5dc795bbd77168e40bfadc340cca9b9f163b88744c10e1a788f4e166d3d"
DEFAULT_ARTIFACT = (
    Path.home() / ".cache/huggingface/hub/models--vllm-sr--d3-edge/snapshots" / SOURCE_REVISION
)

HIDDEN = 1024
INTERMEDIATE = 3584
LAYERS = 12
HEADS, KV_HEADS, HEAD_DIM = 8, 2, 256
ROPE_DIMS = 64
GDN_HEADS, GDN_DK, GDN_DV = 16, 128, 128
GDN_KEY = GDN_HEADS * GDN_DK
GDN_VALUE = GDN_HEADS * GDN_DV
CONV_KERNEL = 4
READOUT_ROWS = 255

DEFAULT_BUCKETS = (128, 256, 512, 1024)
MAX_BUCKET = 1024
DEFAULT_CHUNK = 64
INVERSE_BLOCK = 16
# fp16-safe additive key mask: exp(-1e4) is exactly zero in float16/float32.
MASK_NEGATIVE = -1.0e4
PLAN_NOTE = "anticipated placement, not observed execution"
# Same compute-op set as runtime.coreml_encoder_build (MiniLM) for comparability.
PLACEMENT_COMPUTE_OPS = frozenset({"matmul", "conv", "linear", "softmax", "layer_norm", "gelu"})
GPU_LOCK_OWNER = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")


def gpu_lock_state(path=GPU_LOCK_OWNER):
    """Read (never take) the shared GPU lock; benchmark-class holders forbid latency runs."""
    try:
        owner = json.loads(Path(path).read_text())
    except (FileNotFoundError, NotADirectoryError):
        return {"held": False, "label": None, "latency_allowed": True}
    except (OSError, ValueError):
        return {"held": None, "label": None, "latency_allowed": False}
    label = str(owner.get("label", "")) if isinstance(owner, dict) else ""
    blocked = any(word in label.lower() for word in ("e2e", "bench", "ladder"))
    return {"held": True, "label": label, "latency_allowed": not blocked}


# ------------------------------------------------------------ parity cases ---
# Realistic, varied D3 requests (never repeated-string filler). Each entry is
# (state, question) in the public request shape; ``score`` questions go
# through ``d3.prepare_question`` exactly as served.

_TICKET_THREAD = (
    "Ticket #48213, opened Monday 09:12 by Dana Whitfield (Pro plan, annual).\n"
    "Dana: Since the update on Friday our team can no longer export invoices as PDF. The button "
    "spins for about a minute and then shows 'Something went wrong'. CSV export still works.\n"
    "Agent Marco: Thanks Dana. Could you tell me which browser you use and whether the problem "
    "happens for every invoice or only some?\n"
    "Dana: Chrome 129 on macOS and Edge on Windows, same result. It fails for invoices with more "
    "than about forty line items. Short invoices export fine.\n"
    "Agent Marco: I reproduced it with a 52-line invoice. I have escalated to engineering with "
    "the request IDs. As a workaround you can print to PDF from the invoice preview.\n"
    "Dana: The print workaround loses our logo and the payment QR code, so our accountant cannot "
    "use those files. Month-end close is Thursday and we have 230 invoices to send.\n"
    "Agent Marco: Understood. Engineering confirmed the renderer times out on long tables after "
    "the font change in Friday's release. A fix is scheduled for Wednesday evening.\n"
    "Dana: Wednesday evening is cutting it close. If it slips we will have to send invoices by "
    "hand, which takes two people a full day. Can you at least credit us for the lost time?"
)

_INCIDENT_LOG = {
    "service": "checkout-api",
    "region": "eu-west-1",
    "window_utc": ["2026-10-14T02:10:00Z", "2026-10-14T02:58:00Z"],
    "error_rate_percent": [0.2, 0.3, 14.8, 31.5, 29.9, 12.1, 0.6],
    "p99_latency_ms": [180, 190, 2400, 5100, 4900, 1300, 210],
    "events": [
        "02:14 primary database stopped accepting writes (volume 100% full)",
        "02:15 automated failover promoted replica db-2",
        "02:16 replica inherited disk alarm threshold; WAL cleanup job disabled since migration",
        "02:31 on-call engineer paged manually by support lead",
        "02:44 WAL cleanup re-enabled, 410 GB reclaimed",
        "02:52 payment service replayed 11,204 failed orders",
    ],
    "customer_reports": 386,
    "data_loss": False,
}

_LEASE_CLAUSE = (
    "Section 14. Early termination. The Tenant may terminate this lease before the end of the "
    "initial term by giving the Landlord not less than ninety (90) days' written notice and paying "
    "an early termination fee equal to two months of the then-current base rent. The fee is waived "
    "if the Tenant terminates because of a documented job relocation of more than one hundred "
    "kilometres, provided the notice includes a letter from the employer. Any prepaid rent for the "
    "period after the termination date shall be refunded within thirty (30) days. The security "
    "deposit is returned under Section 9 and is not applied against the termination fee unless the "
    "Tenant agrees in writing. Notice delivered by email is valid only if the Landlord acknowledges "
    "receipt within five business days; otherwise notice must be sent by registered mail."
)

_MEETING_NOTES = (
    "Planning sync, platform team, 15 October.\n"
    "Attendees: Priya (lead), Tomas, Aiko, Femi, Laura (product).\n"
    "1. Release 7.3: Tomas reported that the migration tool passed on staging but took four hours "
    "on the largest customer snapshot. Aiko suggested batching index rebuilds, which cut the time "
    "to seventy minutes in her local test. Decision: ship 7.3 with batched rebuilds, behind a flag "
    "that defaults on for new tenants only.\n"
    "2. On-call load: Femi summarized last month: 41 pages, 29 outside business hours, 12 caused by "
    "the flaky disk-usage alert. Priya asked for the alert threshold to move from 80% to 90% with a "
    "rate-of-change condition. Femi will draft the change; Laura asked that customer-visible alerts "
    "stay unchanged.\n"
    "3. Hiring: two backend candidates are in final rounds. Laura wants one of them to start before "
    "the January reporting work. Priya cautioned that onboarding during the holidays is slow and "
    "proposed a buddy rotation.\n"
    "4. Tech debt: Aiko listed the three oldest services still on the deprecated queue client. "
    "Tomas estimated two sprints to migrate all three. The team agreed to migrate the billing "
    "consumer first because it causes most retries.\n"
    "5. Budget: cloud spend rose 11% month over month, mostly from staging clusters left running "
    "over weekends. Femi will add an automatic weekend scale-down for non-production clusters.\n"
    "Action items: Tomas owns the 7.3 release notes; Femi owns the alert change and staging "
    "scale-down; Aiko writes the queue migration plan; Priya schedules interviews; Laura confirms "
    "the January reporting scope with finance by Friday."
)

_RESEARCH_ABSTRACTS = (
    "Abstract 1. We study whether small language models can act as routers for larger ones. On "
    "eleven public benchmarks a 0.8B router trained on preference labels selects the cheaper model "
    "for 62% of queries while keeping 97% of the larger model's accuracy. Errors concentrate on "
    "multi-step arithmetic and long-context retrieval.\n"
    "Abstract 2. Coastal wetlands store carbon at rates several times higher than temperate "
    "forests per hectare. Using sediment cores from forty-two sites, we estimate burial rates and "
    "show that drained marshes lose most stored carbon within two decades. Restoration recovers "
    "burial within eight years where tidal flow is reinstated.\n"
    "Abstract 3. A randomized trial in 1,240 adults compared morning and evening dosing of a "
    "once-daily blood pressure medication. After twelve months there was no significant difference "
    "in cardiovascular events, but evening dosing produced slightly lower night-time systolic "
    "pressure and more frequent reports of nocturia.\n"
    "Abstract 4. We present a compiler pass that fuses attention, normalization and gating for "
    "on-device inference. On three phone chips the pass reduces memory traffic by 38% and latency "
    "by 21% for sequences up to two thousand tokens, with no accuracy change.\n"
    "Abstract 5. Using archival shipping manifests from 1820 to 1860, we reconstruct the spread of "
    "potato varieties across northern Europe. Port cities with frequent Atlantic traffic adopted "
    "new varieties up to fifteen years earlier than inland markets, and adoption slowed sharply "
    "after the blight years."
)

_OPS_RUNBOOK = (
    "Runbook: elevated queue lag on the ingestion pipeline.\n"
    "Symptoms: consumer lag above 50,000 messages for more than ten minutes, dashboards show "
    "processing time per batch rising while CPU on the consumers stays below 40%. Customers see "
    "delayed analytics but no data loss.\n"
    "Step 1. Check whether a deploy happened in the last hour. If it did and lag started within "
    "fifteen minutes of it, roll back first and investigate afterwards.\n"
    "Step 2. Inspect the downstream warehouse. Lock contention on the events table is the most "
    "common cause; the warehouse console lists blocked sessions. Kill long-running ad hoc queries "
    "from analysts only after notifying the data channel.\n"
    "Step 3. If the warehouse is healthy, look for a hot partition. A single tenant sending a burst "
    "can pin one consumer; the per-partition lag panel shows this as one bar far above the rest. "
    "Temporarily raise that tenant's rate limit penalty rather than adding consumers, because extra "
    "consumers do not help a single hot partition.\n"
    "Step 4. If lag is spread across partitions and the warehouse is fine, scale consumers from six "
    "to twelve. Scaling beyond twelve requires more partitions and a maintenance window.\n"
    "Step 5. Record the timeline, the cause and the action in the incident channel. Open a ticket "
    "for any manual step you had to take so it can be automated.\n"
    "Escalation: page the data platform on-call if lag exceeds 500,000 messages or if any step "
    "above fails. Never purge the queue; purging loses customer events permanently.\n"
    "Context for this alert: deploy of ingestion-consumer v2.18 at 13:05, lag started 13:12, "
    "warehouse console shows no blocked sessions, per-partition lag is roughly even across all "
    "thirty-two partitions, consumer CPU 35%."
)

_DEPARTMENTS = [
    "Accounts payable", "Accounts receivable", "Payroll", "Tax compliance", "Treasury",
    "Procurement", "Vendor management", "Facilities", "Workplace safety", "Travel desk",
    "IT helpdesk", "Network operations", "Identity and access", "Security operations",
    "Data platform", "Analytics", "Product management", "User research", "Design",
    "Mobile engineering", "Web engineering", "Platform engineering", "Quality assurance",
    "Release management", "Technical writing", "Customer support tier 1",
    "Customer support tier 2", "Customer success", "Renewals", "Sales operations",
    "Enterprise sales", "Partner channel", "Marketing operations", "Brand", "Communications",
    "Investor relations", "Legal contracts", "Privacy office", "Internal audit", "Recruiting",
    "Learning and development", "Benefits", "Employee relations", "Office reception",
]

_GENRES = [
    "Literary fiction", "Historical fiction", "Science fiction", "Fantasy", "Horror",
    "Mystery", "Thriller", "Romance", "Young adult", "Children's picture book", "Poetry",
    "Drama", "Biography", "Memoir", "Popular science", "History", "Philosophy", "Religion",
    "Self-help", "Business", "Economics", "Politics", "True crime", "Travel writing",
    "Cookbook", "Gardening", "Art and photography", "Music", "Sports", "Graphic novel",
    "Essay collection", "Short story collection", "Reference", "Language learning",
    "Parenting", "Health and fitness", "Nature writing", "Mathematics", "Computer programming",
    "Architecture", "Film studies", "Mythology", "Westerns", "Satire", "Dystopian fiction",
    "Classics", "Military history", "Crafts and hobbies", "Education", "Law",
    "Psychology", "Anthropology", "Linguistics", "Astronomy", "Geology", "Medicine",
    "Nursing", "Veterinary", "Agriculture", "Engineering", "Transportation", "Urban planning",
]

PARITY_CASES = (
    ("I was charged twice for the same order. Refund me now.",
     {"type": "choice", "instructions": "What does the customer want?",
      "criteria": {"refund": "wants money back", "track": "asks where an order is",
                   "cancel": "wants to cancel an order", "other": "anything else"}}),
    ("I was charged twice for the same order. Refund me now.",
     {"type": "noul", "instructions": "Is the customer angry?"}),
    ("I was charged twice for the same order. Refund me now.",
     {"type": "score", "instructions": "How urgent is this?",
      "criteria": ["can wait", "today", "right now"]}),
    ({"temperature_c": 38.9, "age_years": 4, "symptoms": ["cough", "rash on chest"],
      "duration_days": 2},
     {"type": "noul", "instructions": "Should the parent seek same-day medical advice?"}),
    ("git push was rejected with 'non-fast-forward' after a teammate pushed to main.",
     {"type": "choice", "instructions": "What is the safest next step?",
      "criteria": {"pull_rebase": "fetch and rebase onto the remote branch",
                   "force": "force push over the remote branch",
                   "delete": "delete the local repository and clone again"}}),
    (None,
     {"type": "choice", "instructions": "Which greeting fits a formal business letter?",
      "criteria": {"hey": "Hey there!", "dear": "Dear Ms. Laurent,", "yo": "Yo,"}}),
    ("Bonjour, ma commande n'est jamais arrivée et le suivi n'a pas bougé depuis dix jours.",
     {"type": "choice", "instructions": "Which language is the message written in?",
      "criteria": {"en": "English", "fr": "French", "de": "German", "es": "Spanish",
                   "it": "Italian"}}),
    (("The battery life is great and the screen is sharp, but the speakers crackle at high volume "
     "and the charger stopped working after a week."),
     {"type": "score", "instructions": "Overall, how satisfied is this reviewer?",
      "criteria": ["very unhappy", "unhappy", "mixed", "happy", "very happy"]}),
    ({"cpu_percent": [92, 95, 97, 96], "memory_percent": 61, "queue_depth": 1840,
      "instances": 4, "max_instances": 12, "error_rate_percent": 0.4},
     {"type": "choice", "instructions": "What should the autoscaler do?",
      "criteria": {"scale_up": "add instances", "scale_down": "remove instances",
                   "hold": "keep the current size", "page": "page the on-call engineer"}}),
    (("Subject: Your mailbox is almost full\nFrom: it-support@micros0ft-helpdesk.co\n"
     "Your mailbox has reached 98% of its quota. To avoid losing incoming mail, verify your "
     "account within 24 hours using the link below and enter your password to confirm ownership. "
     "Failure to verify will result in permanent suspension."),
     {"type": "noul", "instructions": "Is this email likely a phishing attempt?"}),
    (("Student essay: The industrial revolution changed how people worked because factories "
     "brought workers together and machines did tasks faster. However it also created bad "
     "conditions like long hours and child labor. In conclusion it had good and bad effects."),
     {"type": "score", "instructions": "Grade the depth of historical analysis.",
      "criteria": ["missing", "minimal", "basic", "adequate", "strong", "exceptional"]}),
    (_TICKET_THREAD,
     {"type": "choice", "instructions": "What is the customer's main concern now?",
      "criteria": {"deadline": "the fix may land too late for month-end invoicing",
                   "price": "the subscription price is too high",
                   "bug_report": "reporting a new, unrelated bug",
                   "cancel": "wants to cancel the subscription"}}),
    (_TICKET_THREAD,
     {"type": "noul", "instructions": "Does the customer ask for compensation?"}),
    (_TICKET_THREAD,
     {"type": "score", "instructions": "How likely is this customer to churn if the fix slips?",
      "criteria": ["very unlikely", "unlikely", "possible", "likely", "very likely"]}),
    (_INCIDENT_LOG,
     {"type": "choice", "instructions": "What was the root cause of the outage?",
      "criteria": {"disk": "database storage filled up and cleanup was disabled",
                   "deploy": "a bad application deploy",
                   "network": "a regional network partition",
                   "ddos": "a denial-of-service attack"}}),
    (_INCIDENT_LOG,
     {"type": "noul", "instructions": "Was customer data lost?"}),
    (_LEASE_CLAUSE,
     {"type": "choice", "instructions": "A tenant moving 300 km for a new job gives 90 days' "
      "notice with an employer letter. What fee applies?",
      "criteria": {"none": "no termination fee", "one_month": "one month of rent",
                   "two_months": "two months of rent", "deposit": "the security deposit"}}),
    (_LEASE_CLAUSE,
     {"type": "noul", "instructions": "Is an unacknowledged email a valid notice?"}),
    (_MEETING_NOTES,
     {"type": "choice", "instructions": "Who owns the weekend staging scale-down?",
      "criteria": {"priya": "Priya", "tomas": "Tomas", "aiko": "Aiko", "femi": "Femi",
                   "laura": "Laura"}}),
    (_MEETING_NOTES,
     {"type": "score", "instructions": "How much schedule risk does release 7.3 carry?",
      "criteria": ["none", "low", "moderate", "high"]}),
    (_MEETING_NOTES,
     {"type": "noul", "instructions": "Did the team agree to change customer-visible alerts?"}),
    (_RESEARCH_ABSTRACTS,
     {"type": "choice", "instructions": "Which abstract is about medicine?",
      "criteria": {"one": "Abstract 1", "two": "Abstract 2", "three": "Abstract 3",
                   "four": "Abstract 4", "five": "Abstract 5"}}),
    (_RESEARCH_ABSTRACTS,
     {"type": "choice", "instructions": "Which abstract reports a latency reduction?",
      "criteria": {"one": "Abstract 1", "two": "Abstract 2", "three": "Abstract 3",
                   "four": "Abstract 4", "five": "Abstract 5"}}),
    (_OPS_RUNBOOK,
     {"type": "choice", "instructions": "Following the runbook and the context, what should the "
      "on-call engineer do first?",
      "criteria": {"rollback": "roll back the recent deploy",
                   "kill_queries": "kill analyst queries in the warehouse",
                   "penalize_tenant": "penalize a hot tenant",
                   "scale": "scale consumers to twelve",
                   "purge": "purge the queue"}}),
    (_OPS_RUNBOOK,
     {"type": "noul", "instructions": "Should the data platform on-call be paged right now?"}),
    (("A user wrote: 'My manager keeps scheduling meetings at 7 a.m. my time because the rest of "
     "the team is in another timezone.'"),
     {"type": "choice", "instructions": "Which department should handle this request?",
      "criteria": dict.fromkeys(_DEPARTMENTS)}),
    (("Expense claim: hotel in Lisbon, 3 nights, booked through the corporate portal, receipt "
     "attached, flagged because the nightly rate exceeds the city cap by 18%."),
     {"type": "choice", "instructions": "Which department should review this?",
      "criteria": dict.fromkeys(_DEPARTMENTS)}),
    (("Back cover: In 1851, a young surveyor maps the unclaimed valleys of the Sierra while a "
     "mining company races to buy them first. Based on real letters and land records."),
     {"type": "choice", "instructions": "Which shelf does this book belong on?",
      "criteria": dict.fromkeys(_GENRES)}),
    (("Back cover: Twelve recipes for sourdough, from a basic country loaf to rye with caraway, "
     "with step-by-step photographs and a troubleshooting chapter."),
     {"type": "choice", "instructions": "Which shelf does this book belong on?",
      "criteria": dict.fromkeys(_GENRES)}),
    (_MEETING_NOTES + "\n\n" + _OPS_RUNBOOK,
     {"type": "choice", "instructions": "Which team member's action item relates to the alerting "
      "problem described in the runbook context?",
      "criteria": {"priya": "Priya", "tomas": "Tomas", "aiko": "Aiko", "femi": "Femi",
                   "laura": "Laura"}}),
    (_TICKET_THREAD + "\n\n" + _RESEARCH_ABSTRACTS,
     {"type": "noul", "instructions": "Does the ticket mention a release date for the fix?"}),
    ({"incident": _INCIDENT_LOG, "runbook": _OPS_RUNBOOK},
     {"type": "score", "instructions": "How closely did the incident response follow good "
      "practice for paging?", "criteria": ["poorly", "partly", "mostly", "fully"]}),
    (_LEASE_CLAUSE + "\n\n" + _MEETING_NOTES,
     {"type": "choice", "instructions": "Which department should review the document about "
      "early termination?", "criteria": dict.fromkeys(_DEPARTMENTS)}),
    (_RESEARCH_ABSTRACTS + "\n\n" + _LEASE_CLAUSE,
     {"type": "choice", "instructions": "Which shelf fits a collection of these abstracts?",
      "criteria": dict.fromkeys(_GENRES)}),
)


# ----------------------------------------------------------------- buckets ---

def validate_buckets(buckets):
    """Strictly ascending unique ints, multiples of the GDN chunk, max 1024."""
    if isinstance(buckets, (str, bytes)) or not hasattr(buckets, "__iter__"):
        raise TypeError("buckets must be a sequence of integers")
    values = tuple(buckets)
    if not 1 <= len(values) <= 8:
        raise ValueError("buckets must contain 1 to 8 sequence lengths")
    if any(type(value) is not int for value in values):
        raise TypeError("bucket sequence lengths must be plain integers")
    if any(not DEFAULT_CHUNK <= value <= MAX_BUCKET or value % DEFAULT_CHUNK for value in values):
        raise ValueError(
            f"bucket lengths must be multiples of {DEFAULT_CHUNK} in [{DEFAULT_CHUNK}, {MAX_BUCKET}]"
        )
    if any(left >= right for left, right in itertools.pairwise(values)):
        raise ValueError("buckets must be strictly ascending and unique")
    return values


def bucket_for(token_count, buckets=DEFAULT_BUCKETS):
    """Smallest bucket that holds ``token_count`` tokens; refuse beyond the largest."""
    if type(token_count) is not int or token_count < 1:
        raise ValueError("token count must be a positive integer")
    for size in buckets:
        if token_count <= size:
            return size
    raise ValueError("token count exceeds the largest Core ML bucket; no truncation")


def length_distribution(counts, buckets=DEFAULT_BUCKETS):
    """Summary of measured prompt lengths and their bucket assignment."""
    values = sorted(int(count) for count in counts)
    if not values:
        raise ValueError("no token counts")
    assigned = {}
    for count in values:
        try:
            key = str(bucket_for(count, buckets))
        except ValueError:
            key = "over"
        assigned[key] = assigned.get(key, 0) + 1

    def quantile(fraction):
        index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
        return values[index]

    return {
        "count": len(values), "min": values[0], "max": values[-1],
        "p50": quantile(0.5), "p90": quantile(0.9), "per_bucket": assigned,
        "padding_fraction_mean": round(
            sum((bucket_for(v, buckets) - v) / bucket_for(v, buckets)
                for v in values if v <= buckets[-1]) / max(1, sum(v <= buckets[-1] for v in values)),
            4,
        ),
    }


def readout_select(mask):
    """One-hot of the last real position from a right-padded 0/1 mask (host mirror)."""
    import numpy as np

    mask = np.asarray(mask)
    following = np.concatenate([mask[..., 1:], np.zeros_like(mask[..., :1])], axis=-1)
    return mask - following


def check_right_padded(mask):
    import numpy as np

    mask = np.asarray(mask).reshape(-1)
    count = int(mask.sum())
    if count < 1 or not (mask[:count] == 1).all() or (mask[count:] != 0).any():
        raise ValueError("mask must be one or more ones followed only by zeros")
    return count


# ------------------------------------------------------ GDN formulations ---
# Shapes (single question, width one): q, k [T, H, Dk] with q already carrying
# the delta-rule 1/sqrt(Dk) scale; v [T, H, Dv]; log_g [T, H] (log decay,
# <= 0); beta [T, H]. Output y [T, H, Dv] and the final state [H, Dk, Dv].

def gdn_sequential_numpy(q, k, v, log_g, beta, state=None):
    """Token-by-token reference scan (the MLX ``gated_delta_ops`` law)."""
    import numpy as np

    T, H, Dk = k.shape
    Dv = v.shape[-1]
    dtype = np.result_type(q, k, v, np.float32)
    state = np.zeros((H, Dk, Dv), dtype) if state is None else state.astype(dtype).copy()
    out = np.empty((T, H, Dv), dtype)
    for t in range(T):
        state *= np.exp(log_g[t])[:, None, None]
        memory = np.einsum("hkv,hk->hv", state, k[t])
        delta = (v[t] - memory) * beta[t][:, None]
        state += k[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hkv,hk->hv", state, q[t])
    return out, state


def decay_sum_matrix(chunk):
    """Binary [C*C, C]: row (i, j) selects positions l with j < l <= i.

    ``log_g @ W.T`` gives every within-chunk log-decay difference as a sum
    of same-signed terms, so no cumulative-sum cancellation occurs in fp16.
    """
    import numpy as np

    index = np.arange(chunk)
    i, j, l = index[:, None, None], index[None, :, None], index[None, None, :]
    matrix = ((j < l) & (l <= i)).astype(np.float32)
    return matrix.reshape(chunk * chunk, chunk)


def unit_lower_inverse(strict, *, block=INVERSE_BLOCK):
    """``(I + A)^-1`` for strictly lower-triangular ``A`` [..., C, C], matmul-only.

    Diagonal ``block`` blocks are inverted with the nilpotent product
    ``(I - A)(I + A^2)(I + A^4)...``; the strictly block-lower remainder
    ``N = D^-1 L`` is nilpotent of degree ``C / block`` and is removed with
    the same product. Bounded intermediate growth versus a degree-C series.
    """
    import numpy as np

    C = strict.shape[-1]
    if block > C or C % block or block & (block - 1) or (C // block) & (C // block - 1):
        raise ValueError("chunk and block must be powers of two with block <= chunk")
    eye = np.eye(C, dtype=strict.dtype)
    blocks = np.arange(C) // block
    same = (blocks[:, None] == blocks[None, :]).astype(strict.dtype)
    diagonal_part = strict * same
    lower_part = strict - diagonal_part

    def nilpotent_inverse(matrix, degree):
        result = eye - matrix
        power = matrix
        span = 2
        while span < degree:
            power = power @ power
            result = result @ (eye + power)
            span *= 2
        return result

    inverse_diagonal = nilpotent_inverse(diagonal_part, block)
    if C == block:
        return inverse_diagonal
    remainder = inverse_diagonal @ lower_part
    return nilpotent_inverse(remainder, C // block) @ inverse_diagonal


def gdn_chunked_numpy(q, k, v, log_g, beta, *, chunk=DEFAULT_CHUNK, block=INVERSE_BLOCK,
                      inverse="blocked", state=None):
    """Chunkwise-parallel WY form of the same recurrence (matmul-only algebra)."""
    import numpy as np

    T, H, Dk = k.shape
    Dv = v.shape[-1]
    dtype = np.result_type(q, k, v, np.float32)
    pad = (-T) % chunk
    if pad:
        # Identity steps: zero key/value/beta and zero log decay.
        q = np.concatenate([q, np.zeros((pad, H, Dk), dtype)])
        k = np.concatenate([k, np.zeros((pad, H, Dk), dtype)])
        v = np.concatenate([v, np.zeros((pad, H, Dv), dtype)])
        log_g = np.concatenate([log_g, np.zeros((pad, H), dtype)])
        beta = np.concatenate([beta, np.zeros((pad, H), dtype)])
    N = (T + pad) // chunk
    # [H, N, C, D]
    qc = q.reshape(N, chunk, H, Dk).transpose(2, 0, 1, 3).astype(dtype)
    kc = k.reshape(N, chunk, H, Dk).transpose(2, 0, 1, 3).astype(dtype)
    vc = v.reshape(N, chunk, H, Dv).transpose(2, 0, 1, 3).astype(dtype)
    lg = log_g.reshape(N, chunk, H).transpose(2, 0, 1).astype(dtype)
    bt = beta.reshape(N, chunk, H).transpose(2, 0, 1).astype(dtype)
    lower_incl = np.tril(np.ones((chunk, chunk), dtype))
    strict_lower = np.tril(np.ones((chunk, chunk), dtype), -1)
    gamma = lg @ lower_incl.T  # [H,N,C] cumulative log decay from chunk start
    tail = lg @ strict_lower  # [l, i] = 1 for l > i: sum_{l>i} lg_l
    differences = (lg @ decay_sum_matrix(chunk).T.astype(dtype)).reshape(H, N, chunk, chunk)
    decay = np.exp(differences) * lower_incl
    a = (bt[..., :, None] * (kc @ kc.swapaxes(-1, -2))) * decay * strict_lower
    if inverse == "blocked":
        t_matrix = unit_lower_inverse(a, block=block)
    elif inverse == "exact":
        t_matrix = np.linalg.inv(np.eye(chunk, dtype=dtype) + a)
    else:
        raise ValueError("inverse must be 'blocked' or 'exact'")
    u = t_matrix @ (vc * bt[..., None])
    w = t_matrix @ (kc * (bt * np.exp(gamma))[..., None])
    intra = (qc @ kc.swapaxes(-1, -2)) * decay
    q_decayed = qc * np.exp(gamma)[..., None]
    k_tail = kc * np.exp(tail)[..., None]
    chunk_decay = np.exp(gamma[..., -1])  # [H,N]
    state = np.zeros((H, Dk, Dv), dtype) if state is None else state.astype(dtype).copy()
    out = np.empty((H, N, chunk, Dv), dtype)
    for n in range(N):
        fresh = u[:, n] - w[:, n] @ state
        out[:, n] = q_decayed[:, n] @ state + intra[:, n] @ fresh
        state = state * chunk_decay[:, n][:, None, None] + k_tail[:, n].swapaxes(-1, -2) @ fresh
    y = out.transpose(1, 2, 0, 3).reshape(N * chunk, H, Dv)[:T]
    return y, state


# ----------------------------------------------------------------- weights ---

def _sha256_file(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _safetensors_header(path):
    with open(path, "rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= 1 << 26:
            raise ValueError("implausible safetensors header length")
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


class D3Weights:
    """Verified D3 trunk weights: FP32 copies of the BF16 trunk tensors (minus the
    embedding), a read-only BF16 view of the embedding table, and the FP32 readout."""

    def __init__(self, root, tensors, embedding_bits, readout, temperature, identity):
        self.root = root
        self.tensors = tensors
        self.embedding_bits = embedding_bits
        self.readout = readout
        self.temperature = temperature
        self.identity = identity

    def __getitem__(self, name):
        return self.tensors[name]

    def embed(self, token_ids):
        """FP32 rows of the BF16 embedding (exactly representable in fp16 too)."""
        import numpy as np

        rows = np.asarray(self.embedding_bits[np.asarray(token_ids, dtype=np.int64)])
        return (rows.astype(np.uint32) << 16).view(np.float32)


def load_weights(root=DEFAULT_ARTIFACT, *, verify_sha256=True, include_trunk=True):
    """Inspect with the served route's own contracts, verify bytes, decode BF16.

    ``include_trunk=False`` keeps only the embedding view and the readout (the
    host-side parts a Core ML execution process needs)."""
    import numpy as np

    from .d3 import inspect_artifact, validate_backbone_headers

    artifact = inspect_artifact(root)
    validate_backbone_headers(artifact)
    root = Path(artifact["path"])
    backbone, readout_path = root / BACKBONE_FILE, root / READOUT_FILE
    identity = {"revision": SOURCE_REVISION, "backbone_sha256": BACKBONE_SHA256,
                "readout_sha256": READOUT_SHA256, "sha256_verified": False}
    if verify_sha256:
        if _sha256_file(backbone) != BACKBONE_SHA256:
            raise ValueError("D3 backbone bytes differ from the pinned revision")
        if _sha256_file(readout_path) != READOUT_SHA256:
            raise ValueError("D3 readout bytes differ from the pinned revision")
        identity["sha256_verified"] = True
    header, base = _safetensors_header(backbone)
    mapped = np.memmap(backbone, dtype=np.uint8, mode="r")
    tensors, embedding = {}, None
    for name, entry in header.items():
        if not name.startswith("language_model."):
            continue  # vision tensors are never executed
        if entry["dtype"] != "BF16":
            raise ValueError(f"unexpected dtype for {name}")
        start, end = entry["data_offsets"]
        bits = mapped[base + start:base + end].view("<u2").reshape(entry["shape"])
        short = name.removeprefix("language_model.")
        if short == "embed_tokens.weight":
            embedding = bits
            continue
        if not include_trunk:
            continue
        value = (bits.astype(np.uint32) << 16).view(np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite D3 weight {name}")
        tensors[short] = value
    if embedding is None:
        raise ValueError("D3 embedding table missing")
    r_header, r_base = _safetensors_header(readout_path)
    entry = r_header["weight"]
    start, end = entry["data_offsets"]
    readout = np.frombuffer(readout_path.read_bytes()[r_base + start:r_base + end],
                            dtype="<f4").reshape(entry["shape"]).astype(np.float32)
    if readout.shape != (READOUT_ROWS, HIDDEN) or not np.isfinite(readout).all():
        raise ValueError("D3 readout must be a finite 255 x 1024 FP32 matrix")
    temperature = float(artifact["decision_config"]["temperature"])
    return D3Weights(root, tensors, embedding, readout, temperature, identity)


def _kinds(weights):
    return ["linear_attention" if f"layers.{i}.linear_attn.A_log" in weights.tensors
            else "full_attention" for i in range(LAYERS)]


# -------------------------------------------------------- numpy reference ---

def _rms_raw(x, weight, eps=1e-6):
    import numpy as np

    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (1.0 + weight)


def _silu(x):
    import numpy as np

    return x / (1.0 + np.exp(-x))


def _softplus(x):
    import numpy as np

    return np.logaddexp(0.0, x)


def rope_tables(length, dims=ROPE_DIMS, theta=1.0e7):
    import numpy as np

    inverse = theta ** (-np.arange(0, dims, 2, dtype=np.float32) / dims)
    angles = np.arange(length, dtype=np.float32)[:, None] * inverse[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles), np.sin(angles)


def _rope(x, cosine, sine):
    import numpy as np

    rotated, tail = x[..., :ROPE_DIMS], x[..., ROPE_DIMS:]
    half = np.concatenate([-rotated[..., ROPE_DIMS // 2:], rotated[..., :ROPE_DIMS // 2]], axis=-1)
    return np.concatenate([rotated * cosine + half * sine, tail], axis=-1)


def gdn_inputs_numpy(weights, layer, x):
    """Projections, causal depthwise conv, q/k normalization and gates (FP32)."""
    import numpy as np

    p = f"layers.{layer}.linear_attn."
    S = x.shape[0]
    qkv = x @ weights[p + "in_proj_qkv.weight"].T
    z = x @ weights[p + "in_proj_z.weight"].T
    b = x @ weights[p + "in_proj_b.weight"].T
    a = x @ weights[p + "in_proj_a.weight"].T
    kernel = weights[p + "conv1d.weight"][:, 0, :]  # [C, K]
    padded = np.concatenate([np.zeros((CONV_KERNEL - 1, qkv.shape[1]), qkv.dtype), qkv])
    conv = sum(kernel[:, j] * padded[j:j + S] for j in range(CONV_KERNEL))
    conv = _silu(conv)
    q = conv[:, :GDN_KEY].reshape(S, GDN_HEADS, GDN_DK)
    k = conv[:, GDN_KEY:2 * GDN_KEY].reshape(S, GDN_HEADS, GDN_DK)
    v = conv[:, 2 * GDN_KEY:].reshape(S, GDN_HEADS, GDN_DV)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * GDN_DK ** -0.5
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    beta = 1.0 / (1.0 + np.exp(-b))
    log_g = -np.exp(weights[p + "A_log"]) * _softplus(a + weights[p + "dt_bias"])
    return q, k, v, log_g, beta, z.reshape(S, GDN_HEADS, GDN_DV)


def numpy_forward(weights, hidden, *, gdn="chunked", stats=None):
    """FP32 numpy D3 trunk over real tokens only; returns the final normed hidden."""
    import numpy as np

    x = hidden.astype(np.float32)
    S = x.shape[0]
    cosine, sine = rope_tables(S)
    for layer, kind in enumerate(_kinds(weights)):
        p = f"layers.{layer}."
        h = _rms_raw(x, weights[p + "input_layernorm.weight"])
        if kind == "linear_attention":
            q, k, v, log_g, beta, z = gdn_inputs_numpy(weights, layer, h)
            if gdn == "chunked":
                y, _ = gdn_chunked_numpy(q, k, v, log_g, beta)
            else:
                y, _ = gdn_sequential_numpy(q, k, v, log_g, beta)
            normed = y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + 1e-6)
            normed = normed * weights[p + "linear_attn.norm.weight"] * _silu(z)
            mixed = normed.reshape(S, GDN_VALUE) @ weights[p + "linear_attn.out_proj.weight"].T
        else:
            a = p + "self_attn."
            projected = (h @ weights[a + "q_proj.weight"].T).reshape(S, HEADS, 2 * HEAD_DIM)
            queries, gate = projected[..., :HEAD_DIM], projected[..., HEAD_DIM:]
            keys = (h @ weights[a + "k_proj.weight"].T).reshape(S, KV_HEADS, HEAD_DIM)
            values = (h @ weights[a + "v_proj.weight"].T).reshape(S, KV_HEADS, HEAD_DIM)
            queries = _rope(_rms_raw(queries, weights[a + "q_norm.weight"]).transpose(1, 0, 2),
                            cosine, sine)
            keys = _rope(_rms_raw(keys, weights[a + "k_norm.weight"]).transpose(1, 0, 2),
                         cosine, sine)
            values = values.transpose(1, 0, 2)
            group = HEADS // KV_HEADS
            keys, values = np.repeat(keys, group, axis=0), np.repeat(values, group, axis=0)
            scores = queries @ keys.transpose(0, 2, 1) * HEAD_DIM ** -0.5
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            attended = (scores / scores.sum(-1, keepdims=True)) @ values  # [H,S,D]
            attended = attended.transpose(1, 0, 2).reshape(S, HEADS * HEAD_DIM)
            attended = attended * (1.0 / (1.0 + np.exp(-gate.reshape(S, -1))))
            mixed = attended @ weights[a + "o_proj.weight"].T
        x = x + mixed
        h = _rms_raw(x, weights[p + "post_attention_layernorm.weight"])
        m = p + "mlp."
        up = _silu(h @ weights[m + "gate_proj.weight"].T) * (h @ weights[m + "up_proj.weight"].T)
        x = x + up @ weights[m + "down_proj.weight"].T
        if stats is not None:
            stats.append({"layer": layer, "kind": kind, "max_abs_residual": float(np.abs(x).max()),
                          "max_rms_residual": float(np.sqrt(np.mean(x * x, axis=-1)).max())})
    return _rms_raw(x, weights["norm.weight"])


def decision_probabilities(hidden_row, readout, labels, temperature=1.0):
    """The served readout law: FP32 logits over the first ``labels`` rows, softmax."""
    import numpy as np

    logits = hidden_row.astype(np.float32) @ readout[:labels].T / np.float32(temperature)
    logits = logits - logits.max()
    weights = np.exp(logits.astype(np.float64))
    return (weights / weights.sum()).astype(np.float64)


# ------------------------------------------------------------- MIL graph ---
# GDN core ops (everything between the input and output projections) carry
# the name prefix below so a precision variant can keep exactly them in FP32.
GDN_CORE_PREFIX = "gdncore"
# The gated GDN RMSNorm is scale invariant; producing y pre-scaled by 2**6
# (folded into q) keeps mean(y^2) out of the fp16 subnormal range. The eps is
# scaled by the same factor squared, so the law is unchanged.
GDN_OUTPUT_SCALE = 64.0
RMS_EPS = 1e-6
INVERSE_TAG = "inv"
PRECISION_VARIANTS = ("fp16", "fp32_gdn", "fp32_inverse")


class _Named:
    """``mb`` proxy that names every op ``<prefix>_<n>_<op>``."""

    def __init__(self, mb, prefix):
        self._mb, self._prefix, self._count = mb, prefix, 0

    def __getattr__(self, operator):
        function = getattr(self._mb, operator)

        def call(**kwargs):
            self._count += 1
            return function(name=f"{self._prefix}_{self._count}_{operator}", **kwargs)

        return call


def rope_full_tables(length):
    """cos/sin over the whole 256-dim head (identity on the 192-dim tail) and the
    signed permutation P with ``x @ P`` = the half-rotation of the first 64 dims."""
    import numpy as np

    cosine, sine = rope_tables(length)
    full_cos = np.ones((length, HEAD_DIM), np.float32)
    full_sin = np.zeros((length, HEAD_DIM), np.float32)
    full_cos[:, :ROPE_DIMS], full_sin[:, :ROPE_DIMS] = cosine, sine
    half = ROPE_DIMS // 2
    permutation = np.zeros((HEAD_DIM, HEAD_DIM), np.float32)
    for column in range(half):
        permutation[column + half, column] = -1.0  # out[c] = -x[c + 32]
        permutation[column, column + half] = 1.0  # out[c + 32] = x[c]
    return full_cos, full_sin, permutation


def _chunk_constants(chunk, block):
    import numpy as np

    lower_incl = np.tril(np.ones((chunk, chunk), np.float32))
    strict_lower = np.tril(np.ones((chunk, chunk), np.float32), -1)
    blocks = np.arange(chunk) // block
    same_block = (blocks[:, None] == blocks[None, :]).astype(np.float32)
    return {
        "lower_incl": lower_incl, "strict_lower": strict_lower, "same_block": same_block,
        "eye": np.eye(chunk, dtype=np.float32), "decay_sum": decay_sum_matrix(chunk),
    }


def _graph_rms(mb, x, gain, eps=RMS_EPS):
    """x * rsqrt(mean(x^2) + eps) * gain over the last axis."""
    import numpy as np

    mean_square = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
    scaled = mb.mul(x=x, y=mb.rsqrt(x=mean_square, epsilon=np.float32(eps)))
    return mb.mul(x=scaled, y=gain)


def _nilpotent_inverse(ops, matrix, degree, eye):
    """(I + A)^-1 for strictly lower nilpotent A of power-of-two degree, matmul-only.

    ANE finding (macOS 26.7, coremltools 9.0, M5 Max): a matmul whose ``y``
    is ``x`` itself or a pointwise function of it (``A @ A``) makes
    ``ANECCompile() FAILED`` for the whole model, after which Core ML runs
    everything on the CPU; rewriting it as ``A @ (A * strict_lower)`` makes
    the ANE compile hang in ``aned`` instead. The ``fp32_gdn`` and
    ``fp32_inverse`` precision variants keep this subgraph off the ANE.
    """
    result = ops.sub(x=eye, y=matrix)
    power, span = matrix, 2
    while span < degree:
        power = ops.matmul(x=power, y=power)
        result = ops.matmul(x=result, y=ops.add(x=eye, y=power))
        span *= 2
    return result


def _gdn_layer(mb, weights, layer, h, S, chunk, block):
    import numpy as np

    p = f"layers.{layer}.linear_attn."
    ops = _Named(mb, f"{GDN_CORE_PREFIX}{layer}")
    H, Dk, Dv, C = GDN_HEADS, GDN_DK, GDN_DV, chunk
    N = S // C
    consts = _chunk_constants(C, block)
    qkv = mb.linear(x=h, weight=weights[p + "in_proj_qkv.weight"])  # [1,S,6144]
    z = mb.linear(x=h, weight=weights[p + "in_proj_z.weight"])  # [1,S,2048]
    gates = mb.linear(
        x=h, weight=np.concatenate([weights[p + "in_proj_b.weight"], weights[p + "in_proj_a.weight"]])
    )  # [1,S,32]
    # Causal depthwise conv: zero history, kernel taps as per-channel scales.
    kernel = weights[p + "conv1d.weight"][:, 0, :]  # [6144, 4]
    padded = ops.pad(x=qkv, pad=np.array([0, 0, CONV_KERNEL - 1, 0, 0, 0], np.int32))
    conv = None
    for tap in range(CONV_KERNEL):
        window = ops.slice_by_index(x=padded, begin=[0, tap, 0], end=[1, tap + S, qkv.shape[-1]])
        term = ops.mul(x=window, y=kernel[:, tap])
        conv = term if conv is None else ops.add(x=conv, y=term)
    conv = ops.silu(x=conv)
    q = ops.reshape(x=ops.slice_by_index(x=conv, begin=[0, 0, 0], end=[1, S, GDN_KEY]),
                    shape=[1, S, H, Dk])
    k = ops.reshape(x=ops.slice_by_index(x=conv, begin=[0, 0, GDN_KEY], end=[1, S, 2 * GDN_KEY]),
                    shape=[1, S, H, Dk])
    v = ops.reshape(x=ops.slice_by_index(x=conv, begin=[0, 0, 2 * GDN_KEY], end=[1, S, 2 * GDN_KEY + GDN_VALUE]),
                    shape=[1, S, H, Dv])

    def l2(value, scale):
        total = ops.reduce_sum(x=ops.mul(x=value, y=value), axes=[-1], keep_dims=True)
        return ops.mul(x=ops.mul(x=value, y=ops.rsqrt(x=total, epsilon=np.float32(1e-6))),
                       y=np.float32(scale))

    q = l2(q, GDN_DK ** -0.5 * GDN_OUTPUT_SCALE)
    k = l2(k, 1.0)
    b = ops.slice_by_index(x=gates, begin=[0, 0, 0], end=[1, S, H])
    a = ops.slice_by_index(x=gates, begin=[0, 0, H], end=[1, S, 2 * H])
    beta = ops.sigmoid(x=b)  # [1,S,H]
    rate = -np.exp(weights[p + "A_log"]).astype(np.float32)
    log_g = ops.mul(x=ops.softplus(x=ops.add(x=a, y=weights[p + "dt_bias"])), y=rate)  # [1,S,H]

    def chunks4(value, depth):  # [1,S,H,D] -> [H,N,C,D]
        return ops.transpose(x=ops.reshape(x=value, shape=[N, C, H, depth]), perm=[2, 0, 1, 3])

    def chunks3(value):  # [1,S,H] -> [H,N,C]
        return ops.transpose(x=ops.reshape(x=value, shape=[N, C, H]), perm=[2, 0, 1])

    qc, kc, vc = chunks4(q, Dk), chunks4(k, Dk), chunks4(v, Dv)
    lg, bt = chunks3(log_g), chunks3(beta)
    gamma = ops.matmul(x=lg, y=consts["lower_incl"], transpose_y=True)  # [H,N,C]
    tail = ops.matmul(x=lg, y=consts["strict_lower"])  # sum_{l>i}
    differences = ops.reshape(
        x=ops.matmul(x=lg, y=consts["decay_sum"], transpose_y=True), shape=[H, N, C, C]
    )
    decay = ops.mul(x=ops.exp(x=differences), y=consts["lower_incl"])
    bt4 = ops.expand_dims(x=bt, axes=[-1])  # [H,N,C,1]
    kk = ops.matmul(x=kc, y=kc, transpose_y=True)
    strict = ops.mul(x=ops.mul(x=ops.mul(x=kk, y=bt4), y=decay), y=consts["strict_lower"])
    # Blocked unit-lower inverse (see unit_lower_inverse).
    # Its ops are named ``<core prefix><layer>inv_*`` so a precision variant
    # can keep only this subgraph in FP32 (and therefore on the CPU).
    inv = _Named(mb, f"{GDN_CORE_PREFIX}{layer}{INVERSE_TAG}")
    eye = consts["eye"]
    diagonal_part = inv.mul(x=strict, y=consts["same_block"])
    inverse = _nilpotent_inverse(inv, diagonal_part, block, eye)
    if C != block:
        lower_part = inv.sub(x=strict, y=diagonal_part)
        remainder = inv.matmul(x=inverse, y=lower_part)
        inverse = inv.matmul(
            x=_nilpotent_inverse(inv, remainder, C // block, eye), y=inverse
        )
    exp_gamma = ops.exp(x=gamma)
    u = ops.matmul(x=inverse, y=ops.mul(x=vc, y=bt4))
    w = ops.matmul(x=inverse, y=ops.mul(x=kc, y=ops.mul(x=bt4, y=ops.expand_dims(x=exp_gamma, axes=[-1]))))
    intra = ops.mul(x=ops.matmul(x=qc, y=kc, transpose_y=True), y=decay)
    q_decayed = ops.mul(x=qc, y=ops.expand_dims(x=exp_gamma, axes=[-1]))
    k_tail = ops.mul(x=kc, y=ops.expand_dims(x=ops.exp(x=tail), axes=[-1]))
    chunk_decay = ops.slice_by_index(x=exp_gamma, begin=[0, 0, C - 1], end=[H, N, C])  # [H,N,1]

    def at(value, n, depth):  # [H,N,C,depth] -> [H,C,depth]
        return ops.reshape(x=ops.slice_by_index(x=value, begin=[0, n, 0, 0], end=[H, n + 1, C, depth]),
                           shape=[H, C, depth])

    outputs, state = [], None
    for n in range(N):
        u_n, intra_n = at(u, n, Dv), at(intra, n, C)
        if state is None:
            fresh = u_n
            out = ops.matmul(x=intra_n, y=fresh)
        else:
            fresh = ops.sub(x=u_n, y=ops.matmul(x=at(w, n, Dk), y=state))
            out = ops.add(x=ops.matmul(x=at(q_decayed, n, Dk), y=state),
                          y=ops.matmul(x=intra_n, y=fresh))
        outputs.append(out)
        if n + 1 < N:
            update = ops.matmul(x=at(k_tail, n, Dk), y=fresh, transpose_x=True)  # [H,Dk,Dv]
            if state is None:
                state = update
            else:
                factor = ops.reshape(x=ops.slice_by_index(x=chunk_decay, begin=[0, n, 0], end=[H, n + 1, 1]),
                                     shape=[H, 1, 1])
                state = ops.add(x=ops.mul(x=state, y=factor), y=update)
    y = outputs[0] if N == 1 else ops.concat(values=outputs, axis=1)  # [H,S,Dv]
    y = ops.transpose(x=y, perm=[1, 0, 2])  # [S,H,Dv]
    mean_square = ops.reduce_mean(x=ops.mul(x=y, y=y), axes=[-1], keep_dims=True)
    normed = ops.mul(x=y, y=ops.rsqrt(x=mean_square, epsilon=np.float32(RMS_EPS * GDN_OUTPUT_SCALE ** 2)))
    normed = ops.mul(x=normed, y=weights[p + "norm.weight"])
    gate = ops.silu(x=ops.reshape(x=z, shape=[S, H, Dv]))
    mixed = ops.reshape(x=ops.mul(x=normed, y=gate), shape=[1, S, GDN_VALUE])
    return mb.linear(x=mixed, weight=weights[p + "out_proj.weight"])


def _attention_layer(mb, weights, layer, h, S, key_bias, rope):
    import numpy as np

    a = f"layers.{layer}.self_attn."
    full_cos, full_sin, permutation = rope
    q_weight = weights[a + "q_proj.weight"].reshape(HEADS, 2, HEAD_DIM, HIDDEN)
    queries = mb.reshape(x=mb.linear(x=h, weight=q_weight[:, 0].reshape(-1, HIDDEN)),
                         shape=[1, S, HEADS, HEAD_DIM])
    gate = mb.sigmoid(x=mb.linear(x=h, weight=q_weight[:, 1].reshape(-1, HIDDEN)))  # [1,S,2048]
    keys = mb.reshape(x=mb.linear(x=h, weight=weights[a + "k_proj.weight"]),
                      shape=[1, S, KV_HEADS, HEAD_DIM])
    values = mb.reshape(x=mb.linear(x=h, weight=weights[a + "v_proj.weight"]),
                        shape=[1, S, KV_HEADS, HEAD_DIM])
    # Fold the 1/sqrt(d) score scale into the query norm gain (RoPE is linear).
    q_gain = ((1.0 + weights[a + "q_norm.weight"]) * HEAD_DIM ** -0.5).astype(np.float32)
    k_gain = (1.0 + weights[a + "k_norm.weight"]).astype(np.float32)
    cos4, sin4 = full_cos[None, :, None, :], full_sin[None, :, None, :]

    def rotate(value):  # [1,S,heads,256]
        half = mb.matmul(x=value, y=permutation)
        return mb.add(x=mb.mul(x=value, y=cos4), y=mb.mul(x=half, y=sin4))

    queries = rotate(_graph_rms(mb, queries, q_gain))
    keys = rotate(_graph_rms(mb, keys, k_gain))
    group = HEADS // KV_HEADS
    # Heads h = kv * group + r: [1,S,8,D] -> [1,2,group*S,D] for grouped scores.
    queries = mb.reshape(x=mb.transpose(x=mb.reshape(x=queries, shape=[S, KV_HEADS, group, HEAD_DIM]),
                                        perm=[1, 2, 0, 3]),
                         shape=[1, KV_HEADS, group * S, HEAD_DIM])
    keys = mb.transpose(x=keys, perm=[0, 2, 1, 3])  # [1,2,S,D]
    values = mb.transpose(x=values, perm=[0, 2, 1, 3])
    scores = mb.add(x=mb.matmul(x=queries, y=keys, transpose_y=True), y=key_bias)  # [1,2,gS,S]
    attended = mb.matmul(x=mb.softmax(x=scores, axis=-1), y=values)  # [1,2,gS,D]
    attended = mb.transpose(x=mb.reshape(x=attended, shape=[KV_HEADS, group, S, HEAD_DIM]),
                            perm=[2, 0, 1, 3])  # [S,2,group,D] == heads in order
    attended = mb.reshape(x=attended, shape=[1, S, HEADS * HEAD_DIM])
    return mb.linear(x=mb.mul(x=attended, y=gate), weight=weights[a + "o_proj.weight"])


def build_program(weights, sequence_length, *, chunk=DEFAULT_CHUNK, block=INVERSE_BLOCK,
                  opset="macOS15", layer_subset=None):
    """MIL program for one bucket: (embeddings fp16 [1,S,1024], mask int32 [1,S]) ->
    (hidden [1,1024] final-normed at the last real token, logits [1,255])."""
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    S = sequence_length
    if S % chunk:
        raise ValueError("bucket length must be a multiple of the GDN chunk")
    kinds = _kinds(weights)
    rope = rope_full_tables(S)
    target = getattr(ct.target, opset)
    specs = [mb.TensorSpec(shape=(1, S, HIDDEN), dtype=types.fp16),
             mb.TensorSpec(shape=(1, S), dtype=types.int32)]

    @mb.program(input_specs=specs, opset_version=target)
    def program(embeddings, mask):
        x = mb.cast(x=embeddings, dtype="fp32")
        mask_f = mb.cast(x=mask, dtype="fp32")
        key_bias = mb.reshape(
            x=mb.mul(x=mb.sub(x=np.float32(1.0), y=mask_f), y=np.float32(MASK_NEGATIVE)),
            shape=[1, 1, 1, S],
        )
        following = mb.concat(values=[mb.slice_by_index(x=mask_f, begin=[0, 1], end=[1, S]),
                                      np.zeros((1, 1), np.float32)], axis=1)
        select = mb.reshape(x=mb.sub(x=mask_f, y=following), shape=[1, 1, S])
        for layer, kind in enumerate(kinds):
            if layer_subset is not None and layer not in layer_subset:
                continue  # diagnostic sub-graphs only; never a served shape
            p = f"layers.{layer}."
            h = _graph_rms(mb, x, 1.0 + weights[p + "input_layernorm.weight"])
            if kind == "linear_attention":
                mixed = _gdn_layer(mb, weights, layer, h, S, chunk, block)
            else:
                mixed = _attention_layer(mb, weights, layer, h, S, key_bias, rope)
            x = mb.add(x=x, y=mixed)
            h = _graph_rms(mb, x, 1.0 + weights[p + "post_attention_layernorm.weight"])
            m = p + "mlp."
            up = mb.mul(x=mb.silu(x=mb.linear(x=h, weight=weights[m + "gate_proj.weight"])),
                        y=mb.linear(x=h, weight=weights[m + "up_proj.weight"]))
            x = mb.add(x=x, y=mb.linear(x=up, weight=weights[m + "down_proj.weight"]))
        row = mb.reshape(x=mb.matmul(x=select, y=x), shape=[1, HIDDEN])
        final = _graph_rms(mb, row, 1.0 + weights["norm.weight"])
        out_hidden = mb.identity(x=final, name="hidden")
        logits = mb.linear(x=final, weight=weights.readout, name="logits")
        return out_hidden, logits

    return program


def is_inverse_op(name):
    """``gdncore<layer>inv_<n>_<op>``: the blocked unit-lower inverse subgraph."""
    if not name.startswith(GDN_CORE_PREFIX):
        return False
    head = name[len(GDN_CORE_PREFIX):].split("_", 1)[0]
    return head.endswith(INVERSE_TAG) and head[: -len(INVERSE_TAG)].isdigit()


def precision_for(variant):
    import coremltools as ct

    if variant == "fp16":
        return ct.precision.FLOAT16
    if variant == "fp32_gdn":
        return ct.transform.FP16ComputePrecision(
            op_selector=lambda op: not op.name.startswith(GDN_CORE_PREFIX)
        )
    if variant == "fp32_inverse":
        return ct.transform.FP16ComputePrecision(op_selector=lambda op: not is_inverse_op(op.name))
    raise ValueError(f"precision variant must be one of {PRECISION_VARIANTS}")


def convert_package(program, package_path, variant, *, opset="macOS15"):
    """Convert one bucket program to an mlpackage; CPU_ONLY and no model load."""
    import coremltools as ct

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, opset),
        compute_precision=precision_for(variant),
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )
    model.save(str(package_path))
    del model
    return Path(package_path)


def merge_multifunction(packages, destination):
    """One mlpackage with a function per bucket; identical weight blobs are shared."""
    import coremltools as ct

    descriptor = ct.utils.MultiFunctionDescriptor()
    for size, package in packages.items():
        descriptor.add_function(str(package), src_function_name="main",
                                target_function_name=function_name(size))
    descriptor.default_function_name = function_name(min(packages))
    ct.utils.save_multifunction(descriptor, str(destination))
    return Path(destination)


def compile_package(package, destination):
    import coremltools as ct

    compiled = Path(ct.models.utils.compile_model(str(package), destination_path=str(destination)))
    if compiled.resolve() != Path(destination).resolve():
        raise RuntimeError("Core ML compiler wrote outside the requested path")
    return compiled


def function_name(size):
    return f"seq{size}"


def tree_digest(path):
    """SHA-256 over relative paths and file digests of a compiled model tree."""
    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError("compiled model tree contains a symlink")
        if item.is_file():
            digest.update(item.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def tree_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


# ------------------------------------------------------------- execution ---

def load_compiled(path, compute_unit, function=None):
    import coremltools as ct

    units = {"cpu_only": ct.ComputeUnit.CPU_ONLY, "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE}
    if compute_unit not in units:  # never ALL / CPU_AND_GPU
        raise ValueError("compute unit must be cpu_only or cpu_and_ne")
    return ct.models.CompiledMLModel(str(path), compute_units=units[compute_unit],
                                     function_name=function)


def pad_token_id(tokenizer):
    value = tokenizer.pad_token_id
    if type(value) is not int:
        raise ValueError("D3 tokenizer declares no pad token")
    return value


def graph_inputs(weights, token_ids, sequence_length, pad_id):
    """Host embedding gather + right padding with a real (finite) pad embedding."""
    import numpy as np

    count = len(token_ids)
    if not 1 <= count <= sequence_length:
        raise ValueError("token ids do not fit the bucket")
    ids = list(token_ids) + [pad_id] * (sequence_length - count)
    hidden = weights.embed(ids).astype(np.float16)[None]
    mask = np.zeros((1, sequence_length), np.int32)
    mask[0, :count] = 1
    return {"embeddings": hidden, "mask": mask}


def checked_outputs(result):
    import numpy as np

    if not isinstance(result, dict) or set(result) != {"hidden", "logits"}:
        raise ValueError("Core ML D3 trunk returned unexpected output features")
    hidden = np.asarray(result["hidden"], dtype=np.float32)
    logits = np.asarray(result["logits"], dtype=np.float32)
    if hidden.shape != (1, HIDDEN) or logits.shape != (1, READOUT_ROWS):
        raise ValueError("Core ML D3 trunk output shapes are wrong")
    if not (np.isfinite(hidden).all() and np.isfinite(logits).all()):
        raise ValueError("Core ML D3 trunk output is nonfinite")
    return hidden[0], logits[0]


def compute_plan_counts(model_path, functions):
    """Anticipated CPU_AND_NE placement per function; ``const`` ops excluded.

    Same counting as ``runtime.coreml_encoder_build.compute_plan_counts``
    (MiniLM compute-op set) plus per-operator rows; anticipated, not observed.
    """
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    compute_ops = PLACEMENT_COMPUTE_OPS
    names = {"MLCPUComputeDevice": "cpu", "MLNeuralEngineComputeDevice": "neural_engine"}
    plan = MLComputePlan.load_from_path(path=str(model_path),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    program = plan.model_structure.program
    if program is None:
        raise ValueError("compute plan is not an ML Program")
    result = {}
    for fname, function in program.functions.items():
        if fname not in functions:
            continue
        counts = {"neural_engine": 0, "cpu": 0, "unknown": 0}
        compute = {"neural_engine": 0, "cpu": 0, "unknown": 0}
        gdn_core = {"neural_engine": 0, "cpu": 0, "unknown": 0}
        inverse_ops = {"neural_engine": 0, "cpu": 0, "unknown": 0}
        by_operator = {}
        for operation in function.block.operations:
            operator = operation.operator_name.rsplit(".", 1)[-1]
            if operator == "const":
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            device = None if usage is None else usage.preferred_compute_device
            kind = type(device).__name__
            if kind == "MLGPUComputeDevice":
                raise ValueError("CPU_AND_NE compute plan unexpectedly prefers GPU")
            where = names.get(kind, "unknown")
            counts[where] += 1
            if operator in compute_ops:
                compute[where] += 1
            outputs = [getattr(value, "name", "") for value in operation.outputs]
            if any(str(name).startswith(GDN_CORE_PREFIX) for name in outputs):
                gdn_core[where] += 1
            if any(is_inverse_op(str(name)) for name in outputs):
                inverse_ops[where] += 1
            slot = by_operator.setdefault(operator, {})
            slot[where] = slot.get(where, 0) + 1
        total = sum(compute.values())
        result[fname] = {
            **counts,
            "neural_engine_compute_fraction": round(compute["neural_engine"] / total, 6) if total else 0.0,
            "compute_ops": compute, "gdn_core_ops": gdn_core, "gdn_inverse_ops": inverse_ops,
            "by_operator": by_operator,
        }
    return result


ANE_FAILURE_MARKERS = ("ANECCompile() FAILED", "failed to compile ANE model")


def ane_compile_failed(stderr_text):
    """E5RT reports an ANE compile failure only on native stderr (through a
    descriptor an in-process redirect cannot see), after which Core ML runs
    the whole model on the CPU. Callers run Core ML in a child process and
    scan its stderr."""
    return any(marker in (stderr_text or "") for marker in ANE_FAILURE_MARKERS)
