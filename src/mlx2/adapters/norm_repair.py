"""Per-group RMSNorm convention: decide, repair, and fail closed.

Qwen3.5/3.6/3.8-family and qwen4_exp checkpoints store RMSNorm gammas as
``w`` for a runtime scale of ``1 + w``; MLX runtimes expect the shifted value.
Converters add the +1 when they convert, but not always uniformly: oQ decides
per tensor with ``add_if_mean_lt_0_5``, so a raw gamma whose mean is already
>= 0.5 is left unshifted. On Qwen3.6-35B-A3B that skips four of the seven
MTP-head norms; on Jundot/Qwen3.6-27B-oQ4e-mtp it skips q_norm, k_norm and
mtp.norm. The head then loads cleanly and drafts badly (cf. omlx#3750,
MTPLX#511).

The convention is therefore decided per group, not per checkpoint:

* **Trunk.** The layout trigger decides (a ``conv1d.weight`` whose last dim
  is not 1 is raw HF layout, so every trunk norm is raw). The decision is then
  verified by :func:`check_pooled_convention`: the count-weighted runtime
  family means must sit decisively closer to 1 than to 0 (and than to the
  opposite fold). A trunk has 12-97 tensors per family, so this is robust.

* **Head** (``mtp.*``, or any separately loaded head given ``trunk_means``).
  Each tensor is decided on its own, in this order of evidence:

  1. A known hash, computed on the float32 upcast so bf16, fp16 and fp32
     copies of the same values all match: :data:`KNOWN_UNSHIFTED_NORMS`
     (raw official tensors) means raw, :data:`KNOWN_SHIFTED_NORMS` (the
     qualified runtime values of heads the statistics cannot place) means
     shifted.
  2. Its stored mean ``m`` against the trunk's same-family runtime mean
     ``mu`` (``gap = mu - m``). A raw head norm sits roughly a unit below its
     trunk family (observed +0.69..+1.04), so ``gap >= HEAD_RAW_MIN_GAP`` is
     raw. A shifted one sits near or above it (observed -0.67..+0.04 outside
     Flash-Next), so ``HEAD_SHIFTED_MIN_GAP <= gap <= HEAD_SHIFTED_MAX_GAP``
     is shifted. A head far *above* its trunk (Flash-Next q/k_norm +2.2,
     hc_norm +1.0..+1.4) is not decisive either way, since the raw
     alternative is equally far off; it needs a hash.
  3. Families the trunk does not have (``pre_fc_norm_*``) use the absolute
     stored mean (only when trunk statistics exist at all): every observed raw value is negative (-0.76..-0.16) and
     every shifted one is >= 0.23, so ``m <= UNPAIRED_RAW_MAX`` is raw and
     ``m >= UNPAIRED_SHIFTED_MIN`` is shifted.

  In raw HF layout the layout is itself evidence that the head is raw; the
  statistics are then only a contradiction check. In converted layout the
  statistics must be decisive. Anything between the thresholds raises
  :class:`NormConventionError` naming the tensors: the Qwen3.6-35B
  ``post_attention_layernorm`` (raw, yet gap +0.44) is exactly such a case and
  is resolved only because its raw hash is known.

Everything is idempotent: a repaired tensor no longer matches its raw hash,
and it sits at or above its trunk family, so a second pass changes nothing.

Residual blind spot: a *raw* tensor whose true runtime value sits 0.75-1.8
above its trunk family reads as shifted (its stored value lands inside the
shifted window). Among the local artifacts only Flash-Next's
``mlp_hyper_connection`` / mixer ``hc_norm`` (+1.36 / +1.04) are in that band;
for them only a change of bytes away from the pinned values is visible, and
it is not flagged. Raw Qwen3.6-27B, Qwen3.8 and Flash-Next sources are not
available locally, so no raw hashes exist for those heads.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class UnshiftedNorm:
    key: str  # without any "language_model." prefix
    sha256: str  # of the float32 upcast of the values (see tensor_sha256)
    reference: str


_QWEN36_35B_RAW = "Qwen/Qwen3.6-35B-A3B@995ad96 model-00026-of-00026.safetensors (raw HF)"

# Float32-upcast SHA-256 of the raw official MTP-head norms, recomputed
# 2026-09-25 from the HF shard above. The same digests match the unshifted
# tensors in Qwen3.6-35B-A3B-...-oQ4e-mtp (bf16) and in
# Noctalin/Qwen3.6-35B-A3B-oQ4-fp16-mtp (fp16).
QWEN36_35B_UNSHIFTED_MTP_NORMS = (
    UnshiftedNorm(
        "mtp.layers.0.input_layernorm.weight",
        "c4df2a20e165909588f7abaae3bb6f532c035df74642155c5a92b08c37b3b983",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.layers.0.post_attention_layernorm.weight",
        "a2fe7de809d3d1868b3b85bcb5ffbc03560877d2e643b5bd985e178d95a03c96",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.layers.0.self_attn.q_norm.weight",
        "64c82070316ecb863867bb560d5e5aae660011992f36fb86bd6af33a165b5ba2",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.layers.0.self_attn.k_norm.weight",
        "e21d7abc5248bbd93176c10f2b7c07ced020445d8e9680b0212e1208c3cf198a",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.norm.weight",
        "0046ba0ddfef6fbab565e4c8a955aac2312d340f9a7780e02542e88ef58fee49",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.pre_fc_norm_embedding.weight",
        "1270f06407b20678147853a67e50014f0805b2b031e684a61e4a4a98a07ca0e0",
        _QWEN36_35B_RAW,
    ),
    UnshiftedNorm(
        "mtp.pre_fc_norm_hidden.weight",
        "2caa58c2cfaef487c7e11a4e27143d3f8f67c2c8be919a39f4ef3f3bb0a3a4c4",
        _QWEN36_35B_RAW,
    ),
)

# No raw Qwen3.6-27B (or Qwen3.8/Flash-Next) HF source is available locally,
# so those heads are decided by the statistical rule alone.
KNOWN_UNSHIFTED_NORMS = QWEN36_35B_UNSHIFTED_MTP_NORMS

_FLASH_NEXT_SHIFTED = (
    "Qwen3.8-Flash-Next-MLX-4bit-MTP (float32-upcast SHA-256, 2026-09-25; "
    "identical in -Uncensored-MLX2-4bit-MTP, -Uncensored-MLX-Serve-4bit and "
    "-MLX-4bit-MTP-VLM; self-MTP acceptance 0.742)"
)

# Correctly shifted head norms that sit too far above their trunk family for
# the statistics to place (gap < HEAD_SHIFTED_MIN_GAP). A match means shifted.
KNOWN_SHIFTED_NORMS = (
    UnshiftedNorm(
        "mtp.hyper_connection_mixer.hc_norm.weight",
        "6377449b6b03b0e5c4f0e342d07fdd8aca34767791fe940abbcd76ed9b57cd0e",
        _FLASH_NEXT_SHIFTED,
    ),
    UnshiftedNorm(
        "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
        "4ca2b04404fb103d70917acac965e28af891fe7b40ae6728d98fcac3ae3d5742",
        _FLASH_NEXT_SHIFTED,
    ),
    UnshiftedNorm(
        "mtp.layers.0.self_attn.q_norm.weight",
        "7e5631a4de5baf3bb6cd4ef9310abe07f8a50b5a1e454b2637e42afed58ba96d",
        _FLASH_NEXT_SHIFTED,
    ),
    UnshiftedNorm(
        "mtp.layers.0.self_attn.k_norm.weight",
        "e2b7fbe2e0ffdd06febd9e114f646fdf69aacafe32f8672b907ec5e33106ed03",
        _FLASH_NEXT_SHIFTED,
    ),
)

# Head thresholds, calibrated 2026-09-25 on every local artifact (see
# docs/PROVENANCE.md, "Per-group norm convention"). gap = trunk family runtime
# mean - head stored mean.
HEAD_SHIFTED_MAX_GAP = 0.25  # shifted observed <= +0.04
HEAD_SHIFTED_MIN_GAP = -0.80  # shifted observed >= -0.67 (Flash-Next pinned instead)
HEAD_RAW_MIN_GAP = 0.55  # raw observed >= +0.69 (35B post_attn +0.44 is ambiguous)
UNPAIRED_RAW_MAX = 0.05  # raw pre_fc observed <= -0.16
UNPAIRED_SHIFTED_MIN = 0.15  # shifted pre_fc observed >= +0.23
POOLED_MARGIN = 0.25

_PREFIXES = ("model.language_model.", "language_model.")
_LAYER = re.compile(r"(^|\.)layers\.\d+\.")


class NormConventionError(ValueError):
    """The RMSNorm convention of some group could not be decided safely."""


def tensor_sha256(value) -> str:
    """SHA-256 of ``value`` upcast to float32 (canonical across bf16/fp16/fp32)."""
    import mlx.core as mx
    import numpy as np

    return hashlib.sha256(np.array(value.astype(mx.float32)).tobytes()).hexdigest()


def _normalize(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_head_key(key: str) -> bool:
    name = _normalize(key)
    return name.startswith("mtp.") or name.startswith("model.mtp.")


def norm_family(key: str) -> str:
    """Key with prefixes, ``mtp.``/``model.`` and layer indices removed.

    ``mtp.layers.0.self_attn.q_norm.weight`` and
    ``language_model.model.layers.7.self_attn.q_norm.weight`` share the family
    ``layers.*.self_attn.q_norm.weight``; ``mtp.norm.weight`` pairs with the
    trunk's final ``model.norm.weight``.
    """
    name = _normalize(key).removeprefix("model.").removeprefix("mtp.")
    return _LAYER.sub(r"\1layers.*.", name)


def _mean(value) -> float:
    import mlx.core as mx

    return float(value.astype(mx.float32).mean().item())


@dataclass(frozen=True)
class NormDecision:
    key: str
    group: str  # "trunk" or "head"
    raw: bool  # True: +1 is (to be) applied
    evidence: str
    stored_mean: float


@dataclass
class NormConventionReport:
    trunk_raw: bool
    decisions: list[NormDecision] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)

    @property
    def raw_keys(self) -> list[str]:
        return [d.key for d in self.decisions if d.raw]

    @property
    def repaired_head_keys(self) -> list[str]:
        """Head tensors shifted although the trunk was not (a skipped fold)."""
        return [
            d.key for d in self.decisions if d.group == "head" and d.raw and not self.trunk_raw
        ]

    def summary(self) -> dict:
        return {
            "trunk": "raw (+1 applied)" if self.trunk_raw else "converted",
            "head": {
                d.key: {
                    "raw": d.raw,
                    "evidence": d.evidence,
                    "stored_mean": round(d.stored_mean, 4),
                }
                for d in self.decisions
                if d.group == "head"
            },
            "repaired_head": self.repaired_head_keys,
        }


def check_pooled_convention(families: Mapping[str, list[float]], raw: bool, *,
                            margin: float = POOLED_MARGIN, hint: str = "") -> None:
    """Require the runtime (post-fold) family means to be decisively one-centered.

    ``families`` maps a family name to the runtime means of its tensors.
    Three count-weighted aggregates over the summed n:

        A_one  = sum(n * |mean - 1|) / sum(n)          # what we applied
        A_zero = sum(n * |mean|) / sum(n)              # +1 fold missing
        A_alt  = sum(n * |mean + shift - 1|) / sum(n)  # opposite fold

    where ``shift`` un-applies the fold (-1 if it ran, +1 if it did not). The
    applied convention must beat both by ``margin``. Raises
    :class:`NormConventionError` on a decisive wrong convention and on
    ambiguity alike.
    """
    if not families:
        return
    shift = -1.0 if raw else 1.0
    total = a_one = a_zero = a_alt = 0.0
    rows = []
    for name, means in families.items():
        count = len(means)
        mean = sum(means) / count
        a_one += count * abs(mean - 1.0)
        a_zero += count * abs(mean)
        a_alt += count * abs(mean + shift - 1.0)
        total += count
        rows.append(
            (abs(mean - 1.0) - min(abs(mean), abs(mean + shift - 1.0)), name, count, mean)
        )
    a_one /= total
    a_zero /= total
    a_alt /= total
    if a_one + margin <= a_zero and a_one + margin <= a_alt:
        return
    rows.sort(reverse=True)
    worst = ", ".join(f"{name} (n={count}, mean {mean:.3f})" for (_, name, count, mean) in rows[:4])
    applied = "raw (+1 offset applied)" if raw else "converted (no offset applied)"
    if a_zero + margin < a_one:
        verdict = "the stored RMSNorm gains are decisively zero-centered, so the +1 fold is missing"
    elif a_alt + margin < a_one:
        verdict = "the opposite fold fits the stored RMSNorm gains decisively better, so the offset has been applied the wrong number of times"
    else:
        verdict = "the stored RMSNorm gains do not decisively favour either convention, so the applied fold cannot be verified"
    raise NormConventionError(
        f"norm convention check failed: the fold trigger chose the {applied} convention, but {verdict} (A_one {a_one:.3f} vs A_zero {a_zero:.3f} vs A_alt {a_alt:.3f}; required A_one + {margin:.2f} <= both, over n={int(total)} gains in {len(families)} families). Worst families: {worst}.{(' ' + hint) if hint else ''}"
    )


def _head_statistical_verdict(mean: float, trunk_mu: float | None):
    """Return (raw: bool | None, evidence) for one head tensor."""
    if trunk_mu is None:
        if mean <= UNPAIRED_RAW_MAX:
            return True, f"unpaired mean {mean:.3f} <= {UNPAIRED_RAW_MAX}"
        if mean >= UNPAIRED_SHIFTED_MIN:
            return False, f"unpaired mean {mean:.3f} >= {UNPAIRED_SHIFTED_MIN}"
        return None, (
            f"unpaired mean {mean:.3f} between {UNPAIRED_RAW_MAX} and {UNPAIRED_SHIFTED_MIN}"
        )
    gap = trunk_mu - mean
    detail = f"mean {mean:.3f} vs trunk family {trunk_mu:.3f} (gap {gap:+.3f})"
    if HEAD_SHIFTED_MIN_GAP <= gap <= HEAD_SHIFTED_MAX_GAP:
        return False, f"{detail} in [{HEAD_SHIFTED_MIN_GAP}, {HEAD_SHIFTED_MAX_GAP}]"
    if gap >= HEAD_RAW_MIN_GAP:
        return True, f"{detail} >= {HEAD_RAW_MIN_GAP}"
    if gap < HEAD_SHIFTED_MIN_GAP:
        return None, f"{detail} < {HEAD_SHIFTED_MIN_GAP}: too far above the trunk to place"
    return None, f"{detail} between {HEAD_SHIFTED_MAX_GAP} and {HEAD_RAW_MIN_GAP}"


def trunk_family_means(weights: Mapping, fold_suffixes: Iterable[str], *, raw: bool) -> dict[str, float]:
    """Runtime (post-fold) mean of every trunk norm family in ``weights``."""
    families: dict[str, list[float]] = {}
    suffixes = tuple(fold_suffixes)
    for key, value in weights.items():
        if is_head_key(key) or value.ndim != 1 or not key.endswith(suffixes):
            continue
        families.setdefault(norm_family(key), []).append(_mean(value) + (1.0 if raw else 0.0))
    return {name: sum(v) / len(v) for name, v in families.items()}


def decide_norm_convention(
    weights: Mapping,
    *,
    fold_suffixes: Iterable[str],
    trunk_raw: bool,
    unshifted: Iterable[UnshiftedNorm] = KNOWN_UNSHIFTED_NORMS,
    shifted: Iterable[UnshiftedNorm] = KNOWN_SHIFTED_NORMS,
    trunk_means: Mapping[str, float] | None = None,
    check_trunk: bool = True,
    on_ambiguous: str = "raise",
    hint: str = "",
) -> NormConventionReport:
    """Decide, per group and per head tensor, which stored norms are raw.

    Reads ``weights`` only. ``trunk_means`` (family -> runtime mean) replaces
    the trunk statistics for a head loaded without its trunk.
    ``on_ambiguous="follow_trunk"`` keeps the legacy one-decision-per-checkpoint
    behaviour for undecidable head tensors instead of raising.
    """
    if on_ambiguous not in ("raise", "follow_trunk"):
        raise ValueError(f"on_ambiguous must be 'raise' or 'follow_trunk', got {on_ambiguous!r}")
    suffixes = tuple(fold_suffixes)
    known: dict[tuple[str, str], tuple[bool, str]] = {}
    for entry in shifted:
        known[(entry.key, entry.sha256)] = (False, entry.reference)
    for entry in unshifted:
        known[(entry.key, entry.sha256)] = (True, entry.reference)
    known_keys = {key for key, _ in known}
    report = NormConventionReport(trunk_raw=bool(trunk_raw))
    trunk_runtime: dict[str, list[float]] = {}
    head = []
    for key in sorted(weights):
        value = weights[key]
        if getattr(value, "ndim", None) != 1 or not key.endswith(suffixes):
            continue
        mean = _mean(value)
        if is_head_key(key):
            head.append((key, value, mean))
            continue
        report.decisions.append(NormDecision(key, "trunk", bool(trunk_raw), "layout", mean))
        trunk_runtime.setdefault(norm_family(key), []).append(mean + (1.0 if trunk_raw else 0.0))
    if check_trunk:
        check_pooled_convention(trunk_runtime, bool(trunk_raw), hint=hint)
    family_mu = (
        dict(trunk_means)
        if trunk_means is not None
        else {name: sum(v) / len(v) for name, v in trunk_runtime.items()}
    )
    problems = []
    for key, value, mean in head:
        name = _normalize(key).removeprefix("model.")
        hit = known.get((name, tensor_sha256(value))) if name in known_keys else None
        if hit is not None:
            is_raw, reference = hit
            if trunk_raw and not is_raw:
                problems.append(
                    f"{key}: trunk decided raw, but the head tensor matches a known shifted tensor ({reference})"
                )
                continue
            label = "known raw" if is_raw else "known shifted"
            report.decisions.append(
                NormDecision(key, "head", is_raw, f"sha256 {label}: {reference}", mean)
            )
            continue
        if family_mu:
            verdict, evidence = _head_statistical_verdict(mean, family_mu.get(norm_family(key)))
        else:
            # No trunk statistics at all (a head loaded on its own without
            # ``trunk_means``): a head family cannot be told from an unpaired
            # one, so nothing but a hash is decisive.
            verdict, evidence = None, f"mean {mean:.3f}, no trunk statistics to compare against"
        if trunk_raw:
            if verdict is False:
                problems.append(
                    f"{key}: trunk decided raw, but the head tensor looks already shifted ({evidence})"
                )
                continue
            report.decisions.append(
                NormDecision(key, "head", True, f"trunk decided raw; {evidence}", mean)
            )
            continue
        if verdict is None:
            if on_ambiguous == "follow_trunk":
                report.decisions.append(
                    NormDecision(key, "head", False, f"ambiguous, followed trunk; {evidence}", mean)
                )
                continue
            problems.append(f"{key}: {evidence}")
            continue
        report.decisions.append(NormDecision(key, "head", verdict, evidence, mean))
    if problems:
        raise NormConventionError(
            "cannot decide the RMSNorm (+1) convention of "
            f"{len(problems)} head tensor(s) (trunk: "
            f"{'raw' if trunk_raw else 'converted'}); no known-raw hash "
            "matched and the statistics are not decisive: "
            + "; ".join(problems)
            + ". Add the raw tensor's float32 SHA-256 to norm_repair.KNOWN_UNSHIFTED_NORMS "
            "or fix the checkpoint."
            + (f" {hint}" if hint else "")
        )
    return report


def apply_norm_convention(weights: dict, report: NormConventionReport) -> NormConventionReport:
    """Add 1.0 in place to every tensor ``report`` marks raw (dtype preserved)."""
    for decision in report.decisions:
        if decision.raw:
            weights[decision.key] = weights[decision.key] + 1.0
            report.applied.append(decision.key)
    return report


def resolve_norm_convention(weights: dict, **kwargs) -> NormConventionReport:
    """:func:`decide_norm_convention` then :func:`apply_norm_convention`."""
    return apply_norm_convention(weights, decide_norm_convention(weights, **kwargs))


def repair_unshifted_norms(weights: dict, table=QWEN36_35B_UNSHIFTED_MTP_NORMS) -> list[str]:
    """Add 1.0 in place to every 1-D tensor whose canonical hash is in ``table``.

    Kept for callers that only want the hash evidence; loaders use
    :func:`resolve_norm_convention`. ``weights`` may carry a
    ``language_model.`` prefix. Returns the repaired keys.
    """
    wanted = {entry.key: entry.sha256 for entry in table}
    repaired = []
    for name in sorted(weights):
        digest = wanted.get(name.removeprefix("language_model."))
        value = weights[name]
        if digest is None or value.ndim != 1:
            continue
        if tensor_sha256(value) == digest:
            weights[name] = value + 1.0
            repaired.append(name)
    return repaired


def norm_means(weights: dict, prefix: str) -> dict[str, float]:
    """Mean runtime gamma of every 1-D ``*norm*`` tensor under ``prefix``."""
    return {
        name: round(_mean(value), 4)
        for name, value in sorted(weights.items())
        if name.removeprefix("language_model.").startswith(prefix)
        and "norm" in name
        and value.ndim == 1
    }
