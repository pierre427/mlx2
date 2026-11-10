# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import math
import os
from functools import lru_cache
from typing import Callable, Dict, List, Optional
import mlx.core as mx

from ..sampling_defaults import SAMPLING_EPS


def _probe_safe(function):
    """Declare a stateless transform safe to reuse for provisional tokens.

    Stateless also means ``history_pure`` (P5): the output is a function of
    the token history alone, so a replay can rebuild it from the tokens.
    """
    function.probe = function
    function.history_pure = True
    return function


class LaneRNG:
    """A per-request random key that advances once per draw.

    Batched decode must not draw from the global ``mx.random`` stream: with one
    stream, a lane that joins or leaves reorders every other lane's draws, so a
    lane's tokens depend on the traffic scheduled beside it. A lane carries a
    ``LaneRNG`` instead, and every draw takes a subkey from it, so its draw
    sequence is a function of its own seed and its own history only.

    Two rules keep that true:

    * **Carry and split, never re-derive.** ``mx.random.key(seed)`` returns the
      same key each call, so re-deriving repeats draws. ``next_key`` splits the
      carried key, which advances it.
    * **Never rewind.** A rejected draft rewinds tokens and caches, not the
      key. Reusing a consumed key would couple the correction draw to the
      proposal it replaces, and the residual acceptance rule needs those
      independent. There is deliberately no rewind method.

    ``key`` is the carried key. Put it in a snapshot and rebuild the lane with
    ``from_key`` so a restored request continues its stream instead of
    repeating it; use ``fork`` (not a copy) where one lane becomes several.
    """

    __slots__ = ("_key", "_draws")

    def __init__(self, seed: int):
        self._key = mx.random.key(int(seed))
        self._draws = 0

    @classmethod
    def from_key(cls, key: mx.array, draws: int = 0) -> "LaneRNG":
        """Rebuild a lane from a carried key (snapshot restore)."""
        lane = cls.__new__(cls)
        lane._key = key
        lane._draws = int(draws)
        return lane

    @property
    def key(self) -> mx.array:
        """The carried key: the lane's position in its own stream."""
        return self._key

    @property
    def draws(self) -> int:
        """Key consumptions so far. Monotone — a rollback never lowers it."""
        return self._draws

    def next_key(self) -> mx.array:
        """Advance the carried key and return a subkey for exactly one draw."""
        (self._key, sub) = mx.random.split(self._key)
        self._draws += 1
        return sub

    def fork(self, n: int) -> List["LaneRNG"]:
        """Make ``n`` independent lanes and advance this one.

        For parallel sampling (``n>1`` branches) and any other place one lane
        becomes several: copying the object would replay one stream in every
        copy.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        keys = mx.random.split(self._key, n + 1)
        self._key = keys[0]
        self._draws += 1
        return [LaneRNG.from_key(keys[i + 1]) for i in range(n)]


def draw_key(rng: Optional[LaneRNG]) -> Optional[mx.array]:
    """A subkey for one draw, or ``None`` to use the global stream.

    ``key=None`` is the ``mx.random`` default, so a call site that always
    forwards this stays byte-identical when no lane key is supplied.
    """
    return None if rng is None else rng.next_key()


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    xtc_special_tokens: List[int] = [],
) -> Callable[[mx.array], mx.array]:
    """
    Make a sampler function for use with ``generate_step``.

    Args:
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        top_k (int, optional): The top k tokens ranked by probability to constrain
          the sampling to.
        xtc_probability (float, optional): The probability of applying XTC
            sampling.
        xtc_threshold (float, optional): The threshold the probs need to reach
            for being sampled.
        xtc_special_tokens (list(int), optional): List of special tokens IDs to
            be excluded from XTC sampling.


    Returns:
        Callable[mx.array, mx.array]:
            A sampler which takes log-probabilities and returns tokens.
    """
    # resolve_sampling already maps 0 < temp < SAMPLING_EPS to 0; guard here
    # too so a direct caller cannot scale by an overflowing 1 / temp.
    if 0 <= temp < SAMPLING_EPS:
        argmax_sampler = lambda x: mx.argmax(x, axis=-1)
        argmax_sampler.batch_groupable = True
        argmax_sampler.deterministic = True  # draws nothing
        argmax_sampler.greedy_argmax = True  # trusted normalized-greedy candidate
        return argmax_sampler
    sampling_methods = []
    if _fuse_top_p_top_k(top_p, top_k, min_p, xtc_probability > 0.0):
        _count_sampling_filter("sampler_cutoff")
        sampling_methods.append(
            lambda x, p=top_p, k=top_k: apply_top_p_top_k_cutoff(x, p, k)
        )
        top_p, top_k = 0.0, 0
    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        sampling_methods.append(
            lambda x: apply_xtc(x, xtc_probability, xtc_threshold, xtc_special_tokens)
        )
    if top_k > 0:
        sampling_methods.append(lambda x: apply_top_k(x, top_k))

    def sampler(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        return categorical_sampling(logprobs, temp)

    compiled = mx.compile(sampler, inputs=mx.random.state, outputs=mx.random.state)

    def sampler(logprobs):
        return compiled(logprobs)

    sampler.batch_groupable = xtc_probability <= 0.0
    return sampler


def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability.

    Args:
        logprobs: A vector of log probabilities.
        top_k (int): Top k tokens to sample from.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not 0 < top_k < vocab_size:
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}) interval, but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return masked_logprobs


def apply_min_p(
    logprobs: mx.array, min_p: float, min_tokens_to_keep: int = 1
) -> mx.array:
    """
    Apply min-p sampling to the logprobs.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.

    Args:
        logprobs: A vector of log probabilities.
        min_p (float): Minimum token probability. Typical values are in the
            0.01-0.2 range, comparably selective as setting `top_p` in the
            0.99-0.8 range.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
            be filtered. Default: ``1``.

    """
    if not 0 <= min_p <= 1.0:
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )
    if min_p == 0:
        return logprobs
    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = logprobs < scaled_min_p
    if min_tokens_to_keep > 1:
        top_indices = mx.argpartition(logprobs, kth=-min_tokens_to_keep, axis=-1)
        top_indices = top_indices[..., -min_tokens_to_keep:]
        tokens_to_remove = mx.put_along_axis(
            tokens_to_remove, top_indices, False, axis=-1
        )
    return mx.where(tokens_to_remove, -float("inf"), logprobs)


def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logprobs: A vector of log probabilities.
        top_p: The cumulative probability threshold for top-p filtering.
    Returns:
        token selected based on the top-p criterion.
    """
    # The nucleus is a cumulative-mass decision over the whole vocabulary.
    # Served logits are bfloat16, whose 8-bit mantissa cannot accumulate a
    # 248K-term prefix sum: on the CPU backend the running total stalls and
    # the nucleus collapses (55 tokens kept where float32 keeps 47,502; flat
    # rows keep nothing and sample NaN).  The GPU backend happens to
    # accumulate wider.  Do the mass arithmetic in float32 on every device and
    # apply the resulting mask to the caller's dtype.
    #
    # The threshold is a fraction of the row's own accumulated mass, not of
    # 1.0: rounded (bf16/fp16) logprobs sum to slightly less than one, and a
    # tiny top_p rounds ``1 - top_p`` to 1.0 in float32, so an absolute
    # threshold masked every token and the sampler drew uniform noise
    # (mlx-lm#1912).  The most likely token always survives, which is the
    # correct nucleus as top_p approaches zero.
    probs = mx.exp(logprobs.astype(mx.float32))
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    total = cumulative_probs[..., -1:]
    width = sorted_indices.shape[-1]
    keep_sorted = (cumulative_probs > (1 - top_p) * total) | (
        mx.arange(width) == width - 1
    )
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(width, dtype=sorted_indices.dtype),
        axis=-1,
    )
    keep = mx.take_along_axis(keep_sorted, inverse_indices, axis=-1)
    return mx.where(keep, logprobs, -float("inf"))



# Opt-in fused top-p + top-k filter (idea from vllm-project/vllm#66534; see
# docs/PROVENANCE.md).  Off by default: MLX2_SAMPLING_TOPK_CUTOFF=1 enables it.
SAMPLING_TOPK_CUTOFF_ENV = "MLX2_SAMPLING_TOPK_CUTOFF"
SAMPLING_FILTER_STATS: Dict[str, int] = {}


def sampling_topk_cutoff_enabled() -> bool:
    """Read at sampler construction, never at import (no import-time flag)."""
    return os.environ.get(SAMPLING_TOPK_CUTOFF_ENV, "0").strip() == "1"


def _fuse_top_p_top_k(top_p: float, top_k: int, min_p: float, xtc: bool = False) -> bool:
    """The fused filter covers exactly top-p followed by top-k.

    min-p and XTC sit between them in the chain; those requests keep the
    reference chain so the fused path never has to argue about composition.
    """
    return (
        0.0 < top_p < 1.0
        and 0 < top_k <= _FUSE_MAX_TOP_K
        and min_p == 0.0
        and not xtc
        and sampling_topk_cutoff_enabled()
    )


# vLLM caps its fused cutoff at k <= 1024 too; past it ``mx.topk`` stops
# beating the sort and the float32 boundary error grows with k.
_FUSE_MAX_TOP_K = 1024


def _count_sampling_filter(path: str) -> None:
    """Counts sampler/transform constructions that selected the fused path
    (a construction, not a draw: memoized transforms count each request)."""
    SAMPLING_FILTER_STATS[path] = SAMPLING_FILTER_STATS.get(path, 0) + 1


def sampling_filter_status() -> dict:
    return {
        "configured": os.environ.get(SAMPLING_TOPK_CUTOFF_ENV, "unset"),
        "default": "0",
        "policy": "top_p_top_k_cutoff_v1" if sampling_topk_cutoff_enabled() else "reference",
        "qualification": "unqualified_opt_in",
        "counts_unit": "constructions",
        "max_top_k": _FUSE_MAX_TOP_K,
        "counts": dict(SAMPLING_FILTER_STATS),
    }


def _row_mass(logprobs: mx.array, chunk: int = 256) -> mx.array:
    """The row's float32 probability mass, summed in two levels.

    ``apply_top_p`` accumulates ascending, so the long tail of tiny
    probabilities survives; one flat float32 sum in index order can lose
    ~1e-5 of it on a 248K vocabulary and move the nucleus boundary by a
    token.  Chunked partial sums keep the error near float32 epsilon.
    """
    probs = mx.exp(logprobs.astype(mx.float32))
    width = probs.shape[-1]
    pad = (-width) % chunk
    if pad:
        probs = mx.pad(probs, [(0, 0)] * (probs.ndim - 1) + [(0, pad)])
    partial = probs.reshape(*probs.shape[:-1], -1, chunk).sum(axis=-1)
    return partial.sum(axis=-1, keepdims=True)


def sampling_filter_receipt(sampling: Optional[dict]) -> dict:
    """Per-request route-receipt label: which filter law this request's
    effective sampling selects (labelled unqualified when fused)."""
    sampling = sampling or {}
    temp = sampling.get("temperature") or 0.0
    fused = temp > 0 and _fuse_top_p_top_k(
        sampling.get("top_p") or 0.0, int(sampling.get("top_k") or 0), sampling.get("min_p") or 0.0
    )
    if not fused:
        return {"policy": "reference"}
    return {"policy": "top_p_top_k_cutoff_v1", "qualification": "unqualified_opt_in"}


def apply_top_p_top_k_cutoff(logprobs: mx.array, top_p: float, top_k: int) -> mx.array:
    """``apply_top_k(apply_top_p(x, top_p), top_k)`` as one scalar cutoff per row.

    Both filters keep a prefix of the descending order, so their composition
    is ``logprobs >= cutoff``.  ``mx.topk`` finds the ``k`` largest values;
    the nucleus is decided on those ``k`` alone against the row's full float32
    mass (the same relative threshold ``apply_top_p`` uses), so no
    vocabulary-wide sort or scatter runs.

    Never keeps more than ``top_k`` tokens (the sparse speculative target law
    relies on it) and keeps the reference chain's count, except when the
    nucleus boundary lies within float32 rounding of the threshold: both
    accumulate in float32 in different orders, so there they can differ by
    one token (measured margins <= 3e-7 of the row mass, this side matching a
    float64 evaluation each time).  The kept set is the reference's except
    at a value tied across the boundary, where the reference keeps an
    argsort / argpartition-order subset and this keeps the lowest-index
    tokens.  The most likely token always survives.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not 0 < top_k < vocab_size:
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}) interval, but is {top_k}."
        )
    values = mx.sort(mx.topk(logprobs, top_k, axis=-1), axis=-1)[..., ::-1]
    probs = mx.exp(values.astype(mx.float32))
    total = _row_mass(logprobs)
    # apply_top_p keeps a token when the ascending cumulative mass up to and
    # including it exceeds (1 - top_p) of the row total.  Mirror that order:
    # the mass outside the top k (summed accurately), then the top-k values
    # accumulated smallest first.
    kth = values[..., -1:]
    outside = _row_mass(mx.where(logprobs < kth, logprobs, -float("inf")))
    tied_outside = (logprobs == kth).astype(mx.int32).sum(axis=-1, keepdims=True) - (
        values == kth
    ).astype(mx.int32).sum(axis=-1, keepdims=True)
    outside = outside + tied_outside.astype(mx.float32) * probs[..., -1:]
    tail = outside + mx.cumsum(probs[..., ::-1], axis=-1)[..., ::-1]
    keep = tail > (1 - top_p) * total
    kept = mx.maximum(keep.astype(mx.int32).sum(axis=-1, keepdims=True), 1)
    cutoff = mx.take_along_axis(values, kept - 1, axis=-1)
    # Everything strictly above the cutoff is in; tokens equal to it fill the
    # remaining slots in index order, so the count matches the reference.
    above_cutoff = logprobs > cutoff
    tied = logprobs == cutoff
    slots = kept - above_cutoff.astype(mx.int32).sum(axis=-1, keepdims=True)
    tie_rank = mx.cumsum(tied.astype(mx.int32), axis=-1)
    survive = above_cutoff | (tied & (tie_rank <= slots))
    return mx.where(survive, logprobs, mx.array(-float("inf"), logprobs.dtype))


def apply_xtc(
    logits: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: List[int],
) -> mx.array:
    """
    Apply XTC sampling to the logits.

    Args:
        logits: The logits from the model's output.
        xtc_probability (float): Probability of XTC sampling to happen for each token
        xtc_threshold (float): The threshold the probs need to reach for being sampled.
        special_tokens_ids (list(int)): List of special tokens IDs to be excluded from XTC sampling.
    """
    if not 0 <= xtc_threshold <= 0.5:
        raise ValueError(
            f"`threshold` has to be a float in the [0, 0.5] interval, but is {xtc_threshold}"
        )
    if not 0 <= xtc_probability <= 1.0:
        raise ValueError(
            f"`probability` has to be a float in the [0, 1] interval, but is {xtc_probability}"
        )
    probs = mx.softmax(logits, -1)
    mask = probs > mx.where(probs > xtc_threshold, probs, mx.inf).min(
        axis=-1, keepdims=True
    )
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False
    return mx.where(
        mx.random.uniform(0, 1) > xtc_probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


def _scale_by_temperature(logits, temp):
    # float16 tops out at 65504: a flat row's max logprob (~-12) times
    # 1 / 1e-4 is -inf, every token then is, and categorical returns id 0.
    # bfloat16 and float32 keep their dtype so seeded draws do not move.
    if logits.dtype == mx.float16:
        logits = logits.astype(mx.float32)
    return logits * (1 / temp)


def categorical_sampling(logits, temp):
    return mx.random.categorical(_scale_by_temperature(logits, temp))


def make_transformed_logprobs(
    temp: float,
    *,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
) -> Callable[[mx.array], mx.array]:
    """See ``_make_transformed_logprobs``; the filter policy joins the memo key
    so toggling ``MLX2_SAMPLING_TOPK_CUTOFF`` never returns a stale chain."""
    fused = _fuse_top_p_top_k(top_p, top_k, min_p)
    if fused:
        _count_sampling_filter("transform_cutoff")
    return _make_transformed_logprobs(
        temp,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        min_tokens_to_keep=min_tokens_to_keep,
        fused=fused,
    )


@lru_cache(maxsize=32)
def _make_transformed_logprobs(
    temp: float,
    *,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
    fused: bool = False,
) -> Callable[[mx.array], mx.array]:
    """Make a map from raw logits to the log-probabilities of the
    distribution ``make_sampler(temp, top_p, min_p, top_k)`` samples from.

    Memoized at module level per parameter tuple: repeat requests (the few
    serving profiles dominate) reuse one compiled chain, and MLX's own trace
    cache then covers the per-shape traces inside it. Safe to share — the
    transform is deterministic (no random state is captured, unlike
    ``make_sampler``, which must compile per call to bind the calling
    thread's RNG state).

    Mirrors the sampler exactly: normalization first promotes the logits to
    float32 so low-precision subtraction cannot collapse distinct logits into
    artificial ties. The filter chain — top-p, then min-p, then top-k, in
    ``make_sampler`` order; XTC is not supported — plus the temperature scale
    runs inside ``mx.compile``, so knife-edge filter ties resolve with the same
    fused rounding as the compiled sampler. Filtered tokens are exactly
    ``-inf``. Requires ``temp > 0``; batched over leading axes.
    """
    if not temp or temp <= 0:
        raise ValueError(f"make_transformed_logprobs requires temp > 0, got {temp}")
    sampling_methods = []
    if fused:
        sampling_methods.append(
            lambda x, p=top_p, k=top_k: apply_top_p_top_k_cutoff(x, p, k)
        )
        top_p, top_k = 0.0, 0
    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if top_k > 0:
        sampling_methods.append(lambda x: apply_top_k(x, top_k))

    def chain(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        return logprobs * (1 / temp)

    compiled = mx.compile(chain)

    def transform(logits):
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        scaled = compiled(logprobs)
        return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)

    return _probe_safe(transform)


def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    presence_penalty: Optional[float] = None,
    presence_context_size: Optional[int] = 20,
    frequency_penalty: Optional[float] = None,
    frequency_context_size: Optional[int] = 20,
    penalty_generation_start: Optional[int] = None,
):
    """
    Make logits processors for use with ``generate_step``.

    Args:
        repetition_penalty (float, optional): A (sign-aware) multiplicative
          penalty for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        presence_penalty (float, optional): An additive penalty to reduce
          repeating tokens.
        presence_context_size (int, optional): The number of tokens to consider
          for the presence penalty. Default: ``20``.
        frequency_penalty (float, optional): An additive penalty to reduce
          repeating tokens. The tokens are penalized proportionally to their
          frequency.
        frequency_context_size (int, optional): The number of tokens to consider
          for the frequency penalty. Default: ``20``.
        logit_bias (dictionary, optional): Additive logit bias.
        penalty_generation_start (int, optional): When set, the presence and
          frequency penalties count only tokens at or after this position of
          the processor's token history (the prompt length), i.e. generated
          tokens, as vLLM and the OpenAI API define them.  The repetition
          penalty always covers prompt and generated tokens (HF/vLLM).

    Returns:
        List[Callable[[mx.array, mx.array], mx.array]]:
            A list of logits processors. Each processor in the list is a
            callable which takes an array of tokens and an array of logits
            and returns the updated logits.
    """
    logits_processors = []
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(_, logits):
            return logits.at[:, indices].add(values)

        logits_processors.append(_probe_safe(logit_bias_processor))

    if repetition_penalty is not None and repetition_penalty != 0:
        logits_processors.append(
            make_repetition_penalty(repetition_penalty, repetition_context_size)
        )
    for make_penalty, penalty, context_size in (
        (make_presence_penalty, presence_penalty, presence_context_size),
        (make_frequency_penalty, frequency_penalty, frequency_context_size),
    ):
        if penalty is not None and penalty != 0:
            logits_processors.append(
                make_penalty(
                    penalty,
                    context_size,
                    generation_start=penalty_generation_start,
                )
            )

    return logits_processors


def make_repetition_penalty(penalty: float, context_size: int = 20):
    """
    Make repetition penalty processor.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        penalty (float): The repetition penalty factor to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]:
            The repetition penalty processor.
    """
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be a non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[:, tokens] = selected_logits
        return logits

    return _probe_safe(repetition_penalty_processor)


def _generated_window(tokens, context_size, generation_start):
    if generation_start is not None:
        tokens = tokens[int(generation_start):]
    return tokens[-context_size:]


def make_presence_penalty(
    penalty: float, context_size: int = 20, *, generation_start: Optional[int] = None
):
    """
    Make a presence penalty processor.

    Corresponds to the OpenAI option with the same name. Namely, subtracts
    ``penalty`` from a logit if the token has occured at least once in the
    ``context_size`` previous tokens.

    Args:
        penalty (float): The presence penalty to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.  ``0`` means every counted token.
        generation_start (int, optional): Count only tokens at or after this
            history position (generated tokens when it is the prompt length).

    Returns:
        Callable[[mx.array, List[int]], mx.array]
    """

    def presence_penalty_processor(tokens, logits):
        tokens = _generated_window(tokens, context_size, generation_start)
        if len(tokens) > 0:
            logits[:, tokens] -= penalty
        return logits

    # Declared contract for batched verifiers: the output is ``logits -
    # penalty`` on exactly the distinct tokens of
    # ``_generated_window(history, context_size, generation_start)``.
    presence_penalty_processor.presence_window = (
        penalty,
        int(context_size),
        None if generation_start is None else int(generation_start),
    )
    # Private progressive/AST branches may evaluate it: the law is a pure
    # function of the supplied history with no mutable state, and it forbids
    # no token at any length (it only lowers logits).  Every batched-law
    # consumer reads ``presence_window`` before ``forbidden_token_ids_at_length``
    # (``_batched_law_contract``, ``_launch_tree_laws``), so the empty forbidden
    # set never stands in for the penalty itself.  Qwen instruct sampling
    # profiles default presence_penalty to 1.5, which kept every Qwen AST round
    # on its primary chain until this was declared (AST smokes 2026-10-18).
    presence_penalty_processor.progressive_branch_safe = True
    presence_penalty_processor.forbidden_token_ids_at_length = lambda length: ()
    return _probe_safe(presence_penalty_processor)


def make_frequency_penalty(
    penalty: float, context_size: int = 20, *, generation_start: Optional[int] = None
):
    """
    Make a frequency penalty processor.

    Corresponds to the OpenAI option with the same name. Namely, subtracts
    ``penalty`` from a logit for every time that the token has occured in the
    ``context_size`` previous tokens.

    The difference with the presence penalty is that the more often a token
    occurs the more it will be penalized.

    Args:
        penalty (float): The frequency penalty to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]
    """

    def frequency_penalty_processor(tokens, logits):
        tokens = _generated_window(tokens, context_size, generation_start)
        if len(tokens) > 0:
            logits = logits.at[:, tokens].subtract(penalty)
        return logits

    return _probe_safe(frequency_penalty_processor)
