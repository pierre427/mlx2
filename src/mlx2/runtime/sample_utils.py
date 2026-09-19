# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import math
from functools import lru_cache
from typing import Callable, Dict, List, Optional
import mlx.core as mx


def _probe_safe(function):
    """Declare a stateless transform safe to reuse for provisional tokens."""
    function.probe = function
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
    if temp == 0:
        argmax_sampler = lambda x: mx.argmax(x, axis=-1)
        argmax_sampler.batch_groupable = True
        return argmax_sampler
    sampling_methods = []
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
    probs = mx.exp(logprobs.astype(mx.float32))
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)
    return mx.where(cumulative_probs > 1 - top_p, logprobs, -float("inf"))


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


def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))


@lru_cache(maxsize=32)
def make_transformed_logprobs(
    temp: float,
    *,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
) -> Callable[[mx.array], mx.array]:
    """Make a map from raw logits to the log-probabilities of the
    distribution ``make_sampler(temp, top_p, min_p, top_k)`` samples from.

    Memoized at module level per parameter tuple: repeat requests (the few
    serving profiles dominate) reuse one compiled chain, and MLX's own trace
    cache then covers the per-shape traces inside it. Safe to share — the
    transform is deterministic (no random state is captured, unlike
    ``make_sampler``, which must compile per call to bind the calling
    thread's RNG state).

    Mirrors the sampler exactly: normalization runs eagerly in the logits'
    native dtype (as ``generate_step`` does before calling the sampler), and
    the filter chain — top-p, then min-p, then top-k, in ``make_sampler``
    order; XTC is not supported — plus the temperature scale runs inside
    ``mx.compile``, so knife-edge filter ties resolve with the same fused
    rounding as the compiled sampler. Filtered tokens are exactly ``-inf``.
    Only the final renormalization is float32: it does not change the
    represented distribution, it only makes the returned values accurate.
    Requires ``temp > 0``; batched over leading axes.
    """
    if not temp or temp <= 0:
        raise ValueError(f"make_transformed_logprobs requires temp > 0, got {temp}")
    sampling_methods = []
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
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        scaled = compiled(logprobs).astype(mx.float32)
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
