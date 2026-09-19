"""Bounded, opt-in serialization of the execution engine's target probabilities.

No tensor backend is imported here. The worker supplies its existing backend;
CPU tests use NumPy. Returned values own no device arrays or cache references.
"""

import math


MAX_TOP_LOGPROBS = 11


def wants_logprobs(request):
    return bool(request.get("logprobs", False) or request.get("top_logprobs", 0))


def token_logprob(logprobs, token, tokenizer, *, top_n=0, array_module):
    """Serialize the emitted token and its top alternatives from the same row.

Accepted MTP drafts and replacement/bonus tokens already carry their respective
target verifier rows. Never substitute draft or residual-sampling probabilities.
MTP target rows include its sampling transform; ordinary rows precede sampling,
matching the unified runtime's existing response contract.
"""
    if len(logprobs.shape) != 1:
        raise ValueError("token logprobs require one vocabulary row")
    vocab = int(logprobs.shape[0])
    token = int(token)
    if not 0 <= token < vocab:
        raise ValueError("emitted token is outside the probability vocabulary")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 0 <= top_n <= MAX_TOP_LOGPROBS:
        raise ValueError("top_logprobs must be an integer from 0 to 11")
    count = min(top_n, vocab)
    top_ids = (
        array_module.argpartition(-logprobs, kth=count - 1)[:count].tolist()
        if count else []
    )
    ids = [token, *map(int, top_ids)]
    values = logprobs[ids].tolist()
    labels = tokenizer.convert_ids_to_tokens(ids)

    def entry(index):
        value = float(values[index])
        if math.isnan(value) or value == math.inf:
            raise ValueError("execution returned an invalid token log probability")
        return {"id": ids[index], "token": labels[index], "logprob": max(value, -9999.0)}

    result = entry(0)
    if top_n:
        result["top_logprobs"] = sorted(
            (entry(i) for i in range(1, len(ids))),
            key=lambda item: (-item["logprob"], item["id"]),
        )
    return result
