"""Host-only interpretation of APCv2 lookup evidence in serving receipts."""


def prompt_boundary_replay_evidence(
    cold, warm, *, expected_prompt_tokens, cold_prompt_sha256, warm_prompt_sha256
):
    """Check reuse of a cold request's complete committed prompt boundary.

    ``cache_checkpoint_role`` is the role read by that request's APCv2 lookup.
    A cold miss legitimately has no role. A same-prompt warm lookup covering
    all but the final prompt token with the committed-boundary role witnesses
    earlier publication; it is not an output-publication receipt for the warm
    request. Attribution to the cold arm additionally requires the caller to
    establish an initially empty cache and exclude other producers.

    This checks cache-boundary evidence only, not generated-token identity,
    numerical equivalence, runtime identity, or route qualification.
    """
    errors = []
    valid_length = type(expected_prompt_tokens) is int and expected_prompt_tokens >= 2
    expected_prefix = expected_prompt_tokens - 1 if valid_length else None
    if not valid_length:
        errors.append("expected prompt must contain a reusable prefix")
    if (
        not isinstance(cold_prompt_sha256, str)
        or len(cold_prompt_sha256) != 64
        or any(c not in "0123456789abcdef" for c in cold_prompt_sha256)
        or cold_prompt_sha256 != warm_prompt_sha256
    ):
        errors.append("cold/warm prompt hashes are absent or differ")
    for arm, receipt in (("cold", cold), ("warm", warm)):
        if type(receipt.get("prompt_tokens")) is not int or receipt.get("prompt_tokens") != expected_prompt_tokens:
            errors.append(f"{arm}: prompt token count differs")
        if "cache_checkpoint_role" not in receipt:
            errors.append(f"{arm}: cache lookup role field absent")
    if type(cold.get("cached_tokens")) is not int or cold.get("cached_tokens") != 0:
        errors.append("cold: expected an APCv2 miss")
    if cold.get("cache_checkpoint_role") is not None:
        errors.append("cold: a cache miss cannot name a reused checkpoint")
    if type(warm.get("cached_tokens")) is not int or warm.get("cached_tokens") != expected_prefix:
        errors.append("warm: complete prompt boundary was not reused")
    if warm.get("cache_checkpoint_role") != "committed_prompt_boundary":
        errors.append("warm: reused checkpoint is not a committed prompt boundary")
    return {
        "schema": "mlx2.prompt-boundary-replay.v1",
        "passed": not errors,
        "cold_lookup_role": cold.get("cache_checkpoint_role"),
        "warm_lookup_role": warm.get("cache_checkpoint_role"),
        "reused_boundary_tokens": warm.get("cached_tokens"),
        "publication_evidence": "same_prompt_warm_lookup" if not errors else None,
        "errors": errors,
    }
