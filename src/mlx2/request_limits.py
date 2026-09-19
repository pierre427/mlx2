"""Shared output-token limits and prompt-aware default resolution."""

from __future__ import annotations

from collections.abc import Mapping


MAX_OUTPUT_TOKENS = 2_097_152
DEFAULT_OUTPUT_TOKENS = 65_536


def validate_default_max_tokens(value: object) -> int:
    """Validate the operator-configured omitted-request output default."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_OUTPUT_TOKENS
    ):
        raise ValueError(
            f"default_max_tokens must be an integer between 1 and {MAX_OUTPUT_TOKENS}"
        )
    return value


def resolve_output_limit(
    request: Mapping,
    *,
    prompt_tokens: int,
    effective_context: int,
    default_max_tokens: int = DEFAULT_OUTPUT_TOKENS,
) -> tuple[int, bool]:
    """Return ``(effective max_tokens, was_defaulted)`` for admission.

    Explicit limits remain fail-closed against the effective context.  An
    omitted limit is the smaller of the configured default and the context
    remaining after the rendered prompt.  With the server's 65,536-token
    default, any request that fit main's former 512-token default still fits;
    operators may deliberately choose a smaller default.
    """
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("prompt_tokens must be a nonnegative integer")
    if type(effective_context) is not int or effective_context < 1:
        raise ValueError("effective_context must be a positive integer")
    default_max_tokens = validate_default_max_tokens(default_max_tokens)

    defaulted = "max_tokens" not in request
    remaining = effective_context - prompt_tokens
    if defaulted:
        if remaining < 1:
            raise ValueError(
                f"prompt plus output must fit {effective_context} tokens"
            )
        maximum = min(default_max_tokens, remaining)
    else:
        maximum = request["max_tokens"]
        if prompt_tokens + maximum > effective_context:
            raise ValueError(
                f"prompt plus output must fit {effective_context} tokens"
            )

    minimum = request.get("min_tokens", 0)
    if minimum > maximum:
        raise ValueError("min_tokens must not exceed effective max_tokens")
    return maximum, defaulted
