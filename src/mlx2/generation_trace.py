"""Bounded host-only evidence of sampled tokens and their producing forwards.

Rendered text is not an invertible token trace. Record the response's already
materialized token and width before detokenization, without reading logits or
synchronizing the device. Only an explicit request retains per-token evidence.
"""

from dataclasses import dataclass, field


TOKEN_TRACE_SCHEMA = "mlx2.generated-token-trace.v1"
MAX_TOKEN_TRACE_TOKENS = 4096


@dataclass
class GeneratedTokenTrace:
    token_ids: list[int] = field(default_factory=list)
    execution_widths: list[int] = field(default_factory=list)
    ordinary_execution_widths: list[int | None] = field(default_factory=list)
    completion_tokens: int = 0

    def append(self, token: int, width: int, ordinary_width: int | None) -> None:
        self.completion_tokens += 1
        if len(self.token_ids) < MAX_TOKEN_TRACE_TOKENS:
            self.token_ids.append(int(token))
            self.execution_widths.append(int(width))
            self.ordinary_execution_widths.append(ordinary_width)

    def receipt(self) -> dict:
        return {
            "schema": TOKEN_TRACE_SCHEMA,
            "source": "generation_response",
            "start_completion_token": 1,
            "token_ids": list(self.token_ids),
            "execution_widths": list(self.execution_widths),
            "ordinary_execution_widths": list(self.ordinary_execution_widths),
            "completion_tokens": self.completion_tokens,
            "limit": MAX_TOKEN_TRACE_TOKENS,
            "truncated": self.completion_tokens > len(self.token_ids),
        }


def read_token_trace(receipt: dict, *, expected_tokens: int | None = None) -> dict:
    """Require complete authoritative evidence; never fall back to tokenization."""
    trace = receipt.get("token_trace")
    if not isinstance(trace, dict) or (
        trace.get("schema") != TOKEN_TRACE_SCHEMA
        or trace.get("source") != "generation_response"
        or trace.get("start_completion_token") != 1
        or trace.get("truncated") is not False
    ):
        raise ValueError("complete authoritative generated-token trace required")
    count = trace.get("completion_tokens")
    tokens = trace.get("token_ids")
    widths = trace.get("execution_widths")
    ordinary = trace.get("ordinary_execution_widths")
    if (
        type(count) is not int
        or not 0 <= count <= MAX_TOKEN_TRACE_TOKENS
        or (expected_tokens is not None and count != expected_tokens)
        or not all(isinstance(values, list) and len(values) == count
                   for values in (tokens, widths, ordinary))
        or not all(type(token) is int and token >= 0 for token in tokens)
        or not all(type(width) is int and width >= 1 for width in widths)
        or not all(width is None or (type(width) is int and width == producing)
                   for width, producing in zip(ordinary, widths))
    ):
        raise ValueError("malformed or incomplete generated-token trace")
    return trace
