"""The grammar processor converts only the generated tail, not the whole context.

Converting prompt + history to Python ints cost 3.4 ms per generated token at
128K context on every structured request; the 2026-09-23 audit fixed the same
pattern in generate.py and this copy survived.
"""

import mlx.core as mx

from mlx2.structured_output import StructuredOutputProcessor


class _Context:
    """Stands in for the token array: slicing yields the tail, converting the
    whole thing is the defect."""

    def __init__(self, prompt_length, generated):
        self.prompt_length = prompt_length
        self.generated = generated

    def __getitem__(self, item):
        assert isinstance(item, slice) and item.start == self.prompt_length
        return mx.array(self.generated, dtype=mx.uint32)

    def tolist(self):
        raise AssertionError("the whole context was converted to Python ints")


def test_call_converts_only_the_generated_tail():
    processor = StructuredOutputProcessor.__new__(StructuredOutputProcessor)
    processor.failure = None
    processor.prompt_length = 131072
    processor.block_eos_while_deferred = False
    processor._defer_until = (2**31,)  # never generated: stays deferred
    logits = mx.zeros((8,))
    out = processor(_Context(131072, [5, 6, 7]), logits)
    assert out is logits
    assert processor._generated_token_count == 3
    assert processor._recent_generated_ids == (5, 6, 7)
