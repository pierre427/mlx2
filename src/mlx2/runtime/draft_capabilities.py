"""CPU-safe semantic bounds supplied by a loaded proposal implementation."""

from __future__ import annotations


def maximum_proposal_length(draft) -> int:
    """Return the implemented depth, preserving trained-block bounds by default.

    A model may explicitly support a different physical proposal length from
    the checkpoint's training block.  This is a capacity, not an acceptance,
    quality, or performance claim.  Configuration inspection can pass the
    config directly; loaded wrappers may delegate the same capability.
    """
    config = getattr(draft, "config", draft)
    declared = getattr(draft, "maximum_proposal_length", None)
    if declared is None:
        declared = getattr(config, "maximum_proposal_length", None)
    if declared is not None:
        if type(declared) is not int or declared < 1:
            raise ValueError("maximum_proposal_length must be a positive integer")
        return declared
    block = getattr(config, "block_size", None)
    if type(block) is not int or block < 2:
        raise ValueError("draft block_size must be an integer of at least two")
    return block - 1
