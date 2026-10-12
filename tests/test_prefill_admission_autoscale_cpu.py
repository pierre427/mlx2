"""Ingress must price the same prompt-length chunk as the executor."""

import pytest

from mlx2.serving import prefill_transient_gib


class Budget:
    def __init__(self):
        self.calls = []

    def prefill_transient_bytes(self, context, rows):
        self.calls.append((context, rows))
        return rows * (1 << 20)


@pytest.mark.parametrize(
    "context,tail,maximum,expected",
    [(1100, 1100, 8192, 512), (32768, 32768, 8192, 512),
     (32769, 32769, 8192, 2048), (65537, 65537, 8192, 8192),
     (65537, 3000, 8192, 3000), (65537, 3000, 256, 256)],
)
def test_admission_uses_complete_prompt_schedule(context, tail, maximum, expected):
    budget = Budget()
    value = prefill_transient_gib(
        budget, context_tokens=context, uncached_tokens=tail,
        prefill_step=maximum, prefill_step_autoscale=True,
    )
    assert budget.calls == [(context, expected)]
    assert value == expected / 1024


def test_explicit_chunk_is_not_autoscaled():
    budget = Budget()
    prefill_transient_gib(
        budget, context_tokens=1100, uncached_tokens=1100, prefill_step=8192,
    )
    assert budget.calls == [(1100, 1100)]


def test_cold_media_preserves_whole_prompt_reservation():
    budget = Budget()
    prefill_transient_gib(
        budget, context_tokens=1100, uncached_tokens=1100, prefill_step=8192,
        prefill_step_autoscale=True, single_chunk=True,
    )
    assert budget.calls == [(1100, 1100)]


def test_warm_anchor_needs_no_prefill_workspace():
    budget = Budget()
    assert prefill_transient_gib(
        budget, context_tokens=65537, uncached_tokens=1, prefill_step=8192,
        prefill_step_autoscale=True,
    ) == 0
    assert budget.calls == []
