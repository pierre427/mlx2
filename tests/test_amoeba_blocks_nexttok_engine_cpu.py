"""aMoEba next-token gating through a real ServingEngine (tiny CPU model): fail closed and route receipts."""

import pytest

from mlx2.runtime.amoeba_blocks import nexttok as NT
from test_suspend_replay import POLICY, A_PROMPT, collect, host, make_adapter, tiny_model  # noqa: F401
from mlx2 import serving


def test_engine_fails_closed_without_amoeba_blocks_and_receipts_the_mode(host):  # noqa: F811
    """A real ServingEngine (tiny CPU model): a gate mode on a route without amoeba_blocks is refused; a route
    that declares amoeba_blocks records the gate mode in its route receipt."""
    model = tiny_model()
    for blocks in (None, type("Blocks", (), {"gate_modes": ("same_step", "next_token")})()):
        cls = make_adapter(model)
        if blocks is not None:
            cls.amoeba_blocks = blocks
        engine = serving.ServingEngine("tiny", adapter_factory=cls, qualification_mode=True, mtp=False,
                                       max_lanes=1, prefill_step=8, execution_policy=POLICY)
        try:
            assert engine.ready.wait(60), engine.error
            req = {"tokens": A_PROMPT, "max_tokens": 2, "temperature": 0, "mlx2_amoeba_gate_mode": "next_token"}
            if blocks is None:
                with pytest.raises((AssertionError, NT.GateModeUnavailable)) as e:
                    collect(engine.submit(req))
                err = e.value
                if isinstance(err, NT.GateModeUnavailable):
                    assert err.http_status == 409 and err.code == "amoeba_gate_mode_unavailable"
                else:                                              # the error event carries the exception
                    text = repr(err.args[0])
                    assert "amoeba_blocks" in text and ("409" in text or "amoeba_gate_mode_unavailable" in text
                                                        or "GateModeUnavailable" in text), text
            else:
                _, receipt = collect(engine.submit(req))
                text = repr(receipt)
                # requested, never echoed as executed: this stub installation reports no executed mode
                assert "'amoeba_gate_mode_requested': 'next_token'" in text and "'amoeba_gate_mode': None" in text
                assert "'amoeba_gate_observed': False" in text and "amoeba_gate_qualified" in text
                _, receipt = collect(engine.submit({**req, "mlx2_amoeba_gate_mode": None}))
                assert "'amoeba_gate_mode_requested': None" in repr(receipt)
        finally:
            engine.close()


def test_apc_scope_server_field_and_scorer_refusal():
    from pathlib import Path
    import mlx2.prompt_logprobs as PL
    import mlx2.server as server
    from mlx2.serving import request_apc_scope
    base = request_apc_scope({})
    scoped = request_apc_scope({"_mlx2_amoeba_gate_mode_scope": "next_token"})
    assert scoped != base and scoped == request_apc_scope({"_mlx2_amoeba_gate_mode_scope": "next_token"})
    assert NT.APPROXIMATE_MODES == ("next_token",)          # verify is exact against same-step: no own scope
    assert '"mlx2_amoeba_gate_mode",' in Path(server.__file__).read_text()
    assert "mlx2_amoeba_gate_mode" in PL._REQUEST_WEIGHT_PROGRAMS
