from types import SimpleNamespace

import pytest

from mlx2 import serving


class _UnsupportedInteriorAdapter:
    max_context = 1024
    identity = {"fingerprint": "fixture"}
    environment = {}
    layout = "fixture"
    model = None
    tokenizer = SimpleNamespace(vocab_size=16, eos_token_ids=[0])
    backend = None

    def __init__(self, _path, execution_policy=None):
        del execution_policy

    def profile_name(self, _mtp):
        return "fixture"

    def execution_config(self, **_kwargs):
        result = {"num_draft": 0}
        if self.backend is not None:
            result["backend"] = self.backend
        return result

    def diagnostics(self):
        return {}

    def close(self):
        pass


@pytest.mark.parametrize(
    ("prompt_lookup", "backend", "route"),
    [
        (True, None, "prompt lookup"),
        (False, "external_draft", "external draft"),
    ],
)
def test_interior_checkpoint_policy_fails_closed_on_routes_that_cannot_capture(
    monkeypatch, prompt_lookup, backend, route
):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})

    class Adapter(_UnsupportedInteriorAdapter):
        pass

    Adapter.backend = backend
    engine = serving.ServingEngine(
        "fixture",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        prompt_lookup=prompt_lookup,
        execution_policy={
            "apc_interior_checkpoints": {"count": 2, "min_stride": 8}
        },
    )
    try:
        engine.thread.join(5)
        assert not engine.ready.is_set()
        assert engine.error is not None
        assert "APCv2 interior checkpoints" in engine.error
        assert route in engine.error
    finally:
        engine.close()
