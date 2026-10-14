"""L7-1 (sweep 2026-10-12): offline teacher capture must run each layer in
the trunk forward's own scope.

``Qwen36TextModel.__call__`` opens ``moe_nax_gather.forward_scope`` so the
NAX MoE gather admits prefill forwards.  ``Qwen35CaptureDriver`` runs the
trunk layer by layer and never entered that scope, so with the gather
selected every captured window declined it as "not prefill" while the
capture's execution binding named the selected mode (introduced 51f88bf78).
"""

import mlx.core as mx
import test_disk_weight_streaming as T

from mlx2.adapters.qwen35_capture import Qwen35CaptureDriver
from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.offline_capture import capture_windows


def _phases(monkeypatch, run):
    seen = []
    real_admit = nax.admit_phase
    monkeypatch.setattr(
        nax, "admit_phase", lambda: (seen.append(nax._PREFILL.get()), real_admit())[1]
    )
    old = nax.set_mode("fused")
    try:
        run()
    finally:
        nax.set_mode(old)
        monkeypatch.setattr(nax, "admit_phase", real_admit)
    return seen


def test_capture_reproduces_the_ordinary_prefill_nax_scope(monkeypatch):
    mx.random.seed(4)
    model = T._moe_model(monkeypatch, fused=False)
    T._quantize_all(model)
    window = [int(t) % 60 for t in range(3, 3 + 7 * 48, 7)]  # sorted gather rows

    ordinary = _phases(
        monkeypatch,
        lambda: mx.eval(model(mx.array([window], mx.uint32), cache=model.make_cache())),
    )
    captured = []
    capture = _phases(
        monkeypatch,
        lambda: capture_windows(
            Qwen35CaptureDriver(model), [window], revision="r",
            sink=lambda i, t, h, c: captured.append(h),
            activation_budget_bytes=1 << 30,
        ),
    )
    assert ordinary and set(ordinary) == {True}
    assert len(capture) == len(ordinary)
    assert set(capture) == {True}, set(capture)
    assert nax._PREFILL.get() is None  # the scope closed after every step
    assert captured
