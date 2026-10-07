"""Engine startup must not import runtime model modules before the adapter.

Adapters pin import-time environment flags and refuse a model module that
latched other values first (``assert_profile_applied``).  A policy parser in
``ServingEngine.__init__`` that imported ``runtime.apc_v2`` loaded
``runtime.models.base`` before every adapter and broke all guarded models at
startup; CPU tests with stub adapters never noticed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_engine_constructs_adapter_before_any_model_module():
    script = textwrap.dedent(
        """
        import sys
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        import mlx2.server  # the CLI's import surface
        from mlx2 import serving

        seen = {}

        class Probe:
            def __init__(self, *args, **kwargs):
                seen["loaded"] = sorted(
                    m for m in sys.modules if m.startswith("mlx2.runtime.models.")
                    and m != "mlx2.runtime.models.import_env"
                )
                raise RuntimeError("probe stop")

        engine = serving.ServingEngine(
            "probe",
            adapter_factory=Probe,
            max_lanes=1,
            execution_policy={"apc_retention_policy": "value"},
        )
        engine.ready.wait(30)
        print("LOADED", seen.get("loaded"))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    line = [x for x in out.stdout.splitlines() if x.startswith("LOADED")]
    assert line, out.stdout + out.stderr
    assert line[0] == "LOADED []", line[0]
