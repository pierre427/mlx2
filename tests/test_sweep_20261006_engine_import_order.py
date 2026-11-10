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


def test_status_polled_before_the_adapter_imports_no_model_module():
    """Launchers poll ``/v1/status`` until healthy, so ``status()`` runs before
    the adapter constructs.  When it imported ``runtime.models.cache`` and
    ``standard_decoder`` (2026-11-06), ``models.base`` latched the unpinned
    environment and the Qwen3.8-27B adapter refused every model load."""
    script = textwrap.dedent(
        """
        import sys, threading
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        import mlx2.server  # the CLI's import surface
        from mlx2 import serving

        gate = threading.Event()
        seen = {}

        class Probe:
            def __init__(self, *args, **kwargs):
                gate.wait(60)
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
        status = engine.status()  # what every readiness poll calls
        print("STATUS_KEYS", "qsdpa_verify" in status and "prefill_tail_row" in status)
        gate.set()
        engine.ready.wait(60)
        print("LOADED", seen.get("loaded"))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    line = [x for x in out.stdout.splitlines() if x.startswith("LOADED")]
    assert line, out.stdout + out.stderr
    assert line[0] == "LOADED []", line[0]
    assert "STATUS_KEYS True" in out.stdout, out.stdout


def test_half_imported_model_module_reads_as_not_loaded(monkeypatch):
    import types

    from mlx2 import serving

    name = "mlx2.runtime.models._half_imported_probe"
    module = types.ModuleType(name)
    module.__spec__ = types.SimpleNamespace(_initializing=True)
    monkeypatch.setitem(sys.modules, name, module)
    assert serving._loaded_runtime_model("_half_imported_probe") is None
    module.__spec__ = types.SimpleNamespace(_initializing=False)
    assert serving._loaded_runtime_model("_half_imported_probe") is module
    first, second = serving._not_loaded_section(), serving._not_loaded_section()
    first["counts"]["x"] = 1
    assert second["counts"] == {}
