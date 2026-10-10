"""Collecting the packed-prefill suite must not leave its runtime doubles behind.

tests/test_packed_prefill_serving_cpu.py installs module doubles and import
blockers for its own tests.  Ordinary pytest runs it in an isolated process,
which would hide a leak, so check its import and module fixtures directly.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = r"""
import importlib.util, sys
DOUBLED = "mlx2.runtime.paged_native_atomic_owner"

def leaked():
    found = [name for name, module in sys.modules.items()
             if name.startswith("mlx2.") and getattr(module, "__file__", None) is None
             and getattr(module, "__path__", None) is None]
    found += [name for name in ("hybrid_factory", "native_contract_source") if name in sys.modules]
    return found

finders = list(sys.meta_path)
spec = importlib.util.spec_from_file_location("packed_prefill_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not leaked(), ("after import", leaked())
assert sys.meta_path == finders, "import left meta_path finders behind"

module.setUpModule()
assert DOUBLED in leaked(), "doubles not installed"
module.tearDownModule()
assert not leaked(), ("after tearDownModule", leaked())
assert sys.meta_path == finders, "tearDownModule left meta_path finders behind"
"""


def test_packed_prefill_doubles_are_scoped_to_its_own_tests():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT / "tests/test_packed_prefill_serving_cpu.py")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
