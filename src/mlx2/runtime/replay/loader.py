"""Discovery and admission of the optional native step-replay module.

The module (``native/step_replay``) is built separately from the package by
``scripts/build_step_replay.py`` into ``build/step_replay/stage/`` and is
installed only after its GPU self-test passes (``--install``), as
``<site-packages>/mlx2_step_replay/_step_replay<ext>`` beside the ``mlx``
package, so its ``@loader_path/../mlx/lib`` rpath resolves the wheel's own
``libmlx.dylib``.

``load()`` never raises: it returns ``(module, None)`` or ``(None, reason)``
with one of ``REASONS``. A missing module declines as ``module_missing``
before anything else is inspected; a refused module leaves ``import mlx2``
working with the engine reported unavailable.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import sysconfig
import threading
from pathlib import Path

API_VERSION = 1
PACKAGE = "mlx2_step_replay"
MODULE = "_step_replay"
DISABLE_ENV = "MLX2_STEP_REPLAY_DISABLE"
DIR_ENV = "MLX2_STEP_REPLAY_DIR"  # test/override: directory holding the module

REASONS = (
    "disabled", "module_missing", "not_macos", "wrong_python", "import_failed",
    "mlx_version_mismatch", "api_mismatch", "abi_probe_failed", "libmlx_images",
    "no_gpu",
)

_lock = threading.Lock()
_result = None


def _suffix():
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def install_dir():
    """Where an installed module lives (beside the ``mlx`` package)."""
    override = os.environ.get(DIR_ENV)
    if override:
        return Path(override)
    import mlx

    roots = list(getattr(mlx, "__path__", []))
    base = Path(roots[0]).parent if roots else Path(sysconfig.get_paths()["platlib"])
    return base / PACKAGE


def module_path():
    return install_dir() / (MODULE + _suffix())


def libmlx_image_count():
    """Number of ``libmlx.dylib`` images loaded in this process (dyld list)."""
    if sys.platform != "darwin":
        return 0
    import ctypes

    libc = ctypes.CDLL(None)
    libc._dyld_image_count.restype = ctypes.c_uint32
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    count = 0
    for i in range(libc._dyld_image_count()):
        name = libc._dyld_get_image_name(i) or b""
        if os.path.basename(name.decode(errors="replace")) == "libmlx.dylib":
            count += 1
    return count


def _admit(module):
    """Version gate and ABI probe of an imported module; None = admitted."""
    import mlx.core as mx

    if getattr(module, "MLX_VERSION", None) != mx.__version__:
        return "mlx_version_mismatch"
    if getattr(module, "API_VERSION", None) != API_VERSION:
        return "api_mismatch"
    try:
        probe = module.abi_probe(mx.zeros((1,), mx.float32))
    except Exception:  # noqa: BLE001 - admission never raises
        return "abi_probe_failed"
    if probe is not True:
        return "abi_probe_failed"
    if libmlx_image_count() != 1:
        return "libmlx_images"
    if not mx.metal.is_available():
        return "no_gpu"
    return None


def load(*, refresh=False):
    """``(module, None)`` or ``(None, reason)``; cached per process."""
    global _result
    with _lock:
        if _result is not None and not refresh:
            return _result
        _result = _load()
        return _result


def _load():
    if os.environ.get(DISABLE_ENV):
        return None, "disabled"
    path = module_path()
    if not path.is_file():
        return None, "module_missing"
    if sys.platform != "darwin":
        return None, "not_macos"
    if sys.implementation.cache_tag != "cpython-312":
        return None, "wrong_python"
    try:
        loader = importlib.machinery.ExtensionFileLoader(f"{PACKAGE}.{MODULE}", str(path))
        spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{MODULE}", str(path),
                                                      loader=loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    except Exception:  # noqa: BLE001 - admission never raises
        return None, "import_failed"
    reason = _admit(module)
    if reason is not None:
        return None, reason
    return module, None


def status():
    module, reason = load()
    return {"available": module is not None, "reason": reason, "path": str(module_path()),
            "api": API_VERSION}
