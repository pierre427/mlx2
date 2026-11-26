"""Load a compiled diagnostic extension from a file path (reviewed dispatcher).

The only path-based loader on the serving import closure. It loads a native
``.so`` from outside the source tree (``build/``), so it adds no local Python
source to a route's identity; ``route_dispatch.json`` pins this file's bytes
with no targets. Keep it this small: callers (``round_probe``'s opt-in Metal
replay hooks) change freely without renewing the pin.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util


def load_extension(name: str, path: str):
    """Import the extension module ``name`` from the file ``path``."""
    loader = importlib.machinery.ExtensionFileLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, str(path), loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


__all__ = ["load_extension"]
