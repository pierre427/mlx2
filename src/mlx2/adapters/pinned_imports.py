"""Fail before executing modules which shadow byte-verified Python sources."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path


class PinnedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, expected_hash):
        super().__init__(name, path)
        self.expected_hash = expected_hash

    def get_code(self, fullname):
        # Compile exactly the verified buffer; never consume cached bytecode.
        data = self.get_data(self.path)
        if hashlib.sha256(data).hexdigest() != self.expected_hash:
            raise ImportError(f"pinned source bytes changed: {fullname}")
        return self.source_to_code(data, self.path)


class PinnedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefixes, sources):
        self.prefixes = tuple(prefixes)
        self.sources = {Path(p).resolve(): expected for p, expected in sources.items()}

    def covers(self, name):
        return any(
            name == prefix or name.startswith(prefix + ".") for prefix in self.prefixes
        )

    def validate(self, name, spec):
        if (
            spec is None
            or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)
            or spec.origin is None
            or Path(spec.origin).resolve() not in self.sources
        ):
            raise ImportError(f"pinned import origin differs: {name}")

    def find_spec(self, fullname, path=None, target=None):
        if not self.covers(fullname):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        self.validate(fullname, spec)
        spec.loader = PinnedSourceLoader(
            fullname, spec.origin, self.sources[Path(spec.origin).resolve()]
        )
        return spec

    def validate_loaded(self):
        for name, module in list(sys.modules.items()):
            if self.covers(name):
                self.validate(name, getattr(module, "__spec__", None))

    def __enter__(self):
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *args):
        sys.meta_path.remove(self)
