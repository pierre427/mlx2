"""Diagnostic: run mlx2.server with every mx.array truth test counted by call site.

Each ``bool(array)`` forces a GPU sync. Counts (keyed by the innermost three
Python frames) are written to $BOOL_TRACE_OUT at exit or SIGTERM.
"""
import atexit
import collections
import json
import os
import runpy
import signal
import sys
import traceback

import mlx.core as mx

COUNTS = collections.Counter()
_original = mx.array.__bool__


def _traced(self):
    frames = traceback.extract_stack(limit=5)[:-1]
    COUNTS[" <- ".join(f"{f.filename.rsplit('/src/', 1)[-1]}:{f.lineno}:{f.name}" for f in reversed(frames[-3:]))] += 1
    return _original(self)


mx.array.__bool__ = _traced


def _dump(*_):
    path = os.environ.get("BOOL_TRACE_OUT")
    if path:
        with open(path, "w") as handle:
            json.dump(COUNTS.most_common(60), handle, indent=1)


atexit.register(_dump)
_previous = signal.getsignal(signal.SIGTERM)


def _term(signum, frame):
    _dump()
    if callable(_previous):
        _previous(signum, frame)
    else:
        sys.exit(0)


signal.signal(signal.SIGTERM, _term)
sys.argv = ["mlx2.server", *sys.argv[1:]]
runpy.run_module("mlx2.server", run_name="__main__")
