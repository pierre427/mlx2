"""Backend-neutral registry of deferred native retirement callbacks.

Only an imported backend registers its reaper. The serving lifecycle therefore
does not import model factories merely to poll for unfinished retirement.
"""

from threading import RLock

_LOCK = RLock()
_REAPERS = {}


def register_reaper(name, reaper):
    if not isinstance(name, str) or not name or not callable(reaper):
        raise ValueError("retirement registration requires a name and callback")
    with _LOCK:
        existing = _REAPERS.get(name)
        if existing is not None and existing is not reaper:
            raise ValueError(f"native retirement owner already registered: {name}")
        _REAPERS[name] = reaper


def reap_registered():
    with _LOCK:
        reapers = tuple(_REAPERS.values())
    for reaper in reapers:
        # A backend retains ambiguous owners itself, including on failed polls.
        try:
            reaper()
        except Exception:  # noqa: BLE001, S112 - retain owner; poll other backends
            continue
