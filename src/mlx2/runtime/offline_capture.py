"""Bounded independent-window teacher capture; no serving/APCv2 mutation.

Layer-major scheduling is inspired by Sushi ad562c4 (MIT); tensor and cache
math belongs to the adapter driver. See provenance/sushi-integration-20261012.json.
"""

from __future__ import annotations


def capture_windows(
    driver,
    windows,
    *,
    revision,
    sink,
    order="layer_major",
    activation_budget_bytes=512 << 20,
):
    """Run windows at their original geometry and deliver one result at a time.

    ``driver`` owns layer calls, masks, independent caches and weight scopes.
    A sink must consume results synchronously. No cache/state is published.
    The activation budget covers live hidden arrays and driver cache arrays,
    with the old hidden/layer cache while an update is evaluated; weights and the
    sink's output storage are separate. This is a dedicated offline worker API.
    """
    if order not in ("layer_major", "window_major"):
        raise ValueError("capture order must be layer_major or window_major")
    if not isinstance(revision, str) or not revision:
        raise ValueError("capture requires a bound model revision")
    if type(activation_budget_bytes) is not int or activation_budget_bytes <= 0:
        raise ValueError("capture activation budget must be a positive integer")
    if not callable(sink):
        raise ValueError("capture requires a synchronous output sink")
    windows = tuple(tuple(row) for row in windows)
    if not 1 <= len(windows) <= 32 or any(not row for row in windows):
        raise ValueError("capture needs 1..32 nonempty independent windows")
    if any(type(t) is not int or not 0 <= t < 2**32 for row in windows for t in row):
        raise ValueError("capture token IDs must be uint32 integers")
    # Embeddings/caches are lazy: account them before the first evaluation.
    states = [driver.start(row) for row in windows]
    peak = 0

    def check(extra=0):
        nonlocal peak
        used = (
            sum(driver.state_bytes(state) for state in states if state is not None)
            + extra
        )
        peak = max(peak, used)
        if used > activation_budget_bytes:
            raise MemoryError(
                f"offline capture needs {used} activation/cache bytes; "
                f"budget is {activation_budget_bytes}"
            )

    check()
    if order == "layer_major":
        for layer in range(driver.layer_count):
            with driver.weight_scope(layer):
                for index, state in enumerate(states):
                    old_bytes = driver.transition_bytes(state, layer)
                    states[index] = driver.step(state, layer)
                    check(old_bytes)
                    driver.evaluate(states[index])
        for index, state in enumerate(states):
            check(driver.hidden_bytes(state))  # final norm output coexists
            driver.finish(index, windows[index], state, sink)
            states[index] = None
    else:
        for index, state in enumerate(states):
            for layer in range(driver.layer_count):
                old_bytes = driver.transition_bytes(state, layer)
                state = states[index] = driver.step(state, layer)
                check(old_bytes)
                driver.evaluate(state)
            check(driver.hidden_bytes(state))
            driver.finish(index, windows[index], state, sink)
            states[index] = None
    return {
        "schema": "mlx2.offline-capture.v1",
        "model_revision": revision,
        "order": order,
        "windows": len(windows),
        "layers": driver.layer_count,
        "tokens_per_window": [len(row) for row in windows],
        "activation_budget_bytes": activation_budget_bytes,
        "peak_tracked_activation_cache_bytes": peak,
        "bound_kind": "array-byte-accounting",
        "state_published": False,
    }
