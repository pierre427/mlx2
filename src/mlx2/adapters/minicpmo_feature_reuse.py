"""Exact-input per-item reuse for MiniCPM-o's sequential vision contract.

Only the pinned width-one source encoder is eligible. The adapter opts in
explicitly and supplies tenant/LoRA-scoped keys at the serving boundary.
No generic batch-axis slicing or regrouping is inferred for other families.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def item_keys(pixel_values, tgt_sizes, *, namespace, max_bytes=512 << 20):
    """Hash the processor's exact NumPy inputs without a device-to-host copy.

    Refuse reuse for an unsupported representation or a request exceeding the
    hashing budget. The existing whole-request/source path remains available.
    Shape, dtype, target geometry and policy/source namespace are all bound.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("vision item namespace must be nonempty text")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("vision item hashing budget must be nonnegative")
    if not isinstance(pixel_values, (list, tuple)) or not isinstance(
        tgt_sizes, (list, tuple)
    ):
        return None
    if len(pixel_values) != len(tgt_sizes):
        return None
    total, result = 0, []
    for pixels, targets in zip(pixel_values, tgt_sizes):
        if (
            not isinstance(pixels, (list, tuple))
            or not isinstance(targets, np.ndarray)
            or targets.dtype.hasobject
            or targets.shape != (len(pixels), 2)
        ):
            return None
        keys = []
        for index, value in enumerate(pixels):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype.hasobject
                or value.ndim != 3
            ):
                return None
            target = targets[index]
            total += int(value.nbytes) + int(target.nbytes)
            if total > max_bytes:
                return None
            digest = hashlib.sha256(
                b"mlx2-minicpmo-vision-item-v1\0" + namespace.encode()
            )
            for name, array in (("pixels", value), ("target", target)):
                digest.update(json.dumps((name, array.dtype.str, array.shape)).encode())
                array = np.ascontiguousarray(array)
                digest.update(memoryview(array).cast("B"))
            keys.append(digest.hexdigest())
        result.append(tuple(keys))
    return tuple(result)


def cached_sequential_features(
    model, cache, pixel_values, tgt_sizes, keys, *, array_module
):
    """Reuse only complete independently encoded items, in original order."""
    if (
        not isinstance(pixel_values, (list, tuple))
        or not isinstance(tgt_sizes, (list, tuple))
        or not isinstance(keys, (list, tuple))
        or len(pixel_values) != len(keys)
        or len(tgt_sizes) != len(keys)
        or any(len(sample) != len(names) for sample, names in zip(pixel_values, keys))
    ):
        raise ValueError(
            "MiniCPM-o per-item cache geometry does not match processor inputs"
        )
    outputs = []
    for sample, targets, names in zip(pixel_values, tgt_sizes, keys):
        if not isinstance(targets, np.ndarray) or targets.shape != (len(sample), 2):
            raise ValueError(
                "MiniCPM-o per-item target geometry does not match processor inputs"
            )
        features = []
        for index, (pixels, key) in enumerate(zip(sample, names)):
            if not isinstance(key, str) or not key:
                raise ValueError("MiniCPM-o per-item cache key must be nonempty text")
            key = ("minicpmo-independent-vision-v1", key)
            feature = cache.get(key)
            if feature is None:
                encoded = model.get_vision_embedding(
                    [[pixels]], [targets[index : index + 1]]
                )
                if (
                    not isinstance(encoded, (list, tuple))
                    or len(encoded) != 1
                    or len(getattr(encoded[0], "shape", ())) != 3
                    or encoded[0].shape[0] != 1
                ):
                    raise ValueError(
                        "MiniCPM-o source encoder violated its one-item feature contract"
                    )
                feature = encoded[0][0]
                array_module.eval(feature)
                cache.put(key, feature)
            features.append(feature)
        outputs.append(array_module.stack(features, axis=0) if features else [])
    return outputs
