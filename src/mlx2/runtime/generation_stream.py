"""The generation thread's MLX stream, importable without the model modules.

``serving.SequenceBatchServer._run`` binds this stream as the generation
thread's default before it constructs the adapter.  Importing it through
``runtime.generate`` pulled ``runtime.models.base`` in first, so model
modules read their import-time GDN/QSDPA flags before an adapter's
environment profile was applied, and every adapter that checks the import
order (Qwen3.8-27B among them) refused to load.  This module imports only
MLX.
"""

from __future__ import annotations

import mlx.core as mx

generation_stream = mx.new_thread_local_stream(mx.default_device())


def bind_generation_thread_stream():
    """Make the calling thread's default stream its generation stream.

    MLX gives every thread its own default stream on first use, and the
    generators run their forwards under ``with mx.stream(generation_stream)``
    (another per-thread stream).  Any lazy array the serving loop builds on
    the generation thread outside that context (an APCv2 restore alias, a
    merged or trimmed lane cache, a quantized KV plane) therefore sat on a
    different GPU stream from the prefill that consumed it, and MLX bridged
    the two with a Fence inside one evaluation.  Under
    ``MLX_METAL_FAST_SYNCH=1`` that fence is a GPU spin kernel whose producer
    is not guaranteed to run, and the GPU wedged on Laguna-S-2.1 and on
    Qwen3.8-27B at B4 (apc-warm-hit-hang-20261021).

    Binding the thread's default stream to the generation stream makes the
    generation thread single-stream: nothing it builds can need a
    cross-stream fence.  It changes which command queue carries the work,
    never the kernels, so no computed bit changes.  Returns the bound stream.
    """
    with mx.stream(generation_stream):
        stream = mx.default_stream(generation_stream.device)
    mx.set_default_stream(stream)
    return stream
