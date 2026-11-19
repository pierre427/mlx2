"""Hy4-flash (``hy_v4_flash``) sweep family: the mlx2 Hy4-flash decoder layers, one whole layer at a time.

Target: ``~/Models/Hy4-flash`` (tencent/Hy4-flash, bf16 safetensors, ~440 GB with its four 31.7 GB
Engram tables; MLX affine artifacts from ``scripts/convert_hy_v4_flash_mlx.py`` load the same way). Numerics are the
in-tree ordinary forward, ``mlx2.runtime.models.hy_v4_flash`` (Model.hidden / DecoderLayer), built against
main after the Hy4-flash integration merged (adapter commit 20b8c9e7c). Nothing numerical is reimplemented: each
layer is the model's own ``DecoderLayer`` with its own ``HY4FlashLayerCache`` (``Model.make_cache``), loaded through
the model's own ``sanitize`` (gate_up split, kv_b fold, fp32 modules) and, for an MLX artifact, ``nn.quantize`` with
the adapter's predicate. This module only reads the bytes and schedules the layers (``README.md``, interface v1 plus
the child_rolling extensions ``begin_layer``, the ``neurons`` payload and ``prefetch_ranges``).

Trunk (``config.json`` text_config): 42 layers on 4 iHC residual streams; gated absorbed MLA with learned sinks,
``indexed_attention`` layers (compressed indexer; dense causal below 2048 tokens) and ``sliding_attention`` layers
(window 512); layer 0 a dense MLP (13824), layers 1..41 latent MoE (512 routed experts top-16, sigmoid router with
correction bias, routed on a 1536-wide latent, one shared expert, clamped SwiGLU); Engram hashed n-gram memories
added to the streams before layers 1, 11, 21, 31.

* carry ``{"ids": [B, T] int32, "h": [B, T, 4, D]}``: the four HC streams in the activation dtype (bf16), exactly
  what ``Model.hidden`` passes from layer to layer. Attention is recomputed per layer from a fresh layer cache over
  the whole text in one call, as the ordinary forward does for a one-chunk prompt (no KV cache kept, no APCv2).
* per-layer call arguments from the token ids: the layer's own fresh cache (``Model.make_cache()[l]``) and, on an
  Engram layer, its ``(rows, mask)`` built with the model's own n-gram/hash helpers (``Model._ngram_tokens``,
  ``EngramHasher.table_rows``) for that layer only (``Model.engram_inputs`` would read all four tables); the rows
  come from the adapter's SSD reader (``open_engram_tables``, bf16 tables, PLE-offload pool), never resident. In
  the map pass ``prepare_layer`` first reads every row the chunk's texts will look up once, in file order (spans of
  the table, not one random read per row per batch), decoded by the reader's own ``_decode``; a row outside that
  set raises.
* weights: each layer's tensors (routed experts included; the Engram table excluded) read by byte range from their
  shards on ``begin_layer``/first use, held for every batch of the layer, dropped on ``release_layer``.
  Embeddings, final norm, ``hc_head`` and the fp32 LM head stay resident (~2.3 GB bf16 source). Vision tower and
  DFly2 layers are never read. ``prefetch_ranges`` lists each layer's own byte ranges for the driver's read-ahead.
* units: layer 0 ``dense`` (``act`` = down_proj input, 13824); MoE layers ``neurons`` by default (``e * F + f``,
  F = 1536: ``{"inds", "w", "h", "F"}``, h the clamped-SwiGLU activation per routed slot, w the router weight the
  model combines with) or ``experts`` with ``moe_units="expert"`` (``{"inds", "w", "rows"}``, rows the unweighted
  per-slot expert outputs on the latent).
* ``reference_logits(ids, layers=K)``: an ordinary ``Model`` of the first K layers (``ModelArgs`` cut to K layers,
  Engram layers < K), loaded resident from the same bytes and run text-major with ``Model.__call__``. That is the
  parity reference on real weights (K layers resident; the whole model does not fit); the CPU tests also check it,
  at full depth, against the adapter's own ``load_model``.

Refused (SystemExit or ValueError, never zero-filled): an artifact the adapter's ``inspect_artifact`` refuses (any
indexed shard missing, e.g. a download in progress; unimplemented topology; FP8 checkpoints); a tensor the index
names that its shard does not hold; image / video tokens; a sequence over ``max_position_embeddings``; the grouped-
expert research kernels (``MLX2_MOE_GROUPED_EXPERTS``, they bypass the activation the neuron units read); an Engram
table format other than bf16; a payload the layer did not produce.
"""
from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

DEFAULT_MODEL = Path.home() / "Models/Hy4-flash"
TABLE = "engram_layer.multi_head_embedding.embedding.weight"
_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I32": np.int32, "I64": np.int64,
       "U32": np.uint32, "U8": np.uint8, "I8": np.int8, "U16": np.uint16}
_SKIP = ("vit.", "model.dfly_layers.", "model.vision_tower", "mtp.")
READ_AHEAD = True              # the plan gives this family a read-ahead window (it exposes prefetch_ranges)


def checkpoint_fingerprint(path: Path) -> str:
    """The adapter's artifact fingerprint (the identity every mlx2 receipt names)."""
    from mlx2.adapters.hy_v4_flash import inspect_artifact
    return inspect_artifact(path)["identity"]["fingerprint"]


class _Tok:
    """The adapter's tokenizer (``AutoTokenizer`` + ``repair_loaded_tokenizer``), loaded on first use."""

    def __init__(self, path: Path):
        self.path = path
        self._t = None

    def _get(self):
        if self._t is None:
            try:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
            except (OSError, ValueError) as e:
                raise SystemExit(f"hy_v4_flash: no tokenizer in {self.path} ({e})") from None
            from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer
            repair_loaded_tokenizer(t, self.path)
            self._t = t
        return self._t

    def encode(self, text):
        return list(self._get().encode(text, add_special_tokens=False))

    def decode(self, ids):
        return self._get().decode(list(ids))


@contextmanager
def _taps(module_calls):
    """Swap each module's class for one whose __call__ is the given wrapper (the host family's pattern)."""
    saved = []
    try:
        for mod, wrap in module_calls:
            base = type(mod)
            mod.__class__ = type(f"Tap{base.__name__}", (base,), {"__call__": wrap(base.__call__)})
            saved.append((mod, base))
        yield
    finally:
        for mod, base in reversed(saved):
            mod.__class__ = base


class _PreRead:
    """The rows of one Engram table a chunk will look up, read once in file order (``prepare_layer``).

    Rows are grouped into spans (a new span past a ``gap``-byte hole or a ``window``-byte boundary) read by ``pread``
    on a few threads, then kept as raw row bytes: in memory up to ``budget`` bytes, else in a scratch file. ``read``
    decodes them with the adapter reader's own ``_decode``; a row outside the set raises (never a zero row)."""

    def __init__(self, reader, rows, scratch: Path, layer: int, *, budget: int, gap=1 << 18, window=64 << 20,
                 workers=8):
        import os
        import time
        from concurrent.futures import ThreadPoolExecutor
        self.reader, self.rows, self.n = reader, np.asarray(rows, np.int64), len(rows)
        rb = int(reader.row_bytes)
        if self.n and (self.rows[0] < 0 or self.rows[-1] >= reader.rows):
            raise IndexError("Engram row outside its table")
        self.file = None
        if self.n * rb > budget:
            scratch.mkdir(parents=True, exist_ok=True)
            self.file = scratch / f"engram_L{layer:02d}.npy"
            self.vals = np.lib.format.open_memmap(self.file, mode="w+", dtype=np.uint8, shape=(self.n, rb))
        else:
            self.vals = np.empty((self.n, rb), np.uint8)
        offs = int(reader.start) + self.rows * rb
        cut = np.flatnonzero((np.diff(offs) > gap) | (np.diff(offs // window) != 0)) + 1 if self.n > 1 else []
        groups = [g for g in np.split(np.arange(self.n), cut) if g.size]
        self.spans, self.bytes_read = len(groups), 0
        t0 = time.time()
        fd = os.open(reader.path, os.O_RDONLY)

        def read(idx):
            b, e = int(offs[idx[0]]), int(offs[idx[-1]]) + rb
            buf = bytearray(e - b)
            mv, pos = memoryview(buf), 0
            while pos < len(buf):
                got = os.preadv(fd, [mv[pos:]], b + pos)
                if got <= 0:
                    raise OSError(f"short Engram table read in {reader.path} at {b + pos}")
                pos += got
            raw = np.frombuffer(buf, np.uint8)
            rel = (offs[idx] - b)[:, None] + np.arange(rb)
            self.vals[idx] = raw[rel]
            return e - b
        try:
            with ThreadPoolExecutor(workers) as ex:
                self.bytes_read = int(sum(ex.map(read, groups)))
        except BaseException:
            self.close()                    # a failed read leaves no scratch file behind
            raise
        finally:
            os.close(fd)
        if self.file is not None:
            self.vals.flush()
        self.seconds = round(time.time() - t0, 2)

    def read(self, rows):
        from mlx2.runtime.models.hy_v4_flash_engram import _decode
        rows = np.asarray(rows, np.int64).reshape(-1)
        pos = np.searchsorted(self.rows, rows)
        if rows.size and (pos.max() >= self.n or not np.array_equal(self.rows[pos], rows)):
            raise RuntimeError("hy_v4_flash: an Engram row was not pre-read for this chunk (texts changed?)")
        raw = np.ascontiguousarray(self.vals[pos])
        return mx.array(_decode(raw.tobytes(), self.reader.dtype, int(rows.size), int(self.reader.dim)))

    def close(self):
        self.vals = None
        if self.file is not None:
            self.file.unlink(missing_ok=True)


class HyV4FlashFamily:
    batch_rows = 4096

    def __init__(self, path=DEFAULT_MODEL, *, moe_units="neuron", engram_cache_rows=1 << 16, engram_preread_gb=8.0,
                 reference_gb=48.0,
                 configure_env=True):
        from mlx2.adapters import hy_v4_flash as A
        from mlx2.runtime.models import hy_v4_flash as H
        from mlx2.runtime.models import switch_grouped_experts as G
        from mlx2.runtime.weight_stream import SafetensorsIndex, read_safetensors_header
        if moe_units not in ("neuron", "expert"):
            raise ValueError("hy_v4_flash: moe_units must be neuron or expert")
        if G.MODE != "off":
            raise SystemExit("hy_v4_flash sweep: the grouped-expert kernels (MLX2_MOE_GROUPED_EXPERTS) bypass the "
                             "expert activation the units read; unset it")
        self.path = Path(path).expanduser()
        try:
            artifact = A.inspect_artifact(self.path)        # topology + every indexed shard present
        except (ValueError, OSError) as e:
            raise SystemExit(f"hy_v4_flash: {self.path}: {e}") from None
        if configure_env:
            A.configure_environment()                       # the adapter's process numerics profile
        self.H, self.moe_units = H, moe_units
        self.config = artifact["config"]
        self.quant = self.config.get("quantization")
        self.fingerprint = artifact["identity"]["fingerprint"]
        self.checkpoint = f"{self.path.name}@{self.fingerprint[:8]}"
        wm = artifact["weight_map"]
        shards = sorted(set(wm.values()))
        self.tensors = SafetensorsIndex.from_headers(
            [(self.path / s, read_safetensors_header(self.path / s), (self.path / s).stat().st_size) for s in shards])
        missing = [n for n in wm if n not in self.tensors]
        if missing:
            raise SystemExit(f"hy_v4_flash: the index names {missing[0]} but its shard does not hold it")
        self.args = H.ModelArgs.from_hf_config(self.config)     # engram precision transformers, latent KV native
        a = self.args
        self.L = a.num_hidden_layers
        self.F = a.moe_intermediate_size
        self.kinds = [("dense" if a.mlp_layer_types[l] == "dense" else
                       ("neurons" if moe_units == "neuron" else "experts")) for l in range(self.L)]
        self.widths = [a.intermediate_size if k == "dense" else
                       (a.n_routed_experts * self.F if k == "neurons" else a.n_routed_experts) for k in self.kinds]
        self._names = {l: [] for l in range(self.L)}
        self._top = []
        for n in wm:
            if n.startswith(_SKIP):
                continue
            m = n.replace("model.language_model.", "model.", 1)
            if m.startswith("model.layers."):
                l = int(m.split(".")[2])
                if not m.endswith(TABLE):
                    self._names[l].append(n)
            else:
                self._top.append(n)
        self.tok = _Tok(self.path)
        self.counters = {"layer_loads": 0, "layer_bytes": 0, "reference_layers_loaded": 0,
                         "engram_preread_rows": 0, "engram_preread_bytes": 0}
        self._layers = {}
        self._preread = {}
        self.preread_gb = float(engram_preread_gb)
        self.reference_gb = float(reference_gb)       # cap on the resident parity reference (layer bytes)
        self._prefetch = None
        self.model = H.Model(a)                              # shell: top modules, hashers, make_cache, sanitize
        top = self.model.sanitize({n: self._read(n) for n in self._top})
        self._quantize(self.model, "", top, only=lambda name: not name.startswith("model.layers."))
        self.model.load_weights(list(top.items()), strict=False)
        from mlx.utils import tree_flatten
        want = {k for k, _ in tree_flatten(self.model.parameters()) if not k.startswith("model.layers.")}
        if want != set(top):
            raise SystemExit(f"hy_v4_flash: resident weights mismatch: missing {sorted(want - set(top))[:3]}, "
                             f"unexpected {sorted(set(top) - want)[:3]}")
        self._top_items = top
        mx.eval(list(top.values()))
        for i in a.engram_layers:                           # the hashers read each Engram layer's lookup table
            name = f"model.layers.{i}.engram_layer.lookup_table"
            self.model.model.layers[i].engram_layer.lookup_table = self._read(self._ckpt_name(name))
        from mlx2.runtime.models.hy_v4_flash_engram import open_engram_tables
        self.readers = open_engram_tables(self.path, wm, a.engram_layers, cache_rows=int(engram_cache_rows),
                                            table_format="bf16", formats=self.config.get("mlx2_engram_formats"))
        try:
            self.model.bind_engram(self.readers)
        except BaseException:
            self.close()
            raise
        self._round_trip = {i: self.model.model.layers[i].engram_layer.table_round_trip_dtype for i in a.engram_layers}
        self.dtype = self.model.model.embed_tokens(mx.array([[0]])).dtype

    # ------------------------------------------------------------------ bytes
    def _ckpt_name(self, name):
        if name in self.tensors:
            return name
        alt = name.replace("model.", "model.language_model.", 1)
        if alt in self.tensors:
            return alt
        raise SystemExit(f"hy_v4_flash: tensor {name} is not in the checkpoint")

    def _read(self, name) -> mx.array:
        loc = self.tensors[name]
        if loc.dtype not in _NP:
            raise SystemExit(f"hy_v4_flash: {name} has unsupported dtype {loc.dtype}")
        data = np.fromfile(loc.path, dtype=np.uint8, count=loc.end - loc.begin, offset=loc.begin)
        if data.size != loc.end - loc.begin:
            raise SystemExit(f"hy_v4_flash: short read of {name} from {loc.path}")
        arr = data.view(_NP[loc.dtype]).reshape(tuple(loc.shape))
        out = mx.array(arr)
        return out.view(mx.bfloat16) if loc.dtype == "BF16" else out

    def _quantize(self, module, prefix, weights, only=None):
        """The adapter's load_model predicate, on ``module`` (paths relative to it, ``prefix`` for overrides)."""
        q = self.quant
        if q is None:
            return

        def pred(name, m):
            if only is not None and not only(name):
                return False
            override = q.get(prefix + name)
            if isinstance(override, dict):
                return override
            return hasattr(m, "to_quantized") and f"{name}.scales" in weights
        nn.quantize(module, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"),
                    class_predicate=pred)

    def _layer_weights(self, l) -> dict:
        raw = {n: self._read(n) for n in self._names[l]}
        out = self.model.sanitize(raw)
        raw.clear()
        self.counters["layer_bytes"] += sum(self.tensors[n].end - self.tensors[n].begin for n in self._names[l])
        return out

    @property
    def prefetch_ranges(self):
        """{layer: [(file, begin, end)]}: layer l's own tensors (adjacent ranges merged), no Engram table."""
        if self._prefetch is None:
            out = {}
            for l in range(self.L):
                spans = []
                for loc in sorted((self.tensors[n] for n in self._names[l]), key=lambda t: (str(t.path), t.begin)):
                    if spans and spans[-1][0] == str(loc.path) and loc.begin <= spans[-1][2]:
                        spans[-1][2] = max(spans[-1][2], loc.end)
                    else:
                        spans.append([str(loc.path), int(loc.begin), int(loc.end)])
                out[l] = [tuple(s) for s in spans]
            self._prefetch = out
        return self._prefetch

    # ------------------------------------------------------------------ layers
    def _load_layer(self, l):
        if l in self._layers:
            return self._layers[l]
        H, p = self.H, f"model.layers.{l}."
        w = {k[len(p):]: v for k, v in self._layer_weights(l).items()}
        layer = H.DecoderLayer(self.args, l)
        self._quantize(layer, p, w)
        layer.load_weights(list(w.items()), strict=True)
        if l in self._round_trip:
            layer.engram_layer.table_round_trip_dtype = self._round_trip[l]
        layer.eval()
        mx.eval(layer.parameters())
        self._layers[l] = layer
        self.counters["layer_loads"] += 1
        return layer

    def begin_layer(self, l):
        self._load_layer(l)

    def release_layer(self, l, exc=None):
        self._layers.pop(l, None)
        pre = self._preread.pop(l, None)
        if pre is not None:
            pre.close()
        mx.clear_cache()

    def _engram_rows(self, l, tokens):
        ngrams = self.model._ngram_tokens(tokens, 0, np.zeros((tokens.shape[0], 0), np.int64), None)
        return self.model.engram_hashers[l].table_rows(ngrams)

    def prepare_layer(self, l, seqs, scratch):
        """Engram layers: read every table row the chunk's texts look up, once, in file order (sequential spans of the
        31.7 GB table instead of one random read per row per batch; USB random reads ran at 65 IOPS on 2026-11-18).
        The rows are decoded exactly as the adapter's reader decodes them (``hy_v4_flash_engram._decode``)."""
        if l not in self.args.engram_layers:
            return None
        reader = self.model.engram_rows[l]
        if getattr(reader, "format", None) != "bf16" or not hasattr(reader, "row_bytes"):
            return {"engram_preread": "skipped (not an SSD bf16 table reader)"}
        old = self._preread.pop(l, None)
        if old is not None:
            old.close()
        by_len = {}
        for s in seqs:
            by_len.setdefault(len(s), []).append(s)
        need = np.unique(np.concatenate(
            [self._engram_rows(l, np.array(group, np.int64)).reshape(-1) for group in by_len.values()]))
        pre = _PreRead(reader, need, Path(scratch), l, budget=int(self.preread_gb * 2 ** 30))
        self._preread[l] = pre
        self.counters["engram_preread_rows"] += pre.n
        self.counters["engram_preread_bytes"] += pre.bytes_read
        return {"engram_preread_rows": pre.n, "engram_preread_bytes": pre.bytes_read,
                "engram_preread_spans": pre.spans, "engram_preread_s": pre.seconds, "on_disk": pre.file is not None}

    def _check_ids(self, ids):
        a = self.args
        tokens = np.array(ids).astype(np.int64)
        if tokens.ndim != 2:
            raise ValueError("hy_v4_flash sweep: ids must be [B, T]")
        if tokens.shape[1] > a.max_position_embeddings:
            raise ValueError("hy_v4_flash sweep: sequence exceeds max_position_embeddings")
        for special, what in ((a.image_token_id, "image"), (a.video_token_id, "video")):
            if special is not None and np.any(tokens == special):
                raise ValueError(f"hy_v4_flash sweep: {what} tokens need a media prefill (text only)")
        return tokens

    def _engram(self, l, tokens):
        """``Model.engram_inputs`` for layer l alone: one lane set from position 0, empty token tail."""
        m, a = self.model, self.args
        B, T = tokens.shape
        mask = tokens != a.engram_pad_id
        for special in (a.image_token_id, a.video_token_id):
            if special is not None:
                mask &= tokens != special
        rows = self._engram_rows(l, tokens)
        reader = self._preread.get(l) or m.engram_rows[l]
        data = reader.read(rows.reshape(-1)).reshape(B, T, rows.shape[-1], -1)
        return data.astype(mx.float32), mx.array(mask)

    def embed(self, ids):
        ids = mx.array(ids).astype(mx.int32) if not isinstance(ids, mx.array) else ids.astype(mx.int32)
        self._check_ids(ids)
        h = self.model.model.embed_tokens(ids)
        streams = mx.contiguous(mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.args.hc_mult, h.shape[-1])))
        mx.eval(streams)
        return {"ids": ids, "h": streams}

    def run_layer(self, l, carry):
        layer = self._load_layer(l)
        ids, streams = carry["ids"], carry["h"]
        tokens = self._check_ids(ids)
        cache = self.model.make_cache()[l]
        engram = self._engram(l, tokens) if l in self.args.engram_layers else None
        pay, calls = {}, []
        if self.kinds[l] == "dense":
            def tap_down(call):
                def f(self_, x):
                    pay["act"] = x
                    return call(self_, x)
                return f
            calls.append((layer.mlp.down_proj, tap_down))
        elif self.kinds[l] is not None:
            def tap_gate(call):
                def f(self_, x):
                    pay["inds"], pay["w"] = out = call(self_, x)
                    return out
                return f

            def tap_experts(call):
                def f(self_, x, indices):
                    y = call(self_, x, indices)
                    pay["rows"] = y
                    return y
                return f

            def tap_act(call):
                def f(self_, up, gate):
                    h = call(self_, up, gate)
                    pay["h_raw"] = h
                    return h
                return f
            calls += [(layer.mlp.gate, tap_gate), (layer.mlp.experts, tap_experts)]
            if self.kinds[l] == "neurons":
                calls.append((layer.mlp.experts.activation, tap_act))
        with _taps(calls):
            out = layer(streams, cache, engram)
        payload = None
        if self.kinds[l] == "dense":
            if "act" not in pay:
                raise RuntimeError(f"hy_v4_flash L{l}: the dense MLP gave no down_proj input")
            payload = {"act": pay["act"]}
        elif self.kinds[l] is not None:
            if not {"inds", "w", "rows"} <= set(pay):
                raise RuntimeError(f"hy_v4_flash L{l}: the MoE block gave no routing / expert rows")
            inds = pay["inds"].astype(mx.int32)
            if self.kinds[l] == "experts":
                payload = {"inds": inds, "w": pay["w"].astype(mx.float32), "rows": pay["rows"]}
            else:
                payload = {"inds": inds, "w": pay["w"].astype(mx.float32),
                           "h": self._slot_order(l, pay.get("h_raw"), inds), "F": self.F}
        mx.eval(out, *(v for v in (payload or {}).values() if isinstance(v, mx.array)))
        return {"ids": ids, "h": out}, payload

    def _slot_order(self, l, h, inds):
        """The activation as [B, T, k, F] in routed-slot order. SwitchGLU computes it unsorted ([B, T, k, 1, F]
        squeezed later) or, past its sort threshold, in expert-sorted order ([n, 1, F], optionally tail-padded): the
        sort is ``argsort(indices.flatten())`` (switch_layers._gather_sort), undone here as _scatter_unsort does."""
        if h is None:
            raise RuntimeError(f"hy_v4_flash L{l}: the expert activation was not observed (neuron units)")
        B, T, k = inds.shape
        if h.ndim == 5:
            return h.squeeze(-2)
        if h.ndim == 3 and h.shape[0] >= B * T * k:
            inv = mx.argsort(mx.argsort(inds.flatten()))
            return h[inv].reshape(B, T, k, h.shape[-1])
        raise RuntimeError(f"hy_v4_flash L{l}: unexpected expert activation shape {h.shape}")

    def final_logits(self, carry):
        m = self.model
        return m.logits(m.model.norm(m.model.hc_head(carry["h"])))

    def reference_logits(self, ids, layers=None):
        """The ordinary ``Model.__call__`` over the first ``layers`` blocks (all when None), loaded resident from the
        same bytes, one text-major forward over the batch."""
        H, a = self.H, self.args
        ids = mx.array(ids).astype(mx.int32) if not isinstance(ids, mx.array) else ids.astype(mx.int32)
        self._check_ids(ids)
        full = a.num_hidden_layers                  # not self.L: the driver's parity lowers self.L to ``layers``
        K = full if not layers or layers >= full else int(layers)
        args = a if K == full else dataclasses.replace(
            a, num_hidden_layers=K, layer_types=list(a.layer_types[:K]), mlp_layer_types=list(a.mlp_layer_types[:K]),
            engram_layer_ids=[lid for lid in (a.engram_layer_ids or []) if lid - 1 < K])
        need = sum(self.tensors[n].end - self.tensors[n].begin for l in range(K) for n in self._names[l])
        if need > self.reference_gb * 2 ** 30:
            raise SystemExit(f"hy_v4_flash: a resident reference of {K} layers needs {need / 2 ** 30:.0f} GiB, over "
                             f"reference_gb {self.reference_gb:g} (use --parity-layers 6 or fewer on real weights)")
        ref = H.Model(args)
        weights = dict(self._top_items)
        for l in range(K):
            weights.update(self._layer_weights(l))
            self.counters["reference_layers_loaded"] += 1
        self._quantize(ref, "", weights)
        ref.load_weights(list(weights.items()), strict=True)
        weights.clear()
        ref.bind_engram({i: self.readers[i] for i in args.engram_layers})
        ref.eval()
        try:
            out = ref(ids)
            mx.eval(out)
        finally:
            del ref
            mx.clear_cache()
        return out

    # ------------------------------------------------------------------ receipts
    @property
    def _blocks(self):              # child_rolling's weight_reads_observed counts streamed layers by this
        return {l: None for l in range(self.L)}

    def stream_counters(self):
        return {"page_ins": self.counters["layer_loads"], "page_in_bytes": self.counters["layer_bytes"],
                "evictions": None, "bypasses": None, "hits": None}

    def stream_receipt(self):
        return {"mode": "hy_v4_flash sweep family: whole decoder layers, routed experts included, by byte range",
                "engram": "bf16 tables on SSD (adapter EngramTableRows); in the map pass each chunk's rows are pre-read "
                          "once in file order (prepare_layer), in memory up to engram_preread_gb, else on scratch",
                "engram_preread_gb": self.preread_gb,
                "resident": "embeddings, final norm, hc_head, LM head",
                "quantization": self.quant, "moe_units": self.moe_units,
                "trunk": "mlx2.runtime.models.hy_v4_flash (Model.sanitize, DecoderLayer, make_cache)"}

    def close(self):
        for p in getattr(self, "_preread", {}).values():
            p.close()
        self._preread.clear()
        for r in getattr(self, "readers", {}).values():
            r.close()
        self._layers.clear()
        mx.clear_cache()


def load(model_dir=DEFAULT_MODEL, *, moe_units="neuron", **options) -> HyV4FlashFamily:
    return HyV4FlashFamily(model_dir, moe_units=moe_units, **options)


def describe(model_dir=DEFAULT_MODEL) -> dict:
    """Header-only summary (no tensor bytes read)."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
    return {"layers": cfg["num_hidden_layers"], "n_routed_experts": cfg["n_routed_experts"],
            "moe_intermediate_size": cfg["moe_intermediate_size"], "engram_layer_ids": cfg.get("engram_layer_ids")}
