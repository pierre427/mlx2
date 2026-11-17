# aMoEba sweep capture (layer streaming)

Forward-only topic capture for models too large to hold in memory. The driver runs the model
**layer-major**: for each layer it loads that layer's weights once, pushes every probe text's
carried state through it in batches, writes the layer's `resp_L<l>.npy` rows, stores the next
carried state in a disk-backed array, frees the layer and moves on. Output is the same capture
format `scripts/amoeba_topics_capture.py --backend carve` writes, so the existing CPU cluster and
label scripts read it unchanged.

```
python -m research.amoeba.sweep.capture --family deepseek_v41 --model DIR \
    --corpus .../topic-probes-16e8232ad/corpus.jsonl --out OUT --state-dir SCRATCH [--resume]
```

## Family interface (stable contract, v1)

A family adapts one model family to the driver. Register it in `research/amoeba/sweep/family.py`
`FAMILIES` as `name -> "module:loader"`; the loader is `load(model_dir, **options) -> SweepFamily`.

```python
class SweepFamily:                     # one per family; nothing big loaded in __init__
    checkpoint: str                    # "<artifact>@<fingerprint8>"
    fingerprint: str                   # full fingerprint
    L: int                             # layers the sweep walks, 0..L-1
    widths: list[int]                  # units per layer (0 where kinds[l] is None)
    kinds: list[str | None]            # "dense" | "experts" | None, per layer
    tok                                # .encode(text) -> list[int]; .decode(ids) -> str
    batch_rows: int = 1024             # optional: max B*T per run_layer call (driver honours it)

    def embed(self, ids) -> Carry                         # ids: mx.array int32 [B, T]
    def run_layer(self, l, carry) -> tuple[Carry, dict | None]
    def release_layer(self, l) -> None
    def prepare_layer(self, l, seqs, scratch) -> dict | None  # optional: called once before layer l
                                                          # with every token sequence (e.g. to read
                                                          # side tables in file order); the dict
                                                          # goes into the receipt
    def final_logits(self, carry) -> mx.array             # optional: [B, T, V] after layer L-1
    def reference_logits(self, ids) -> mx.array           # optional: the family's ordinary
                                                          # whole-model forward, for parity
    def close(self) -> None                               # optional
```

**Carry.** Either an `mx.array` `[B, T, d]` (the hidden state) or a `dict[str, mx.array]` of named
arrays that all have the batch as their leading axis. A dict lets a family carry everything a
later layer reads besides the residual stream (hyper-connection streams, shared KV pools, token
ids). The driver stores each key in its own disk array and hands the same keys back for the next
layer. Rules:
- Every array has leading axis `B`. Its trailing shape may depend only on `T` (the batch's
  sequence length) and must be non-decreasing in `T`; the driver allocates each key from the
  longest batch and slices shorter texts back to their own shape.
- dtypes: any numeric mlx dtype, including bfloat16 (stored losslessly as its bit pattern).
- The set of keys may change from layer to layer (a key can first appear at layer `l`), but within
  one layer every batch returns the same keys.
- The driver batches texts **of one length**: within a call all texts have the same `T`, so a
  family never sees padding.

**run_layer(l, carry).** Runs layer `l` on the batch, loading the layer's weights on first use
(the driver calls it many times per layer and calls `release_layer(l)` once after the last batch).
Returns the next carry and the layer's unit payload, carve-style (`research/amoeba/carve/units.py`):
- `kinds[l] == "dense"`: `{"act": [B, T, widths[l]]}`, the down_proj input. The driver computes
  `mu` on the fit split in a first pass of that layer, then the unit signal `|act - mu|`.
- `kinds[l] == "experts"`: `{"inds": [B, T, k], "w": [B, T, k], "rows": [B, T, k, d]}` (signal
  `w * ||rows||`), or the cheaper equivalent `{"inds": [B, T, k], "signal": [B, T, k]}` with the
  per-route signal already reduced. Signals accumulate per expert id.
- `kinds[l] is None`: payload `None`.

The driver writes `resp_L<l>` row `s` = mean over positions `>= 1` of the unit signal, fp16,
zero-padded to the widest layer `W = max(widths)`.

**Laws.** `run_layer` must reuse the family's ordinary layer code; the sweep's final hidden state
(or `final_logits`) must equal `reference_logits` within that route's own numerical tolerance.
`capture --parity N [--parity-layers K]` checks it on N texts (against the ordinary forward of the
first K blocks when given) and writes `parity.json`.

## Output (OUT/)

`resp_L<l>.npy` fp16 `[S, W]`, `items.json` (id, topics, split, n_tokens), `mu.npy` `[L, W]` f32
(zero rows for non-dense layers), `meta.json` (`widths`, `layers`, `neurons` = W, `sequences`,
`checkpoint`, `fingerprint`, `corpus_sha256`, `mlx2_source_revision`, `unit`, `unit_kinds`),
`nll_full.npy` when the family has `final_logits`, `progress.json` and `receipt.json` (per-layer
seconds, peak memory, swap at start and end). Tokenisation is
`amoeba_topics_lib.tokenize_items(family.tok.encode, items, seq_len=64, min_len=48)`.

Carried state lives in `--state-dir` (`L<l>/<key>.npy`), at most two layers at a time (parity keeps all), stamped with a run id so a stale state is never reused. `--resume`
continues after the last layer marked complete in `progress.json`; without it a non-empty OUT is
refused.

## Families

| name | module | units | notes |
|---|---|---|---|
| `deepseek_v41` | `deepseek_v41.py` | 384 routed experts x 40 layers | Source checkpoint; experts stream from SSD through the Source LFU store |
| `glm5_next` | `glm5_next.py` | 3 dense layers x 12288 neurons, then 288 routed experts x 42 layers | Sage-S2.1-Flash (bf16 safetensors, text only; MTP + vision never read; registered under the checkpoint's `model_type`, mlx2 has no glm5_next adapter). The GLM-5.3 ordinary forward (`glm53_gguf.Glm53GGUFModel`) over an HF-named weights provider. Units are the v1 contract (mean signal, whole experts), not comparable with the CLI's peak/neuron maps. Carry `{ids, h [B,T,4,d] f32}` (f32 is the reference's own precision): ~71 GB a layer boundary for 17.7K 64-token texts, two live, so `--state-dir` on internal NVMe (refused under 40 GB free). Parity: `--parity-layers 3` (dense) or `4` with `--expert-budget-gb 40`; the reference reads experts one at a time, so it also checks the stacked GPU path. Fails closed above 2048 tokens. **Implemented, CPU-tested on a tiny checkpoint against transformers `Glm5NextTextModel`; unverified on real weights.** |
