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
`nll_full.npy` when the family has `final_logits` (per-text mean NLL of targets 2..T-1, the resident
capture's convention; `meta.json` then has `nll_full.convention`, and a capture without that key scored
targets 1..T-1), `progress.json` and `receipt.json` (per-layer
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
| `hy_v4_flash` | `hy_v4_flash.py` | layer 0: 13824 dense neurons; layers 1..41: 512 routed experts x 1536 neurons (`e*F+f`, default) or 512 whole experts (`--moe-units expert`) | Hy4-flash (bf16 HF or the MLX affine conversion). The in-tree ordinary forward (`runtime/models/hy_v4_flash.py`: `Model.sanitize`, `DecoderLayer`, `make_cache`, the Engram hash helpers); the family only reads each layer's bytes (routed experts included, Engram tables excluded) once per chunk and exposes `prefetch_ranges`. Carry `{ids, h [B,T,4,d]}` in the activation dtype (the 4 iHC streams, as `Model.hidden` passes them); each layer gets a fresh `HY4FlashLayerCache` and, on Engram layers 1/11/21/31, its rows from the adapter's SSD reader (in the map pass, pre-read once per chunk in file order by `prepare_layer`). Registered by `model_type` (the adapter's family is `hy-v4-flash`). Uses the child_rolling extensions (`begin_layer`, `neurons` payload); `capture.py` does not know them. Refuses: an artifact the adapter's `inspect_artifact` refuses (a missing indexed shard, e.g. a download in progress), image/video tokens, the grouped-expert kernels, non-bf16 Engram tables. `reference_logits(layers=K)` is an ordinary `Model` of the first K layers, resident (refused above `reference_gb`, 48 GiB: K <= 6 on real weights). **Implemented, CPU-tested bitwise on a tiny checkpoint (bf16 trunk with float32 routed experts, since CPU GatherMM is float32-only; f32; the MLX 8-bit conversion) against the family's reference and the adapter's `load_model`; unverified on real weights.** |
| `kimi_linear` | `kimi_linear.py` (on `layer_swap.py`) | layer 0 dense (9216), 26 MoE layers x 256 experts: neurons `e*F+f` (F = 1024) or whole experts | Whole-layer streaming of mlx2's own Kimi Linear model, built as `KimiLinearAdapter._load` builds it. Admits every layout the adapter stages, judged by its own `inspect_artifact`: the REAP hub BF16 and the 48B MLX q8, which is the parity model. Also admits the original 48B hub BF16 (`~/Models/Kimi-Linear-48B-A3B-Instruct`, about 92 GB): the adapter's hub contract with the expert count widened to 256 (config, tensor index, dtype law), fingerprinted by mlx2's artifact identity. Refuses `fused_projections`, `tensorfold_qmv` and `moe_nax`. **CPU-tested** on tiny hub-BF16 and q8 checkpoints: bitwise equal to the resident model the adapter itself loads. A real q8 layer and a real 48B BF16 layer bind on the CPU. **Not yet run on real weights end to end.** |
| `nemotron_h` | `nemotron_h.py` (on `layer_swap.py`) | Nemotron-H MoE layers in whole-expert units (Super VL: 40 x 512; Lightning: 23 x 128); Mamba and attention layers carry none | Whole-layer streaming of mlx2's Nemotron-H model for `Nemotron35SuperAdapter` artifacts and `Nemotron35LightningAdapter`, the parity model; Bolt and anything else refuse. For the Super VL NVFP4 export it follows `_load_super_model` and its checkpoint defaults: NVFP4 modules, vendor FP8 dense projections recovered exactly per streamed layer, the per-expert `weight_scale_2` fold redone at every load, the NAX prefill scope, exact attention rows. The vision tower, projector and MTP head are dropped from the tree and never read. Neuron units refuse because the experts are SwitchMLP: pass `--moe-units expert`. **CPU-tested** on tiny Lightning-affine and Super-NVFP4-export checkpoints: bitwise equal to the resident model the adapter itself loads. Real Super VL and Lightning layers bind and run on the CPU. **Not yet run on real weights end to end.** |

### Whole-layer streaming (`layer_swap.py`)

The families above that sit on `layer_swap.py` (`HOST_UNIT_FAMILIES` in `family.py`) stream a whole decoder layer,
every routed expert included, once per chunk. They use the host family's scheduling, call arguments and unit
payloads (`host_family.py`). Their model is the adapter's own module tree, built by a per-artifact spec. The rules:

- Every layer parameter holds a shape-true placeholder (a broadcast scalar), so nothing allocates until the layer
  is bound. The non-layer parameters load resident and must match the model exactly.
- `begin_layer(l)` reads layer l's checkpoint tensors lazily (`mx.load`, no cached shard mapping). They go through
  the adapter's own `sanitize`. The names must be exactly the layer's parameters. Shapes must match, and so must
  the packed dtypes; floats bind in the checkpoint's dtype, as `load_weights` does. The spec's per-layer
  installers run, then the tensors are evaluated.
- `release_layer(l)` restores the placeholders and drops every private array cache a module derived from its
  weights (an id-keyed cache could match a later array that reuses the id).
- `reference_logits` is the trunk's own text-major forward, each layer bound when called and released after.
  `prefetch_ranges` is each layer's merged checkpoint byte ranges. `stream_counters()["page_ins"]` counts layer
  loads, so `weight_reads_observed` holds.
- The loader takes `moe_units` and `log`, as the host family does. `child_rolling` passes them, and the run gate
  runs at full depth.
