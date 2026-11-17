# Gemma 4 resident calculator integration

The resident calculator ran on the Gemma 4 A4B ordinary serving route and
improved numeric accuracy from **59/112 to 103/112**. It is an opt-in,
unqualified experiment. CPU tests cover admission and unchanged feed-forward
computation in small dense and MoE models. The served run establishes GPU
engagement on the checkpoint below.

The intended target is
`gemma-4-26B-A4B-it-MLX-8bit@0364a378`, full artifact fingerprint
`0364a378f7c6b1c4685e43c3bde9d5cec69185d68dd2dab47fe43bd99f9caca6`.
This is the local 30-layer Gemma 4 A4B checkpoint, with 2,816 hidden dimensions
and 128 routed experts. The dense 31B adapter shares the admission code but is
not the model chosen for the first GPU demonstration.

## Adapter boundary

The policy takes a manifest path:

```json
{"amoeba_det_tiles": "/absolute/path/to/manifest.json"}
```

The server's `--execution-policy` option takes the **path to that JSON file**.
It does not accept inline JSON. The generic calculator harness must include
the CLI repair in `30f7740ae` or a descendant.

Gemma admits exactly one `calc-parse-v5-resident` calculator with force writing,
GPU arithmetic, and the validated-syntax gate. Other deterministic tile types
and learned-probe calculator versions are refused before model weights load.
This avoids claiming that a Qwen-trained probe or fact-table projection
transfers to Gemma. The manifest's fingerprint and layer count must match the
actual artifact. Its digest is rechecked after installation to detect a
manifest changed while the model was loading.

Gemma A4B has parallel dense and sparse feed-forward branches. The adapter
wraps the existing dense MLP only to capture the current batch/forward binding.
The resident calculator reads no learned feature there and changes no tensor
in either branch. The selected manifest joins both the adapter layout and its
execution-numerics contract, so APCv2 identity distinguishes stock, calculator,
and different calculator manifests.

The shared calculator controller receives Gemma's tokenizer and the actual
request. Vocabulary initialization uses the existing byte-faithful helper,
which preserves SentencePiece word-boundary spaces and byte fallback pieces.
The parser, arithmetic, and actuation gate remain in the shared tile; this
adapter adds no Gemma-specific math parsing. Unsupported candidates cannot
be promoted by a Gemma-specific fallback.

## Reproduce preparation and testing

CPU preparation does not load model weights or dispatch GPU work:

```sh
PYTHONPATH="$PWD/src" .venv/bin/python \
  scripts/research/amoeba_calc_gemma4.py \
  --out output/research/calculator-gemma4-20261117/tile
```

It writes `manifest.json`, a hash-bound `resident.json` with no learned
parameters, `execution-policy.json`, and a `setup.json` naming the exact
artifact, current commit, environment, and benchmark command. A nonempty
output directory is never overwritten. The manifest contains no answers or
benchmark data.

Inside an owned GPU window, first run the resident kernel and
tokenizer checks, then the generated benchmark command. The serving harness
runs stock batches before calculator batches at width four, a 4,096-token
limit, and run-on/spiral defenses. It performs width-one and width-four
controls before scored requests and records route/counter receipts. Its
logprob-instrumented timing is diagnostic, not a publishable throughput
measurement. The frozen 128 questions are reused diagnostic evidence, not a
new held-out benchmark.

Focused CPU integration tests:

```sh
PYTHONPATH="$PWD/src" .venv/bin/python -m pytest \
  tests/test_gemma4_calculator_cpu.py tests/test_gemma4_adapter.py \
  tests/test_gemma4_moe_globals_cpu.py -q
```

The new tests cover exact artifact admission, refusal before loading,
manifest changes during loading, request forwarding, numerical/cache identity,
unchanged dense/MoE feed-forward outputs, untouched sparse experts, and
weight-free manifest preparation. Production defaults are unchanged.

## Served result

The complete A/B ran at source `77c9ce408da4a097f0f1d28985f9b80e5bf8db49`
with four lanes and the 4,096-token cap. All three supported calculator controls
returned the expected result alone at widths one and four. The calculator
recorded 82 `gpu_resident_metal` plans and 1,230 written tokens, including the
controls. The device checks used real Qwen and Gemma tokenizers and left logits
unchanged on all unsupported and exact-copy candidates.

| Case group | Stock | Resident calculator |
|---|---:|---:|
| Hard literal expressions | 0/32 | 32/32 |
| Hard word problems | 0/8 | 0/8 |
| Basic arithmetic | 20/20 | 20/20 |
| Large numbers and decimals | 4/16 | 16/16 |
| Noisy word problems | 8/8 | 8/8 |
| GSM8K sample | 19/20 | 19/20 |
| Literal counting and list sums | 8/8 | 8/8 |
| **Numeric total** | **59/112** | **103/112** |
| Exact-copy controls | 12/12 | 12/12 |

The tile recovered 44 numeric cases and preserved every correct stock answer.
All four prose outputs matched token for token. Their mean per-request decode
rates were 54.7 t/s stock and 50.8 t/s with the tile. These four served prompts
use logprob diagnostics and supply a directional comparison.

[A/B results, route counters and ownership receipts](../../qualification/runs/calculator-gemma4-20261117/served/)
