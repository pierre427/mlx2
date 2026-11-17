# Resident calculator and speculative decoding

This change admits the GPU-resident calculator (`calc-parse-v5-resident`) on
Qwen's external-draft route. The Qwen 27B and DFlash2 pair exercised it on the
M5 Max. The combined route remains an opt-in candidate.

DFlash proposes tokens. The target model still decides which tokens are valid.
The calculator modifies that target decision, using exactly the same device
parser and result-writing code as ordinary decoding. If complete syntax or exact
arithmetic fails validation, the calculator gate stays closed. The speculative
wrapper never parses text or reads a result, gate bit, or output token on the CPU.

The state contract has four parts:

1. **Draft probes are disposable.** Each proposal position starts from a detached
   snapshot of device state. The provisional suffix is replayed on device, so a
   multi-digit answer advances through its digits rather than repeatedly forcing
   the first digit. Proposal guidance cannot change the authoritative tile or its
   counters. The verifier remains authoritative if draft guidance differs.
2. **Verification records device snapshots.** Each reachable target row uses its
   own complete prefix. The verifier saves the device state, initial-request flag,
   and prefix position after each row. A skipped unreachable row never calls the
   calculator.
3. **Only committed rows survive.** After draft acceptance, rejection, stop, or
   budget truncation, the existing verifier ledger restores the last used row.
   Gate openings and tokens-written counters are published only at settlement.
4. **Failure recovery preserves ownership.** Recovery restores the original
   request's processor and device state while retaining its registered counter
   entry and shared immutable vocabulary. The model's KV cache and RNG continue
   to use the existing external-decoder recovery mechanism.

Dynamic `tree15` verification follows only the selected root-to-leaf path and
uses this same ledger. It uses the existing row-wise target-law verifier;
calculator processors do not claim the pure, vectorized mask contract. Private
progressive branches remain unsupported and use the existing primary-chain
fallback where that route permits one. Explicit private provider/oracle,
continuation-pool, and special packed-tree combinations fail before model load.
Native MTP, PLD, draft portfolios, legacy CPU-parser tiles, and fact/mixed tile sets
remain refused by this compatibility layer.

The Qwen adapter retains checkpoint/manifest-bound cache identity. Production
defaults are unchanged. An ordinary serving reference remains available.

Validation includes owned Metal parser tests and an exact
checkpoint/source-bound DFlash A/B. Run `scripts/spec_acceptance_gsm8k.py` against
both DFlash alone and DFlash plus the tile, recording accuracy, mean acceptance
length, per-position acceptance, route receipts, and counters. Also exercise
plain prose/copy, unsupported expressions, single and batched requests, early
stop, and cold/warm APCv2 replay. CPU transaction tests substitute only the
calculator kernel; they run the real tiny-model DFlash executor with greedy and
sampled decoding at widths one and two.

## Reproducible preparation and owned smoke

From the integrated worktree, preparation is CPU-only. It creates a hashed
contract file and manifest, with no learned tensor weights or benchmark answers:

```sh
PYTHONPATH="$PWD/src" .venv/bin/python \
  scripts/research/amoeba_calc_spec_prepare.py \
  --out "$PWD/output/research/calculator-speculative-setup"
```

`setup.json` records the exact target fingerprint and source commit. The external
loader computes and checks target/drafter payload revision pins on startup;
retain the returned status and route receipts. For the installed November 17
target, the metadata identity is `Qwen3.8-27B-BF16-q8-mlx2@2dd8d304`.
The parent independently observed its DFlash baseline drafter as `478902e6`;
that observation does not qualify the combination.

Inside the normal CPG lease and paired-lock window, the chain control is:

```sh
PYTHONPATH="$PWD/src" MLX_METAL_FAST_SYNCH=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m mlx2.server \
  --model ~/mlx-models/Qwen3.8-27B-BF16-q8-mlx2 \
  --external-draft --port 8393 --max-lanes 4 \
  --execution-policy "$PWD/output/research/calculator-speculative-setup/chain-policy.json"
```

Stop that whole server group before the tree15 arm. Its command uses the current
AST drafter defaults, including whichever artifact/device-bound optimizations
the adapter selects:

```sh
PYTHONPATH="$PWD/src" MLX_METAL_FAST_SYNCH=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m mlx2.server \
  --model ~/mlx-models/Qwen3.8-27B-BF16-q8-mlx2 \
  --ast --ast-drafter ~/mlx-models/Qwen3.8-27B-DFlash2 \
  --port 8393 --max-lanes 4 \
  --execution-policy "$PWD/output/research/calculator-speculative-setup/ast-policy.json"
```

A first-token arithmetic smoke must return `189` and a normal terminal finish:

```sh
curl -sS http://127.0.0.1:8393/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Calculate (19+8)*7."}],"temperature":0,"enable_thinking":false,"max_tokens":64}'
```

Repeat at B1 and B4 with `How many r's are in strawberry?`, a nested fraction,
`Repeat exactly this text, with no explanation:\n2 + 3 =`, and unsupported
`Calculate sin(2)+3.` / `Calculate 1/0.`. Inspect the live calculator receipt's
`plan_executions.gpu_resident_metal`, `gate_open`, `tokens_written` and
`resident_refusals`: unsupported/copy requests must not add gate openings.
Do not infer engagement merely from correct text. Preserve each request's actual
`speculative_receipt` and verify the concrete route, proposal counts,
`verify_accept_hist`, `verify_span_hist`, tree row-law counters and any fallback.

The standard acceptance probe against the owned server is:

```sh
PYTHONPATH="$PWD/src" .venv/bin/python \
  scripts/spec_acceptance_gsm8k.py --base-url http://127.0.0.1:8393 \
  --checkpoint Qwen3.8-27B-BF16-q8-mlx2@2dd8d304 \
  --output "$PWD/output/research/calculator-dflash-gsm8k" \
  --num-questions 100 --format chat-fewshot --max-tokens 4096 --draft-depth 7 \
  --request '{"enable_thinking":false}'
```

Run the same fixed prompts against the DFlash-only baseline and retain both
results. The harness reports mean accepted length and per-position acceptance
from verified request counters; instrumented functional timings are not a
controlled throughput result. Use the full fingerprint from `setup.json` in the
archived checkpoint receipt rather than the abbreviated label in this example.


## Served comparison

The run used `Qwen3.8-27B-BF16-q8-mlx2@2dd8d304`, the local `Qwen3.8-27B-DFlash2@478902e6`
drafter, and source `1b182463e1e99e2f119798d49bc1e3abda040a60`. Both arms passed all 12 exact
copy cases and produced identical token IDs for all four prose prompts. The
combined arm passed the supported arithmetic, fraction and counting checks at
one and four lanes.

| GSM8K and DFlash metric | Stock DFlash | DFlash with resident calculator |
|---|---:|---:|
| Five-shot GSM8K sample | 31/32 | 31/32 |
| Verification rounds | 753 | 757 |
| Mean committed tokens per round | 8.070 | 8.036 |
| First draft position accepted | 98.67% | 98.68% |
| Eighth draft position accepted | 43.29% | 42.80% |
| Median diagnostic decode t/s | 86.2 | 78.1 |

The receipt includes the full per-position acceptance curve. The combined
route recorded eight GPU calculator plans with no aborts, including six
control plans. The accuracy comparison uses the same 32 prompts in both arms.
The decode rate comes from this diagnostic sample.

[Source-bound served receipts](../../qualification/runs/calculator-speculative-20261117/served/)
