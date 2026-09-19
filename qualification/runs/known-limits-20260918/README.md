# Closing four documented limits — GPU verification (2026-09-18)

Viability and correctness, not a performance qualification. One server at a
time on port 8297, `--qualification-mode`, served from the worktree that holds
these changes. Production mlx-serve was already stopped on request and was not
touched. Driver `run_campaign.py`; superseded passes under `first-pass/`.

| Limit | Fix | GPU evidence |
|---|---|---|
| Spomin published no prefix after it applied, so repeated long prompts re-prefilled | The **exact** pre-surgery boundary is published to APCv2 under the original tokens; the compacted state is still never published. Another session landed an equivalent fix on `main` (`77a8d20`) while this ran, so this branch keeps that implementation and contributes the GPU evidence (re-verified on the rebased tree: North TTFT 3.73 s → 0.07 s) | 13.2K-token prompt, 3 runs. North: `cached_tokens` 0 → 13,278 → 13,278, TTFT **3.77 s → 0.08 s**; Muse: 0 → 13,228, TTFT **14.95 s → 0.11 s**. Surgery re-applied every run (13.2K → 10.2K), identical answers, 3 approximate end-state stores skipped |
| North had no grammar-deferral marker, so structured requests defaulted to thinking off | North declares `<|END_THINKING|>` as its marker and a `<|START_TEXT|>` / `<|END_TEXT|>` answer **envelope**: the markers are ordinary vocabulary, so the processor keeps them out of the grammar's view and admits one opener before any text, a closer once the answer is complete, and only end-of-turn after it. Structured requests now think by default | 6 concurrent strict-schema requests: all valid, `deferred: true` (230–1,905 reasoning chars), reasoning and answer separated, no `<|` leaks; `json_object` and a raw grammar also pass; thinking-off still enforced |
| Approximate KV over a warm exact prefix was never exercised | Probe only (the product path already existed). Needs `start_tokens > 0` so a short prompt stays an exact, published lane, and raw completions so the long prompt is a true token extension | Qwen3.6, `kv_k8v4`, `start_tokens` 2500: 1.4K prompt declined/exact/warm; 5.3K prompt warm-hits **1,423 exact tokens**, is quantized privately (`requantized_prefix_hits` 1 → 2), answers `5521`; the exact entry afterwards returns the identical text with the same 1,410 cached tokens; leases drain to 0 |
| Prompt lookup verified one lane at a time | Lanes whose planes are all stock `KVCache` / `RotatingKVCache` share one target forward per round through the segmented KV transaction (ragged verify blocks, exact per-lane commit from K/V already computed, no replay forward for rejected tails). The round is split into plan / verify / commit steps used by both drivers | 20×20 sanity: Muse **400/400**, width 20, **41.6 tok/s (was 25)**; North 398/400 (the two known thinking-off arithmetic slips), width 20, **102 tok/s (was 94)** |

The external-draft route was already batched (observed width 9 of 20; cohorts
group lanes with equal draft counts), so nothing changed there except the
shared transaction fix below.

## Defect found on the way

The segmented KV transaction deep-copied every lane's whole cache each round.
On a served lane this raised (`cannot pickle '_thread.lock'`: caches restored
from APCv2 carry COW bookkeeping) and killed the generation worker; on any lane
it copied the full context per round. An append-only `KVCache` is now restored
by its offset alone, a rotating ring by a copy of its window-sized arrays plus
shallow host state, and masks are built from payload-free twins. This also
benefits the DFlash2 route, which uses the same transaction.

## Observations

- North with thinking on by default (`north-ordinary-thinking`, 20×20): 386/400,
  arithmetic 20/20. The 16 empties are reasoning that outran a 700–900 token
  budget on vague prompts (`finish_reason: length`) — clients of a thinking
  model need a larger `max_tokens`.
- North still pads JSON values with the (now bounded) whitespace after a key
  even with its answer envelope available, and gets some facts wrong inside
  valid JSON (4/6 coastal, tiny populations). Model behaviour; the documents are
  valid and terminate.
- Batched prompt-lookup gains are modest because attention stays per-row and
  prompt-lookup prefill is still one lane at a time; the shared part is the
  projections/MLP/MoE. Exactness is verified against the per-lane driver and
  plain greedy on CPU.
- APCv2 reuses whole stored boundaries; a different question over the same long
  context is a cold (correct, compacted) request on sliding-window models.
