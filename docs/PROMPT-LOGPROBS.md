# Prompt logprobs: oracle and family coverage (2026-11-14)

Use `--prompt-logprobs on` to enable opt-in prompt scoring. The code is in:

- `src/mlx2/prompt_logprobs.py`;
- `ServingEngine.submit_prompt_scoring` / `_step_prompt_scoring`;
- `server.wait_prompt_scoring`.

It is original mlx2 code; nothing was mined. The route is `implemented`, not
`qualified`, and the flag is off by default.

## Oracle

- **Checkpoint:** `Qwen3-0.6B-BF16-microscaling-source@93ff4d9ed094199001756a120664cc2ce379e79dfe357f71978e8228664d76c6`
- **mlx2 commit:** `9db353833`, after the second review (`oracle-v3-chunk256.json`), and `82f2abf35`, after the first review fixes (`oracle-v2-chunk256.json`). The first run was at `7b8a7ec0a` (`oracle-chunk256.json`, `oracle-chunk8.json`, `chunkcheck.json`).
- **Device:** MLX CPU, on dev port 8391, run with `--prompt-logprobs on --max-lanes 2`.
  - `cpu_server.py` pins the default device to the CPU.
  - On the CPU device `mx.device_info()` has no recommended working set, so `execution_headroom` reads 0 and every admission refuses. That is why generation returned 429 in the first run. For this smoke only, the shim reports host-available memory as headroom.
- **Reference:** the adapter's `model(tokens[None])` over the whole prompt in one forward, then fp32 `log_softmax`. It is compared against the served `token_logprobs` and against every served `prompt_logprobs` entry (actual token plus top 5).

Rerun at `9db353833` after the second review (`oracle-v3-chunk256.json`):
- **Values:** every cell below repeated, with max abs diff 0.0 against the one-shot reference and cold and warm scoring identical.
- **Memory:** charged through lane admission (`memory.method: lane_admission`, `kv_charge: lane_envelope_gib_per_1k_tokens`). Qwen3-0.6B's standard decoder has no cache projection.
- **Cache:** `kv_cache.classes: [KVCache]`.
- **Multi-chunk prompt:** a 2199-token prompt on the idle worker ran 9 chunks back to back in 9.38 s, 1.04 s per chunk on the MLX CPU device, with 0.03 s of queue time. This is compute, not the former 50 ms idle wait.

Results at `82f2abf35`, default chunk 256:

| Cell | Result |
|---|---|
| 4 completions probes (20/34/37/37 tokens), echo + `prompt_logprobs`, `max_tokens: 0` | max abs diff **0.0**; top-5 id sets identical |
| Same prompts scored before any generation (cold) and again after an ordinary generation populated APCv2 (warm) | identical values (APCv2 is never read) |
| APCv2 after scoring | a later generation on the same prompt hit 19 cached tokens: the store and its gates are intact |
| Echo + `max_tokens: 6` (ordinary generation path) | prompt part diff **0.0**; 6 generated tokens appended; offsets monotonic; generation starts at offset 162 = prompt length |
| Chat `prompt_logprobs` (17 templated tokens) | **0.0** |
| 2 scoring requests racing 3 generations at lane width 2 | all 200, diff **0.0**. Five 503 `busy` answers to the scorer retries show the one-at-a-time limit |
| `prompt_logprobs: 21` | 400 `prompt_logprobs_top_cap` |

The receipts report:
- `numerics.fidelity: exact`, `int8_prefill.active: false`;
- `memory.checked: true`, with required and available bytes.

**Chunk boundaries (first run, `--prompt-logprobs-chunk 8`).** The served values equal a direct chunk-8 cached forward exactly (0.0). They differ from the one-shot bf16 forward by up to 0.54 nat. That is bf16 shape sensitivity:
- the one-shot bf16 forward is itself 0.20 nat from fp32 on the same weights;
- summed NLL is 79.71 (bf16 one-shot), 80.51 (bf16 chunk 8) and 80.44 (fp32).

The scoring math did not change in the review fixes. A prompt longer than the chunk is scored the way the server's own chunked prefill computes it.

## Family coverage

The classes below come from a static read of every text-registry adapter (no GPU loads), plus the CPU run above. "Expected" means the default contract applies and nothing declares a refusal; it is not "observed".

**Expected to serve** (`adapter.model(tokens, cache=)` returns every row):
- Standard decoder (qwen3 / qwen3_moe / qwen2 / llama, Nanbeige), Qwen3.5 4B/9B, Qwen3.6/3.8 27B, Qwen3.6 35B, Qwen3.5 122B;
- Flash-Next, North, Laguna, Xing, Nemotron-H, Mellum, Inkling, Agnes, Ling, HY v3, gpt-oss, Granite SWA, LFM2.5 MoE, HySparse2 export, Kimi Linear, Ripple.

Recurrent state and sliding windows ride `make_prompt_cache`. The CPU tests check that chunked equals one-shot on a tiny Qwen3.8 hybrid GDN model and a tiny Muse sliding-window model. **Observed served: Qwen3-0.6B only.**

**Expected to serve through `prefill_forward`:** Flint, Gemma 3n, Nemotron 3.5 Super-VL/Bolt, GLM-5.3 GGUF, DeepSeek V4 GGUF/MLX. Any row pruning fails closed (501 `forward_not_full_sequence`).

**Expected to serve, text-only prompts:** Gemma 4, MiniCPM-o, LFM2.5-VL, SmolVLM2, Qwen2.5-VL, Muse Glimmer. Chat media parts are refused.

**Fail closed (501 `route_not_exact`):**
- Any route whose APCv2 reuse is disabled for a fidelity reason:
  - Flash-Next soft reap, and the `moe_keep_topk_stock` control;
  - DS4.1 Source approximate operations, q1-on-miss, EXL3 experts and dynamic precision;
  - DS4.1 Source adaptive prefill partitions (`adaptive_prefill_partitions_not_qualified`, the Source default). This is conservative.
- DS4.1 Source's declared `prompt_logprobs_fidelity`: cold-Q1 hot maps, yoco_v1, and the policies above. Dynamic precision is refused, so scoring cannot update its expert-cache history.
- Microscaling prefill.
- Any adapter whose `numeric_profile` or execution numerics contract says approximate.

**Other DS4.1 routes:** GGUF/MLX/IQ1/omlx serve on their exact profiles. Under CED `dependency_exact`, the prompt forward returns one row, so it fails closed with 501.

**W8A8 int8 prefill** (e.g. Flash-Next on :8282): served. The scoring chunk is capped below the row threshold (512 gives 511 rows), so the int8 kernels never engage. The receipt records `int8_prefill: {active, row_threshold, engaged: false, mechanism: chunk_below_row_threshold}`.

**Approximate KV (`kv_q8`):** scoring uses a fresh unquantized cache (`kv_cache: fresh_request_private_unquantized`). It scores the exact model, not the quantized-KV serving numerics.

**Fail closed or not reachable:**
- MiniLM embeddings/reranker: 501 `not_autoregressive_text`.
- Kimi-K3.1: not loadable.
- TensorFold-owned header route: not reached; that route is a separate router.
- `tensor_fa_research`: imported by nothing, so never on this path.

An adapter can override the default with `prompt_logprobs_binding()`, which returns `(forward, cache)`. It can refuse with `prompt_logprobs_refusal`, or declare `prompt_logprobs_fidelity`. All three are declared attributes, so serving never branches on a model name.

## Known limits

- **Decode latency.** Scoring chunks interleave with decode steps. While lanes decode, each chunk adds one forward of at most 256 rows (511 with int8 prefill) to the decode step it runs beside. With the worker idle, chunks run back to back. Scoring never drains or blocks admission.
- **Exclusive operations.** Embeddings, rerank, LoRA or overlay install, and prefill-handoff export preempt a running or queued scoring task (503 `preempted_by_<operation>`, with a receipt) instead of waiting up to their 30 s drain. Steady embeddings traffic can starve scoring; retrying with backoff is the contract.
- **Access.** `--prompt-logprobs-tenants` requires tenant authentication. Under it, the dedicated `prompt-logprobs` scope or a listed tenant may score. A non-loopback bind needs tenant auth, an API key or `--prompt-logprobs-allow-unauthenticated`.
- **Quantized KV routes.** Routes whose private cache would be quantized (GLM-5.3 `mla_kv_q8`) are refused.
- **One task at a time.** Any other scoring request gets 503 `busy` immediately.
- **No streaming and no `n > 1`.**
- **Chat scoring includes the generation prompt.** To score an assistant reply, use `/apply-template` and `/v1/completions`.
