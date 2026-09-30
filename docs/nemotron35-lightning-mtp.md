# Nemotron 3.5 Lightning 30B-A3B MLX 8-bit with original BF16 MTP

The `nemotron_h` resolver selects `Nemotron35LightningAdapter` for the pinned
52-layer MLX 8-bit conversion. The ordinary path is available without an MTP
attachment. Native MTP becomes an implemented candidate only when the target
contains `mlx2-nemotron35-mtp.json` and the separate original NVIDIA shard
passes SHA-256, safetensors, and tensor-inventory checks. The target's config
declares one MTP head, but its seven converted shards contain no MTP tensors.

For the locally downloaded artifacts, the manifest is:

```json
{
  "schema": "mlx2-nemotron35-mtp-v1",
  "target_revision": "a9db86e1fe5baf448346efd33541ce117b5b8403",
  "source_repository": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
  "source_revision": "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
  "source_sha256": "64577b275ca4e7e5266eae0903674f7f46ec2a8cbf4f4f1a3207f80d503cd1d0",
  "path": "../sidecars/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-MTP-original/model-00014-of-00014.safetensors"
}
```

The adapter verifies the target config and weight-index hashes, all seven shard
headers, the tokenizer/tool-template contract, and the entire MTP source hash.
The MTP file stays outside the target weight directory, so generic weight
discovery cannot load it as an extra target shard. The artifact fingerprint
includes the source attachment. Invalid or stale attachments fail closed;
without a manifest the descriptor exposes ordinary decode only.

## Settings and runtime behavior

The adapter uses the checkpoint's affine 8-bit, group-64 target weights and
keeps the original MTP weights in BF16. It uses the existing Nemotron H Mamba,
MoE, GQA and two-layer MTP math, target `lm_head` and embeddings, tokenwise
single-row target verification, exact recurrent rollback, APCv2 prefix state,
and shard-by-shard buffer-cache eviction during loading. The configured
context ceiling is **262,144 tokens**, the value in this conversion's config.
Actual usable context is still limited by memory and qualification.
The Mamba recurrence uses an FP32 state and clamps the inference time step to
the config's `time_step_limit`, which is unset, so (0, inf), as NVIDIA's own
Nemotron H forward, vLLM and mlx-lm do. The config's 0.001 minimum and 0.1
maximum apply to time-step bias initialization only. transformers' native
`nemotron_h` floors the step at 0.001 instead (transformers #48989); an earlier
mlx2 revision copied that floor, which raised the step of 133 of 1,472 heads
(those with `softplus(dt_bias) < 0.001`) 10-100x on every token.

Generation defaults are **temperature 1.0** and **top-p 0.95**, matching the
checkpoint and [NVIDIA's model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).
The chat template enables thinking by default, supports the Qwen3 Coder XML
tool-call format, and stops on token IDs 2 and 11. The engine's 2,048-token
prefill chunks and APCv2 cache are retained. Native MTP proposes two tokens
per cycle by default; an explicit `num_draft` may select one through three.
That depth is a bounded implementation setting, **not** a measured optimum
for this 8-bit target. Ordinary decode stays the default reference route.

NVIDIA's CUDA recipes describe DSpark, MTP and CUDA-specific Mamba/attention
settings for different hardware and concurrency. They do not establish an
Apple Silicon performance setting for this MLX conversion. The adapter does
not select those backends or adopt their FP16 recurrent-cache and stochastic
rounding settings without parity and device evidence.

## Validation state

CPU-only checks cover source-bound artifact dispatch, absent/invalid sidecar
refusal, cache geometry, the Mamba time-step clamp, exact MTP parameter-key correspondence after expert
stacking, and a tiny MTP forward. They do not load the full checkpoint or
establish full-model output parity, speed, memory peaks, cached-prefix MTP
parity, or serving qualification. An explicit MTP route still requires a fresh
artifact/runtime/settings-bound qualification receipt before production
selection. No GPU resource or full-model execution was used for this adapter work.
