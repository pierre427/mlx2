# Qwen3.6 35B-A3B port

The `qwen3_5_moe` adapter provides an ordinary text/APCv2 baseline for the
40-layer 35B-A3B topology: 30 Gated DeltaNet layers, 10 full-attention layers,
256 routed experts with top-8 selection, and one shared expert. Model-specific
tensor construction and normalization stay behind the adapter.

Implemented does not mean selected. The baseline deliberately uses eager
decode, the stock MoE router/expert path and the shared GDN implementation.
Compiled replay, fused router/expert kernels and an embedded MTP head are
separate candidate arms in
[`qualification/qwen36-gpu-plan.json`](../../qualification/qwen36-gpu-plan.json).
The single-token fused GDN kernel is implemented as an opt-in trial but remains
off in the adapter baseline.

Route selection follows the inspected artifact: a complete embedded MTP head
defaults to `native_mtp`, while a headless artifact defaults to `ordinary`.
Explicit route flags still win and remain part of operational provenance, but
qualification identity is the resolved route rather than the flag spelling.

## Evidence

- GPU-free inspection validates topology, indexed shards and actual MTP
  tensors without opening tensor payloads.
- The first real GPU probe loaded the 23 GiB four-bit artifact, exercised all
  30 GDN and 10 full-attention layers, produced finite logits and obtained
  exact full-vs-split replay (`max_abs_difference=0`, same argmax).
- A candidate HTTP server returned `READY` on cold and 15-token APCv2-warm
  requests. Strict JSON-schema output returned valid JSON. TTFT/ITL and
  mechanism telemetry populated without swap growth.
- Ordinary cold/warm context checks passed at 32,015 tokens (8.38/0.36 s TTFT)
  and 64,015 tokens (21.06/0.41 s TTFT); warm hits reused 32,014 and 64,014
  tokens respectively. These are bounded context cells, not a near-limit soak.
- The native-MTP artifact passed a warm two-lane candidate smoke with six true
  batched cycles, 72 segmented-attention calls and zero failures, full-prefix
  materializations or physical-B2 formations.
- The fused GDN decode trial passed 32 ordinary-artifact and 16 MTP-artifact
  matched steps. Logits, convolution state and float32 recurrent state were
  bit-identical on every step. The runs recorded 960 and 480 fused layer calls,
  zero fallbacks, and warm whole-model decode speedups of 1.077x and 1.107x.
  This is promising bounded evidence, but does not cover speculative verify,
  padded batching, long-duration soak or fused output projection.

The probe is recorded in
[`ordinary-probe.json`](../../qualification/runs/qwen36-35b-a3b/ordinary-probe.json).
This is bounded correctness evidence, not a complete qualification receipt;
the route remains unselected.

The ordered, inert checklist for the 2026-09-16/17 overnight window is
[`QWEN36-35B-A3B-OVERNIGHT-2026-09-16.md`](QWEN36-35B-A3B-OVERNIGHT-2026-09-16.md).
