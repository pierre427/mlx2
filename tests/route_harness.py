"""Real ServingEngines over tiny random models, one per serving route (CPU).

Tests that compare routes (ordinary, native MTP, prompt lookup, external
draft) build their engines here so every route sees the same fake adapter,
tokenizer and output parser.  Detokenized text is the token ids separated by
spaces, so a response's content round-trips to its token list.
"""

import mlx.core as mx

from mlx2 import memory, serving
from mlx2.contracts import Capability, ModelDescriptor, StatePlane
from mlx2.runtime import os_memory
from mlx2.serving import ServingEngine


def patch_host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(serving, "execution_headroom", lambda: 100 * 2**30, raising=False)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def tiny_qwen38_mtp(seed=7, vocab=128):
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=vocab, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=512,
    )
    mx.random.seed(seed)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model, vocab


def tiny_muse_dflash(seed=8, vocab=128):
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(seed)
    model = Model(ModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=vocab, sliding_window=8, max_position_embeddings=512,
    ))
    config = DFlash2Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=vocab, num_target_layers=4, target_layer_ids=[0, 3],
        conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=4, mask_token_id=vocab - 1,
        max_position_embeddings=512, sliding_window=8,
        layer_types=["sliding_attention"] * 2,
    )
    draft = DFlash2DraftModel(config).bind(model)
    mx.eval(model.parameters(), draft.parameters())
    return model, draft, vocab


class Detok:
    def __init__(self):
        self._pending = ""

    def reset(self):
        self._pending = ""

    def add_token(self, token):
        self._pending += f"{int(token)} "

    def finalize(self):
        pass

    @property
    def last_segment(self):
        out, self._pending = self._pending, ""
        return out


class Parser:
    stopped = False
    tool_count = 0

    def push(self, text, final=False):
        return [{"content": text}] if text else []


PIECES = ["<eos>"] + [f"<s{i}>" for i in range(1, 32)] + [chr(i) for i in range(32, 127)] + ["</think>"]
PIECES += [f"<x{i}>" for i in range(len(PIECES), 256)]

CAPS = frozenset({
    Capability.TEXT, Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE,
    Capability.APC_V2, Capability.MTP, Capability.SEGMENTED_MTP,
    Capability.PROMPT_LOOKUP, Capability.STREAMING, Capability.GRAMMAR,
})


def make_adapter(model, vocab, *, eos=(), num_draft=2, extra=None, close_id=None,
                 adapter_mixin=None):
    extra = dict(extra or {})

    class Tokenizer:
        vocab_size = vocab
        eos_token_ids = list(eos)

        @property
        def detokenizer(self):
            return Detok()

        def convert_ids_to_tokens(self, ids):
            return [PIECES[int(i)] if int(i) < len(PIECES) else f"<x{int(i)}>" for i in ids]

        def decode(self, ids, **_kw):
            return "".join(self.convert_ids_to_tokens(ids))

        def encode(self, text, **_kw):
            return [PIECES.index(ch) for ch in text]

        def __len__(self):
            return vocab

    class Adapter:
        max_context = 512
        identity = {"fingerprint": "tiny-hybrid-mtp"}
        environment = {}
        layout = "tiny-hybrid-mtp-layout"
        tokenizer = Tokenizer()
        descriptor = ModelDescriptor(
            model_type="tiny", family="tiny", variant="v",
            state_planes=frozenset({StatePlane.ATTENTION_KV}),
            capabilities=CAPS, cache_layout="tiny-layout",
        )

        def __init__(self, _path):
            self.model = model

        def profile_name(self, mtp):
            return "tiny-mtp" if mtp else "tiny-ordinary"

        def execution_config(self, *, max_lanes, prefill_step):
            return {"persistent": True, "num_draft": num_draft, "rate_gate": False,
                    "prefill_step_size": prefill_step, **extra}

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        if close_id is not None:
            def thinking_enabled(self, request):
                return "messages" in request and request.get("enable_thinking", True)

            def thinking_close_token_ids(self):
                return tuple(close_id) if isinstance(close_id, (tuple, list)) else (close_id,)

        def close(self):
            pass

    if adapter_mixin is not None:
        Adapter = type("Adapter", (adapter_mixin, Adapter), {})
    return Adapter


def make_engine(model, vocab, *, mtp=True, prompt_lookup=False, max_lanes=1,
                eos=(), num_draft=2, extra=None, close_id=None, adapter_mixin=None, **kw):
    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(
            model, vocab, eos=eos, num_draft=num_draft, extra=extra,
            close_id=close_id, adapter_mixin=adapter_mixin,
        ),
        qualification_mode=True, mtp=mtp, prompt_lookup=prompt_lookup,
        max_lanes=max_lanes, max_inflight=max(32, max_lanes), prefill_step=16, **kw)
    assert engine.ready.wait(120), engine.error
    if engine.error:
        raise RuntimeError(engine.error)
    return engine


def make_external_engine(model, draft, vocab, *, num_draft=2, max_lanes=1, eos=(),
                         close_id=None, **kw):
    caps = (set(CAPS) | {Capability.EXTERNAL_DRAFT}) - {Capability.MTP, Capability.SEGMENTED_MTP}

    class Mixin:
        descriptor = ModelDescriptor(
            model_type="tiny", family="tiny", variant="v",
            state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.DRAFT}),
            capabilities=frozenset(caps), cache_layout="tiny-layout",
        )

        def execution_config(self, *, max_lanes, prefill_step):
            return {"persistent": True, "num_draft": num_draft,
                    "backend": "external_draft", "rate_gate": False,
                    "prefill_step_size": prefill_step, "segment_aware_live_tip": False,
                    "segment_aware_cohort_size": max_lanes}

        def profile_name(self, mtp):
            return "tiny-ext"

        def create_external_batch(self, **kwargs):
            from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

            return ExternalDraftBatchGenerator(
                self.model, draft_model=draft, binding="tiny", num_draft=num_draft, **kwargs
            )

    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(model, vocab, eos=eos, close_id=close_id, adapter_mixin=Mixin),
        qualification_mode=True, mtp=False, max_lanes=max_lanes,
        max_inflight=max(32, max_lanes), prefill_step=16, **kw)
    assert engine.ready.wait(120), engine.error
    if engine.error:
        raise RuntimeError(engine.error)
    return engine


def collect(job, timeout=120):
    tokens, finish, events = [], None, []
    while True:
        event = job.events.get(timeout=timeout)
        events.append(event)
        if "error" in event:
            return {"error": event["error"], "status": event.get("status"),
                    "tokens": tokens, "events": events}
        if "delta" in event:
            text = event["delta"].get("content", "")
            tokens.extend(int(t) for t in text.split())
        if "finish_reason" in event:
            return {"tokens": tokens, "finish": event["finish_reason"],
                    "receipt": event.get("receipt") or {}, "events": events}


def run(engine, request, timeout=120):
    return collect(engine.submit(dict(request)), timeout)
