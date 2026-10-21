"""Unqualified ordinary-text adapter for exported HySparse2 research checkpoints.

Original mlx2 code.  It serves a checkpoint trained by
``mlx2.experimental.hysparse2.train`` through the generic scheduler, APCv2 and
prompt-lookup machinery.  The artifact is a serving export produced by
``scripts/export_hysparse2_serving.py``: hard links to the immutable training
outputs plus a generated ``config.json``, the training tokenizer and a
manifest binding every source hash.  The training outputs are never modified.

Scope (truthful states, AGENTS.md):

* ordinary decode, continuous batching, APCv2 prefix reuse and adaptive
  prompt lookup are implemented; none is qualified;
* the MTP head is a training objective in this research model.  The only
  inference path (``experimental.hysparse2.speculation``) is a sequential
  mechanics reference without an MTP history cache, so no native-MTP serving
  route is declared and ``--ast`` resolves to adaptive prompt lookup;
* the diffusion student and semantic capsules are not serving routes;
* the chat format is the training corpus' own ``role: content`` turns
  (``corpus.messages_text``); roles absent from training fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, fields
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import (
    PROCESS_NUMERICS,
    assert_process_profile_applied,
    clear_inherited_profile,
    require_process_numerics,
)
from .artifact_paths import shard_within_artifact
from .ordinary_text import OrdinaryTextAdapter

MANIFEST = "hysparse2-serving.json"
MANIFEST_SCHEMA = "mlx2.hysparse2-serving-export.v1"
MODEL_TYPES = frozenset(
    {"mlx2_hysparse2_research", "mlx2_hysparse2_memory_diffusion_research"}
)
ARCHITECTURE = "HySparse2ResearchModel"
CHECKPOINT_FILES = ("model.safetensors", "state.json")
PLE_FILE = "semantic-ple.safetensors"
TOKENIZER_FILE = "tokenizer.json"
# The 16K context ladder is the longest validated cached execution of these
# checkpoints (docs/research/hysparse2/memory-diffusion-training-1m.json);
# the configured 2M address limit is a capacity target, not evidence.
SERVING_MAX_CONTEXT = 16384
# Training formats turns as ``role: content`` joined by blank lines
# (corpus.messages_text); a generation prompt opens the assistant turn.
CHAT_ROLES = ("user", "assistant")
CHAT_TURN_STOP = "\n\nuser:"

DESCRIPTOR = ModelDescriptor(
    model_type="mlx2_hysparse2_research",
    family="hysparse2-research",
    variant="toy-ordinary",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.PROMPT_LOOKUP,
        }
    ),
    cache_layout="hysparse2-serving-ple-swa-fa-cross-kv-v1",
    metadata={
        "execution": "mlx2.adapters.hysparse2.HySparse2Adapter",
        "qualification": "pending",
        "scope": "ordinary text and prompt lookup; research toy checkpoint",
        "native_mtp": {
            "implemented": False,
            "reason": (
                "the MTP head is a training objective; the research inference "
                "reference is sequential and keeps no MTP history cache"
            ),
        },
        "diffusion_student": {"implemented": False, "serving_route": False},
    },
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def research_config(config: dict):
    """Build the research ``Config`` from a serving ``config.json``."""
    from ..experimental.hysparse2.config import Config

    names = {field.name for field in fields(Config)}
    return Config(**{key: value for key, value in config.items() if key in names})


def inspect_artifact(model_path: str | Path) -> dict:
    """CPU-only artifact checks; never opens tensor payloads."""
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    manifest = _json(path / MANIFEST)
    state = _json(path / "state.json")
    if config.get("model_type") not in MODEL_TYPES:
        raise ValueError("not a HySparse2 research serving export")
    if config.get("architectures") != [ARCHITECTURE]:
        raise ValueError("HySparse2 export architecture mismatch")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported HySparse2 serving manifest")
    if state.get("schema") not in {
        "mlx2.hysparse2-checkpoint.v1",
        "mlx2.hysparse2-model-checkpoint.v1",
    }:
        raise ValueError("unsupported HySparse2 checkpoint schema")
    cfg = research_config(config)
    from ..experimental.hysparse2.config import Config

    if asdict(Config(**state["config"])) != asdict(cfg):
        raise ValueError("serving config.json differs from the checkpoint configuration")
    if state.get("capsule_binding") is not None:
        raise ValueError("semantic capsule reads are not a HySparse2 serving route")
    required = list(CHECKPOINT_FILES) + [TOKENIZER_FILE]
    if cfg.semantic_ple_rows:
        required.append(PLE_FILE)
    records = []
    for name in required:
        item = path / name
        if not item.is_file() or not shard_within_artifact(path, item.resolve()):
            raise ValueError(f"missing HySparse2 export file: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(required):
        raise ValueError("HySparse2 manifest must hash every export file")
    # Small files are verified on every inspection; the multi-GiB weights
    # carry their export-time hash and size/mtime in the fingerprint, and
    # load strictly through the research checkpoint loader.
    for name in ("state.json", TOKENIZER_FILE):
        if _sha256(path / name) != hashes[name]:
            raise ValueError(f"HySparse2 export file changed since export: {name}")
    tokenizer_pin = (state.get("run") or {}).get("tokenizer_sha256")
    if tokenizer_pin is not None and tokenizer_pin != hashes[TOKENIZER_FILE]:
        raise ValueError("tokenizer differs from the one the checkpoint was trained with")
    sidecar = state.get("permanent_sidecar")
    if cfg.semantic_ple_rows and (
        not isinstance(sidecar, dict) or sidecar.get("sha256") != hashes[PLE_FILE]
    ):
        raise ValueError("semantic PLE sidecar hash differs from the checkpoint binding")
    digest = hashlib.sha256()
    for name in ("config.json", MANIFEST, "state.json", TOKENIZER_FILE):
        digest.update(name.encode())
        digest.update((path / name).read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    identity = {
        "path": str(path),
        "fingerprint": digest.hexdigest(),
        "files": records,
        "tokenizer_digest": hashes[TOKENIZER_FILE],
    }
    return {
        "config": config,
        "research_config": asdict(cfg),
        "manifest": manifest,
        "identity": identity,
        "has_mtp": False,
        "checkpoint": {
            "step": state.get("step"),
            "model_sha256": hashes["model.safetensors"],
            "semantic_ple_sha256": hashes.get(PLE_FILE),
            "state_sha256": hashes["state.json"],
        },
    }


def render_chat(messages, add_generation_prompt=True, **_kwargs):
    """The training corpus' turn format (``corpus.messages_text``)."""
    from ..experimental.hysparse2.corpus import text_content

    turns = []
    for message in messages:
        role = message.get("role")
        if role not in CHAT_ROLES:
            raise ValueError(
                f"HySparse2 training format has no {role!r} role; "
                "send user/assistant turns only"
            )
        if message.get("tool_calls"):
            raise ValueError("tools are not declared for this adapter")
        content = text_content(message.get("content", ""))
        if not isinstance(message.get("content", ""), (str, list)):
            raise ValueError("HySparse2 chat content must be text")
        turns.append(f"{role}: {content}")
    text = "\n\n".join(turns)
    if add_generation_prompt:
        text = (text + "\n\n" if text else "") + "assistant:"
    return text


class HySparse2Adapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    profile = "hysparse2-research-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._init(model_path, execution_policy=execution_policy)
        )

    def _init(self, model_path, *, execution_policy=None):
        policy = dict(execution_policy or {})
        dtype_name = policy.pop("dtype", "bfloat16")
        if policy:
            raise ValueError(f"HySparse2 adapter policy keys not supported: {sorted(policy)}")
        if dtype_name not in ("bfloat16", "float32"):
            raise ValueError("HySparse2 dtype must be bfloat16 or float32")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.checkpoint = artifact["checkpoint"]
        self.dtype_name = dtype_name
        require_process_numerics("the HySparse2 profile")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("the HySparse2 profile")
        path = Path(self.identity["path"])

        import mlx.core as mx
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from ..runtime.models.hysparse2 import Model
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        cfg = research_config(self.config)
        self.model = Model(cfg)
        self.load_receipt = self.model.load_research_checkpoint(path)
        if self.load_receipt["model_sha256"] != self.checkpoint["model_sha256"]:
            raise ValueError("HySparse2 weights differ from the export manifest")
        if dtype_name == "bfloat16":
            self.model.set_dtype(mx.bfloat16)
        self.model.eval()
        mx.eval(self.model.parameters())
        mx.clear_cache()
        # The cache namespace binds the attention/PLE layout fingerprint, the
        # explicit no-capsule identity and the exact PLE sidecar digest
        # (research config ``apcv2_identity``), plus this serving layout.
        apc = cfg.apcv2_identity(None, self.checkpoint["semantic_ple_sha256"])
        semantic = hashlib.sha256(
            json.dumps(apc["semantic_fingerprint"], separators=(",", ":")).encode()
        ).hexdigest()
        self.layout = (
            f"{DESCRIPTOR.cache_layout}|{apc['cache_layout_fingerprint']}|"
            f"semantic:{semantic}|dtype:{dtype_name}"
        )
        raw = Tokenizer.from_file(str(path / TOKENIZER_FILE))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw,
            eos_token="<eos>",
            bos_token="<bos>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        eos = raw.token_to_id("<eos>")
        if eos is None:
            raise ValueError("HySparse2 tokenizer has no <eos>")
        if raw.get_vocab_size() > cfg.vocab_size:
            raise ValueError("HySparse2 tokenizer vocabulary exceeds the model's")
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[eos],
            chat_template=render_chat,
        )
        self.max_context = min(SERVING_MAX_CONTEXT, int(cfg.max_context))

    def prompt_tokens(self, request):
        if "messages" in request:
            return self.tokenizer.encode(
                render_chat(request["messages"]), add_special_tokens=False
            )
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request):
        if "messages" in request:
            return render_chat(request["messages"])
        return request["prompt"]

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools

        if callable_tools(request) or request.get("enable_thinking"):
            raise ValueError("tools and reasoning are not declared for this adapter")
        stops = tuple(request.get("stop", ()) or ())
        if isinstance(request.get("stop"), str):
            stops = (request["stop"],)
        if "messages" in request:
            # The next training-format user turn ends the assistant answer.
            stops = stops + (CHAT_TURN_STOP,)
        return OutputParser(chat="messages" in request, thinking=False, tools=None, stops=stops)

    def diagnostics(self):
        return {
            "route": "ordinary",
            "qualification": "pending",
            "family": DESCRIPTOR.family,
            "checkpoint_step": self.checkpoint.get("step"),
            "model_sha256": self.checkpoint.get("model_sha256"),
            "dtype": self.dtype_name,
            "native_mtp": dict(DESCRIPTOR.metadata["native_mtp"]),
        }

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None
