"""Kimi Linear sweep family: whole-layer streaming (layer_swap.py) of the mlx2 Kimi Linear model.

Artifacts (``mlx2.adapters.kimi_linear`` owns the math and the contracts):
- every layout the adapter stages (the REAP 35B hub BF16, the 48B MLX affine q8): its own ``inspect_artifact``
  (topology, the complete tensor index, shard dtypes, tokenizer-bound fingerprint), so a map's fingerprint is the
  resident capture's;
- the original 48B hub BF16 (``moonshotai/Kimi-Linear-48B-A3B-Instruct`` layout, 256 experts, ~92 GB: it does not
  fit under the 80 GB cap): the adapter does not stage it, because its hub layout pins the REAP expert count (180).
  The family admits it only when it is the adapter's hub contract with 256 experts in every other respect: the
  adapter's ``validate_config`` with the expert count substituted, the adapter's required-tensor set with the
  expert range widened, the adapter's shard dtype law. The fingerprint is mlx2's artifact identity
  (``thinking_calibration.artifact_identity``), as for any artifact without an adapter.

The model is built exactly as ``KimiLinearAdapter._load`` builds it: ``runtime.models.kimi_linear.Model``, MLX affine
quantization of the modules whose ``.scales`` the checkpoint carries, the adapter's default policy
(``parse_policy({})``, ``configure``), the process numerics it requires and its stock MoE claim. Each layer's tensors
(every expert) go through the model's own ``sanitize``. Refused: ``fused_projections`` (it precomputes joined weights
at configure time), MoE NAX gather and TensorFold (they need the affine banks at load). A policy override (``--family-opt``; the CPU tests use the reference KDA and
expanded MLA, which run on the CPU) is recorded in the receipt; the resident capture runs the defaults.

Units: the carve host's sites (research/amoeba/carve/units.py): layer 0 dense ``act``; layers 1..26 MoE
(``sigmoid_shared``/``ling`` routing) in neuron units ``e*F+f`` (F = 1024) or whole experts.
"""
from __future__ import annotations

import json
from pathlib import Path

from .layer_swap import LayerSwapFamily

HUB48_EXPERTS = 256         # the original 48B hub release; the adapter stages only the REAP (180) hub layout
_LAYER = "model.layers."
ADAPTERS = ("KimiLinearAdapter", None)   # plan.py: the adapters this family builds (None = no adapter: hub48)


def _config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())


def _hub48_required(full, experts):
    from mlx2.adapters import kimi_linear as K
    base = {k for k in K._required("hub_bf16", full) if ".block_sparse_moe.experts." not in k}
    return base | {f"model.layers.{i}.block_sparse_moe.experts.{e}.{n}.weight"
                   for i in range(1, 27) for e in range(experts) for n in ("w1", "w2", "w3")}


def inspect(path):
    """(artifact dict, how): the adapter's own inspection when it stages the artifact, else the 48B hub contract."""
    from mlx2.adapters import kimi_linear as K
    path = Path(path).expanduser().resolve()
    try:
        return K.inspect_artifact(path), "adapter"
    except (ValueError, OSError) as staged_error:
        cfg = _config(path)
        experts = cfg.get("num_experts")
        # only the 256-expert release falls back: a 180-expert (REAP) artifact the adapter refuses stays refused
        if cfg.get("quantization") or cfg.get("quantization_config") or experts != HUB48_EXPERTS:
            raise SystemExit(f"kimi_linear sweep family: {path}: {staged_error}; and it is not the 256-expert hub "
                             "BF16 layout either") from None
        try:
            layout = K.validate_config({**cfg, "num_experts": K.LAYOUTS["hub_bf16"]["num_experts"]})
        except ValueError as e:
            raise SystemExit(f"kimi_linear sweep family: {path}: {e}") from None
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        full = set(cfg["linear_attn_config"]["full_attn_layers"])
        if set(index) != _hub48_required(full, experts):
            raise SystemExit(f"kimi_linear sweep family: {path}: tensor index differs from the {experts}-expert hub "
                             "topology")
        from safetensors import safe_open

        from mlx2.adapters.artifact_paths import shard_within_artifact
        observed = set()
        for shard in sorted(set(index.values())):
            if Path(shard).name != shard or not shard_within_artifact(path, (path / shard).resolve()):
                raise SystemExit(f"kimi_linear sweep family: shard {shard!r} escapes the artifact {path}")
            with safe_open(path / shard, framework="np") as f:
                for k in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                    if index.get(k) != shard:
                        raise SystemExit(f"kimi_linear sweep family: {k} is in {shard} but the index says "
                                         f"{index.get(k)}")
                    if f.get_slice(k).get_dtype() != K._expected_dtype(layout, k, set()):
                        raise SystemExit(f"kimi_linear sweep family: {k} dtype differs from the hub BF16 contract")
                    observed.add(k)
        if observed != set(index):
            raise SystemExit(f"kimi_linear sweep family: {path}: shard headers lack indexed tensors "
                             f"{sorted(set(index) - observed)[:4]}")
        for name in ("tiktoken.model", "special_tokens_map.json", "tokenizer_config.json", "chat_template.jinja"):
            if not (path / name).is_file():
                raise SystemExit(f"kimi_linear sweep family: {path}: missing {name}")
        return {"config": cfg, "weight_map": index, "layout": layout,
                "identity": {"path": str(path), "fingerprint": checkpoint_fingerprint(path)}}, "hub48"


def checkpoint_fingerprint(path) -> str:
    """The adapter's fingerprint when it stages the artifact, else mlx2's artifact identity (inspection.py calls
    this for a checkpoint with no adapter)."""
    from mlx2.adapters import kimi_linear as K
    path = Path(path).expanduser().resolve()
    try:
        return K.inspect_artifact(path)["identity"]["fingerprint"]
    except (ValueError, OSError):
        from mlx2.thinking_calibration import artifact_identity
        return artifact_identity(path)


class KimiSpec:
    def __init__(self, path, artifact, how, *, policy=None, tokenizer=None):
        from mlx2.adapters import kimi_linear as K
        from mlx2.experimental.amoeba.inspection import artifact_name
        self.path = Path(artifact["identity"]["path"])
        # the name the resident capture and the inspection give the path as passed (a symlink, an HF snapshot)
        self.name = artifact_name(Path(path).expanduser()).split("/")[-1]
        self.fingerprint = artifact["identity"]["fingerprint"]
        self.config = artifact["config"]
        self.weight_map = artifact["weight_map"]
        self.layout = artifact["layout"]
        self.how = how
        if isinstance(policy, str):                  # --family-opt policy={"kda": "reference"} arrives as text
            policy = json.loads(policy)
        self.policy_override = dict(policy or {})
        try:
            resolved = K.parse_policy(self.policy_override)
        except ValueError as e:
            raise SystemExit(f"kimi_linear sweep family: {e}") from None
        # judged on the RESOLVED policy, so a changed adapter default cannot slip past
        if resolved["fused_projections"] or resolved["tensorfold_qmv"] or resolved["moe_nax"] != "off":
            raise SystemExit("kimi_linear sweep family: fused_projections, tensorfold_qmv and moe_nax precompute or "
                             "re-tile weights at load; a streamed layer cannot carry them (refused)")
        self._tokenizer = tokenizer
        self._scaled = None
        self._L = int(self.config["num_hidden_layers"])

    def layer_of(self, key):
        if key.startswith(_LAYER):
            i = int(key[len(_LAYER):].split(".", 1)[0])
            return i if 0 <= i < self._L else None
        return "resident"

    def skeleton(self):
        """KimiLinearAdapter._load's construction, under the same guarded process-global claim."""
        from mlx2.adapters.process_globals import guarded_construction
        out = {}
        guarded_construction(self, lambda: out.update(built=self._build()))
        return out["built"]

    def _build(self):
        from mlx import nn

        from mlx2.adapters import kimi_linear as K
        from mlx2.adapters.process_globals import claim_stock_moe
        from mlx2.process_env import require_process_numerics
        from mlx2.runtime.models.kimi_linear import Model, ModelArgs
        require_process_numerics("Kimi Linear ordinary (sweep family)")
        self.policy = K.parse_policy(self.policy_override)
        claim_stock_moe(self, "the kimi_linear sweep family")
        model = Model(ModelArgs.from_dict(self.config))
        if self.layout == "mlx_affine":
            scaled = {k[:-len(".scales")] for k in model.sanitize(
                {k: _tiny() for k in self.weight_map if k.endswith(".scales")})}
            nn.quantize(model, **K.QUANTIZATION, class_predicate=lambda name, module:
                        hasattr(module, "to_quantized") and name in scaled)
            self._scaled = len(scaled)
        model.eval()
        model.configure(self.policy)
        self._model = model
        return model, self._tokenizer or self._load_tokenizer()

    def _load_tokenizer(self):
        from mlx2.runtime.kimi_tokenizer import TikTokenTokenizer
        from mlx2.runtime.tokenizer_utils import TokenizerWrapper
        tok = TikTokenTokenizer.from_pretrained(self.path, local_files_only=True)
        tok.chat_template = (self.path / "chat_template.jinja").read_text()
        return TokenizerWrapper(tok, eos_token_ids=[163586])

    def transform(self, raw):
        # the model's own sanitize: renames, per-expert stacking (hub layouts), the HF depthwise conv layout
        return self._model.sanitize(raw)

    def layer_scope(self, ids, cache):
        from mlx2.runtime.models import moe_nax_gather
        return moe_nax_gather.forward_scope(ids, cache)

    def receipt(self):
        return {"family": "kimi_linear", "layout": self.layout, "admitted_by": self.how,
                "policy": self.policy,
                "policy_override": self.policy_override or None, "quantized_modules": self._scaled,
                "experts": self.config.get("num_experts")}


def _tiny():
    import mlx.core as mx
    return mx.zeros((1,), mx.uint8)


def load(model_dir, *, moe_units="neuron", log=print, policy=None, tokenizer=None):
    artifact, how = inspect(model_dir)
    return LayerSwapFamily(KimiSpec(model_dir, artifact, how, policy=policy, tokenizer=tokenizer),
                           moe_units=moe_units, log=log)
