"""Adapter-owned shared-root Flash-Next TensorFold tree target.

One traversal of the transformer layers over valid rows. Stateful QSA still
uses one ordinary attention reduction per node; projections, HC, MoE and GDN
are row-packed. This is not a claim of one Metal kernel for the whole model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import mlx.core as mx
import mlx.nn as nn

from ..runtime.cow_cache import snapshot_prompt_cache_descriptors
from ..runtime.models.qwen4_tree_gdn import ancestry


PROTOCOL = "mlx2.flash-next-tensorfold-tree.v1"


class TreeProjectionLinear(nn.QuantizedLinear):
    # Existing format compatibility marker only; this does not enter REV or
    # install its canary/projection machinery.
    _mlx2_row_exact_base = nn.QuantizedLinear

    def __call__(self, x):
        from ..runtime.models.flash_tensorfold_qmv import qmv_rows

        if x.dtype != mx.bfloat16:
            raise ValueError("Flash tree projection law requires bf16 activations")
        flat = x.reshape(-1, x.shape[-1])
        pieces = [qmv_rows(flat[start:start + 16], self)
                  for start in range(0, len(flat), 16)]
        return mx.concatenate(pieces, axis=0).reshape(*x.shape[:-1], -1)


def install_projection_law(model):
    """Cold explicit numerical law, BEFORE any target/APCv2 state exists.

    The TensorFold qmv is not stock MLX's qmv on this build. Both the ordinary
    reference and packed target must use this selected law, including wide
    prefill chunks. No weights are copied. HC's kernels and routed expert
    tables retain their own existing arithmetic.
    """
    from ..runtime.models.flash_tensorfold_qmv import eligible, hc_owned_projection_ids
    from ..runtime.models.qwen4_exp import Attention

    if getattr(model, "_mlx2_flash_tree_law", None) is not None or model._progressive_row_exact:
        raise ValueError("Flash tree projection law already bound or REV selected")
    hc_owned = hc_owned_projection_ids(model)
    selected = []
    for name, module in model.named_modules():
        if (type(module) is nn.QuantizedLinear and id(module) not in hc_owned
            and not any(part in name.split('.') for part in ('switch_mlp', 'shared_expert'))
            and eligible(module)):
            module.__class__ = TreeProjectionLinear
            selected.append(name)
    if not selected:
        raise ValueError("Flash tree projection law found no supported projections")
    # Concatenated stock tables would bypass the selected module arithmetic.
    for layer in model.language_model.model.layers:
        if layer.is_linear:
            layer.linear_attn.set_gdn_fused_inproj(False)
        elif type(layer.self_attn) is Attention:
            layer.self_attn.__class__ = TreeAttention
    material = ("flash_tree_qmv_q4q8g64_bf16_chunk16_v1", tuple(sorted(selected)))
    identity = {"law": material[0], "modules": len(selected),
                "sha256": hashlib.sha256(json.dumps(material).encode()).hexdigest()}
    object.__setattr__(model, "_mlx2_flash_tree_law", tuple(sorted(identity.items())))
    return identity


def install_ast_target_contract(model):
    """Select the tree-only contract, never install a REV fallback."""
    from ..runtime.models.qwen4_exp import Model

    if type(model) is not Model or not getattr(model, "_mlx2_flash_tree_law", None):
        raise ValueError("Flash AST requires its cold-bound TensorFold law")
    model.__class__ = TreeTargetModel


from ..runtime.models.qwen4_exp import Model as _TreeModel


class TreeTargetModel(_TreeModel):
    progressive_external_alternative_row_capacity = 16

    @property
    def progressive_external_exact_max_lanes(self):
        return 1

    @property
    def progressive_external_verify_protocol(self):
        from ..runtime.progressive_external_verify import PROGRESSIVE_TARGET_PROTOCOL_VERSION

        return PROGRESSIVE_TARGET_PROTOCOL_VERSION if getattr(self, "_mlx2_flash_tree_law", None) else None

    def progressive_external_target_rows(self, inputs, cache, capture_layers):
        # A capability/processor mismatch must not turn packed AST into a
        # serial/REV verifier. Ordinary terminal decode stays outside AST.
        raise ValueError("Flash TensorFold AST requires its admitted tree provider; no serial verifier fallback")


# Retain the whole ordinary QSA consumer; only its stock fused table is
# suppressed when the cold tree projection law is explicitly selected.
from ..runtime.models.qwen4_exp import Attention as _Attention


class TreeAttention(_Attention):
    def _fused_projection_table(self):
        return None


def _clone(cache):
    return snapshot_prompt_cache_descriptors([cache], require_committed=False)[0][0]


def _project(module, x, *, cpu_reference=False):
    if cpu_reference:
        return module(x)
    from ..runtime.models.flash_tensorfold_qmv import eligible, qmv_rows

    if eligible(module):
        return qmv_rows(x, module)
    # Scalar/small dense gates cannot use the packed quantized kernel.
    # Preserve their ordinary one-row reduction, not a stock multirow GEMM.
    flat = x.reshape(-1, x.shape[-1])
    return mx.concatenate([module(flat[i:i + 1]) for i in range(len(flat))], axis=0).reshape(
        *x.shape[:-1], -1)


def _norm(module, x):
    # Sequence width one preserves GroupRMSNorm's ordinary fast-norm law;
    # all independent rows still run in the same tensor operation.
    return module(x.reshape(-1, 1, x.shape[-1])).reshape(x.shape)


def _hc(module, x, *, cpu_reference=False):
    if cpu_reference:
        return module(x)
    from ..runtime.models import qwen4_hc_decode as HC

    if HC._cached_static_admission(module) is not None or HC.served_exp_refusal() is not None:
        raise ValueError("Flash tree HC lacks the ordinary one-row kernel law")
    plan = HC._build_plan(module, law=0)
    mixed, inject = HC.hc_decode_launch(module, x.reshape(-1, x.shape[-1]), plan=plan)
    mixed = mixed.reshape(*x.shape[:-1], module.hidden_size)
    return mixed if inject is None else (mixed, x, inject.reshape(*x.shape[:-1], module.hc_count))


def _moe(block, x, *, cpu_reference=False):
    if cpu_reference:
        return block(x)
    from ..runtime.models import qwen3_next as M
    from ..runtime.models import qwen4_moe_window as W

    if (block.shared_folded or block.sharding_group is not None or block.training
        or not block.fused_expert_kernel_enabled
        or getattr(block.switch_mlp, "routed_decode_mode", "off") != "gate_up_down_shared"
        or not M._shared_fold_ok(block, x)):
        raise ValueError("Flash tree requires the served shared-fold MoE law")
    gates = _project(block.gate, x)
    gate = _project(block.shared_expert_gate, x)
    inds, scores = M._stock_routing(block, gates)
    sw = block.switch_mlp
    result = W.shared_rows(x, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj,
                           block.shared_expert, gate)
    return result.reshape(x.shape)


def _ple(layer, x, tokens, base, paths, *, cpu_reference=False):
    p = layer.ple
    rows = len(paths)
    context = p.ple_embedding.context_len
    previous = base[3]
    if previous is None:
        previous = mx.full((1, context), p.ple_embedding.eos_token_id, mx.int64)
    histories = mx.concatenate([previous, mx.array([tokens], mx.int64)], axis=1)
    indices = []
    for path in paths:
        indices.append((tuple(range(context)) + tuple(context + i for i in path[:-1]))[-context:])
    incoming = mx.take(histories[0], mx.array(indices, mx.int32), axis=0)
    scratch = [None, None, None, incoming]
    embeddings = p.ple_embedding(mx.array(tokens, mx.int64)[:, None], scratch)
    hidden = x.reshape(rows, 1, -1)
    key = _norm(p.norm_key, _project(p.key_proj, embeddings, cpu_reference=cpu_reference))
    key = key.reshape(rows, 1, p.hc_count, p.hidden_size)
    value = _project(p.value_proj, embeddings, cpu_reference=cpu_reference)
    query = _norm(p.norm_query, hidden).reshape(rows, 1, p.hc_count, p.hidden_size)
    gate = mx.sum(key * query, axis=-1, keepdims=True) / p.hidden_size ** 0.5
    gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
    gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(rows, 1, -1)
    normed = _norm(p.norm_conv, gated)
    keep = p.short_conv_state_len
    old = base[2]
    if old is None:
        old = mx.zeros((1, keep, hidden.shape[-1]), hidden.dtype)
    full = mx.concatenate([old[0], normed[:, 0]], axis=0)
    windows = [(tuple(range(keep)) + tuple(keep + i for i in path))[-(keep + 1):]
               for path in paths]
    # Left-pad short lineages with the committed conv history, not siblings.
    gathered = mx.take(full, mx.array(windows, mx.int32), axis=0)
    out = gated + nn.silu(p.conv1d(gathered))
    return out.reshape(x.shape), gathered[:, -keep:], scratch[3]


class FlashNextTreeTransaction:
    provider_protocol = PROTOCOL

    def __init__(self, base, successors, paths, revision, metrics):
        self.base, self.successors, self.paths = base, successors, paths
        self.provider_revision_token = revision
        self.metrics = metrics
        self.closed = False

    def prepared_cache(self, path):
        if self.closed:
            raise ValueError("Flash tree transaction closed")
        path = tuple(path)
        if path and (any(type(row) is not int for row in path)
                     or not 0 <= path[-1] < len(self.paths) or self.paths[path[-1]] != path):
            raise ValueError("Flash tree publication is not one root-connected path")
        if not path:
            return [_clone(item) for item in self.base]
        row = path[-1]
        result = []
        for base, record in zip(self.base, self.successors, strict=True):
            if record[0] == "qsa":
                result.append(_clone(record[1][row]))
                continue
            cache = _clone(base)
            cache[0], cache[1] = record[1][row:row + 1], record[2][row:row + 1]
            if record[3] is not None:
                cache[2], cache[3] = record[3][row:row + 1], record[4][row:row + 1]
            cache.advance(len(path))
            result.append(cache)
        return result

    def commit_paths(self, paths):
        if len(paths) != 1:
            raise ValueError("Flash tree owns exactly one request")
        result = self.prepared_cache(paths[0])
        self.closed = True
        self.successors = []
        return [result]

    def abort(self):
        self.closed = True
        self.successors = []


class FlashNextTargetTreeExecutor:
    protocol = PROTOCOL
    maximum_cohort_width = 1

    def __init__(self, model, *, cpu_reference=False):
        self.model = model
        self.cpu_reference = cpu_reference
        root = Path(__file__).resolve().parents[3]
        paths = (
            "src/mlx2/adapters/flash_next_tree.py", "src/mlx2/runtime/models/qwen4_tree_gdn.py",
            "src/mlx2/runtime/models/qwen4_fused_gdn.py", "src/mlx2/runtime/models/qwen4_exp.py",
            "src/mlx2/runtime/models/flash_tensorfold_qmv.py", "src/mlx2/runtime/models/qwen4_hc_decode.py",
            "src/mlx2/runtime/models/qwen4_moe_window.py", "src/mlx2/runtime/models/qwen3_next.py",
            "src/mlx2/runtime/cow_cache.py",
            "src/mlx2/runtime/external_speculative.py", "src/mlx2/runtime/proposal_providers.py",
            "src/mlx2/runtime/drafters/native_mtp.py", "src/mlx2/runtime/ast_tree_verify.py",
            "src/mlx2/runtime/ast_tree_tiles.py", "src/mlx2/adapters/flash_next.py",
            "src/mlx2/adapters/external_draft_policy.py", "src/mlx2/adapters/ast_execution_profile.py",
        )
        self.source_identity = tuple((path, hashlib.sha256((root / path).read_bytes()).hexdigest())
                                     for path in paths)
        self.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        self.source_revision_token = hashlib.sha256(json.dumps(
            [PROTOCOL, self.source_revision, self.source_identity], separators=(",", ":")
        ).encode()).hexdigest()
        self.launches = 0
        self.ast_input_widths = ()
        self.ast_oracle = None
        self.projection_law = getattr(model, "_mlx2_flash_tree_law", None)

    @property
    def ast_tree_tile_rows(self):
        return 16 if self.ast_input_widths else 0

    def oracle_binding(self, *, artifact, capture_layers, target_binding, profile):
        """Cold direct-probe identity; no request tensors or mutable feedback."""
        return {
            "protocol": PROTOCOL, "source_revision": self.source_revision,
            "source_revision_token": self.source_revision_token,
            "source_identity": dict(self.source_identity), "artifact": artifact,
            "projection_law": dict(self.projection_law or ()),
            "capture_layers": list(capture_layers), "target_binding": target_binding,
            "profile_binding": profile.binding, "device": profile.device, "build": profile.build,
        }

    def with_ast_oracle(self, path, *, artifact, capture_layers, target_binding):
        document = json.loads(Path(path).read_text())
        if (set(document) != {"schema", "binding", "input_widths", "checks", "receipt_sha256"}
            or document["schema"] != "mlx2.flash-next-tree-oracle.v1"
            or document["input_widths"] != [16]
            or document["checks"] != {key: True for key in (
                "logits", "features", "prepared_state", "continuation", "isolation", "commit_abort")}
            or not isinstance(document["receipt_sha256"], str)
            or len(document["receipt_sha256"]) != 64):
            raise ValueError("Flash tree numerical oracle is incomplete")
        self.ast_oracle = (document, str(Path(path).resolve()), artifact,
                           tuple(capture_layers), target_binding)
        return self

    def ast_execution_backend(self, profile):
        if self.ast_oracle is None or self.cpu_reference:
            return None
        from ..runtime.ast_planner import AlternativeExecutionBackend, NumericalEquivalenceOracle

        document, path, artifact, layers, binding = self.ast_oracle
        expected = self.oracle_binding(artifact=artifact, capture_layers=layers,
                                       target_binding=binding, profile=profile)
        if document["binding"] != expected or not self.projection_law:
            raise ValueError("Flash tree numerical oracle does not match the live source/artifact/profile/law")
        self.ast_input_widths = (16,)
        return AlternativeExecutionBackend(
            backend="target_tree_single_session", profile_binding=profile.binding,
            device=profile.device, build=profile.build,
            source_revision_token=self.source_revision_token, input_widths=(16,),
            row_costs=profile.row_costs, launch_cost=profile.launch_cost,
            context_row_cost=profile.context_row_cost, context_windows=profile.context_windows,
            cost_authority="unmeasured_ordinary_prior", timing_boundary="target_successor",
            oracle=NumericalEquivalenceOracle(profile.binding, profile.device, profile.build,
                                              self.source_revision_token, path, True),
        )

    def validate_cohort(self, width, configured_limit):
        if width != 1 or configured_limit < 1:
            raise ValueError("Flash tree supports one request owner")

    def validate_revision(self):
        if self.model.external_feature_source != "mtp_seed" or self.model._progressive_row_exact:
            raise ValueError("Flash tree requires native MTP seed without the REV target")
        if self.projection_law != getattr(self.model, "_mlx2_flash_tree_law", None):
            raise ValueError("Flash tree projection law changed")

    def forward(self, model, inputs, parents, cache, layers, *, cached=True):
        self.validate_revision()
        if model is not self.model:
            raise ValueError("Flash tree model changed")
        if self.ast_oracle is not None and (len(inputs) not in self.ast_input_widths
                                           or tuple(layers) != self.ast_oracle[3]):
            raise ValueError("Flash tree launch is outside its admitted geometry")
        paths = ancestry(parents)
        if (len(inputs) != len(paths) or len(inputs) < 2
            or any(type(t) is not int or not 0 <= t < model.language_model.args.vocab_size for t in inputs)):
            raise ValueError("Flash tree token topology mismatch")
        trunk = model.language_model.model
        if (any(type(layer) is not int for layer in layers)
            or tuple(layers) != (len(trunk.layers) - 1,) or len(cache) != len(trunk.layers)):
            raise ValueError("Flash tree requires the exact native-MTP tap and cache schema")
        if not self.cpu_reference and mx.default_device() != mx.gpu:
            raise ValueError("Flash tree requires Metal; CPU reference is explicit only")
        if not self.cpu_reference and getattr(model, "_mlx2_flash_tree_law", None) is None:
            raise ValueError("Flash tree requires its cold-bound projection law")
        base = [_clone(item) for item in cache]
        x = mx.tile(trunk.embed_tokens(mx.array([inputs])), (1, 1, trunk.args.hc_count))
        records = []
        metrics = {"rows": len(inputs), "padding": 0, "layer_calls": 0,
                   "gdn_tree_calls": 0, "qsa_node_calls": 0, "moe_packed_calls": 0,
                   "serial_model_calls": 0, "rev_calls": 0}
        from ..runtime.models import qwen4_exp as Q
        for layer, state in zip(trunk.layers, base, strict=True):
            ple_conv = ple_tokens = None
            if layer.ple is not None:
                addition, ple_conv, ple_tokens = _ple(layer, x, inputs, state, paths,
                                                     cpu_reference=self.cpu_reference)
                x = x + addition
            mixed, residual, inject = _hc(layer.attn_hyper_connection, x, cpu_reference=self.cpu_reference)
            if layer.is_linear:
                gdn = layer.linear_attn
                qkv, z, b, a = tuple(_project(getattr(gdn, name), mixed, cpu_reference=self.cpu_reference)
                                     for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"))
                if self.cpu_reference:
                    node_states, output = [], []
                    for row, parent in enumerate(parents):
                        local = _clone(state if parent < 0 else node_states[parent])
                        out = gdn._recurrent_core(qkv[:, row:row + 1], z[:, row:row + 1],
                                                 b[:, row:row + 1], a[:, row:row + 1], None,
                                                 local, dtype=mixed.dtype)
                        node_states.append(local)
                        output.append(out.reshape(1, 1, -1))
                    out = mx.concatenate(output, axis=1)
                    conv = mx.concatenate([item[0] for item in node_states], axis=0)
                    recurrent = mx.concatenate([item[1] for item in node_states], axis=0)
                else:
                    from ..runtime.models.qwen4_tree_gdn import tree
                    out, conv, recurrent = tree(gdn, qkv, z, b, a, state, parents)
                branch = _project(gdn.out_proj, out, cpu_reference=self.cpu_reference)
                records.append(("gdn", conv, recurrent, ple_conv, ple_tokens))
                metrics["gdn_tree_calls"] += 1
            else:
                attn = layer.self_attn
                projections = tuple(_project(module, mixed, cpu_reference=self.cpu_reference)
                                    for module in (attn.q_proj, attn.k_proj, attn.v_proj,
                                                   attn.indexer.index_qk_proj))
                nodes, outputs = [], []
                for row, parent in enumerate(parents):
                    local = _clone(state if parent < 0 else nodes[parent])
                    out = attn(mixed[:, row:row + 1], None, local,
                               _projected=tuple(value[:, row:row + 1] for value in projections),
                               _return_pre_o=True)
                    # Keep the ordinary QSA output gate outside the reduction.
                    if isinstance(out, tuple):
                        value, gate = out
                        value = value.transpose(0, 2, 1, 3).reshape(1, 1, -1) if value.ndim == 4 else value
                        out = value * mx.sigmoid(gate)
                    outputs.append(out)
                    nodes.append(local)
                branch = _project(attn.o_proj, mx.concatenate(outputs, axis=1),
                                  cpu_reference=self.cpu_reference)
                records.append(("qsa", nodes))
                metrics["qsa_node_calls"] += len(inputs)
            x = Q._apply_inject(residual, branch, inject)
            mixed, residual, inject = _hc(layer.mlp_hyper_connection, x, cpu_reference=self.cpu_reference)
            branch = _moe(layer.mlp, mixed, cpu_reference=self.cpu_reference)
            x = Q._apply_inject(residual, branch, inject)
            metrics["moe_packed_calls"] += 1
            metrics["layer_calls"] += 1
        mixed = _hc(trunk.hyper_connection_mixer, x, cpu_reference=self.cpu_reference)
        logits = _project(model.language_model.lm_head, mixed, cpu_reference=self.cpu_reference)
        transaction = FlashNextTreeTransaction(base, records, paths, self.source_revision_token, metrics)
        self.launches += 1
        return logits, x, transaction

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit):
        self.validate_cohort(len(inputs), configured_limit)
        logits, features, transaction = self.forward(model, inputs[0], parents[0], caches[0], layers, cached=cached)
        return logits, features, transaction

    def receipt(self):
        return {"protocol": self.protocol, "source_revision_token": self.source_revision_token,
                "source_revision": self.source_revision,
                "launches": self.launches, "qualified": False, "defaulted": False,
                "cpu_reference": self.cpu_reference}

    def dispatch_metrics(self, width, transaction, *, cached):
        from ..runtime.target_tree_executor import TargetTreeDispatchMetrics

        if (width != 1 or transaction.closed or transaction.provider_protocol != self.protocol
            or transaction.provider_revision_token != self.source_revision_token):
            raise ValueError("Flash tree dispatch metrics require the matching live transaction")
        return TargetTreeDispatchMetrics(1, False, True, cached, self.protocol)
