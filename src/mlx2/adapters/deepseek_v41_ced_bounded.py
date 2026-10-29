# SPDX-License-Identifier: MIT
"""Default-off bounded fresh-prefix CED executor, with a distinct profile.

Dependency-complete execution is not a claim of Source numerical equivalence:
pruning changes dense/expert row geometry. Prefixes above1024 and resumed/all-
row capabilities return an explicit ordinary-required decision. No central
adapter policy, cache engine or serving default is installed here.
Provenance: provenance/deepseek-v41-ced-bounded-executor.json.
"""
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

PROFILE = 'source_ced_bounded_fresh_suffix_v1'
MAX_PROMPT_ROWS = 1024


@dataclass(frozen=True)
class OrdinaryRequired:
    reason: str
    numerical_profile: str = PROFILE
    ordinary_required: bool = True
    mutated_cache: bool = False


@dataclass(frozen=True)
class CedResult:
    logits: object
    cache: tuple
    receipt: dict
    model: object


class CedExecutionError(RuntimeError):
    """Private next-state remains retained until its owner drains GPU work."""
    def __init__(self, message, *, retained_cache):
        super().__init__(message)
        self.retained_cache = retained_cache
        self.published = False


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    names = ('adapters/deepseek_v41_ced_bounded.py',
             'runtime/models/deepseek_v41_ced_plan.py',
             'runtime/models/deepseek_v41.py', 'runtime/models/deepseek_v41_cache.py')
    return hashlib.sha256(''.join(name+':'+hashlib.sha256((root/name).read_bytes()).hexdigest()
                                  for name in names).encode()).hexdigest()


class BoundedCedExecutor:
    """One adapter-owned candidate over a stable ordinary model/weight owner."""
    def __init__(self, model, *, encoder_layers, reference_chunk_rows, numerical_profile):
        if numerical_profile != PROFILE:
            raise ValueError('Bounded CED requires its explicit numerical profile')
        total = model.args.num_hidden_layers
        if type(encoder_layers) is not int or not 0 < encoder_layers < total:
            raise ValueError('Bounded CED requires a nonempty encoder and decoder')
        if type(reference_chunk_rows) is not int or not 1 <= reference_chunk_rows <= 1024:
            raise ValueError('Bounded CED requires an explicit reference row alignment')
        self.base = model
        self.encoder_layers = encoder_layers
        self.reference_chunk_rows = reference_chunk_rows
        self.implementation = implementation_identity()
        binding = dict(profile=PROFILE, implementation=self.implementation,
                       encoder_layers=encoder_layers, reference_chunk_rows=reference_chunk_rows)
        self.identity = model.identity+':ced='+hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        self._model = None
        self._poisoned = False
        self._retained_failure = ()

    def decision(self, inputs, *, cache=None, prompt_logprobs=False, all_hidden_states=False, output_rows=1):
        """CPU geometry/capability decision; no array values are inspected."""
        if self._poisoned:
            return OrdinaryRequired('prior_failed_execution_requires_explicit_drain')
        c, weights = self.base.args, self.base.weights
        if prompt_logprobs or all_hidden_states:
            return OrdinaryRequired('all_row_capability_requires_ordinary')
        if cache is not None:
            return OrdinaryRequired('caller_cache_or_resumed_prefix_requires_ordinary')
        if (getattr(inputs, 'ndim', None) != 2 or inputs.shape[0] != 1
                or not 1 <= inputs.shape[1] <= MAX_PROMPT_ROWS):
            return OrdinaryRequired('fresh_B1_prefix_1_to1024_required')
        if type(output_rows) is not int or not 1 <= output_rows <= inputs.shape[1]:
            return OrdinaryRequired('requested_output_extent_unsupported')
        if any(i > self.encoder_layers for i in c.kv_source_layer_ids):
            return OrdinaryRequired('deeper_decoder_global_pool_owner_unsupported')
        if any(i >= self.encoder_layers for i in c.engram_layer_ids):
            return OrdinaryRequired('decoder_engram_unsupported')
        if self.encoder_layers in c.kv_source_layer_ids and c.compress_ratios[self.encoder_layers] != 1:
            return OrdinaryRequired('boundary_pool_requires_ratio1')
        if getattr(weights, 'source_row_packet_policy', 'reference') != 'reference':
            return OrdinaryRequired('row_packet_frame_geometry_unsupported')
        if getattr(weights, 'source_frame_reuse_policy', 'reference') != 'reference':
            return OrdinaryRequired('prefill_frame_geometry_unsupported')
        if callable(getattr(weights, 'evaluate_block', None)):
            return OrdinaryRequired('transactional_block_executor_unsupported')
        if (getattr(weights, 'dense_prefill_policy', 'reference') != 'reference'
                or getattr(weights, 'expert_prefill_policy', 'reference') != 'reference'):
            return OrdinaryRequired('generic_prefill_context_unsupported')
        if getattr(weights, 'expert_io_policy', 'blocking') != 'blocking':
            return OrdinaryRequired('asynchronous_expert_tickets_unsupported')
        if (getattr(weights, 'source_submission_policy', 'reference') == 'shared_first_v1'
                and getattr(weights, 'source_ffn_input_policy', 'reference') != 'reference'):
            return OrdinaryRequired('shared_first_routed_token_order_unsupported')
        if (getattr(self.base, '_weights_parked', False)
                or getattr(self.base, '_weight_executions', 0)
                or getattr(weights, '_source_current_frame', None) is not None):
            return OrdinaryRequired('weight_owner_not_quiescent')
        return None

    def _view(self):
        if self._model is None:
            from ..runtime.models.deepseek_v41 import Block
            model = copy(self.base)
            model.identity = self.identity
            model.scheduled_kv_calls = {}
            model.scheduled_hc_calls = {}
            model._weight_executions = 0
            model.layers = [Block(model, i) for i in range(model.args.num_hidden_layers)]
            self._model = model
        return self._model

    @staticmethod
    def _crop_shared(shared, previous_first, first):
        if first < previous_first:
            raise ValueError('CED query interval cannot move backward')
        if first == previous_first:
            return
        trim = first-previous_first
        for name in ('indices', 'selected_mask', 'candidates'):
            value = shared.get(name)
            if value is not None:
                if value.shape[1] < trim:
                    raise ValueError('CED shared query metadata does not cover required suffix')
                shared[name] = value[:, trim:]
        shared.pop('_decode_selected_kv', None)
        # The old marker binds identity, length and absolute start. A future
        # index-source layer will issue a new one; a cropped view cannot reuse it.
        shared.pop('_all_retained_chronological', None)

    @staticmethod
    def _ffn(block, h, x, pre, post, comb):
        import mlx.core as mx
        weights, c = block.w, block.c
        indices, scores = block.router(x)
        factory = getattr(weights, 'ffn_input_token', None)
        token = factory(x, block.prefix[:-1]) if factory is not None else None
        def routed():
            return weights.moe(token.flat_input if token is not None else x.reshape(-1, c.hidden_size),
                indices.reshape(-1, c.num_experts_per_tok), scores.reshape(-1, c.num_experts_per_tok),
                block.prefix[:-1], **({'ffn_input_token': token} if token is not None else {})).reshape(x.shape)
        def shared():
            return block.shared_expert(x, **({'ffn_input_token': token} if token is not None else {}))
        try:
            if getattr(weights, 'source_submission_policy', 'reference') == 'shared_first_v1':
                from .deepseek_v41_source_submission import shared_before_route
                branch, common = shared_before_route(shared, routed, submit=mx.async_eval,
                                                     counters=weights.scheduled_dense_calls)
            else:
                branch, common = routed(), shared()
            return block.finish(h, branch, common, post, comb), pre
        finally:
            if token is not None:
                token.close()

    def prefill(self, inputs, *, cache=None, output_rows=1, prompt_logprobs=False,
                all_hidden_states=False, check_live=None):
        """Return a completed private next-state or an ordinary-required decision.

        This entry never mutates or publishes a caller cache. GPU callers must
        supply their existing ownership/live callback, returning True. Final
        model/cache numerics remain unqualified until actual Source gates pass.
        """
        decision = self.decision(inputs, cache=cache, output_rows=output_rows,
            prompt_logprobs=prompt_logprobs, all_hidden_states=all_hidden_states)
        if decision is not None:
            return decision
        import mlx.core as mx
        import numpy as np
        if inputs.dtype not in (mx.int32, mx.uint32, mx.int64):
            return OrdinaryRequired('integer_prompt_tokens_required')
        if mx.default_device() == mx.gpu and (not callable(check_live) or check_live() is not True):
            return OrdinaryRequired('GPU_owner_callback_required')
        if callable(check_live) and check_live() is not True:
            return OrdinaryRequired('live_owner_refused')
        from ..runtime.models.deepseek_v41_ced_plan import build_plan
        c = self.base.args
        plan = build_plan(prompt_tokens=inputs.shape[1], encoder_layers=self.encoder_layers,
            total_layers=c.num_hidden_layers, sliding_window=c.sliding_window,
            reference_chunk_rows=self.reference_chunk_rows, output_rows=output_rows,
            pool_source_layers=c.kv_source_layer_ids, index_source_layers=c.index_source_layer_ids,
            engram_layers=c.engram_layer_ids)
        steps = plan.proposed_decoder_steps
        if not steps:
            return OrdinaryRequired(plan.reason)
        self.base._weight_executions += 1
        model, work, stream = None, None, None
        started_context = False
        failure_error = None
        stages = []
        try:
            model = self._view()
            model._weight_executions += 1
            work, ids = model.validate_forward(inputs, None, 0)
            stream = getattr(model.weights, 'expert_stream', None)
            hashes = tail = None
            if stream is not None and getattr(stream, 'atlas', None) is not None:
                from ..runtime.expert_stream_atlas import ExpertStreamAtlas
                stream.begin_context(ExpertStreamAtlas.keys(model.hasher.token_map[ids[0]],
                                                           np.asarray(work[0].hash_tail)[::-1]))
                started_context = True
            if model.hasher is not None:
                hashes, tail = model.hasher(ids[0], np.asarray(work[0].hash_tail))
                for index, layer in enumerate(c.engram_layer_ids):
                    model.engram_tables[layer].prefetch(hashes[:, index])
            for state in work:
                state._in_forward = True
            execution = getattr(model.weights, 'execution_scope', None)
            with execution() if execution is not None else nullcontext():
                with model._weight_scope():
                    x = model.round(model.weights.embedding(inputs.astype(mx.uint32)))
                    h = mx.broadcast_to(x[:, :, None], (1, inputs.shape[1], c.hc_mult, c.hidden_size))
                    pre = mx.broadcast_to(mx.array([1.0]+[0.0]*(c.hc_mult-1)), h.shape[:-1])
                    model._weight_eval(h, pre)
                shared = {}
                for layer in range(self.encoder_layers):
                    if callable(check_live) and check_live() is not True:
                        raise RuntimeError('CED live owner refused between encoder layers')
                    with model._weight_scope():
                        if layer in c.engram_layer_ids:
                            h = model.engram(h, layer, hashes[:, c.engram_layer_ids.index(layer)])
                        h, pre = model.layers[layer](h, pre, work[layer], work, shared)
                        model._weight_eval(h, pre, [state.state for state in work], shared)
                    stages.append(dict(layer=layer, phase='encoder', input_first=0,
                                       attention_rows=inputs.shape[1], ffn_rows=inputs.shape[1]))
                active_first = shared_first = 0
                for step in steps:
                    if callable(check_live) and check_live() is not True:
                        raise RuntimeError('CED live owner refused between decoder layers')
                    layer, block = step.layer, model.layers[step.layer]
                    # The first decoder owns the boundary pool in canonical
                    # Source. Its complete attention half publishes every KV/
                    # index row before pruning any FFN input or hidden/pre row.
                    first = 0 if layer == self.encoder_layers else step.required_input.first
                    trim = first-active_first
                    if trim < 0:
                        raise ValueError('CED preceding output does not cover warm input')
                    h, pre = h[:, trim:], pre[:, trim:]
                    self._crop_shared(shared, shared_first, first)
                    shared_first = first
                    state = work[layer]
                    state.offset = first
                    with model._weight_scope():
                        h, x, next_pre, post, comb = block.attention_half(h, pre, state, work, shared)
                        keep = step.aligned_output.first-first
                        if keep < 0 or keep >= x.shape[1]:
                            raise ValueError('CED kept output lies outside computed attention rows')
                        h, x, next_pre = h[:, keep:], x[:, keep:], next_pre[:, keep:]
                        post, comb = post[:, keep:], comb[:, keep:]
                        h, pre = self._ffn(block, h, x, next_pre, post, comb)
                        model._weight_eval(h, pre, [row.state for row in work], shared)
                    stages.append(dict(layer=layer, phase='decoder', input_first=first,
                        attention_rows=inputs.shape[1]-first, discarded_warm_outputs=keep,
                        ffn_rows=h.shape[1], output_first=step.aligned_output.first,
                        full_boundary_pool_prepared=layer == self.encoder_layers and state.source))
                    active_first = step.aligned_output.first
                if tail is not None:
                    work[0].hash_tail = mx.array(tail, mx.int32)
                with model._weight_scope():
                    hidden = model.norm(model.collapse(h, pre), model.norm_weight)[:, -output_rows:]
                    logits = model.linear(hidden, 'output.weight', rounded=False)
                    model._weight_eval(logits, hidden, [row.state for row in work], shared)
                    # A generic ordinary fixture may have no evaluator hook.
                    # All result/state roots complete before any publication.
                    mx.eval(logits, [row.state for row in work])
            if started_context:
                started_context = False
                stream.end_context()
            for state in work:
                state.validate_state()
                state._in_forward = False
                if state.pending:
                    raise ValueError('CED completed a pending cache frontier')
            receipt = dict(profile=PROFILE, implementation_sha256=self.implementation,
                model_identity=self.identity, prompt_rows=inputs.shape[1], output_rows=output_rows,
                source_row_limit=MAX_PROMPT_ROWS, fresh_prefix=True, stages=stages,
                dependency_plan=plan.receipt(), qualified=False, default_selected=False,
                numerical_contract='existing operators/owning boundaries; changed row geometry; distinct unqualified profile',
                final_offsets=[state.offset for state in work], caller_cache_mutated=False,
                final_cache_complete=True, source_numerical_equivalence_claimed=False)
            return CedResult(logits, tuple(work), receipt, model)
        except BaseException as error:
            self._poisoned = True
            self._retained_failure = tuple(work or ())
            failure_error = CedExecutionError('Bounded CED failed before publishing private next-state: '+str(error),
                                             retained_cache=self._retained_failure)
            raise failure_error from error
        finally:
            try:
                if started_context and stream is not None:
                    try:
                        stream.end_context()
                    except BaseException as cleanup_error:
                        if failure_error is None:
                            raise
                        failure_error.context_cleanup_error = repr(cleanup_error)
            finally:
                if model is not None:
                    model._weight_executions -= 1
                self.base._weight_executions -= 1

    def drain_failed(self, fence):
        """Owner-only cleanup; a failed drain retains every private state root."""
        if not callable(fence):
            raise ValueError('CED failed-state cleanup requires an explicit completion fence')
        fence(*(value for state in self._retained_failure for value in state.state if value is not None))
        for state in self._retained_failure:
            state.close()
        self._retained_failure = ()
        # This executor remains poisoned. Reconstruct it after inspecting the
        # failed owner/provider rather than silently retrying its state.
