# SPDX-License-Identifier: MIT
"""Official DeepSeek V4.1 Source, ordinary text B1 and APCv2, unqualified."""
from dataclasses import replace
import hashlib
import json
import math
import os

import numpy as np

from .deepseek_v41_mlx import DeepSeekV41MLXAdapter, DESCRIPTOR as NATIVE_DESCRIPTOR, admission_plan, execution_policy
from .deepseek_v41_source_artifact import MODEL_TYPE, inspect_artifact
from .deepseek_v41_source_math import POLICIES

NUMERICS_REVISION = "mlx2.v41-official-mxfp4-fp8-act32-bf16.v1"
DESCRIPTOR = replace(NATIVE_DESCRIPTOR, model_type=MODEL_TYPE, variant="flash-1017-official-source-text",
    metadata={"execution": "mlx2.adapters.deepseek_v41_source.DeepSeekV41SourceAdapter",
              "qualification": "unqualified", "scope": "official checkpoint; ordinary B1 text; streamed MXFP4 experts and split-plane Engram; exact APCv2 within this numerical profile"})

SOURCE_POLICY_CHOICES = {
    'deepseek_source_frame_reuse': ('reference', 'prefill_v1'),
    'deepseek_source_native_mx': ('reference', 'original_pp_half_v1'),
    'deepseek_source_prefill_down': ('reference', 'q1_m1_owner_store', 'q1_tiny_batch_v1'),
    'deepseek_source_ffn_input': ('reference', 'shared_routed_qat_reuse'),
    'deepseek_source_woa_pp': ('reference', 'native_bf16_grouped_pp_v1'),
    'deepseek_source_fp8_pp': ('reference', 'native_bf16_qb_wob_v1'),
    'deepseek_source_hc_boundary': ('reference', 'expand_collapse_seq'),
    'deepseek_source_q_b_prefill': ('reference', 'tile64x32_raw'),
    'deepseek_source_qakv': ('reference', 'qat_reuse'),
    'deepseek_source_attention_math': ('reference', 'all_retained_broadcast', 'omlx_nax_bf16_online_v1'),
    'deepseek_source_prefill_batch': ('reference', 'gate_up', 'gate_up_down'),
    'deepseek_source_prefill_wide_dtype': ('reference', 'bf16_p1'),
    'deepseek_source_prefill_wide_math': ('reference', 'legacy_tiles', 'legacy_tiles_16x64', 'legacy_tiles_fused', 'legacy_tiles_fused_16x64', 'nax_k16_ordered_f32_v1'),
    'deepseek_source_dense_qat': ('reference', 'fused_fp32'),
    'deepseek_source_bf16_loads': ('reference', 'woa_f32', 'direct_f32'),
    'deepseek_source_completion': ('reference', 'bounded_2'),
    'deepseek_source_hc_coefficients': ('reference', 'fused_fp32'),
    'deepseek_index_policy': ('reference', 'finite_all_retained'),
    'deepseek_selected_kv_policy': ('reference', 'decode_reuse'),
}

def source_policy(value=None):
    value=dict(value or {})
    expert=value.pop('deepseek_source_experts',None)
    expert_math=value.pop('deepseek_source_expert_math',None)
    source_options={key:value.pop(key) for key in SOURCE_POLICY_CHOICES if key in value}
    for key,choice in source_options.items():
        if type(choice) is not str or choice not in SOURCE_POLICY_CHOICES[key]:
            raise ValueError('invalid Source policy: '+key)
    if expert_math is not None and expert_math not in POLICIES:
        raise ValueError('invalid Source expert math policy')
    if (source_options.get('deepseek_source_frame_reuse','reference') != 'reference'
            and (expert_math != 'routed_fused_fp32'
                 or source_options.get('deepseek_source_prefill_batch','reference') != 'gate_up')):
        raise ValueError('Source frame reuse requires routed_fused_fp32 and gate_up prefill')
    from .deepseek_v41_source_native_mx import validate_policy as validate_native_mx
    validate_native_mx(source_options.get('deepseek_source_native_mx','reference'),expert_math=expert_math,
        prefill_batch=source_options.get('deepseek_source_prefill_batch','reference'),
        wide_math=source_options.get('deepseek_source_prefill_wide_math','reference'),
        wide_dtype=source_options.get('deepseek_source_prefill_wide_dtype','reference'),
        prefill_down=source_options.get('deepseek_source_prefill_down','reference'),
        ffn_input=source_options.get('deepseek_source_ffn_input','reference'))
    if source_options.get('deepseek_source_native_mx','reference')!='reference' and expert is None:
        raise ValueError('Source native MX requires explicit bounded LFU expert policy')
    if (source_options.get('deepseek_source_ffn_input','reference')!='reference'
            and (source_options.get('deepseek_source_dense_qat','reference')!='fused_fp32'
                 or expert_math!='routed_fused_fp32')):
        raise ValueError('Source FFN input reuse requires fused dense QAT and routed_fused_fp32')
    if (source_options.get('deepseek_source_fp8_pp','reference')!='reference'
            and source_options.get('deepseek_source_q_b_prefill','reference')!='reference'):
        raise ValueError('Source native FP8 PP cannot combine with Q_b prefill tile')
    if (source_options.get('deepseek_source_qakv','reference') != 'reference'
            and source_options.get('deepseek_source_dense_qat','reference') != 'fused_fp32'):
        raise ValueError('Source Qa/KV reuse requires fused_fp32 Source dense QAT')
    if (source_options.get('deepseek_source_attention_math','reference') != 'reference'
            and source_options.get('deepseek_index_policy','reference') != 'finite_all_retained'):
        raise ValueError('Source attention broadcast requires finite_all_retained index policy')
    if source_options.get('deepseek_source_prefill_batch','reference')!='reference' and expert_math!='routed_fused_fp32':
        raise ValueError('Source prefill batching requires routed_fused_fp32 expert math')
    from .deepseek_v41_source_prefill_down import validate_policy
    validate_policy(source_options.get('deepseek_source_prefill_down','reference'),
        expert_math=expert_math, prefill_batch=source_options.get('deepseek_source_prefill_batch','reference'))
    if source_options.get('deepseek_source_prefill_wide_math','reference')!='reference':
        if source_options.get('deepseek_source_prefill_batch','reference') not in ('gate_up','gate_up_down') or expert_math!='routed_fused_fp32':
            raise ValueError('Source wide legacy tiles require routed_fused_fp32 and batched prefill')
        if source_options.get('deepseek_source_prefill_wide_dtype','reference')!='reference':
            raise ValueError('Source wide legacy tiles cannot combine with bf16_p1')
    result=execution_policy(value)
    if source_options.get('deepseek_source_hc_boundary','reference')!='reference'and result['deepseek_hc']!='reference':
        raise ValueError('Source HC boundary requires reference HC')
    if source_options.get('deepseek_source_hc_coefficients')=='fused_fp32' and result['deepseek_hc']!='reference':
        raise ValueError('Source HC coefficients cannot combine with legacy fused HC')
    result.update(source_options)
    if expert_math is not None:
        result['deepseek_source_expert_math']=expert_math
    if expert is not None:
        from .deepseek_v41_source_lfu import parse_policy
        result['deepseek_source_experts']=parse_policy(expert)
    return result


def hasher_metadata(backend, args):
    """Official tokenizer normalization, prime layout and per-layer RNG, CPU only."""
    from tokenizers import Regex, normalizers
    sentinel = "\ue000"
    normalize = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    mapping, lookup = {}, []
    for token in range(backend.get_vocab_size()):
        text = backend.decode([token], skip_special_tokens=False)
        key = backend.id_to_token(token) if "\ufffd" in text else (normalize.normalize_str(text) or text)
        lookup.append(mapping.setdefault(key, len(mapping)))
    if len(mapping) != args.engram_compressed_vocab_size:
        raise ValueError("official Engram compressed vocabulary mismatch")
    seen, primes = set(), []
    for _ in args.engram_layer_ids:
        layer = []
        for _ in range(3):
            current = 15_999_999
            for _ in range(8):
                current += 1
                while current in seen or any(current % n == 0 for n in range(2, math.isqrt(current)+1)):
                    current += 1
                seen.add(current)
                layer.append(current)
        primes.append(layer)
    bound = max(1, (np.iinfo(np.int64).max // len(mapping)) // 2)
    multipliers = [np.random.default_rng(10007*layer).integers(0, bound, size=4, dtype=np.int64)*2+1 for layer in args.engram_layer_ids]
    prefix = "deepseek41.engram."
    return {prefix+"token_map": lookup, prefix+"rows": list(args.engram_num_embeddings), prefix+"primes": primes,
            prefix+"multipliers": np.array(multipliers).tolist(), prefix+"compressed_vocab_size": len(mapping), prefix+"pad_id": lookup[2]}


class DeepSeekV41SourceAdapter(DeepSeekV41MLXAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "deepseek-v41-1017-official-source-ordinary-b1"
    numeric_profile = "official_mxfp4_fp8_bf16"
    numerics_revision = NUMERICS_REVISION
    weight_format = "official-safetensors-mxfp4-fp8-bf16"
    policy_parser = staticmethod(source_policy)
    adapter_owned_default_policy_keys = DeepSeekV41MLXAdapter.adapter_owned_default_policy_keys | {'deepseek_source_experts','deepseek_source_expert_math'} | set(SOURCE_POLICY_CHOICES)
    weight_streaming_modes = frozenset({"moe_experts"})

    def _bind_fp8_pp_identity(self):
        if self.kernel_policy.get('deepseek_source_fp8_pp','reference')=='native_bf16_qb_wob_v1':
            from .deepseek_v41_source_fp8_pp import POLICY,implementation_identity
            self.numerics_revision+=':source_fp8_pp='+POLICY+':implementation='+implementation_identity()
            self.profile+=':fp8pp-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_woa_pp_identity(self):
        if self.kernel_policy.get('deepseek_source_woa_pp','reference')=='native_bf16_grouped_pp_v1':
            from .deepseek_v41_source_woa_pp import POLICY,implementation_identity
            self.numerics_revision+=':source_woa_pp='+POLICY+':implementation='+implementation_identity()
            self.profile+=':woapp-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_ffn_input_identity(self):
        if self.kernel_policy.get('deepseek_source_ffn_input','reference')=='shared_routed_qat_reuse':
            from .deepseek_v41_source_ffn_input import POLICY,implementation_identity
            self.numerics_revision+=':source_ffn_input='+POLICY+':implementation='+implementation_identity()
            self.profile+=':ffninput-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_prefill_down_identity(self):
        selected=self.kernel_policy.get('deepseek_source_prefill_down','reference')
        if selected in ('q1_m1_owner_store','q1_tiny_batch_v1'):
            from .deepseek_v41_source_prefill_down import implementation_identity,tiny_implementation_identity
            identity=tiny_implementation_identity()if selected=='q1_tiny_batch_v1'else implementation_identity()
            self.numerics_revision+=':source_prefill_down='+selected+':implementation='+identity
            self.profile+=':ppdown-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_native_mx_identity(self):
        if self.kernel_policy.get('deepseek_source_native_mx','reference')=='original_pp_half_v1':
            from .deepseek_v41_source_native_mx import POLICY,implementation_identity
            self.numerics_revision+=':source_native_mx='+POLICY+':implementation='+implementation_identity()
            self.profile+=':nativemx-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile+=':original_pp_half_v1'

    def _bind_frame_reuse_identity(self):
        if self.kernel_policy.get('deepseek_source_frame_reuse','reference')=='prefill_v1':
            from .deepseek_v41_source_frame_reuse import POLICY,implementation_identity
            self.numerics_revision+=':source_frame_reuse='+POLICY+':implementation='+implementation_identity()
            self.profile+=':framereuse-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _load(self, model_path):
        from ..process_env import PROCESS_NUMERICS, assert_process_profile_applied, clear_inherited_profile, require_process_numerics
        require_process_numerics("official DeepSeek V4.1 Source")
        clear_inherited_profile(())
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        os.environ.update(self.environment)
        assert_process_profile_applied("official DeepSeek V4.1 Source")
        artifact = inspect_artifact(model_path)
        self.identity, self.config = artifact["identity"], artifact["config"]
        args, root = artifact["args"], artifact["root"]
        if self.weight_streaming_request is None:
            from ..runtime.streamed_load import WeightStreamRequest
            from ..serving import preload_admission_limit_bytes
            self.weight_streaming_request = WeightStreamRequest(mode="moe_experts", budget_bytes=64 << 30,
                admission_limit_bytes=preload_admission_limit_bytes(), read_workers=2, bypass_page_cache=True)
        request = self.weight_streaming_request
        from .deepseek_v41_source_weights import SourceV41Weights, weight_plan
        expert_policy=self.kernel_policy.get('deepseek_source_experts')
        if expert_policy is None:
            self.native_admission = admission_plan(artifact, request, planner=weight_plan)
        else:
            from .deepseek_v41_source_lfu import admission
            self.native_admission = admission(artifact, request, expert_policy)
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        backend = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.metadata = hasher_metadata(backend, args)
        self.tokenizer = TokenizerWrapper(PreTrainedTokenizerFast(tokenizer_object=backend,
            bos_token=backend.id_to_token(0), eos_token=backend.id_to_token(1), pad_token=backend.id_to_token(2),
            model_max_length=args.max_position_embeddings, clean_up_tokenization_spaces=False),
            detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[1])
        self.tokenizer_receipt = {"source": "official tokenizer.json", "token_map_sha256": hashlib.sha256(np.array(self.metadata["deepseek41.engram.token_map"], dtype="<i8").tobytes()).hexdigest()}
        from ..runtime.models.deepseek_v41 import DeepSeekV41Model
        from ..runtime.models.deepseek_v41_engram import EngramHasher, EngramRows, file_stamp
        from ..runtime.weight_residency import wire_serving_weights
        self.preload_residency = wire_serving_weights()
        self.weights = SourceV41Weights(artifact, expert_cache_bytes=self.native_admission["expert_cache_bytes"], expert_read_workers=request.read_workers,
            expert_policy=expert_policy, admitted_bounds=self.native_admission if expert_policy else None,
            expert_math_policy=self.kernel_policy.get('deepseek_source_expert_math','reference'),
            dense_qat_policy=self.kernel_policy.get('deepseek_source_dense_qat','reference'),
            bf16_loads_policy=self.kernel_policy.get('deepseek_source_bf16_loads','reference'),
            completion_policy=self.kernel_policy.get('deepseek_source_completion','reference'),
            prefill_batch_policy=self.kernel_policy.get('deepseek_source_prefill_batch','reference'),
            prefill_wide_dtype=self.kernel_policy.get('deepseek_source_prefill_wide_dtype','reference'),
            prefill_wide_math=self.kernel_policy.get('deepseek_source_prefill_wide_math','reference'),
            attention_math_policy=self.kernel_policy.get('deepseek_source_attention_math','reference'),
            qakv_policy=self.kernel_policy.get('deepseek_source_qakv','reference'),
            q_b_prefill_policy=self.kernel_policy.get('deepseek_source_q_b_prefill','reference'),
            fp8_pp_policy=self.kernel_policy.get('deepseek_source_fp8_pp','reference'),
            woa_pp_policy=self.kernel_policy.get('deepseek_source_woa_pp','reference'),
            ffn_input_policy=self.kernel_policy.get('deepseek_source_ffn_input','reference'),
            prefill_down_policy=self.kernel_policy.get('deepseek_source_prefill_down','reference'),
            source_frame_reuse_policy=self.kernel_policy.get('deepseek_source_frame_reuse','reference'),
            native_mx_policy=self.kernel_policy.get('deepseek_source_native_mx','reference'))
        self.weight_stream = self.weights.expert_stream
        if expert_policy is not None:
            self.numerics_revision += ':'+self.weight_stream.apc_revision()
            self.profile += ':'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            if expert_policy['mode']=='hot_original_q1':
                self.numeric_profile='source_hot_original_cold_q1_g64'
                self.weight_format='official-source-dense-hot-mxfp4-cold-q1g64'
        self.weight_stream.plan.resident_remainder_bytes = self.native_admission["dense_payload_bytes"]+self.native_admission["dense_prepared_bound_bytes"]+self.native_admission["metadata_reserve_bytes"]
        self.weight_stream.plan.ceiling_source = request.budget_mode
        for layer, rows in zip(args.engram_layer_ids, args.engram_num_embeddings):
            weight = artifact["tensors"][f"layers.{layer}.engram.embed.weight"]
            scale = artifact["tensors"][f"layers.{layer}.engram.embed.scale"]
            if (weight["file"] != scale["file"] or weight["shape"] != (rows, 256) or scale["shape"] != (rows, 8)
                    or weight["dtype"] != "F8_E4M3" or scale["dtype"] != "F8_E8M0"):
                raise ValueError("unsupported official Engram plane layout")
            path = root / weight["file"]
            self.engram_tables[layer] = EngramRows(path, offset=weight["offset"], scale_offset=scale["offset"], rows=rows,
                expected_stat=file_stamp(path.stat()), workers=min(8, request.read_workers), cache_format=self.kernel_policy["deepseek_engram_cache"])
        self.weight_stream.bind_ple_providers(self.engram_tables)
        self._bind_fp8_pp_identity()
        self._bind_woa_pp_identity()
        self._bind_ffn_input_identity()
        self._bind_prefill_down_identity()
        self._bind_native_mx_identity()
        self._bind_frame_reuse_identity()
        binding = self.identity["fingerprint"]+":"+self.numerics_revision+":"+hashlib.sha256(json.dumps(self.kernel_policy, sort_keys=True).encode()).hexdigest()
        self.model = DeepSeekV41Model(args, self.weights, identity=binding, hasher=EngramHasher(self.metadata), engram_tables=self.engram_tables,
            hc_policy=self.kernel_policy["deepseek_hc"], qat_policy=self.kernel_policy["deepseek_qat"],
            hc_coeff_policy=('source_fused_fp32' if self.kernel_policy.get('deepseek_source_hc_coefficients','reference')=='fused_fp32' else 'reference'),
            index_policy=self.kernel_policy.get('deepseek_index_policy','reference'),
            selected_kv_policy=self.kernel_policy.get('deepseek_selected_kv_policy','reference'),
            source_attention_math_policy=self.kernel_policy.get('deepseek_source_attention_math','reference'),
            source_hc_boundary_policy=self.kernel_policy.get('deepseek_source_hc_boundary','reference'))
        self.model.verification_commit_policy = self.kernel_policy["deepseek_verify_commit"]
        self.model.verify_rows_policy = self.kernel_policy["deepseek_verify_rows"]
        self.model.pointwise_policy = self.kernel_policy["deepseek_pointwise"]
        self.model.attention_epilogue_policy = self.kernel_policy["deepseek_attention_epilogue"]
        self.model.hc_epilogue_policy = self.kernel_policy["deepseek_hc_epilogue"]
        self.model.verified_commit_stats = {}
        # The inherited constructor retires all owners on any load failure
        # while retaining the original exception if cleanup also fails.
        self.model.preload().eval()
        self.weights.verify_source()
        if expert_policy is not None:
            self.weight_stream.preheat()
        import mlx.core as mx
        if mx.get_active_memory()+self.weight_stream.unfilled_reserve_bytes()+self.native_admission["headroom_bytes"]+self.native_admission["metadata_reserve_bytes"] > request.admission_limit_bytes:
            raise MemoryError("official source retained weights exceed early admission")
        self.layout = self.descriptor.cache_layout
        self.max_context = min(args.max_position_embeddings, 65536)
        self._configure_copy_chain()

    def diagnostics(self):
        value = super().diagnostics()
        value.update(original_source_required=True, expert_storage="original E2M1/E8M0 group32",
                     source_expert_math=self.weights.expert_math_policy,
                     dense_storage={**self.weights.dense_receipt(), 'source_hc_boundary':self.model.source_hc_boundary_policy}, checkpoint_revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
                     native_dspark="not selected", vision="unavailable in this text serving slice",
                     numerical_scope="MLX reference reductions; original weight codes and activation quantization; CUDA parity unproven")
        value.update(index_policy=self.model.index_policy,selected_kv_policy=self.model.selected_kv_policy,
                     source_qakv=self.weights.source_qakv_policy,
                     source_q_b_prefill=self.weights.source_q_b_prefill_policy,
                     source_fp8_pp=getattr(self.weights,'source_fp8_pp_policy','reference'),
                     source_woa_pp=getattr(self.weights,'source_woa_pp_policy','reference'),
                     source_native_mx=getattr(self.weights,'source_native_mx_policy','reference'),
                     source_frame_reuse=getattr(self.weights,'source_frame_reuse_policy','reference'),
                     source_ffn_input=getattr(self.weights,'source_ffn_input_policy','reference'),
                     source_prefill_down=getattr(self.weights,'source_prefill_down_policy','reference'),
                     source_attention_math=self.model.source_attention_math_policy,
                     source_hc_boundary=self.model.source_hc_boundary_policy,
                     source_hc_boundary_counters={k:v for k,v in self.model.scheduled_hc_calls.items()if k.startswith('source_hc_boundary_')},
                     hc_coeff_policy=self.model.hc_coeff_policy,
                     index_score_contract=('finite unmasked index scores required' if self.model.index_policy=='finite_all_retained' else 'ordinary reference behavior'))
        if self.kernel_policy.get('deepseek_source_experts',{}).get('mode')=='hot_original_q1':
            value.update(expert_storage='frozen hot map: original E2M1/E8M0; all other routed experts: affine Q1 g64',
                         numerical_scope='approximate Q1 expert derivative; dense/shared/attention/router/Engram original; unqualified')
        if self.model.source_attention_math_policy == 'omlx_nax_bf16_online_v1':
            from ..runtime.models.deepseek_v41_source_attention import implementation_identity
            from ..runtime.models.deepseek_v41_omlx_nax_attention import PROFILE
            value['source_attention_numerical_profile'] = PROFILE
            value['source_attention_implementation_sha256'] = implementation_identity()
            value['numerical_scope'] = (
                ('Approximate Q1 expert derivative; ' if self.kernel_policy.get('deepseek_source_experts',{}).get('mode')=='hot_original_q1' else 'Original weight codes; ')
                + 'opt-in PP attention BF16 NAX QK/online F32 softmax/FP16 hi+lo PV; distinct numerical profile with existing owner BF16; decode/reference fallback unchanged; unqualified')
        if getattr(self.weights,'source_fp8_pp_policy','reference')=='native_bf16_qb_wob_v1':
            from .deepseek_v41_source_fp8_pp import POLICY,implementation_identity
            value['source_fp8_pp_numerical_profile']=POLICY
            value['source_fp8_pp_implementation_sha256']=implementation_identity()
            value['numerical_scope']=value['numerical_scope'].replace('MLX reference reductions;', 'Existing Source profile;')
            value['numerical_scope']+='; explicit PP99 Q_b/WoB native BF16 arithmetic over unchanged original FP8 codes, distinct numerical profile; decode/reference fallback unchanged; unqualified'
        if getattr(self.weights,'source_prefill_down_policy','reference')!='reference':
            from .deepseek_v41_source_prefill_down import implementation_identity,tiny_implementation_identity,TINY_CONTRACT,TINY_COUNTER
            tiny=getattr(self.weights,'source_prefill_down_policy','reference')=='q1_tiny_batch_v1'
            value['source_prefill_down_implementation_sha256']=tiny_implementation_identity()if tiny else implementation_identity()
            value['source_prefill_down_contract']=TINY_CONTRACT if tiny else 'PP-only Q1 M1 final BF16 owner store fused; unchanged dot program and coefficient codes; explicit/unqualified'
            if tiny:
                value['source_prefill_down_counters']={k:v for k,v in self.weights.scheduled_expert_calls.items()if k.startswith(TINY_COUNTER)}
        if getattr(self.weights,'source_native_mx_policy','reference')!='reference':
            from .deepseek_v41_source_native_mx import POLICY,implementation_identity
            value['source_native_mx_numerical_profile']=POLICY
            value['source_native_mx_implementation_sha256']=implementation_identity()
            value['source_native_mx_counters']={k:v for k,v in self.weights.scheduled_expert_calls.items() if k.startswith('official_source_native_mx_')}
            value['numerical_scope']=value['numerical_scope'].replace('MLX reference reductions;','Existing Source profile;')
            value['numerical_scope']+='; opt-in original PP expert M16..99 native HALF activation consumption (rounding allowed), distinct profile; original coefficient precision/Source middle/owners unchanged; Q1/small-M/decode reference; unqualified'
        return value
