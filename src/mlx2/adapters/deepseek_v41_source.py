# SPDX-License-Identifier: MIT
"""Official V4.1 Source with explicit text, draft, image and batch profiles."""
import hashlib
import json
import math
import os
from dataclasses import replace

import numpy as np

from .deepseek_v41_execution import (
    bind_cache_layout,
    model_options,
    validate_csa_policy,
)
from .deepseek_v41_mlx import DESCRIPTOR as NATIVE_DESCRIPTOR
from .deepseek_v41_mlx import DeepSeekV41MLXAdapter, execution_policy
from .deepseek_v41_source_artifact import MODEL_TYPE, inspect_artifact
from .deepseek_v41_source_math import POLICIES
from .deepseek_v41_source_defaults import default_route_policy_applies, default_route_policy_filter
from .deepseek_v41_source_approx import POLICY_KEYS as APPROXIMATE_POLICY_KEYS

NUMERICS_REVISION = "mlx2.v41-official-mxfp4-fp8-act32-bf16.v1"
DESCRIPTOR = replace(NATIVE_DESCRIPTOR, model_type=MODEL_TYPE, variant="flash-1017-official-source-text",
    metadata={"execution": "mlx2.adapters.deepseek_v41_source.DeepSeekV41SourceAdapter",
              "qualification": "unqualified", "scope": "official checkpoint; ordinary B1 text; streamed MXFP4 experts and split-plane Engram; exact APCv2 within this numerical profile"})

SOURCE_POLICY_CHOICES = {
    'deepseek_source_prefill_partition': ('adaptive_v1', 'canonical_m1_v1', 'position_grid_v1'),
    'deepseek_source_prefill_order': ('reference', 'layer_major_v1'),
    'deepseek_source_prefill_readahead': ('reference', 'layer_ahead_v1'),
    'deepseek_source_dspark_ast': ('off', 'tile3x5_v1'),
    'deepseek_source_dspark_prepare': ('off', 'oracle_v1'),
    'deepseek_source_dspark_target_expert_law': ('ordinary_m16_v1', 'source_ast_m1_rows_v1', 'source_ast_resident_rows_v1', 'source_ast_hybrid_rows_v1'),
    'deepseek_source_dspark_target_forward': ('reference', 'fully_batched_v1', 'fully_batched_m1_v1'),
    'deepseek_source_ast_kv_preparation': ('reference', 'descendant_reuse_v1'),
    'deepseek_source_resident_gather': ('reference', 'resident_bank_v1'),
    'deepseek_source_resident_prefill': ('reference', 'expert_major_v1', 'expert_major_wide_v1'),
    'deepseek_source_submission': ('reference', 'shared_first_v1'),
    'deepseek_source_expert_io': ('reference', 'overlap_v1', 'overlap_demand_v1'),
    'deepseek_source_row_packet': ('reference', 'row_packet_v1'),
    'deepseek_vision': ('reference', 'enabled'),
    'deepseek_physical_batch': ('b1', 'b2', 'b4'),
    'deepseek_source_frame_reuse': ('reference', 'prefill_v1'),
    'deepseek_source_native_mx': ('reference', 'original_pp_half_v1'),
    'deepseek_source_prefill_down': ('reference', 'q1_m1_owner_store', 'q1_tiny_batch_v1'),
    'deepseek_source_ffn_input': ('reference', 'shared_routed_qat_reuse'),
    'deepseek_source_woa_pp': ('reference', 'native_bf16_grouped_pp_v1'),
    'deepseek_source_woa_storage': ('reference', 'mxfp8_gemv_v1'),
    'deepseek_source_fp8_pp': ('reference', 'native_bf16_qb_wob_v1'),
    'deepseek_source_hc_boundary': ('reference', 'expand_collapse_seq'),
    'deepseek_source_hc_mega': ('reference', 'mega_v1'),
    'deepseek_source_q_b_prefill': ('reference', 'tile64x32_raw'),
    'deepseek_source_qakv': ('reference', 'qat_reuse'),
    'deepseek_source_attention_math': ('reference', 'all_retained_broadcast', 'omlx_nax_bf16_online_v1',
        'source_short_decode_joint_f32_v1', 'source_short_decode_grouped_f32_v1'),
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
    if value is not None and not isinstance(value, dict):
        raise ValueError('Source V4.1 execution policy must be a dictionary')
    value=dict(value or {})
    from .deepseek_v41_source_approx import split_policy as split_approximate
    approximate=split_approximate(value)  # opt-in approximate operations; validated last
    from .deepseek_v41_source_compatibility import validate_source_profiles
    validate_source_profiles(value)
    dspark={key:value.pop(key)for key in ('deepseek_source_dspark_oracle','deepseek_source_dspark_cache_bytes')if key in value}
    embedded = value.get('draft_model') == 'embedded_dspark'
    draft_count = value.get('num_draft', 5)
    from .deepseek_v41_dspark_policy import split_chain_policy
    chain = split_chain_policy(value, embedded=embedded)
    if embedded:
        if value.get('ast_prompt_lookup', False) is not False or type(draft_count) is not int or not 1 <= draft_count <= 5:
            raise ValueError('embedded DSpark requires num_draft1..5 and no prompt-copy override')
        value.pop('draft_model')
        value.pop('num_draft', None)
    expert=value.pop('deepseek_source_experts',None)
    expert_math=value.pop('deepseek_source_expert_math',None)
    grid_rows=value.pop('deepseek_source_prefill_grid_rows',None)
    source_options={key:value.pop(key) for key in SOURCE_POLICY_CHOICES if key in value}
    for key,choice in source_options.items():
        if type(choice) is not str or choice not in SOURCE_POLICY_CHOICES[key]:
            raise ValueError('invalid Source policy: '+key)
    ast=source_options.get('deepseek_source_dspark_ast','off')
    preparing=source_options.get('deepseek_source_dspark_prepare','off')=='oracle_v1'
    if preparing:
        # Prepare the future target's identity before its first cache exists.
        # This loads neither the DSpark head nor a serving route/oracle.
        if ast!='off':
            raise ValueError('Source DSpark oracle preparation requires AST off')
        if 'deepseek_source_dspark_oracle' in dspark:
            raise ValueError('Source DSpark preparation cannot consume a serving oracle')
        if (source_options.get('deepseek_source_dspark_target_forward')!='fully_batched_m1_v1'
                or source_options.get('deepseek_source_dspark_target_expert_law')not in ('source_ast_m1_rows_v1','source_ast_resident_rows_v1','source_ast_hybrid_rows_v1')
                or expert_math!='routed_fused_fp32'):
            raise ValueError('Source DSpark preparation requires explicit fully_batched_m1_v1/M1-or-resident expert law/routed_fused_fp32')
        if value.get('draft_model')is not None or source_options.get('deepseek_source_resident_gather','reference')!='reference':
            raise ValueError('Source DSpark preparation requires an ordinary target without foreign draft or resident bank')
        budget=dspark.get('deepseek_source_dspark_cache_bytes',512<<20)
        if type(budget)is not int or budget<18_800_640:
            raise ValueError('Source DSpark expert maximum must hold one original expert')
        dspark['deepseek_source_dspark_cache_bytes']=budget
    elif ast=='tile3x5_v1':
        from pathlib import Path
        oracle=dspark.get('deepseek_source_dspark_oracle');budget=dspark.get('deepseek_source_dspark_cache_bytes',512<<20)
        if type(oracle)is not str or not oracle.strip()or not Path(oracle).is_file():
            raise ValueError('Source DSpark AST requires an explicit existing exact16-row oracle file')
        if type(budget)is not int or budget<18_800_640:
            raise ValueError('Source DSpark expert maximum must hold one original expert')
        batched=source_options.get('deepseek_source_dspark_target_forward','reference')!='reference'
        if value.get('draft_model')is not None or (source_options.get('deepseek_source_resident_gather','reference')!='reference' and not batched):
            raise ValueError('Source DSpark AST cannot combine a foreign draft or B1-only diagnostic bank policy')
        law=source_options.get('deepseek_source_dspark_target_expert_law','ordinary_m16_v1')
        if law in ('source_ast_m1_rows_v1','source_ast_resident_rows_v1','source_ast_hybrid_rows_v1')and expert_math!='routed_fused_fp32':
            raise ValueError('Source DSpark AST M1 law requires routed_fused_fp32 boundaries')
        if law=='source_ast_resident_rows_v1'and not batched:
            raise ValueError('Source resident AST expert law requires fully_batched_v1')
        if law=='source_ast_hybrid_rows_v1'and source_options.get('deepseek_source_dspark_target_forward')!='fully_batched_m1_v1':
            raise ValueError('Source hybrid AST expert law requires fully_batched_m1_v1')
        dspark['deepseek_source_dspark_cache_bytes']=budget
    elif dspark or any(key in source_options for key in ('deepseek_source_dspark_target_expert_law','deepseek_source_dspark_target_forward')):
        raise ValueError('DSpark oracle/cache options require tile3x5_v1')
    if source_options.get('deepseek_source_ast_kv_preparation','reference')!='reference' and (
            source_options.get('deepseek_source_dspark_target_forward')!='fully_batched_m1_v1'
            or not (preparing or ast=='tile3x5_v1')):
        raise ValueError('Source descendant KV preparation requires prepared or serving fully_batched_m1_v1')
    if (source_options.get('deepseek_source_dspark_target_expert_law')=='source_ast_hybrid_rows_v1'
            and source_options.get('deepseek_source_prefill_partition','adaptive_v1')!='canonical_m1_v1'):
        raise ValueError('Source hybrid AST currently requires explicit canonical_m1_v1 prefill partition')
    if expert_math is not None and expert_math not in POLICIES:
        raise ValueError('invalid Source expert math policy')
    from .deepseek_v41_source_canonical_prefill import GRID, GRID_ROWS, GRID_DEFAULT_ROWS
    grid=source_options.get('deepseek_source_prefill_partition','adaptive_v1')==GRID
    if grid_rows is not None and (not grid or type(grid_rows) is not int or grid_rows not in GRID_ROWS):
        raise ValueError('deepseek_source_prefill_grid_rows requires position_grid_v1 and one of '+str(GRID_ROWS))
    if grid:
        # The grid law is an ordinary-B1 prefill profile. Private bank, tree
        # and draft routes own other forward/state transactions.
        for key,reference in (('deepseek_source_dspark_ast','off'),('deepseek_source_dspark_prepare','off'),
                              ('deepseek_source_resident_gather','reference'),('deepseek_source_resident_prefill','reference')):
            if source_options.get(key,reference)!=reference:
                raise ValueError('Source position_grid_v1 cannot combine with '+key)
        source_options['deepseek_source_prefill_grid_rows']=GRID_DEFAULT_ROWS if grid_rows is None else grid_rows
    if source_options.get('deepseek_source_resident_gather','reference')!='reference' and (
            expert_math!='routed_fused_fp32' or source_options.get('deepseek_source_ffn_input','reference')!='reference'):
        raise ValueError('Source resident bank requires routed_fused_fp32 and reference FFN input')
    if source_options.get('deepseek_source_resident_prefill','reference')!='reference' and source_options.get('deepseek_source_resident_gather','reference')!='resident_bank_v1':
        raise ValueError('Source resident prefill requires resident_bank_v1')
    if (source_options.get('deepseek_source_frame_reuse','reference') != 'reference'
            and (expert_math != 'routed_fused_fp32'
                 or source_options.get('deepseek_source_prefill_batch','reference') != 'gate_up')):
        raise ValueError('Source frame reuse requires routed_fused_fp32 and gate_up prefill')
    if (source_options.get('deepseek_source_row_packet','reference') != 'reference'
            and source_options.get('deepseek_source_frame_reuse','reference') != 'prefill_v1'):
        raise ValueError('Source row packet requires authoritative prefill_v1 frame reuse')
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
    if (source_options.get('deepseek_source_submission','reference')=='shared_first_v1'
            and source_options.get('deepseek_source_ffn_input','reference')!='reference'):
        raise ValueError('Source shared-first requires reference FFN input; routed/shared reuse consumes routed first')
    if source_options.get('deepseek_source_woa_storage','reference')!='reference':
        if source_options.get('deepseek_source_bf16_loads','reference') not in ('woa_f32','direct_f32'):
            raise ValueError('Source MXFP8 WoA storage requires deepseek_source_bf16_loads woa_f32 or direct_f32')
        for name,reference in (('deepseek_source_resident_gather','reference'),('deepseek_source_resident_prefill','reference'),
                               ('deepseek_source_dspark_ast','off'),('deepseek_source_dspark_prepare','off')):
            if source_options.get(name,reference)!=reference:
                raise ValueError('Source MXFP8 WoA storage cannot combine with '+name)
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
    if source_options.get('deepseek_source_expert_io','reference')!='reference':
        # Exact-route early reads are implemented for the ordinary B1 block
        # hook over a bounded LFU provider only; everything else fails closed.
        blocked=[name for name,reference in (
            ('deepseek_source_resident_gather','reference'),('deepseek_source_resident_prefill','reference'),
            ('deepseek_source_submission','reference'),('deepseek_source_ffn_input','reference'),
            ('deepseek_physical_batch','b1'),('deepseek_source_dspark_ast','off'),
            ('deepseek_source_dspark_prepare','off'))if source_options.get(name,reference)!=reference]
        if expert is None:blocked.append('deepseek_source_experts(required)')
        if embedded:blocked.append('embedded_dspark')
        if blocked:
            raise ValueError('Source expert overlap_v1 cannot combine with: '+', '.join(blocked))
    result=execution_policy(value)
    if source_options.get('deepseek_source_hc_boundary','reference')!='reference'and result['deepseek_hc']!='reference':
        raise ValueError('Source HC boundary requires reference HC')
    if source_options.get('deepseek_source_hc_coefficients')=='fused_fp32' and result['deepseek_hc']!='reference':
        raise ValueError('Source HC coefficients cannot combine with legacy fused HC')
    if source_options.get('deepseek_source_hc_mega','reference')!='reference' and (
            source_options.get('deepseek_source_hc_coefficients')!='fused_fp32'
            or source_options.get('deepseek_source_hc_boundary','reference')!='reference'
            or result['deepseek_hc']!='reference'):
        raise ValueError('Source HC mega requires fused_fp32 Source coefficients with reference HC and HC boundary')
    result.update(source_options)
    result.update(dspark)
    if embedded:
        result.update(draft_model='embedded_dspark', num_draft=draft_count)
    result.update(chain)
    from .deepseek_v41_dspark_policy import validate_chain_policy
    validate_chain_policy(result)
    if grid and (embedded or result.get('draft_model') is not None or result.get('ast_prompt_lookup')):
        raise ValueError('Source position_grid_v1 requires ordinary decoding without a draft')
    if source_options.get('deepseek_vision', 'reference') == 'enabled' and (embedded or result['ast_prompt_lookup']):
        raise ValueError('V4.1 Source vision requires ordinary decoding')
    validate_csa_policy(result)
    validate_source_profiles(result)
    if expert_math is not None:
        result['deepseek_source_expert_math']=expert_math
    if expert is not None:
        from .deepseek_v41_source_lfu import parse_policy
        result['deepseek_source_experts']=parse_policy(expert)
        from .deepseek_v41_source_residency import validate_combination
        validate_combination(result,result['deepseek_source_experts'])
    if result.get('deepseek_physical_batch', 'b1') != 'b1':
        from .deepseek_v41_batch import validate_physical_batch_policy
        validate_physical_batch_policy(result, width=int(result['deepseek_physical_batch'][1:]))
    from .deepseek_v41_source_layer_major import validate_policy as validate_prefill_order
    validate_prefill_order(result)
    from .deepseek_v41_source_prefill_readahead import validate_policy as validate_prefill_readahead
    validate_prefill_readahead(result)
    from .deepseek_v41_source_approx import validate_policy as validate_approximate
    result.update(validate_approximate(approximate,result))
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
    multimodal_routes = frozenset({"ordinary"})
    descriptor = DESCRIPTOR
    # User-selected bounded profile; exact guard keeps unrelated Source
    # artifacts/precision maps on their existing defaults. Route remains
    # unqualified; see provenance/deepseek-v41-selected-defaults-20261027.json.
    # 2026-10-28: direct_f32 head, MXFP8 WoA and HC mega added after a
    # 3-repetition token-exact checkpoint A/B (+6.4% warm decode); see
    # provenance/deepseek-v41-selected-defaults-20261028.json.
    default_route_policy_applies = staticmethod(default_route_policy_applies)
    default_route_policy_filter = staticmethod(default_route_policy_filter)
    default_route_execution_policy = {
        "ordinary": {
            'deepseek_attention_epilogue': 'fused',
            'deepseek_engram_cache': 'bf16_lut',
            'deepseek_index_policy': 'finite_all_retained',
            'deepseek_pointwise': 'fused',
            'deepseek_qat': 'fused_fp32',
            'deepseek_selected_kv_policy': 'decode_reuse',
            'deepseek_source_attention_math': 'all_retained_broadcast',
            'deepseek_source_bf16_loads': 'direct_f32',
            'deepseek_source_completion': 'bounded_2',
            'deepseek_source_dense_qat': 'fused_fp32',
            'deepseek_source_expert_math': 'routed_fused_fp32',
            'deepseek_source_fp8_pp': 'native_bf16_qb_wob_v1',
            'deepseek_source_hc_coefficients': 'fused_fp32',
            'deepseek_source_hc_mega': 'mega_v1',
            'deepseek_source_prefill_batch': 'gate_up',
            'deepseek_source_qakv': 'qat_reuse',
            'deepseek_source_woa_storage': 'mxfp8_gemv_v1',
        },
    }
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "deepseek-v41-1017-official-source-ordinary-b1"
    numeric_profile = "official_mxfp4_fp8_bf16"
    numerics_revision = NUMERICS_REVISION
    weight_format = "official-safetensors-mxfp4-fp8-bf16"
    policy_parser = staticmethod(source_policy)
    adapter_owned_default_policy_keys = DeepSeekV41MLXAdapter.adapter_owned_default_policy_keys | {'deepseek_source_experts','deepseek_dspark_admission','deepseek_source_expert_math','deepseek_source_dspark_oracle','deepseek_source_dspark_cache_bytes','deepseek_dspark_verify_schedule','deepseek_dspark_verify_costs','deepseek_dspark_draft_cost','deepseek_source_prefill_grid_rows'} | set(SOURCE_POLICY_CHOICES) | APPROXIMATE_POLICY_KEYS
    weight_streaming_modes = frozenset({"moe_experts"})

    @staticmethod
    def weight_streaming_max_lanes(execution_policy, *, mode):
        policy = source_policy(execution_policy)
        return int(policy.get('deepseek_physical_batch', 'b1')[1:]) if mode == 'moe_experts' else 1

    @staticmethod
    def weight_streaming_external_draft(policy):
        if 'draft_models' in policy or policy.get('draft_model') not in ('prompt_lookup', 'embedded_dspark'):
            return False
        source_policy(policy)  # Includes B1-only draft and auxiliary admission policy.
        return True

    def _bind_fp8_pp_identity(self):
        if self.kernel_policy.get('deepseek_source_fp8_pp','reference')=='native_bf16_qb_wob_v1':
            from .deepseek_v41_source_fp8_pp import POLICY, implementation_identity
            self.numerics_revision+=':source_fp8_pp='+POLICY+':implementation='+implementation_identity()
            self.profile+=':fp8pp-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_woa_pp_identity(self):
        if self.kernel_policy.get('deepseek_source_woa_pp','reference')=='native_bf16_grouped_pp_v1':
            from .deepseek_v41_source_woa_pp import POLICY, implementation_identity
            self.numerics_revision+=':source_woa_pp='+POLICY+':implementation='+implementation_identity()
            self.profile+=':woapp-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_woa_storage_identity(self):
        if self.kernel_policy.get('deepseek_source_woa_storage','reference')=='mxfp8_gemv_v1':
            from ..runtime.gguf.source_dense_mxfp8 import POLICY, implementation_identity
            self.numerics_revision+=':source_woa_storage='+POLICY+':implementation='+implementation_identity()
            self.profile+=':woamxfp8-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_ffn_input_identity(self):
        if self.kernel_policy.get('deepseek_source_ffn_input','reference')=='shared_routed_qat_reuse':
            from .deepseek_v41_source_ffn_input import POLICY, implementation_identity
            self.numerics_revision+=':source_ffn_input='+POLICY+':implementation='+implementation_identity()
            self.profile+=':ffninput-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_prefill_down_identity(self):
        selected=self.kernel_policy.get('deepseek_source_prefill_down','reference')
        if selected in ('q1_m1_owner_store','q1_tiny_batch_v1'):
            from .deepseek_v41_source_prefill_down import (
                implementation_identity,
                tiny_implementation_identity,
            )
            identity=tiny_implementation_identity()if selected=='q1_tiny_batch_v1'else implementation_identity()
            self.numerics_revision+=':source_prefill_down='+selected+':implementation='+identity
            self.profile+=':ppdown-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_native_mx_identity(self):
        if self.kernel_policy.get('deepseek_source_native_mx','reference')=='original_pp_half_v1':
            from .deepseek_v41_source_native_mx import POLICY, implementation_identity
            self.numerics_revision+=':source_native_mx='+POLICY+':implementation='+implementation_identity()
            self.profile+=':nativemx-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile+=':original_pp_half_v1'

    def _bind_frame_reuse_identity(self):
        if self.kernel_policy.get('deepseek_source_frame_reuse','reference')=='prefill_v1':
            from .deepseek_v41_source_frame_reuse import POLICY, implementation_identity
            self.numerics_revision+=':source_frame_reuse='+POLICY+':implementation='+implementation_identity()
            self.profile+=':framereuse-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_submission_identity(self):
        if self.kernel_policy.get('deepseek_source_submission','reference') == 'shared_first_v1':
            from .deepseek_v41_source_submission import POLICY, implementation_identity
            self.numerics_revision += ':source_submission='+POLICY+':implementation='+implementation_identity()
            self.profile += ':submission-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_row_packet_identity(self):
        if self.kernel_policy.get('deepseek_source_row_packet','reference') == 'row_packet_v1':
            from .deepseek_v41_source_row_packet import implementation_identity
            self.numerics_revision += ':source_row_packet=row_packet_v1:implementation='+implementation_identity()
            self.profile += ':rowpacket-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_hc_mega_identity(self):
        if self.kernel_policy.get('deepseek_source_hc_mega','reference')=='mega_v1':
            from ..runtime.models.deepseek_v41_source_hc_mega import POLICY, implementation_identity
            self.numerics_revision += ':source_hc_mega='+POLICY+':implementation='+implementation_identity()
            self.profile += ':hcmega-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]

    def _bind_short_attention_identity(self):
        selected=self.kernel_policy.get('deepseek_source_attention_math','reference')
        if selected in ('source_short_decode_joint_f32_v1','source_short_decode_grouped_f32_v1'):
            from ..runtime.models.deepseek_v41_short_attention import implementation_identity
            self.numerics_revision += ':short_attention='+selected+':implementation='+implementation_identity()
            self.profile += ':shortattn-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile += ':'+selected

    def _bind_batched_tree_identity(self):
        selected=self.kernel_policy.get('deepseek_source_dspark_target_forward','reference')
        law=self.kernel_policy.get('deepseek_source_dspark_target_expert_law','ordinary_m16_v1')
        self._source_ast_target_laws=(law,selected)
        if selected!='reference':
            from .deepseek_v41_ast_target import source_revision_token
            from .deepseek_v41_ast_batched_target import PROFILES
            profile=PROFILES[selected]
            binding=source_revision_token(law,selected)
            self.numerics_revision+=':source_ast_forward='+profile+':implementation='+binding
            self.profile+=':astbatch-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile+=':'+profile

    def _bind_prefill_partition_identity(self):
        from .deepseek_v41_source_canonical_prefill import GRID,POLICY,_grid,implementation_identity
        selected=self.kernel_policy.get('deepseek_source_prefill_partition','adaptive_v1')
        self.weights.source_prefill_partition_policy=selected
        if selected==POLICY:
            self.numerics_revision+=':source_prefill_partition='+selected+':implementation='+implementation_identity()
            self.profile+=':canonical-m1-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile+=':'+selected+'_unqualified'
        elif selected==GRID:
            grid=_grid(self.kernel_policy.get('deepseek_source_prefill_grid_rows'))
            # Prompt-width-scoped arithmetic would bind numerics to the outer
            # call rather than to the grid cell; refuse it rather than guess.
            if any(getattr(self.weights,name,'reference')!='reference'
                   for name in ('dense_prefill_policy','expert_prefill_policy')):
                raise ValueError('Source position_grid_v1 requires reference width-scoped prefill arithmetic')
            self.weights.source_prefill_grid_rows=grid
            self.numerics_revision+=(':source_prefill_partition='+selected+':grid_rows='+str(grid)
                                     +':implementation='+implementation_identity())
            self.profile+=':grid'+str(grid)+'-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile+=':'+selected+'_'+str(grid)+'_unqualified'

    def _bind_prefill_order_identity(self):
        from .deepseek_v41_source_layer_major import KEY,POLICY,implementation_identity
        selected=self.kernel_policy.get(KEY,'reference')
        self.weights.source_prefill_order_policy=selected
        if selected==POLICY:
            self.numerics_revision+=':'+KEY+'='+selected+':implementation='+implementation_identity()
            self.profile+=':layer-major-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile+=':'+selected+'_unqualified'

    def _load(self, model_path):
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )
        require_process_numerics("official DeepSeek V4.1 Source")
        clear_inherited_profile(())
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        os.environ.update(self.environment)
        assert_process_profile_applied("official DeepSeek V4.1 Source")
        artifact = inspect_artifact(model_path)
        self.source_artifact = artifact
        self.vision_weights = None
        from .deepseek_v41_identity import effective_source_identity
        self.identity = effective_source_identity(artifact["identity"])
        self.config = artifact["config"]
        args, root = artifact["args"], artifact["root"]
        if self.weight_streaming_request is None:
            from ..runtime.streamed_load import WeightStreamRequest
            from ..serving import preload_admission_limit_bytes
            self.weight_streaming_request = WeightStreamRequest(mode="moe_experts", budget_bytes=64 << 30,
                admission_limit_bytes=preload_admission_limit_bytes(), read_workers=2, bypass_page_cache=True)
        request = self.weight_streaming_request
        auxiliary = {}
        if self.kernel_policy.get('draft_model') == 'embedded_dspark':
            from .deepseek_v41_dspark_artifact import dspark_admission_plan
            auxiliary['dspark'] = dspark_admission_plan(artifact)
        if self.kernel_policy.get('deepseek_vision') == 'enabled':
            from .deepseek_v41_vision import vision_admission_plan
            auxiliary['vision'] = vision_admission_plan(artifact)
        if self.kernel_policy.get('deepseek_source_dspark_ast', 'off') == 'tile3x5_v1':
            from .deepseek_v41_dspark import source_dspark_admission_plan
            auxiliary['dspark_ast'] = source_dspark_admission_plan(artifact,
                expert_cache_bytes=self.kernel_policy.get('deepseek_source_dspark_cache_bytes', 512 << 20))
        from .deepseek_v41_source_weights import SourceV41Weights
        expert_policy=self.kernel_policy.get('deepseek_source_experts')
        from .deepseek_v41_source_lfu import source_admission
        self.native_admission = source_admission(artifact, request, expert_policy)
        from .deepseek_v41_source_auxiliary import stage_auxiliary_plans
        stage_auxiliary_plans(self.native_admission, auxiliary)
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
        from ..runtime.models.deepseek_v41_engram import (
            EngramHasher,
            EngramRows,
            file_stamp,
        )
        from ..runtime.weight_residency import wire_serving_weights
        self.preload_residency = wire_serving_weights()
        self.weights = SourceV41Weights(artifact, expert_cache_bytes=self.native_admission["expert_cache_bytes"], expert_read_workers=request.read_workers,
            expert_policy=expert_policy, admitted_bounds=self.native_admission,
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
            woa_storage_policy=self.kernel_policy.get('deepseek_source_woa_storage','reference'),
            ffn_input_policy=self.kernel_policy.get('deepseek_source_ffn_input','reference'),
            prefill_down_policy=self.kernel_policy.get('deepseek_source_prefill_down','reference'),
            source_frame_reuse_policy=self.kernel_policy.get('deepseek_source_frame_reuse','reference'),
            native_mx_policy=self.kernel_policy.get('deepseek_source_native_mx','reference'),
            submission_policy=self.kernel_policy.get('deepseek_source_submission','reference'),
            source_row_packet_policy=self.kernel_policy.get('deepseek_source_row_packet','reference'),
            resident_gather_policy=self.kernel_policy.get('deepseek_source_resident_gather','reference'),
            resident_prefill_policy=self.kernel_policy.get('deepseek_source_resident_prefill','reference'),
            source_expert_io_policy=self.kernel_policy.get('deepseek_source_expert_io','reference'),
            prefill_readahead_policy=self.kernel_policy.get('deepseek_source_prefill_readahead','reference'))
        self.weight_stream = self.weights.expert_stream
        from ..memory import host_available_bytes
        self.weight_stream.configure_adaptive_residency(
            enabled=True, available_bytes=lambda: int(host_available_bytes(host_signals=True)),
            headroom_bytes=self.native_admission['auxiliary_workspace_bound_bytes'],
            working_set_experts=args.num_experts_per_tok,
            staging_within_cache=True)
        # Adaptive prefill partitions are a distinct unqualified numerical
        # profile; never reuse state published under the former fixed policy.
        self.numerics_revision += ':source-memory=adaptive-demand-v1'
        self.profile += ':adaptive-demand-v1'
        self._bind_prefill_partition_identity()
        self._bind_prefill_order_identity()
        if expert_policy is not None:
            self.numerics_revision += ':'+self.weight_stream.apc_revision()
            self.profile += ':'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            if expert_policy['mode']=='hot_original_q1':
                self.identity = effective_source_identity(artifact["identity"],
                    q1_bank=self.weight_stream.q1_binding, hot_map=self.weight_stream.identity_hot_binding())
                self.numeric_profile='source_hot_original_cold_q1_g64'
                self.weight_format='official-source-dense-hot-mxfp4-cold-q1g64'
                if self._source_dynamic_precision():
                    # Output depends on cache history: distinct, approximate,
                    # unqualified profile; APCv2 publication/reuse disabled.
                    self.numeric_profile='source_dynamic_precision_mxfp4_q1g64_v1_unqualified'
                    self.weight_format='official-source-dense-dynamic-mxfp4-q1g64'
                    self.profile+=':dynamic-precision-v1'
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
        self._bind_woa_storage_identity()
        self._bind_ffn_input_identity()
        self._bind_prefill_down_identity()
        self._bind_native_mx_identity()
        self._bind_frame_reuse_identity()
        self._bind_submission_identity()
        self._bind_row_packet_identity()
        self._bind_short_attention_identity()
        self._bind_hc_mega_identity()
        self._bind_batched_tree_identity()
        from .deepseek_v41_source_approx import bind_identity as bind_approximate_identity
        bind_approximate_identity(self)  # no-op unless an approximate operation is selected
        if self.kernel_policy.get('deepseek_verify_rows','serial')=='lockstep_multirow_v1':
            from .deepseek_v41_dspark_chain import POLICY, implementation_identity
            self.numerics_revision+=':verify_rows='+POLICY+':implementation='+implementation_identity()
            self.profile+=':lockstep-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
        if self.kernel_policy.get('deepseek_source_resident_gather','reference')!='reference':
            from .deepseek_v41_source_bank_session import implementation_identity
            self.numerics_revision+=':source_resident_bank_v1:implementation='+implementation_identity()
            self.profile+=':resident-bank-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile+=':resident_bank_v1_unqualified'
        if self.kernel_policy.get('deepseek_source_resident_prefill','reference')!='reference':
            from .deepseek_v41_source_prefill_session import implementation_identity
            self.numerics_revision+=':source_resident_prefill='+self.kernel_policy['deepseek_source_resident_prefill']+':implementation='+implementation_identity()
            self.profile+=':resident-pp-'+hashlib.sha256(self.numerics_revision.encode()).hexdigest()[:16]
            self.numeric_profile+=':'+self.kernel_policy['deepseek_source_resident_prefill']+'_unqualified'
        # Head selection, its storage maximum and oracle location belong to
        # the composed route. They do not alter target arithmetic or its oracle.
        target_policy={k:v for k,v in self.kernel_policy.items()if not k.startswith('deepseek_source_dspark_')}
        binding = self.identity["fingerprint"]+":"+self.numerics_revision+":"+hashlib.sha256(json.dumps(target_policy, sort_keys=True).encode()).hexdigest()
        model_type=DeepSeekV41Model
        if self.kernel_policy.get('deepseek_source_prefill_partition','adaptive_v1')=='canonical_m1_v1':
            from .deepseek_v41_source_canonical_prefill import canonical_model_class
            model_type=canonical_model_class()
        elif self.kernel_policy.get('deepseek_source_prefill_partition','adaptive_v1')=='position_grid_v1':
            from .deepseek_v41_source_canonical_prefill import grid_model_class
            model_type=grid_model_class()
        elif self.kernel_policy.get('deepseek_source_prefill_order','reference')=='layer_major_v1':
            from .deepseek_v41_source_layer_major import layer_major_model_class
            model_type=layer_major_model_class()
        from .deepseek_v41_source_approx import approximate_model_class
        model_type=approximate_model_class(model_type,self.kernel_policy)
        self.model = model_type(args, self.weights, identity=binding, hasher=EngramHasher(self.metadata), engram_tables=self.engram_tables,
            hc_policy=self.kernel_policy["deepseek_hc"], qat_policy=self.kernel_policy["deepseek_qat"],
            hc_coeff_policy=('source_fused_fp32' if self.kernel_policy.get('deepseek_source_hc_coefficients','reference')=='fused_fp32' else 'reference'),
            index_policy=self.kernel_policy.get('deepseek_index_policy','reference'),
            selected_kv_policy=self.kernel_policy.get('deepseek_selected_kv_policy','reference'),
            source_attention_math_policy=self.kernel_policy.get('deepseek_source_attention_math','reference'),
            source_hc_boundary_policy=self.kernel_policy.get('deepseek_source_hc_boundary','reference'),
            source_hc_mega_policy=self.kernel_policy.get('deepseek_source_hc_mega','reference'),
            **model_options(self.kernel_policy))
        self.model._mlx2_source_ast_target_laws=self._source_ast_target_laws
        self.model.source_ast_kv_preparation_policy=self.kernel_policy.get('deepseek_source_ast_kv_preparation','reference')
        self.model.recommended_prefill_rows = self.weights.recommended_prefill_rows
        if self.kernel_policy.get('deepseek_source_prefill_partition','adaptive_v1')=='position_grid_v1':
            # Compose over whatever cap is in force; never widen it.
            from .deepseek_v41_source_canonical_prefill import grid_row_cap
            self.model.recommended_prefill_rows = grid_row_cap(
                self.model.recommended_prefill_rows, self.weights.source_prefill_grid_rows)
        from .deepseek_v41_source_approx import configure_model as configure_approximate
        configure_approximate(self)
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
        if (not self.weight_stream.adaptive_residency_enabled
                and request.admission_limit_bytes is not None
                and mx.get_active_memory()+self.weight_stream.unfilled_reserve_bytes()+self.native_admission["headroom_bytes"]+self.native_admission["metadata_reserve_bytes"] > request.admission_limit_bytes):
            raise MemoryError("official source retained weights exceed early admission")
        self.layout = self.descriptor.cache_layout
        bind_cache_layout(self)
        self.max_context = self.model.context_limit
        loaded_auxiliary = []
        if 'dspark' in auxiliary:
            self._check_auxiliary_memory(loaded=loaded_auxiliary)
            from .deepseek_v41_dspark import configure_source_dspark
            configure_source_dspark(self)
            loaded_auxiliary.append('dspark')
        else:
            self._configure_copy_chain()
        if 'vision' in auxiliary:
            self._check_auxiliary_memory(loaded=loaded_auxiliary)
            self._configure_source_vision(artifact, auxiliary['vision']['reservation_bytes'])
            loaded_auxiliary.append('vision')
        if self.kernel_policy.get('deepseek_source_dspark_ast','off')=='tile3x5_v1':
            self._check_auxiliary_memory(loaded=loaded_auxiliary)
            self.configure_dspark_ast(oracle_path=self.kernel_policy['deepseek_source_dspark_oracle'],
                expert_cache_bytes=self.kernel_policy['deepseek_source_dspark_cache_bytes'],
                expert_law=self.kernel_policy.get('deepseek_source_dspark_target_expert_law','ordinary_m16_v1'),
                forward_law=self.kernel_policy.get('deepseek_source_dspark_target_forward','reference'))
            loaded_auxiliary.append('dspark_ast')
        if auxiliary:
            self._check_auxiliary_memory(loaded=loaded_auxiliary)
        if self.kernel_policy.get('deepseek_physical_batch', 'b1') != 'b1':
            from .deepseek_v41_batch import configure_physical_batch
            configure_physical_batch(self, width=int(self.kernel_policy['deepseek_physical_batch'][1:]))

    def configure_dspark_ast(self,*,oracle_path,expert_cache_bytes=512<<20,check_live=None,expert_law='ordinary_m16_v1',forward_law='reference'):
        """Explicit loaded-head AST route; actual16row oracle required by batch."""
        from .deepseek_v41_dspark_ast import SourceDSparkASTBinding
        from ..contracts import Capability,StatePlane
        from ..sampling_contract import GREEDY_REFUSAL_SAMPLING_CONTRACT
        from ..sampling_defaults import SamplingDefaults,VendorSampling
        from .deepseek_v41_source_compatibility import validate_source_profiles
        validate_source_profiles({**self.kernel_policy, 'deepseek_source_dspark_ast': 'tile3x5_v1'})
        model=self.model
        def owned():
            if self.model is not model or getattr(self.weights,'_closed',False):
                raise RuntimeError('Source DSpark target owner changed')
            self.weights._ready();return True
        binding=SourceDSparkASTBinding(self,check_live=owned if check_live is None else check_live,
            expert_cache_bytes=expert_cache_bytes,oracle_path=oracle_path,expert_law=expert_law,forward_law=forward_law)
        self._dspark_ast_binding=binding;self.draft_model=binding.draft
        self.target_identity=dict(self.identity)
        self.identity={**self.identity,'target_fingerprint':self.target_identity['fingerprint'],
            'draft_fingerprint':binding.backbone.record['fingerprint'],'fingerprint':binding.binding}
        self.layout+=':dspark-ast3x5:'+binding.binding
        self.profile+=':dspark-ast3x5-'+binding.binding[:16]
        self.numerics_revision+=':dspark-ast3x5:'+binding.binding
        self.default_route='external_draft'
        self.sampling_defaults=VendorSampling.single(SamplingDefaults(temperature=0.0,top_p=1.0,top_k=0,
            source='V4.1 embedded DSpark AST greedy contract'),model='DeepSeek-V4.1-Source+DSpark-AST3x5')
        self.descriptor=replace(self.descriptor,capabilities=self.descriptor.capabilities|{Capability.EXTERNAL_DRAFT},
            state_planes=self.descriptor.state_planes|{StatePlane.DRAFT},
            metadata={**self.descriptor.metadata,'qualification':'unqualified','packed_ast':True,
                'sampling_contract':GREEDY_REFUSAL_SAMPLING_CONTRACT,'dspark':binding.receipt()})
        return binding

    def create_external_batch(self,**kwargs):
        binding=getattr(self,'_dspark_ast_binding',None)
        if binding is not None:return binding.create_external_batch(**kwargs)
        return super().create_external_batch(**kwargs)

    def _check_auxiliary_memory(self, *, loaded):
        import mlx.core as mx
        from ..memory import host_available_bytes
        from .deepseek_v41_source_auxiliary import auxiliary_memory_requirement, check_auxiliary_memory
        required = auxiliary_memory_requirement(self.native_admission, loaded=loaded)
        available = host_available_bytes(host_signals=True)
        if type(available) is not int or available < 0:
            raise MemoryError('Source auxiliary reservation requires measured available memory')
        reclaim = getattr(self.weight_stream, 'rebalance_headroom', None)
        if callable(reclaim):
            limit = self.native_admission.get('admission_limit_bytes')
            protected = required if limit is None else max(required, available + int(mx.get_active_memory()) + required - limit)
            reclaim(available_bytes=available, protected_bytes=protected)
        receipt = check_auxiliary_memory(self.native_admission, loaded=loaded,
            active_bytes=int(mx.get_active_memory()),
            available_bytes=host_available_bytes(host_signals=True))
        self.native_admission['auxiliary_memory_receipt'] = receipt
        return receipt

    def _configure_source_vision(self, artifact, reservation_bytes):
        from ..contracts import Capability
        from .deepseek_v41_vision import SourceVision

        if getattr(self, 'vision_weights', None) is not None:
            raise ValueError('Source vision cannot replace an existing owner')
        # Install ownership before upload, but publish the callback/capability
        # only after every admitted vision parameter has loaded successfully.
        self.vision_weights = SourceVision(artifact, self.model, admitted_bytes=reservation_bytes)
        self.vision_weights.load()
        self.model.vision_inputs = self.vision_weights
        self.descriptor = replace(self.descriptor,
            capabilities=self.descriptor.capabilities | {Capability.VISION},
            metadata={**self.descriptor.metadata, 'vision': 'Source BF16 ordinary B1; unqualified'})

    def prompt_tokens(self, request):
        if '_mlx2_prompt_tokens' in request:
            return list(request['_mlx2_prompt_tokens'])
        return super().prompt_tokens(request)

    def prepare_multimodal_request(self, request, *, file_loader=None):
        from .deepseek_v41_vision import prepare_request
        return prepare_request(self, request, file_loader=file_loader)

    def validate_prefill_inputs(self, request, remaining_tokens, prefill_input):
        if list(remaining_tokens) != list(request['_mlx2_prompt_tokens']):
            raise ValueError('V4.1 images require a complete uncached media prefill')
        if set(prefill_input) != {'image_inputs'}:
            raise ValueError('unexpected V4.1 media prefill inputs')
        return prefill_input

    def close(self):
        # Both auxiliaries borrow target weights. A failed fence must retain
        # every borrowed owner; the target cannot retire underneath them.
        error = None
        binding = getattr(self, '_dspark_ast_binding', None)
        if binding is not None:
            binding.close()
            self._dspark_ast_binding = None
        for name in ('draft_weights', 'vision_weights'):
            owner = getattr(self, name, None)
            if owner is None:
                continue
            try:
                owner.close()
            except BaseException as failure:  # noqa: BLE001 - re-raised before borrowed target retirement
                error = error or failure
            else:
                setattr(self, name, None)
        if error is not None:
            raise error
        super().close()

    def checkpoint_resource_binding(self):
        from .deepseek_v41_source_checkpoint_resources import binding
        return binding(self)

    def checkpoint_resource_hints(self,prompt_cache,tokens):
        from .deepseek_v41_source_checkpoint_resources import snapshot
        return snapshot(self,prompt_cache,tokens)

    def restore_checkpoint_resources(self,hints,prompt_cache):
        from .deepseek_v41_source_checkpoint_resources import restore
        return restore(self,hints,prompt_cache)

    def cache_budget(self, *, mtp):
        from ..runtime.demand_stream_memory import DemandStreamCacheBudget
        return DemandStreamCacheBudget(super().cache_budget(mtp=mtp),
                                       self.model.recommended_prefill_rows)

    def execution_config(self, *, max_lanes, prefill_step):
        from .deepseek_v41_source_layer_major import CHUNK_ROWS, MAX_WAVE_ROWS, POLICY
        wave = (self.kernel_policy.get('deepseek_source_prefill_order','reference')==POLICY
                and type(prefill_step) is int and CHUNK_ROWS < prefill_step <= MAX_WAVE_ROWS)
        if wave and (getattr(self,'model',None) is not None
                     and getattr(type(self.model),'source_layer_major_rows',None) is None):
            raise ValueError('Source layer-major prefill waves were not bound at load')
        # The ordinary validator admits one chunk; a wave is whole chunks.
        from .deepseek_v41_source_approx import apc_refusal_reason, YOCO_KEY, YOCO_MIN_ROWS
        yoco = self.kernel_policy.get(YOCO_KEY,'off')!='off'
        if yoco and (type(prefill_step) is not int or not self.kernel_policy[YOCO_MIN_ROWS] <= prefill_step <= 1048576):
            raise ValueError('Source approximate yoco_v1 requires prefill_step >= its minimum prompt rows')
        result = super().execution_config(max_lanes=max_lanes, prefill_step=CHUNK_ROWS if wave or yoco else prefill_step)
        if wave or yoco:
            result['prefill_step_size'] = prefill_step
        selected=self.kernel_policy.get('deepseek_source_prefill_partition','adaptive_v1')
        if apc_refusal_reason(self.kernel_policy) is not None:
            result['apcv2_reuse_disabled_reason'] = apc_refusal_reason(self.kernel_policy)
        elif self._source_dynamic_precision():
            from .deepseek_v41_source_residency import DYNAMIC_APC_REASON
            result['apcv2_reuse_disabled_reason'] = DYNAMIC_APC_REASON
        elif selected not in ('canonical_m1_v1','position_grid_v1'):
            result['apcv2_reuse_disabled_reason'] = 'adaptive_prefill_partitions_not_qualified'
        elif (getattr(self.model,'_source_canonical_identity',None)!=self.model.identity
                or getattr(self.weights,'source_prefill_partition_policy',None)!=selected):
            raise ValueError('Source canonical APC profile was not bound at load')
        elif selected=='position_grid_v1':
            grid=getattr(self.weights,'source_prefill_grid_rows',None)
            cap=getattr(self.model,'recommended_prefill_rows',None)
            if grid!=self.kernel_policy.get('deepseek_source_prefill_grid_rows') or getattr(cap,'source_prefill_grid_rows',None)!=grid:
                raise ValueError('Source grid APC profile was not bound at load')
            if type(prefill_step) is int and prefill_step<grid:
                # Every chunk would start off the grid: nothing could publish.
                raise ValueError(f'Source position_grid_v1 grid {grid} exceeds prefill step {prefill_step}')
        return result

    def apc_partition_checkpoint_position(self, tokens, *, cached_tokens=0, cuts=()):
        """Deepest publishable grid boundary of a cold prefill, else None."""
        if self.kernel_policy.get('deepseek_source_prefill_partition','adaptive_v1')!='position_grid_v1':
            return None
        from .deepseek_v41_source_canonical_prefill import aligned_checkpoint
        return aligned_checkpoint(len(tokens), self.weights.source_prefill_grid_rows,
                                  cached_tokens=cached_tokens, cuts=cuts)

    def _source_dynamic_precision(self):
        expert=self.kernel_policy.get('deepseek_source_experts') or {}
        return expert.get('residency','shared_v1')=='dynamic_precision_v1'

    def apc_publication_refusal(self, prompt_cache, tokens):
        """Quietly decline off-grid or tainted grid state before APCv2 store."""
        from .deepseek_v41_source_approx import apc_refusal_reason
        if apc_refusal_reason(self.kernel_policy) is not None:
            return apc_refusal_reason(self.kernel_policy)
        if self._source_dynamic_precision():
            from .deepseek_v41_source_residency import DYNAMIC_APC_REASON
            return DYNAMIC_APC_REASON
        if self.kernel_policy.get('deepseek_source_prefill_partition','adaptive_v1')!='position_grid_v1':
            return None
        from .deepseek_v41_source_canonical_prefill import grid_publication_refusal
        rows=list(prompt_cache)
        reason=grid_publication_refusal(rows,self.weights.source_prefill_grid_rows)
        if reason is None and tokens is not None and len(tokens)!=getattr(rows[0],'offset',None):
            return 'Source grid publication length disagrees with cache offset'
        return reason

    def prepare_resident_bank_session(self,bank,*,ownercheck,cancelled=None,prefill=False):
        """Attach an explicitly prepared closed decode bank; no implicit loading."""
        from .deepseek_v41_source_compatibility import validate_source_profiles
        validate_source_profiles({**self.kernel_policy, 'deepseek_source_resident_gather': 'resident_bank_v1'})
        from .deepseek_v41_source_bank_session import PreparedBankSession
        if getattr(bank,'_store',None)is not self.weight_stream:
            raise ValueError('Prepared Source bank belongs to another expert store')
        if self.weights.source_resident_bank_session is not None:
            raise ValueError('Source resident bank session already attached')
        bank.assert_current()
        if type(prefill)is not bool:raise ValueError('Source bank prefill selection must be boolean')
        session_type=PreparedBankSession
        if prefill:
            if self.kernel_policy.get('deepseek_source_resident_prefill','reference')not in ('expert_major_v1','expert_major_wide_v1'):
                raise ValueError('Source PP bank attachment requires explicit expert-major policy')
            from .deepseek_v41_source_prefill_session import PreparedPrefillBankSession
            session_type=PreparedPrefillBankSession
        session=session_type(self.weights,bank,ownercheck=ownercheck,cancelled=cancelled)
        self.weights.source_resident_bank_session=session
        return session

    def diagnostics(self):
        value = super().diagnostics()
        value.update(base_artifact=self.identity.get('base_fingerprint', self.identity['fingerprint']),
                     effective_artifact=self.identity['fingerprint'],
                     artifact_derivation=self.identity.get('derivation'))
        from .deepseek_v41_source_canonical_prefill import partition_receipt
        partition=partition_receipt(self.kernel_policy.get('deepseek_source_prefill_partition','adaptive_v1'),
                                    self.kernel_policy.get('deepseek_source_prefill_grid_rows'))
        value['source_prefill_partition']=partition
        from .deepseek_v41_source_layer_major import receipt as order_receipt
        value['source_prefill_order']=order_receipt(self.kernel_policy.get('deepseek_source_prefill_order','reference'),
                                                    getattr(self,'weights',None))
        binding=getattr(self,'_dspark_ast_binding',None)
        if binding is not None:
            value['speculation']=binding.receipt()
        session=getattr(self.weights,'source_resident_bank_session',None)
        value['source_resident_gather']=dict(policy=self.kernel_policy.get('deepseek_source_resident_gather','reference'),
            session=None if session is None else session.receipt(),qualification='unqualified')
        value['source_resident_prefill']=dict(policy=self.kernel_policy.get('deepseek_source_resident_prefill','reference'),qualification='unqualified')
        value['source_submission'] = getattr(self.weights,'source_submission_policy','reference')
        value['source_row_packet'] = getattr(self.weights,'source_row_packet_policy','reference')
        value['scheduled_kv_calls'] = dict(getattr(self.model,'scheduled_kv_calls',{}))
        value['source_memory'] = {
            'policy':'adaptive-demand-v1',
            'prefill':getattr(self.weights,'source_memory_advice',None),
            'experts':getattr(self.weights,'source_expert_memory_advice',None),
            'completion':getattr(self.weights,'source_completion_memory_advice',None),
            'auxiliary_plans':self.native_admission.get('auxiliary_plans',{}),
            'auxiliary_memory_receipt':self.native_admission.get('auxiliary_memory_receipt'),
            'apcv2_reuse':('disabled: dynamic_precision_v1_output_depends_on_cache_history' if self._source_dynamic_precision()
                           else ('enabled: '+partition['policy']+' (unqualified)') if partition['apcv2_reuse_enabled']
                           else 'disabled: adaptive_prefill_partitions_not_qualified'),
            'expert_residency':(self.weight_stream.residency_receipt() if hasattr(getattr(self,'weight_stream',None),'residency_receipt') else None),
            'hot_map_provenance':getattr(getattr(self,'weight_stream',None),'hot_map_provenance',None),
        }
        from .deepseek_v41_source_resident import resident_capabilities
        value['source_resident_routes'] = resident_capabilities()
        value.update(original_source_required=True, expert_storage="original E2M1/E8M0 group32",
                     source_expert_math=self.weights.expert_math_policy,
                     dense_storage={**self.weights.dense_receipt(), 'source_hc_boundary':self.model.source_hc_boundary_policy}, checkpoint_revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
                     native_dspark="not selected"if binding is None else binding.receipt(), vision="unavailable in this text serving slice",
                     numerical_scope="MLX reference reductions; original weight codes and activation quantization; CUDA parity unproven")
        value['router_precision'] = getattr(self, 'source_artifact', {}).get('router_precision')
        if getattr(self.draft_model, 'kind', None) == 'deepseek_v41_dspark':
            value.update(native_dspark='selected; embedded Source draft; greedy B1; unqualified',
                         speculation=self.external_record)
        if getattr(self, 'vision_weights', None) is not None:
            value.update(vision='selected; native Source BF16 image prefill; ordinary B1; unqualified',
                         vision_admission=self.vision_weights.plan)
        value.update(physical_width=getattr(self, 'physical_batch_width', 1),
                     batch_execution=getattr(self.model, 'batch_execution_profile', None),
                     scheduled_batch_calls=dict(getattr(self.model, 'batch_counters', {})))
        value.update(index_policy=self.model.index_policy,selected_kv_policy=self.model.selected_kv_policy,
                     source_qakv=self.weights.source_qakv_policy,
                     source_q_b_prefill=self.weights.source_q_b_prefill_policy,
                     source_fp8_pp=getattr(self.weights,'source_fp8_pp_policy','reference'),
                     source_woa_pp=getattr(self.weights,'source_woa_pp_policy','reference'),
                     source_woa_storage=getattr(self.weights,'source_woa_storage_policy','reference'),
                     source_native_mx=getattr(self.weights,'source_native_mx_policy','reference'),
                     source_frame_reuse=getattr(self.weights,'source_frame_reuse_policy','reference'),
                     source_ffn_input=getattr(self.weights,'source_ffn_input_policy','reference'),
                     source_prefill_down=getattr(self.weights,'source_prefill_down_policy','reference'),
                     source_attention_math=self.model.source_attention_math_policy,
                     source_hc_boundary=self.model.source_hc_boundary_policy,
                     source_hc_boundary_counters={k:v for k,v in self.model.scheduled_hc_calls.items()if k.startswith('source_hc_boundary_')},
                     hc_coeff_policy=self.model.hc_coeff_policy,
                     source_hc_mega=getattr(self.model,'source_hc_mega_policy','reference'),
                     source_hc_mega_counters={k:v for k,v in self.model.scheduled_hc_calls.items()if k.startswith('source_hc_mega_')},
                     index_score_contract=('finite unmasked index scores required' if self.model.index_policy=='finite_all_retained' else 'ordinary reference behavior'))
        if self.kernel_policy.get('deepseek_source_experts',{}).get('mode')=='hot_original_q1':
            value.update(expert_storage='frozen hot map: original E2M1/E8M0; all other routed experts: affine Q1 g64',
                         numerical_scope='approximate Q1 expert derivative; dense/shared/attention/router/Engram original; unqualified')
            if self._source_dynamic_precision():
                value.update(expert_storage='dynamic_precision_v1: each routed expert served from its resident MXFP4 high-pool copy or Q1 g64; promotions/demotions only at cycle boundaries',
                             numerical_scope='approximate; precision depends on cache history (fixed within each forward); APCv2 disabled; unqualified')
        if self.model.source_attention_math_policy == 'omlx_nax_bf16_online_v1':
            from ..runtime.models.deepseek_v41_omlx_nax_attention import PROFILE
            from ..runtime.models.deepseek_v41_source_attention import (
                implementation_identity,
            )
            value['source_attention_numerical_profile'] = PROFILE
            value['source_attention_implementation_sha256'] = implementation_identity()
            value['numerical_scope'] = (
                ('Approximate Q1 expert derivative; ' if self.kernel_policy.get('deepseek_source_experts',{}).get('mode')=='hot_original_q1' else 'Original weight codes; ')
                + 'opt-in PP attention BF16 NAX QK/online F32 softmax/FP16 hi+lo PV; distinct numerical profile with existing owner BF16; decode/reference fallback unchanged; unqualified')
        if self.model.source_attention_math_policy in ('source_short_decode_joint_f32_v1','source_short_decode_grouped_f32_v1'):
            from ..runtime.models.deepseek_v41_short_attention import implementation_identity
            value['source_attention_numerical_profile'] = self.model.source_attention_math_policy
            value['source_attention_implementation_sha256'] = implementation_identity()
            value['numerical_scope'] += '; explicit short-band F32 attention reduction profile; existing owner BF16; GPU gates pending; unqualified'
        if getattr(self.weights,'source_fp8_pp_policy','reference')=='native_bf16_qb_wob_v1':
            from .deepseek_v41_source_fp8_pp import POLICY, implementation_identity
            value['source_fp8_pp_numerical_profile']=POLICY
            value['source_fp8_pp_implementation_sha256']=implementation_identity()
            value['numerical_scope']=value['numerical_scope'].replace('MLX reference reductions;', 'Existing Source profile;')
            value['numerical_scope']+='; explicit PP99 Q_b/WoB native BF16 arithmetic over unchanged original FP8 codes, distinct numerical profile; decode/reference fallback unchanged; unqualified'
        if getattr(self.weights,'source_prefill_down_policy','reference')!='reference':
            from .deepseek_v41_source_prefill_down import (
                TINY_CONTRACT,
                TINY_COUNTER,
                implementation_identity,
                tiny_implementation_identity,
            )
            tiny=getattr(self.weights,'source_prefill_down_policy','reference')=='q1_tiny_batch_v1'
            value['source_prefill_down_implementation_sha256']=tiny_implementation_identity()if tiny else implementation_identity()
            value['source_prefill_down_contract']=TINY_CONTRACT if tiny else 'PP-only Q1 M1 final BF16 owner store fused; unchanged dot program and coefficient codes; explicit/unqualified'
            if tiny:
                value['source_prefill_down_counters']={k:v for k,v in self.weights.scheduled_expert_calls.items()if k.startswith(TINY_COUNTER)}
        if getattr(self.weights,'source_native_mx_policy','reference')!='reference':
            from .deepseek_v41_source_native_mx import POLICY, implementation_identity
            value['source_native_mx_numerical_profile']=POLICY
            value['source_native_mx_implementation_sha256']=implementation_identity()
            value['source_native_mx_counters']={k:v for k,v in self.weights.scheduled_expert_calls.items() if k.startswith('official_source_native_mx_')}
            value['numerical_scope']=value['numerical_scope'].replace('MLX reference reductions;','Existing Source profile;')
            value['numerical_scope']+='; opt-in original PP expert M16..99 native HALF activation consumption (rounding allowed), distinct profile; original coefficient precision/Source middle/owners unchanged; Q1/small-M/decode reference; unqualified'
        from .deepseek_v41_source_approx import receipt as approximate_receipt, selected_operations
        value['source_approximate_operations']=approximate_receipt(self.kernel_policy,getattr(self,'model',None))
        if selected_operations(self.kernel_policy):
            value['approximate_state']=True
            value['numerical_scope']+='; APPROXIMATE '+'+'.join(selected_operations(self.kernel_policy))+' (opt-in, salted identity, APCv2 publication refused); unqualified'
        return value
