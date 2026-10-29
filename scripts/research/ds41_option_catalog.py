# SPDX-License-Identifier: MIT
"""CPU-only DS4.1 Source option inventory and dependency-complete test arms.

This only parses adapter policy. It never imports MLX, loads a model, allocates
weights, acquires GPU ownership, or claims that a selected branch executed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from mlx2.adapters.deepseek_v41_source import SOURCE_POLICY_CHOICES, source_policy
from mlx2.adapters.deepseek_v41_mlx import POLICY_CHOICES
from mlx2.adapters.deepseek_v41_source_math import POLICIES

R = 'routed_fused_fp32'
F = {'deepseek_source_expert_math': R}
G = {**F, 'deepseek_source_prefill_batch': 'gate_up'}
AST_KEYS = {'deepseek_source_dspark_ast', 'deepseek_source_dspark_prepare',
            'deepseek_source_dspark_target_expert_law', 'deepseek_source_dspark_target_forward',
            'deepseek_source_ast_kv_preparation'}
EXTENSIONS = {'deepseek_kv_storage', 'deepseek_candidate_score', 'deepseek_ced',
              'deepseek_vision', 'deepseek_physical_batch'}
PP_KEYS = {'deepseek_source_prefill_batch', 'deepseek_source_prefill_wide_math',
           'deepseek_source_prefill_wide_dtype', 'deepseek_source_q_b_prefill',
           'deepseek_source_fp8_pp', 'deepseek_source_woa_pp',
           'deepseek_source_native_mx', 'deepseek_source_prefill_down',
           'deepseek_source_frame_reuse', 'deepseek_source_row_packet'}


def prerequisites(key, value, *, oracle=None):
    p = {}
    if key in {'deepseek_source_prefill_batch', 'deepseek_source_resident_gather'}:
        p.update(F)
    if key in {'deepseek_source_prefill_wide_math', 'deepseek_source_prefill_wide_dtype',
               'deepseek_source_prefill_down', 'deepseek_source_native_mx',
               'deepseek_source_frame_reuse'}:
        p.update(G)
    if key == 'deepseek_source_row_packet':
        p.update(G, deepseek_source_frame_reuse='prefill_v1')
    if key == 'deepseek_source_ffn_input':
        p.update(F, deepseek_source_dense_qat='fused_fp32')
    if key == 'deepseek_source_qakv':
        p['deepseek_source_dense_qat'] = 'fused_fp32'
    if key == 'deepseek_source_attention_math':
        p['deepseek_index_policy'] = 'finite_all_retained'
    if key == 'deepseek_attention_epilogue':
        # Parser accepts reference QAT, but the producer branch requires fused QAT.
        p['deepseek_qat'] = 'fused_fp32'
    if key == 'deepseek_hc_epilogue':
        p['deepseek_hc'] = 'fused_fp32'
    if key == 'deepseek_source_resident_prefill':
        p.update(F, deepseek_source_resident_gather='resident_bank_v1')
    if key in AST_KEYS:
        p.update(F)
        if key in {'deepseek_source_dspark_prepare', 'deepseek_source_ast_kv_preparation'}:
            p.update(deepseek_source_dspark_prepare='oracle_v1',
                     deepseek_source_dspark_target_forward='fully_batched_m1_v1',
                     deepseek_source_dspark_target_expert_law='source_ast_m1_rows_v1')
        else:
            p.update(deepseek_source_dspark_ast='tile3x5_v1',
                     deepseek_source_dspark_oracle=oracle or str(Path(__file__).resolve()))
        if key == 'deepseek_source_dspark_target_expert_law':
            if value == 'source_ast_resident_rows_v1':
                p['deepseek_source_dspark_target_forward'] = 'fully_batched_v1'
            elif value == 'source_ast_hybrid_rows_v1':
                p.update(deepseek_source_dspark_target_forward='fully_batched_m1_v1',
                         deepseek_source_prefill_partition='canonical_m1_v1')
    p[key] = value
    return p


def describe(key, value):
    lane = 'ordinary_text'
    trigger = 'Cold ordinary prompt and generated decode; inspect named scheduling counters.'
    resources = []
    if key in PP_KEYS:
        lane = 'adaptive_prefill'
        trigger = 'Cold multi-row PP; include 100 prompt tokens so guard-prefill produces M99, plus M1 decode fallback.'
    if key in {'deepseek_source_q_b_prefill', 'deepseek_source_fp8_pp', 'deepseek_source_woa_pp'}:
        trigger = 'Exact PP M99 required; Q_b [1,99,1280], WoB [1,99,8192], WoA [1,99,8,4096]. Canonical M1 cannot exercise this option.'
    if key == 'deepseek_source_prefill_down':
        trigger = 'Hybrid Q1 route required; owner-store local Q1 M1, tiny-batch local M1..3. Original-only test cannot exercise Q1 branch.'
    if key == 'deepseek_source_native_mx':
        trigger = 'Cold PP with original MXFP4 routed experts and gate_up; verify native eligibility/dispatch and fallbacks separately.'
    if key == 'deepseek_source_attention_math':
        if value == 'omlx_nax_bf16_online_v1':
            lane = 'adaptive_prefill'
            trigger = 'B1/H64/D512 PP M16..128; chronological all-retained pool P1..512, local rows1..255. Decode uses reference fallback.'
        elif value.startswith('source_short_decode'):
            trigger = 'B1/H64/M1/D512 decode; local rows1..128 and selected pool rows0..512; test with and without decode_reuse.'
        else:
            lane = 'adaptive_prefill'
            trigger = 'Multi-row PP with chronological all-retained pooled rows<=512. Larger pools require explicit fallback observation.'
    if key == 'deepseek_index_policy':
        trigger = 'Short pool <=512 exercises finite all-retained bypass; long pool >512 exercises selector fallback.'
    if key == 'deepseek_selected_kv_policy':
        trigger = 'Repeated M1 decoder layers sharing a selected pooled source; verify hit, invalidation and source/frontier binding.'
    if key in EXTENSIONS:
        lane = {'deepseek_vision':'vision', 'deepseek_physical_batch':'physical_batch',
                'deepseek_ced':'ced', 'deepseek_kv_storage':'packed_kv',
                'deepseek_candidate_score':'candidate_score'}[key]
    if key == 'deepseek_ced':
        trigger = 'Ordinary cold prompt longer than decoder suffix/window; compare correct dependency-complete suffix and full reference state, partition-aligned.'
    if key == 'deepseek_candidate_score':
        trigger = 'Compressed pool beyond selection limit with real candidate subset; short <=512 all-retained path alone does not qualify selection savings.'
    if key == 'deepseek_kv_storage':
        trigger = 'Cold PP, decode, native FP8/FP4 code/scale state, snapshots and restoration; reference QAT/attention producers and replay verifier.'
    if key == 'deepseek_vision':
        trigger = 'Actual image decode/encode input, ordinary output and image identity; text-only success is no vision coverage.'
        resources = ['validated image request', 'checkpoint visual tensors']
    if key == 'deepseek_physical_batch':
        trigger = f'Concurrent distinct ordinary text requests filling physical {value}; per-lane output/state and actual execution-width receipts.'
    if key in AST_KEYS:
        lane = 'dspark_prepare' if key in {'deepseek_source_dspark_prepare','deepseek_source_ast_kv_preparation'} else 'dspark_ast'
        trigger = 'Exact16-row target oracle plus actual tile3x5 candidate verification, accepted/rejected prefixes and continuation; preparation alone never executes DSpark.'
        resources = ['fresh exact-source target oracle']
    if key in {'deepseek_source_resident_gather','deepseek_source_resident_prefill'}:
        lane = 'resident_bank'
        trigger = 'Explicit closed-query expert inventory; caller-owned idle bank build, attach prepare_resident_bank_session and actual in-bank route. Ordinary load alone is insufficient.'
        resources = ['owned prepared closed-set bank', 'fresh expert route inventory', 'bank peak/staging budget']
    if key == 'deepseek_verify_commit':
        lane = 'prompt_lookup'
        trigger = 'Prompt-lookup draft with num_draft1..5; induce both acceptance and rejection. Ordinary generation never exercises verifier commit.'
    return lane, trigger, resources


def catalog(*, oracle=None):
    choices = {**POLICY_CHOICES, **SOURCE_POLICY_CHOICES,
               'deepseek_source_expert_math': POLICIES}
    arms = []
    for key, values in choices.items():
        for value in values[1:]:
            policy = prerequisites(key, value, oracle=oracle)
            if key == 'deepseek_verify_commit':
                policy.update(draft_model='prompt_lookup', num_draft=5)
            lane, trigger, resources = describe(key, value)
            validation_policy = dict(policy)
            if key == 'deepseek_source_native_mx':
                validation_policy['deepseek_source_experts'] = {'mode':'original_lfu','budget_bytes':75_000_000_000}
            parsed = source_policy(validation_policy)
            arms.append(dict(id=key.removeprefix('deepseek_')+'__'+value,
                option=key,value=value,policy=policy,lane=lane,coverage_trigger=trigger,
                resources=resources,policy_parser_passed=True,
                requires_sealed_expert_policy=key=='deepseek_source_native_mx',
                parser_oracle_placeholder=(key in AST_KEYS and lane=='dspark_ast' and oracle is None),
                apcv2_reuse='same_exact_profile_only; adaptive default explicitly disables reuse',
                cross_arm_apcv2='incompatible: policy is included in model identity; never transplant state',
                cache_reuse_lane_viable=(lane not in {'adaptive_prefill','packed_kv','candidate_score','ced','vision','physical_batch'}),
                qualification=False,observed_used=False))
    return dict(schema='mlx2.ds41-option-catalog.v1',
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        catalog_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        owning_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'src/mlx2/adapters'/name for name in ('deepseek_v41_source.py','deepseek_v41_mlx.py',
             'deepseek_v41_execution.py','deepseek_v41_source_math.py','deepseek_v41_source_compatibility.py','deepseek_v41_batch.py','deepseek_v41_source_lfu.py')]},
        scope='CPU source inventory and parser admission only',
        defaults={key:values[0]for key,values in choices.items()},choices=choices,
        arms=arms,arm_count=len(arms),option_count=len(choices),
        serving_tunables={
            'prefill_step': {'cells':[1,32,64,128,256], 'coverage':'Adaptive PP numerical partitions may differ; canonical M1 preserves math but excludes wide-PP candidates.'},
            'expert_cache_bytes': {'cells':[32<<30,48<<30,75_000_000_000], 'coverage':'Capacity profile; retain immutable precision map. Constructor supports at most90decimalGB, with one-expert-plus-staging minimum.'},
            'expert_read_workers': {'cells':[1,2,4,8], 'coverage':'ReadPool max_pending1 and synchronous demand can make extra workers ineffective; measure actual read counters.'},
            'expert_preheat': {'cells':[False,True], 'coverage':'Explicit true warms observed atlas experts within measured availability, staging and working headroom; it defers absent/over-budget experts. Default adaptive demand remains. Advisory warmth is not a pin.'},
            'apcv2_cache_bytes': {'cells':[64<<20,256<<20,1<<30], 'coverage':'Canonical profile only for exact reuse. Record actual hit counts and retention/persistence separately.'},
            'context_capacity': {'cells':[1024,2048,4096,8192,16384,32768,65536], 'coverage':'Capacity is not prompt length. Above65536 requires native_packed+candidate_only and separate profile.'},
            'draft_num_draft': {'cells':[1,2,3,4,5], 'coverage':'Separate prompt_lookup/embedded_dspark profiles, acceptance and rejection; AST tile geometry fixed3x5.'},
            'max_lanes_max_inflight': {'cells':[[1,1],[2,2],[4,4]], 'coverage':'Source b1 supports one stream lane; b2/b4 require reference Source provider geometry and actual concurrent requests.'},
        },unsupported_component_only=['source_resident_explicit_resource_pilot_v1 arbitrary-residency serving: require_serving_backend() always refuses; separate from supported closed-set resident-bank path'],
        current_state='All statuses unqualified until exact artifact/source/shape evidence is produced.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',help='Existing oracle candidate path; content/source bindings still require runtime validation')
    args = parser.parse_args()
    print(json.dumps(catalog(oracle=args.oracle),indent=2,sort_keys=True))
