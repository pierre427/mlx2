"""CPU route policy and provider wiring; no actual checkpoint execution."""
from types import SimpleNamespace

import pytest

from mlx2.adapters.deepseek_v41 import execution_policy as gguf_policy
from mlx2.adapters.deepseek_v41_execution import (
    bind_cache_layout,
    execution_receipt,
    model_options,
)
from mlx2.adapters.deepseek_v41_mlx import DESCRIPTOR, DeepSeekV41MLXAdapter
from mlx2.adapters.deepseek_v41_mlx import execution_policy as native_policy
from mlx2.adapters.deepseek_v41_source import source_policy


@pytest.mark.parametrize('parse', [gguf_policy, native_policy, source_policy])
def test_csa_profiles_are_explicit_and_incompatible_producers_refused(parse):
    base = parse({})
    assert model_options(base)['kv_storage_policy'] == 'reference'
    selected = parse({'deepseek_kv_storage': 'native_packed', 'deepseek_candidate_score': 'candidate_only',
                      'deepseek_context_limit': 1048576})
    assert model_options(selected) == {'kv_storage_policy': 'native_packed',
        'candidate_score_policy': 'candidate_only', 'ced_policy': 'reference', 'ced_state_only': False,
        'context_limit': 1048576}
    request = {'deepseek_kv_storage': 'native_packed', 'deepseek_candidate_score': 'candidate_only',
               'deepseek_context_limit': 1048576}
    # Fused QAT/attention-epilogue producers have an exact packed producer.
    fused = parse({**request, 'deepseek_qat': 'fused_fp32', 'deepseek_attention_epilogue': 'fused'})
    assert model_options(fused)['kv_storage_policy'] == 'native_packed'
    assert fused['deepseek_qat'] == 'fused_fp32'
    with pytest.raises(ValueError, match='snapshot/replay'):
        parse({**request, 'deepseek_verify_commit': 'local_frontier', 'draft_model': 'prompt_lookup'})
    for value in (True, 0, -1, 1048577, 2.5, '1048576'):
        with pytest.raises(ValueError):
            parse({'deepseek_context_limit': value})
    with pytest.raises(ValueError, match='above65536'):
        parse({'deepseek_context_limit': 1048576})
    with pytest.raises(ValueError, match='separate ordinary'):
        parse({'deepseek_ced': 'dependency_exact', 'draft_model': 'prompt_lookup'})


def test_native_copy_can_use_exact_local_frontier_and_keeps_layout():
    adapter = DeepSeekV41MLXAdapter.__new__(DeepSeekV41MLXAdapter)
    adapter.model = SimpleNamespace(args=SimpleNamespace(vocab_size=19, hidden_size=8), layers=range(3))
    adapter.kernel_policy = native_policy({'draft_model': 'prompt_lookup', 'num_draft': 3,
                                           'deepseek_verify_commit': 'local_frontier'})
    adapter.descriptor = DESCRIPTOR
    adapter._configure_copy_chain()
    assert adapter.draft_model.receipt_settings['verified_prefix_commit'] == 'v41_local_frontier'
    assert adapter.draft_model.num_draft == 3
    assert adapter.descriptor.cache_layout == DESCRIPTOR.cache_layout


def test_packed_layout_binding_and_receipt_are_route_owned():
    from mlx2.runtime.models.deepseek_v41_packed_cache import CACHE_LAYOUT
    model = SimpleNamespace(kv_storage_policy='native_packed', candidate_score_policy='candidate_only',
                            ced_policy='dependency_exact', context_limit=1048576)
    adapter = SimpleNamespace(model=model, descriptor=DESCRIPTOR, layout=DESCRIPTOR.cache_layout)
    bind_cache_layout(adapter)
    assert adapter.layout == adapter.descriptor.cache_layout == CACHE_LAYOUT
    receipt = execution_receipt(model)
    assert receipt['supported_context'] == 1048576
    assert receipt['full_depth_prefill'] is False
    assert receipt['approximate_replay'] is False
    assert 'unqualified' in receipt['long_context_evidence']
