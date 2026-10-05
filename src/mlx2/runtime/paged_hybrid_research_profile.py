"""Source-bound, default-off hybrid B2 permit; never a performance price."""
from __future__ import annotations
import json
from pathlib import Path
from .paged_pack_price import _identity
SCHEMA = 'mlx2.native-hybrid-b2-research-admission.v1'

def validate_hybrid_stock_pair(profile, context_lengths):
    """Fail before allocation/write for the narrow stock32 virtual-padding ABI."""
    selected = profile.get('stock_reduction', False)
    singleton = profile.get('stock_singleton', False)
    if type(singleton) is not bool or (singleton and not selected):
        raise ValueError('stock_singleton requires exact boolean and stock reduction')
    if type(selected) is not bool:
        raise ValueError('hybrid stock_reduction must be an exact boolean')
    if selected and (profile['q1_split_partition'] != 0 or
            profile['q1_simd_stripes'] not in (8, 16) or
            type(context_lengths) is not tuple or len(context_lengths) != 2 or
            any(type(n) is not int or not 32 <= n <= 128 - profile['max_tokens'] + 1
                for n in context_lengths) or
            context_lengths[0] == context_lengths[1] or
            abs(context_lengths[0] - context_lengths[1]) % 32):
        raise ValueError('stock32 serving requires short aligned ragged B2 and survivor stripes8/16')
    return selected


def require_hybrid_native_capabilities(profile, native_extension):
    """Counter ABI must exist before native arena allocation or bootstrap."""
    if profile.get('stock_singleton', False) and not callable(
            getattr(native_extension, 'q1_stock_singleton_dispatch_count', None)):
        raise ValueError('native singleton stock32 physical counter capability is unavailable')
    if profile.get('stock_reduction', False) and not callable(
            getattr(native_extension, 'q1_stock_reduction_dispatch_count', None)):
        raise ValueError('native stock32 physical counter capability is unavailable')


def load_hybrid_research_profile(path, *, live_identity, context_lengths, environment):
    if environment.get('MLX2_PAGED_Q1_STOCK_LONG', '0') != '0':
        raise ValueError('stock-long serving admission is not enabled')
    data = json.loads(Path(path).read_text())
    fields = {'schema','profile_id','identity','context_bounds','max_tokens','sampling',
              'required_environment','qualified','price_usable','serving_default','warm_apcv2',
              'q1_simd_stripes','q1_split_partition','storage_dtype','memory_budget_bytes'}
    if (type(data) is not dict or not fields <= set(data) or
            set(data) - fields - {'stock_reduction', 'stock_singleton'} or data['schema'] != SCHEMA):
        raise ValueError('hybrid research profile schema/fields differ')
    if (_identity(data['identity']) != _identity(live_identity) or
        not isinstance(data['profile_id'],str) or not data['profile_id'] or
        any(data[key] is not False for key in ('qualified','price_usable','serving_default','warm_apcv2')) or
        data['sampling'] != {'mode':'greedy','processors':False} or
        type(data['max_tokens']) is not int or not 1 <= data['max_tokens'] <= 256 or
        type(data['memory_budget_bytes']) is not int or not 0 < data['memory_budget_bytes'] <= 12 << 30 or
        type(data['q1_simd_stripes']) is not int or data['q1_simd_stripes'] not in (8,16) or
        type(data['q1_split_partition']) is not int or data['q1_split_partition'] not in (0,128,256) or
        data['storage_dtype'] not in ('float16','bfloat16')):
        raise ValueError('hybrid research profile identity/scope differs')
    bounds = data['context_bounds']; limit = 8192 if data['q1_split_partition'] else 128
    if (type(bounds) is not dict or set(bounds) != {'minimum','maximum','distinct'} or
        type(bounds['minimum']) is not int or type(bounds['maximum']) is not int or
        not 32 <= bounds['minimum'] <= bounds['maximum'] <= limit-data['max_tokens']+1 or
        bounds['distinct'] is not True or type(context_lengths) is not tuple or len(context_lengths)!=2 or
        context_lengths[0]==context_lengths[1] or
        any(type(n) is not int or not bounds['minimum'] <= n <= bounds['maximum'] for n in context_lengths)):
        raise ValueError('hybrid complete-request context bound differs')
    stock_reduction = validate_hybrid_stock_pair(data, context_lengths)
    expected = {'MLX2_PAGED_HYBRID_B2':'1','MLX2_PAGED_Q1_SIMD_TILE':'1',
        'MLX2_PAGED_GROUPED_Q1_WRITE':'1','MLX2_PAGED_PRIVATE_TAIL_REUSE':'1',
        'MLX2_PAGED_Q1_SIMD_STRIPES':str(data['q1_simd_stripes']),
        'MLX2_PAGED_Q1_SPLIT_KV':str(data['q1_split_partition']),
        'MLX2_PAGED_Q1_STOCK_REDUCTION':'1' if stock_reduction else '0',
        **{key:'0' for key in ('MLX2_PAGED_Q1_STOCK_SDPA','MLX2_PAGED_Q1_INLINE_METADATA',
            'MLX2_PAGED_B2_DEFERRED_EVAL','MLX2_PAGED_B2_DEFERRED_WRITE_EVAL',
            'MLX2_PAGED_GROUPED_SAMPLER','MLX2_PAGED_GROUPED_DIRECT_FENCE')}}
    singleton = data.get('stock_singleton', False)
    if 'stock_singleton' in data:
        expected['MLX2_PAGED_Q1_STOCK_SINGLETON'] = '1' if singleton else '0'
    if environment.get('MLX2_PAGED_Q1_STOCK_SINGLETON', '0') != ('1' if singleton else '0'):
        raise ValueError('hybrid singleton stock environment differs from profile')
    if data['required_environment'] != expected or any(environment.get(key,'0') != value for key,value in expected.items()):
        raise ValueError('hybrid physical environment differs from profile')
    return data
