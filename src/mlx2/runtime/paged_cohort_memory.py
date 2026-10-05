"""Generic source-bound shared adapter cost contract, without model arithmetic."""
import hashlib,json
from .paged_pack_price import _identity

def token_digest(row):return hashlib.sha256(json.dumps(list(row),separators=(',',':')).encode()).hexdigest()

def validate_shared_cohort_bound(bound,requests):
    fields={'schema','route','source_identity','source_input_ids','token_ids_sha256','output_caps','runtime_bytes','components','loaded_parameter_bytes','process_headroom_bytes','process_bound_bytes','max_process_bytes','profile_id','qualified'}
    if type(bound) is not dict or set(bound)!=fields or bound['schema']!='mlx2.adapter-shared-cohort-memory.v1' or bound['qualified'] is not False or not isinstance(bound['route'],str) or not bound['route'] or not isinstance(bound['profile_id'],str) or not bound['profile_id']:
        raise ValueError('complete adapter shared cost receipt required')
    _identity(bound['source_identity'])
    if type(requests) is not tuple or not 1<=len(requests)<=20 or any(type(r) is not tuple or len(r)!=3 or type(r[0]) is not str or not r[0] or type(r[1]) is not tuple or not r[1] or any(type(t) is not int or t<0 for t in r[1]) or type(r[2]) is not int or r[2]<1 for r in requests) or len({r[0] for r in requests})!=len(requests):raise ValueError('distinct concrete source input/token/cap tuples required')
    if (bound['source_input_ids']!=tuple(r[0] for r in requests) or bound['token_ids_sha256']!=tuple(token_digest(r[1]) for r in requests) or bound['output_caps']!=tuple(r[2] for r in requests)):
        raise ValueError('shared memory receipt token/cap identity drifted')
    numbers=('runtime_bytes','loaded_parameter_bytes','process_headroom_bytes','process_bound_bytes','max_process_bytes')
    if any(type(bound[k]) is not int or bound[k]<=0 for k in numbers) or type(bound['components']) is not dict or not bound['components'] or any(type(v) is not int or v<0 for v in bound['components'].values()) or sum(bound['components'].values())!=bound['runtime_bytes']:
        raise ValueError('complete exact byte bound required')
    if bound['process_bound_bytes']!=bound['runtime_bytes']+bound['loaded_parameter_bytes']+bound['process_headroom_bytes'] or bound['process_bound_bytes']>bound['max_process_bytes']:
        raise MemoryError('shared cohort process/model/headroom bound does not fit')
    return bound
