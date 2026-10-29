# SPDX-License-Identifier: MIT
"""CPU-only experiment preparation and paired analysis for DS4.1.

No MLX import, model loading, subprocess launch or GPU work. Actual execution
continues to use the existing CPG/radio-owned serving campaign. Provenance:
provenance/deepseek-v41-plan-performance.json.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import tempfile

SCHEMA = 'mlx2.ds41-performance-plan.v1'
REQUIRED_GATES = ('ownership_valid', 'ordinary_reference_passed', 'source_unchanged',
                  'route_observed', 'budget_retained', 'released', 'output_passed',
                  'timing_observer_clear', 'thermal_control_passed', 'contamination_clear')


def _sha(value, count, label):
    if type(value) is not str or re.fullmatch('[0-9a-f]{'+str(count)+'}', value) is None:
        raise ValueError('invalid '+label+' identity')


@dataclass(frozen=True)
class Binding:
    checkpoint: str
    source_revision: str
    source_snapshot: str
    hotmap_sha256: str
    atlas_seed_sha256: str
    hardware: str
    runtime: str
    numerics: str

    def __post_init__(self):
        if type(self.checkpoint) is not str or '@' not in self.checkpoint:
            raise ValueError('checkpoint must be artifact@fingerprint')
        name, digest = self.checkpoint.rsplit('@', 1)
        if not name: raise ValueError('checkpoint name is empty')
        _sha(digest, 64, 'checkpoint')
        _sha(self.source_revision, 40, 'source revision')
        for label in ('source_snapshot', 'hotmap_sha256', 'atlas_seed_sha256'):
            _sha(getattr(self, label), 64, label)
        if any(type(v) is not str or not v for v in (self.hardware, self.runtime, self.numerics)):
            raise ValueError('hardware/runtime/numerics binding required')


@dataclass(frozen=True)
class Shape:
    cell: str
    actual_prompt_tokens: int = 100
    output_tokens: int = 32
    cached_tokens: int = 99
    context_capacity: int = 16384
    physical_width: int = 1
    inflight: int = 1
    apc_bytes: int = 256 << 20
    apc_persistence: str = 'same-as-reference'
    expert_bytes: int = 75_000_000_000
    original_bytes: int = 60_000_000_000
    live_host_reserve_bytes: int = 9 << 30
    residency: str = 'closed_preheated'
    speculation: bool = False

    def __post_init__(self):
        if self.cell not in ('pp', 'decode', 'long_decode', 'context_decode'):
            raise ValueError('unsupported rate cell')
        for key in ('actual_prompt_tokens','output_tokens','context_capacity','physical_width',
                    'inflight','expert_bytes','original_bytes','live_host_reserve_bytes'):
            if type(getattr(self,key)) is not int or getattr(self,key) <= 0:
                raise ValueError('invalid shape '+key)
        if (type(self.cached_tokens) is not int or not 0 <= self.cached_tokens <= self.actual_prompt_tokens
                or type(self.apc_bytes) is not int or self.apc_bytes < 0
                or self.actual_prompt_tokens + self.output_tokens > self.context_capacity
                or self.original_bytes >= self.expert_bytes
                or self.residency not in ('closed_preheated','ordinary_dynamic')
                or type(self.speculation) is not bool or self.speculation
                or type(self.apc_persistence) is not str or not self.apc_persistence):
            raise ValueError('incompatible ordinary shape/cache/residency')
        if self.cell == 'pp' and (self.cached_tokens or self.output_tokens != 1):
            raise ValueError('PP must be uncached and retain exactly one output')
        if self.cell != 'pp' and self.output_tokens < 2:
            raise ValueError('decode needs at least two generated timestamps')
        if self.cell == 'long_decode' and self.output_tokens < 513:
            raise ValueError('long decode needs at least 512 intervals')


def decode_intervals(timestamps):
    """N generated-token timestamps contain N-1 steady generated intervals."""
    values=tuple(timestamps)
    if (len(values)<2 or any(type(v) not in (float,int) or not math.isfinite(v) for v in values)
            or any(b<=a for a,b in zip(values,values[1:]))):
        raise ValueError('generated timestamps must be finite and strictly increasing')
    intervals=tuple(b-a for a,b in zip(values,values[1:]))
    elapsed=values[-1]-values[0]
    if (any(not math.isfinite(v) or v<=0 for v in intervals)
            or not math.isfinite(elapsed) or elapsed<=0):
        raise ValueError('derived decode interval/elapsed must be finite and positive')
    rate=len(intervals)/elapsed
    if not math.isfinite(rate) or rate<=0:
        raise ValueError('derived decode rate must be finite and positive')
    return {'intervals':intervals, 'interval_count':len(intervals),
            'elapsed_seconds':elapsed,
            'tokens_per_second':rate,
            'first_token_included':False}


def policy_identity(policy, *, numerical_profile, implementation_sha256):
    if (not isinstance(policy,dict) or any(type(k) is not str or not k or type(v) is not str
                                         for k,v in policy.items())
            or type(numerical_profile) is not str or not numerical_profile):
        raise ValueError('explicit policy and numerical profile required')
    _sha(implementation_sha256,64,'policy implementation')
    result={'policy':dict(sorted(policy.items())),'numerical_profile':numerical_profile,
            'implementation_sha256':implementation_sha256}
    result['identity_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,
        separators=(',',':')).encode()).hexdigest()
    return result


def _validate_arm_identity(value):
    if not isinstance(value,dict) or set(value)!={'policy','numerical_profile','implementation_sha256','identity_sha256'}:
        raise ValueError('incomplete preregistered arm identity')
    expected=policy_identity(value['policy'],numerical_profile=value['numerical_profile'],
                             implementation_sha256=value['implementation_sha256'])
    if value!=expected:raise ValueError('preregistered policy identity hash mismatch')


def campaign(binding, candidate_policy, *, seed, pairs=5, requests=3,
             reference_policy=None, candidate_numerical_profile=None):
    """Preregister matched independent ownership windows; never execute them."""
    if not isinstance(binding,Binding) or type(seed) is not int:
        raise ValueError('validated binding and explicit random seed required')
    if type(pairs) is not int or pairs<5 or type(requests) is not int or requests<3:
        raise ValueError('controlled campaign requires >=5 pairs and >=3 requests/cell')
    if (not isinstance(candidate_policy,dict) or not candidate_policy
            or any(type(k) is not str or type(v) is not str for k,v in candidate_policy.items())):
        raise ValueError('candidate must be an explicit nonempty policy mapping')
    rng=random.Random(seed); windows=[]
    reference_policy={} if reference_policy is None else reference_policy
    reference=policy_identity(reference_policy,numerical_profile=binding.numerics,
                              implementation_sha256=binding.source_snapshot)
    delta_digest=hashlib.sha256(json.dumps(candidate_policy,sort_keys=True).encode()).hexdigest()
    candidate=policy_identity({**reference_policy,**candidate_policy},
        numerical_profile=(candidate_numerical_profile if candidate_numerical_profile is not None
                           else binding.numerics+':policy-'+delta_digest),
        implementation_sha256=binding.source_snapshot)
    if reference['policy']==candidate['policy']:
        raise ValueError('candidate policy does not change the preregistered reference')
    arms={'reference':reference,'candidate':candidate}
    cells=[asdict(Shape('pp',output_tokens=1,cached_tokens=0)),asdict(Shape('decode'))]
    for pair in range(pairs):
        order_arms=['reference','candidate'];rng.shuffle(order_arms)
        for order,arm in enumerate(order_arms):
            windows.append({'pair':pair,'order':order,'arm':arm,
                'unit':f'pair-{pair:02d}-{order}-{arm}', 'requests_per_cell':requests,
                'policy_delta':dict(candidate_policy) if arm=='candidate' else {},
                'arm_binding':arms[arm],
                'cells':cells, 'release_after_window':True})
    # Two A/A windows are attribution controls, not candidate measurements.
    controls=[{'unit':'control-aa-'+str(i),'arm':'reference','policy_delta':{},
               'cells':cells,'requests_per_cell':requests,'release_after_window':True} for i in range(2)]
    return {'schema':SCHEMA,'binding':asdict(binding),'arm_bindings':arms,'seed':seed,'windows':windows,
            'aa_controls':controls,'executed':False,'gpu_authorization':'pending',
            'minimum_speed_ratio':1.05, 'statistics_unit':'independent matched ownership-window pair',
            'gates':REQUIRED_GATES,'no_overwrite':True,'full_preflight_authorization':'separate',
            'execution_contract':'existing paired CPG/radio ownership; one artifact; ordinary smoke; one loopback server; atomic receipts; finally process-group cleanup; release/yield'}


def validate_receipt(receipt, *, binding, shape, expected_arm=None):
    if not isinstance(binding,Binding) or not isinstance(shape,Shape):
        raise ValueError('receipt validation requires exact binding and shape')
    if receipt.get('binding')!=asdict(binding) or receipt.get('shape')!=asdict(shape):
        raise ValueError('receipt checkpoint/source/hardware/shape mismatch')
    if expected_arm is not None:
        _validate_arm_identity(expected_arm)
        if receipt.get('arm_binding') != expected_arm:
            raise ValueError('receipt selected policy/numerical profile differs from preregistered arm')
    if receipt.get('completed') is not True:
        raise ValueError('partial or refused window cannot supply a rate')
    gates=receipt.get('gates',{})
    failed=[name for name in REQUIRED_GATES if gates.get(name) is not True]
    if failed: raise ValueError('performance gates failed: '+','.join(failed))
    samples=receipt.get('requests')
    if not isinstance(samples,list) or len(samples)<3:
        raise ValueError('window requires three complete request samples')
    latencies=[]
    for sample in samples:
        if any(type(sample.get(k)) is not int for k in ('actual_prompt_tokens','cached_tokens','output_tokens')):
            raise ValueError('actual token/cache counters must be integers')
        if sample.get('actual_prompt_tokens')!=shape.actual_prompt_tokens or sample.get('cached_tokens')!=shape.cached_tokens:
            raise ValueError('actual prompt/cache counters do not match shape')
        if sample.get('output_tokens')!=shape.output_tokens:
            raise ValueError('generated token count mismatch')
        if shape.cell=='pp':
            elapsed=sample.get('prefill_seconds')
            if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<=0:
                raise ValueError('invalid PP duration')
            latencies.append(elapsed/shape.actual_prompt_tokens)
        else:
            timestamps=sample.get('generated_timestamps',())
            if len(timestamps)!=shape.output_tokens:
                raise ValueError('decode timestamps do not match output count')
            timed=decode_intervals(timestamps)
            latencies.append(timed['elapsed_seconds']/timed['interval_count'])
    if any(not math.isfinite(t) or t<=0 for t in latencies):
        raise ValueError('derived per-token latency must be finite and positive')
    latency=statistics.median(latencies)
    if not math.isfinite(latency) or latency<=0:
        raise ValueError('derived window latency must be finite and positive')
    return latency


def paired_analysis(pairs, *, binding, shape, seed, arm_bindings,
                    bootstrap_samples=2000, minimum_speed_ratio=1.05):
    """Window-level paired bootstrap. Within-window requests are not replicas."""
    if type(seed) is not int or type(bootstrap_samples) is not int or not 100<=bootstrap_samples<=100000:
        raise ValueError('explicit seed and bounded bootstrap count required')
    if type(minimum_speed_ratio) not in (int,float) or not math.isfinite(minimum_speed_ratio) or minimum_speed_ratio<1:
        raise ValueError('invalid preregistered minimum speed ratio')
    if not isinstance(arm_bindings,dict) or set(arm_bindings)!={'reference','candidate'}:
        raise ValueError('preregistered reference/candidate arm bindings required')
    for arm in arm_bindings.values():_validate_arm_identity(arm)
    if arm_bindings['reference']['policy']==arm_bindings['candidate']['policy']:
        raise ValueError('candidate must change the reference policy')
    pairs=tuple(pairs)
    if len(pairs)<5: raise ValueError('at least five independent completed pairs required')
    ids=set();units=set();ratios=[];raw=[]
    for reference,candidate in pairs:
        pair_id=reference.get('pair')
        if (type(pair_id) is not int or type(candidate.get('pair')) is not int
                or candidate.get('pair')!=pair_id or pair_id in ids
                or reference.get('arm')!='reference' or candidate.get('arm')!='candidate'
                or reference.get('unit')==candidate.get('unit')):
            raise ValueError('invalid or repeated independent pair')
        for receipt in (reference,candidate):
            unit=receipt.get('unit')
            if type(unit) is not str or not unit or unit in units:
                raise ValueError('nonempty globally unique independent unit IDs required')
            units.add(unit)
        ids.add(pair_id)
        a=validate_receipt(reference,binding=binding,shape=shape,expected_arm=arm_bindings['reference'])
        b=validate_receipt(candidate,binding=binding,shape=shape,expected_arm=arm_bindings['candidate'])
        ratio=a/b
        if not math.isfinite(ratio) or ratio<=0:
            raise ValueError('derived paired speed ratio must be finite and positive')
        ratios.append(ratio);raw.append({'pair':pair_id,'reference_seconds_per_token':a,
                                     'candidate_seconds_per_token':b,'speed_ratio':ratio})
    rng=random.Random(seed)
    boot=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(bootstrap_samples))
    lo,hi=boot[int(.025*(bootstrap_samples-1))],boot[int(.975*(bootstrap_samples-1))]
    point=statistics.median(ratios)
    return {'binding':asdict(binding),'shape':asdict(shape),'arm_bindings':arm_bindings,'pairs':raw,
            'median_paired_speed_ratio':point,'window_bootstrap_95_interval':[lo,hi],
            'seed':seed,'bootstrap_samples':bootstrap_samples,
            'minimum_speed_ratio':minimum_speed_ratio,
            'verdict':'measured_win' if point>=minimum_speed_ratio and lo>1 else 'inconclusive_or_no_win',
            'performance_qualified':False,'qualification':'separate correctness/default contract',
            'tail_latency_claim':False}


def write_new_json(path, value):
    """Create a new evidence file without replacing an earlier attempt."""
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,
                prefix='.'+path.name+'-',delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        # An atomic no-replace link keeps retries distinct, unlike replace().
        os.link(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def source_snapshot(root, paths):
    root=Path(root).resolve(); records=[]; stamps=[]; lexical_stamps=[]
    def stamp(path):
        s=path.stat()
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    for name in sorted(set(paths)):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('snapshot paths must stay within source root')
        lexical=root/relative
        lexical_stat=lexical.lstat()
        lexical_stamp=(lexical_stat.st_dev,lexical_stat.st_ino,lexical_stat.st_size,
                       lexical_stat.st_mtime_ns,lexical_stat.st_ctime_ns)
        path=lexical.resolve(strict=True)
        if not path.is_relative_to(root): raise ValueError('snapshot source escapes root')
        before=stamp(path);payload=path.read_bytes()
        if stamp(path)!=before:
            raise ValueError('source changed during snapshot: '+str(relative))
        records.append({'path':str(relative),'sha256':hashlib.sha256(payload).hexdigest()})
        stamps.append((path,before))
        lexical_stamps.append((lexical,path,lexical_stamp))
    if not records:raise ValueError('empty source snapshot')
    if any(stamp(path)!=before for path,before in stamps):
        raise ValueError('source changed during snapshot')
    for lexical,resolved,before in lexical_stamps:
        s=lexical.lstat()
        if ((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=before
                or lexical.resolve(strict=True)!=resolved):
            raise ValueError('source lexical path changed during snapshot')
    digest=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'sha256':digest,'files':records,'scope':'explicit source list; not a model-run receipt'}


def runtime_source_paths(root, *, additional=()):
    """Include Python, shader, descriptor and harness inputs; no model payloads."""
    root=Path(root)
    suffixes={'.py','.metal','.h','.hpp','.cpp','.c','.mm','.m','.swift','.sh','.rs',
              '.json','.toml','.txt','.yaml','.yml','.cfg','.ini'}
    paths=set()
    for base in ('src/mlx2','scripts'):
        for path in (root/base).rglob('*'):
            if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts:
                paths.add(str(path.relative_to(root)))
    for path in (root/'.local-source-lfu').glob('*.py'):
        paths.add(str(path.relative_to(root)))
    if (root/'pyproject.toml').is_file():paths.add('pyproject.toml')
    paths.update(additional)
    return tuple(sorted(paths))
