# SPDX-License-Identifier: MIT
"""Private physical prefix checkpoint, never APCv2 publication.

The six-plane/capacity save/restore below is mined unchanged from the sealed
v33 ladder helper; additional shards bind exact target taps and PP profile.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from pathlib import Path
import hashlib
import json
import os
import time


def _write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x')as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def _close(cache):
    if cache is not None:
        for row in cache:row.close()


_PLANES=('keys','buf_kv','buf_gate','_pooled','_index_keys','hash_tail')
_CHECKPOINT_QUARANTINE=[]


def _memory_snapshot(store,mx):
    from mlx2.runtime.os_memory import host_memory_snapshot
    host=host_memory_snapshot()
    return dict(at_unix=time.time(),cache_payload_bytes=store._cache_bytes,
        allocator_active_bytes=mx.get_active_memory(),allocator_cached_bytes=mx.get_cache_memory(),
        allocator_peak_bytes=mx.get_peak_memory(),actual_host_available_bytes=None if host is None else host.available_bytes,
        adaptive_state=dict(store.adaptive_state),free_memory_credit_bytes=0)


def _file_hash(path,live):
    digest=hashlib.sha256()
    with Path(path).open('rb')as f:
        while True:
            part=f.read(16<<20)
            if not part:break
            digest.update(part);live()
    return digest.hexdigest()


def _save_discovery(path,mx,prefix,reference,logits,ids,seen,binding,live,*,stage="discovery"):
    """Private disk frontier, not APC state. Full backing capacity is explicit."""
    if stage not in('prefix','discovery'):raise ValueError('Unknown private checkpoint stage')
    if stage=='prefix'and(reference is not None or logits or ids or seen):
        raise ValueError('Prefix checkpoint requires zero reference/logits/IDs/keys')
    if stage=='discovery'and(reference is None or len(logits)!=64 or len(ids)!=64):
        raise ValueError('Discovery checkpoint requires complete64 reference')
    path=Path(path)
    if path.exists():raise FileExistsError('Discovery checkpoint already exists')
    path.mkdir();arrays={};meta={}
    for label,cache in (('prefix',prefix),('reference',reference)):
        if cache is None:continue
        rows=[]
        for i,state in enumerate(cache):
            state.validate_state()
            if state.pending:raise ValueError('Cannot save pending discovery cache')
            planes={}
            for name in _PLANES:
                value=getattr(state,name,None);key=f'{label}.{i}.{name}'
                planes[name]=None if value is None else key
                if value is not None:arrays[key]=value
            rows.append(dict(meta=list(state.meta_state),capacity=state._capacity,pool_count=state._pool_count,planes=planes))
        meta[label]=rows
    for i,value in enumerate(logits):arrays[f'logits.{i}']=value
    live();mx.eval(*arrays.values())
    shard=path/'frontier.safetensors';temporary=path/'frontier.partial.safetensors'
    mx.save_safetensors(str(temporary),arrays)
    with temporary.open('rb')as f:os.fsync(f.fileno())
    os.replace(temporary,shard);live()
    result=dict(schema='mlx2.ds41-private-discovery-checkpoint.v1',stage=stage,binding=binding,caches=meta,
        logits_count=len(logits),output_ids=list(ids),expert_keys=[list(k)for k in sorted(seen)],
        shard_sha256=_file_hash(shard,live),generator_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        APCv2_publication=False,qualified=False)
    _write(path/'checkpoint.json',result)
    return dict(path=str(path.resolve()),manifest_sha256=_file_hash(path/'checkpoint.json',live),
        shard_sha256=result['shard_sha256'],stage=stage,binding=binding)


def _load_discovery(path,mx,model,binding,live,*,expected_manifest_sha256=None):
    path=Path(path)
    if expected_manifest_sha256 is not None and _file_hash(path/'checkpoint.json',live)!=expected_manifest_sha256:
        raise ValueError('Private discovery checkpoint manifest digest differs')
    manifest=json.loads((path/'checkpoint.json').read_text())
    if manifest.get('schema')!='mlx2.ds41-private-discovery-checkpoint.v1'or manifest.get('binding')!=binding:
        raise ValueError('Private discovery checkpoint source/profile/prompt binding differs')
    if _file_hash(path/'frontier.safetensors',live)!=manifest['shard_sha256']:
        raise ValueError('Private discovery checkpoint shard digest differs')
    stage=manifest.get('stage','discovery')
    labels=('prefix',)if stage=='prefix'else('prefix','reference')
    if stage not in('prefix','discovery')or set(manifest.get('caches',{}))!=set(labels):
        raise ValueError('Checkpoint stage/cache inventory differs')
    expected_outputs=0 if stage=='prefix'else 64
    arrays={};caches=[]
    try:
        arrays=mx.load(str(path/'frontier.safetensors'));mx.eval(*arrays.values());live()
        for label in labels:
            rows=manifest['caches'][label];cache=model.make_cache();caches.append(cache)
            if len(cache)!=len(rows):raise ValueError('Checkpoint layer inventory differs')
            for state,row in zip(cache,rows):
                if tuple(state.meta_state[:9])!=tuple(row['meta'][:9]):raise ValueError('Checkpoint model identity/topology differs')
                offset=row['meta'][9]
                if type(offset)is not str or str(int(offset))!=offset or int(offset)<0:
                    raise ValueError('Checkpoint offset is not canonical')
                capacity,count=row['capacity'],row['pool_count']
                if any(type(v)is not int or v<0 for v in(capacity,count))or count>capacity:
                    raise ValueError('Checkpoint pool capacity/count differs')
                state.offset=int(offset);state._capacity=capacity;state._pool_count=count
                for name in _PLANES:
                    key=row['planes'][name];setattr(state,name,None if key is None else arrays[key])
                state.validate_state()
                if state.pending:raise ValueError('Restored checkpoint frontier is pending')
                if not state.source and(capacity or count or state._pooled is not None or state._index_keys is not None):
                    raise ValueError('Checkpoint nonsource pool owner differs')
                for name,dim in (('_pooled',state.head_dim),('_index_keys',state.index_head_dim)):
                    value=getattr(state,name)
                    if value is None:
                        if capacity:raise ValueError('Restored full pool capacity is missing')
                    elif value.shape!=(1,capacity,dim)or value.dtype!=mx.float32:
                        raise ValueError('Restored full pool capacity differs')
        expected={k for rows in manifest['caches'].values()for row in rows for k in row['planes'].values()if k is not None}
        expected.update(f'logits.{i}'for i in range(manifest['logits_count']))
        if set(arrays)!=expected or manifest['logits_count']!=expected_outputs or len(manifest['output_ids'])!=expected_outputs:
            raise ValueError('Checkpoint tensor/output inventory differs')
        vocab=model.args.vocab_size
        if any(type(t)is not int or not 0<=t<vocab for t in manifest['output_ids']):
            raise ValueError('Checkpoint output token IDs differ')
        for i in range(expected_outputs):
            value=arrays[f'logits.{i}']
            if value.shape!=(1,1,vocab)or value.dtype not in(mx.float32,mx.bfloat16):
                raise ValueError('Checkpoint logit geometry/dtype differs')
        keys=manifest['expert_keys']
        if stage=='prefix'and keys:raise ValueError('Prefix checkpoint cannot contain discovery keys')
        if (any(not isinstance(k,list)or len(k)!=2 or any(type(v)is not int for v in k)
                or not 0<=k[0]<40 or not 0<=k[1]<384 for k in keys)
                or len(set(tuple(k)for k in keys))!=len(keys)):
            raise ValueError('Checkpoint expert key inventory differs')
        if _file_hash(path/'frontier.safetensors',live)!=manifest['shard_sha256']:
            raise ValueError('Checkpoint shard changed during restore')
        if expected_manifest_sha256 is not None and _file_hash(path/'checkpoint.json',live)!=expected_manifest_sha256:
            raise ValueError('Checkpoint manifest changed during restore')
        return (caches[0],None if stage=='prefix'else caches[1],
            [arrays[f'logits.{i}']for i in range(expected_outputs)],list(manifest['output_ids']),
            set(tuple(k)for k in manifest['expert_keys']))
    except BaseException:
        try:mx.synchronize()
        except BaseException:
            _CHECKPOINT_QUARANTINE.append((arrays,tuple(caches)));raise
        for cache in caches:_close(cache)
        raise





def save_prefix(path,mx,prefix,logits,features,binding,partition,live):
    path=Path(path);stage=path.with_name('.'+path.name+'.partial');feature_tmp=path.with_name('.'+path.name+'.features.partial.safetensors')
    if any(p.exists()or p.is_symlink()for p in(path,stage,feature_tmp)):raise FileExistsError('Private prefix checkpoint already exists')
    if(any(row.offset!=binding['prefix_tokens']for row in prefix)
            or features.shape!=(1,binding['prefix_tokens'],binding['feature_width'])
            or logits.shape!=(1,1,binding['vocab_size'])):raise ValueError('Private prefix checkpoint feature/frontier geometry differs')
    live();mx.eval(logits,features);mx.save_safetensors(str(feature_tmp),dict(seed_logits=logits,seed_features=features))
    with feature_tmp.open('rb')as stream:os.fsync(stream.fileno())
    feature_sha=_file_hash(feature_tmp,live)
    profile=(json.dumps(partition,sort_keys=True,separators=(',',':'))+'\n').encode();profile_sha=hashlib.sha256(profile).hexdigest()
    extended={**binding,'features_shard_sha256':feature_sha,'partition_receipt_sha256':profile_sha}
    result=_save_discovery(stage,mx,prefix,None,[],[],set(),extended,live,stage='prefix')
    profile_path=stage/'prefix-profile.json'
    with profile_path.open('xb')as stream:stream.write(profile);stream.flush();os.fsync(stream.fileno())
    os.replace(feature_tmp,stage/'features.safetensors');os.rename(stage,path)
    return{**result,'path':str(path.resolve()),'features_shard_sha256':feature_sha,'partition_receipt_sha256':profile_sha,
        'private_physical_prefix':True,'APCv2_publication':False}


def load_prefix(path,mx,model,binding,live,*,expected_manifest_sha256):
    path=Path(path)
    if _file_hash(path/'checkpoint.json',live)!=expected_manifest_sha256:raise ValueError('Private prefix manifest SHA differs')
    manifest=json.loads((path/'checkpoint.json').read_text());stored=manifest.get('binding',{})
    extended={**binding,'features_shard_sha256':stored.get('features_shard_sha256'),
        'partition_receipt_sha256':stored.get('partition_receipt_sha256')}
    if stored!=extended or any(stored.get(k) is None for k in('features_shard_sha256','partition_receipt_sha256')):
        raise ValueError('Private prefix source/profile/ids binding differs')
    if _file_hash(path/'features.safetensors',live)!=stored['features_shard_sha256']:raise ValueError('Private prefix feature shard SHA differs')
    profile_bytes=(path/'prefix-profile.json').read_bytes()
    if hashlib.sha256(profile_bytes).hexdigest()!=stored['partition_receipt_sha256']:raise ValueError('Private prefix partition receipt SHA differs')
    prefix=None;arrays={}
    try:
        prefix,reference,logits,ids,keys=_load_discovery(path,mx,model,extended,live,expected_manifest_sha256=expected_manifest_sha256)
        if reference is not None or logits or ids or keys:raise ValueError('Private prefix checkpoint contains discovery data')
        arrays=mx.load(str(path/'features.safetensors'));mx.eval(*arrays.values());live()
        if(set(arrays)!= {'seed_logits','seed_features'}
                or arrays['seed_features'].shape!=(1,binding['prefix_tokens'],binding['feature_width'])
                or arrays['seed_logits'].shape!=(1,1,binding['vocab_size'])
                or arrays['seed_features'].dtype!=mx.float32
                or arrays['seed_logits'].dtype not in(mx.float32,mx.bfloat16)
                or any(row.offset!=binding['prefix_tokens']for row in prefix)):
            raise ValueError('Restored private prefix target feature/frontier geometry differs')
        if _file_hash(path/'features.safetensors',live)!=stored['features_shard_sha256']:raise ValueError('Feature shard changed duringrestore')
        return prefix,arrays['seed_logits'],arrays['seed_features'],json.loads(profile_bytes)
    except BaseException:
        try:mx.synchronize()
        except BaseException:_CHECKPOINT_QUARANTINE.append((prefix,arrays));raise
        _close(prefix);raise
