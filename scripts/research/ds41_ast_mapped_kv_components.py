# SPDX-License-Identifier: MIT
"""Parent-owned mapped-KV component replay over saved actual M64 operands.

Common tables/maps are CPU-derived FIXTURE setup only, not a serving path.
Complete timings include map staging and validation; prepared-consumer timings
are separate lower bounds using immutable prevalidated maps/status.
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

_QUARANTINE=[]


def _write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x')as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def deduplicate(right,transpose_b):
    """Byte-exact fixture table; preserve signed zero and every row order."""
    import numpy as np
    if(right.ndim!=3 or right.dtype!=np.float32 or type(transpose_b)is not bool
            or not np.isfinite(right).all()):raise ValueError('Mapped fixture coefficient geometry/dtype invalid')
    rows=np.ascontiguousarray(np.swapaxes(right,-1,-2)if transpose_b else right)
    mapping=np.empty(rows.shape[:2],np.uint32);seen={};table=[]
    for batch in range(rows.shape[0]):
        for column in range(rows.shape[1]):
            value=np.ascontiguousarray(rows[batch,column]);key=value.tobytes()
            index=seen.get(key)
            if index is None:index=len(table);seen[key]=index;table.append(value.copy())
            mapping[batch,column]=index
    if not table:raise ValueError('Mapped fixture table cannot be empty')
    physical=np.stack(table);reconstructed=physical[mapping]
    if reconstructed.tobytes()!=rows.tobytes():raise ValueError('Mapped fixture lost coefficient bytes/order')
    return physical,mapping


def _owning(mx,left,right,role,transpose_b):
    """Actual ordinary 4D reshape or owning selected einsum, never None-M."""
    outputs=[];g,h,k=left.shape;n=right.shape[-1]
    for batch in range(g):
        query=mx.contiguous(left[batch:batch+1]).reshape(1,h,1,k)
        source=right[batch:batch+1]
        if role=='QK_selected':
            values=mx.contiguous(mx.swapaxes(source,-1,-2)).reshape(1,1,n,k)
            value=mx.einsum('bhld,blkd->bhlk',query,values)
        elif role=='PV_selected':
            values=mx.contiguous(source).reshape(1,1,k,n)
            value=mx.einsum('bhlk,blkd->bhld',query,values)
        else:
            values=(mx.swapaxes(mx.contiguous(mx.swapaxes(source,-1,-2)).reshape(1,1,n,k),-1,-2)
                    if transpose_b else mx.contiguous(source).reshape(1,1,k,n))
            value=query@values
        outputs.append(value.reshape(1,h,n))
    return mx.concatenate(outputs,axis=0)


def _prepared_mapped(mx,kernel,left,table,maps,status,descriptor,live):
    """Diagnostic lower bound only: skip already-run map staging/validation."""
    live();g,m,k=left.shape;n=maps.shape[1]if descriptor['transpose_b']else table.shape[1]
    stages=kernel._kernels(g,m,n,k,descriptor['transpose_b'],table.shape[0])
    if descriptor['mode']=='splitk':
        partial=stages[1](inputs=[left,table,maps,status],grid=descriptor['grid'],threadgroup=descriptor['threadgroup'],
            output_shapes=[descriptor['split_shape']],output_dtypes=[mx.float32])[0]
        return stages[2](inputs=[partial],grid=(n,m,g),threadgroup=(min(16,n),min(16,m),1),
            output_shapes=[descriptor['output_shape']],output_dtypes=[mx.float32])[0]
    return stages[1](inputs=[left,table,maps,status],grid=descriptor['grid'],threadgroup=descriptor['threadgroup'],
        output_shapes=[descriptor['output_shape']],output_dtypes=[mx.float32])[0]


def run_replay(adapter,unit,check_live,fixture_path,fixture_sha):
    if not callable(check_live):raise ValueError('Mapped replay needs owned live callback')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Mapped replay requires parent-owned GPU')
    import numpy as np
    from ds41_ast_attention_matmul_components import _compare,_literal
    from ds41_resident_prefill_model_gate import _checkpoint,_counters
    from mlx2.runtime.gguf import source_ast_batched_splitk as ordinary,source_dense
    def live():
        if check_live()is False:raise RuntimeError('Mapped replay ownership lost')
        return True
    unit=Path(unit);path=unit/'ast-mapped-kv-components.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Mapped replay evidence exists')
    spec=json.loads((unit/'spec.json').read_text());result=dict(schema='mlx2.ds41-ast-mapped-kv-components.v1',
        checkpoint=_checkpoint(adapter),source_revision=spec['source_revision'],source_snapshot_sha256=spec['source_snapshot_sha256'],
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='fixture_validation',
        completed=False,passed=False,qualified=False,rate_eligible=False,APCv2_publication=False,
        model_capture_performed=False,runtime_CPU_rowmaps=False,fixture_CPU_byte_deduplication=True)
    cases=[];roots=[];drained=True
    try:
        live();_write(path,result);fixture_path=Path(fixture_path).resolve();raw=fixture_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=fixture_sha:raise ValueError('Actual attention fixture SHA differs')
        fixture=json.loads(raw);producer=Path(fixture['producer_unit']).resolve();archive=Path(fixture['archive']).resolve()
        if(fixture.get('schema')!='mlx2.ds41-ast-attention-matmul-fixture.v1'or fixture.get('checkpoint')!=result['checkpoint']
                or fixture.get('ordinary_controls_completed')is not True or fixture.get('candidate_shader_attempted')is not False
                or len(fixture.get('cases',[]))!=32 or fixture_path.parent!=producer or archive.parent!=producer):
            raise ValueError('Actual mapped-input fixture identity/completion differs')
        source=json.loads((producer/'source/desktop-source.json').read_text());old=producer/'source'
        if source['source_sha256']!=fixture['source_snapshot_sha256']or source['source_revision']!=fixture['source_revision']:
            raise ValueError('Actual fixture producer source binding differs')
        for name in('src/mlx2/runtime/gguf/source_ast_batched_splitk.py','src/mlx2/runtime/gguf/source_dense.py'):
            if hashlib.sha256((old/name).read_bytes()).hexdigest()!=source['files'][name]:raise ValueError('Producer arithmetic source seal differs')
        if(_literal(old/'src/mlx2/runtime/gguf/source_ast_batched_splitk.py','HASHES')!=ordinary.HASHES
                or _literal(old/'src/mlx2/runtime/gguf/source_dense.py','HASHES')!=source_dense.HASHES
                or _literal(old/'src/mlx2/runtime/gguf/source_dense.py','LIBRARY_SHA256')!=source_dense.LIBRARY_SHA256):
            raise ValueError('Mapped fixture installed header/library closure differs')
        ordinary._installed()
        if os.environ.get('MLX_ENABLE_TF32')!='0':raise ValueError('Mapped replay requires TF32 disabled')
        if archive.stat().st_size>130<<20 or hashlib.sha256(archive.read_bytes()).hexdigest()!=fixture['archive_sha256']:
            raise ValueError('Actual matrix archive bound/SHA differs')
        expected={f'case{i}_{name}'for i in range(32)for name in('left','right','ordinary_control','old_regular_batch')}
        derived={};metadata=[];total=0
        with np.load(archive,allow_pickle=False)as values:
            if set(values.files)!=expected:raise ValueError('Actual matrix archive inventory differs')
            for index,meta in enumerate(fixture['cases']):
                left=values[f'case{index}_left'];right=values[f'case{index}_right'];role=meta['role'];nt=meta['transpose_b']
                if(left.dtype!=np.float32 or right.dtype!=np.float32 or not np.isfinite(left).all()or not np.isfinite(right).all()
                        or left.shape[:2]!=(3,64)or right.shape[0]!=3 or left.shape[-1]!=right.shape[1]
                        or role not in('QK_local','QK_selected','PV_local','PV_selected')or nt!=role.startswith('QK')):
                    raise ValueError('Actual matrix archive role/geometry differs')
                table,mapping=deduplicate(right,nt)
                cases.append(dict(index=index,layer=meta['layer'],role=role,transpose_b=nt,left_cpu=left.copy(),
                    right_cpu=right.copy(),table_cpu=table,map_cpu=mapping))
                for name,array in(('left',left),('table',table),('row_map',mapping)):
                    derived[f'case{index}_{name}']=array.copy();total+=array.nbytes
                metadata.append(dict(index=index,layer=meta['layer'],role=role,transpose_b=nt,left_shape=list(left.shape),
                    right_shape=list(right.shape),table_shape=list(table.shape),map_shape=list(mapping.shape),
                    logical_rows=int(mapping.size),unique_rows=table.shape[0],logical_coefficient_bytes=right.nbytes,
                    physical_table_bytes=table.nbytes,map_bytes=mapping.nbytes))
        if total>128<<20:raise ValueError('Derived mapped fixture exceeds bound')
        derived_path=unit/'ast-mapped-kv-fixture.npz'
        with derived_path.open('xb')as stream:np.savez(stream,**derived)
        manifest=dict(schema='mlx2.ds41-ast-mapped-kv-fixture.v1',checkpoint=result['checkpoint'],
            source_revision=result['source_revision'],source_snapshot_sha256=result['source_snapshot_sha256'],
            original_fixture=str(fixture_path),original_fixture_sha256=fixture_sha,
            archive=str(derived_path.resolve()),archive_sha256=hashlib.sha256(derived_path.read_bytes()).hexdigest(),
            array_bytes=total,cases=metadata,persisted_before_mapped_compile=True,
            CPU_byte_deduplication='Fixture preparation only; no runtime CPU indices or persistent owner sharing claimed')
        manifest_path=unit/'ast-mapped-kv-fixture.json';_write(manifest_path,manifest)
        result.update(status='derived_fixture_saved',fixture=dict(path=str(manifest_path),sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),array_bytes=total),
            original_fixture=dict(path=str(fixture_path),sha256=fixture_sha,archive_sha256=fixture['archive_sha256']),case_count=32)
        _write(path,result)
        # Inputs/maps/common tables are durable before first mapped shader.
        from mlx2.runtime.gguf import source_ast_mapped_kv_matmul as mapped
        before=_counters(adapter);rows=[]
        for case,meta in zip(cases,metadata):
            live();left=mx.array(case['left_cpu']);right=mx.array(case['right_cpu']);table=mx.array(case['table_cpu']);maps=mx.array(case['map_cpu'])
            a=mx.contiguous(left);b=mx.contiguous(table);nt=case['transpose_b'];g,m,k=left.shape
            expected=_owning(mx,left,right,case['role'],nt)
            roots=[left,right,table,maps,a,b,expected]
            pinned=ordinary.private_matmul(left,right,transpose_b=nt,check_live=live)
            roots.append(pinned)
            drained=False;positive=mapped.private_matmul(left,table,maps,transpose_b=nt,check_live=live)
            roots.append(positive)
            mx.eval(expected,pinned,positive.output,positive.status);drained=True;live()
            status=np.asarray(positive.status);checks=dict(owning_vs_pinned=_compare(pinned,expected),mapped_vs_owning=_compare(positive.output,expected),mapped_vs_pinned=_compare(positive.output,pinned))
            controls=[]
            for label,bad_value in(('negative_one',-1),('equal_table_rows',table.shape[0])):
                bad=case['map_cpu'].astype(np.int32);bad[0,0]=bad_value
                bad_gpu=mx.array(bad);roots.append(bad_gpu)
                drained=False;invalid=mapped.private_matmul(left,table,bad_gpu,transpose_b=nt,check_live=live)
                roots.append(invalid)
                mx.eval(invalid.output,invalid.status);drained=True;live();flags=np.asarray(invalid.status);output=np.asarray(invalid.output)
                expect_flags=np.zeros(g,np.uint32);expect_flags[0]=1
                zero=np.zeros_like(output[0]);other=_compare(output[1:],np.asarray(expected)[1:])
                controls.append(dict(label=label,status_exact=bool(np.array_equal(flags,expect_flags)),
                    invalid_batch_defined_zero=output[0].tobytes()==zero.tobytes(),other_batches=other,
                    passed=bool(np.array_equal(flags,expect_flags))and output[0].tobytes()==zero.tobytes()and other['equal']))
            row={**meta,'descriptor':positive.descriptor,'checks':checks,'positive_status_zero':bool(np.array_equal(status,np.zeros(g,np.uint32))),
                'invalid_controls':controls,'passed':all(v['equal']and v['finite']for v in checks.values())and not status.any()and all(v['passed']for v in controls)}
            rows.append(row);result.update(status='raw_candidate_cases',cases=rows);_write(path,result)
            if not row['passed']:continue
            # Materialize shared inputs and both lower-bound prerequisites.
            gathered=table[maps];prepared_right=mx.swapaxes(gathered,-1,-2)if nt else gathered
            roots.extend((gathered,prepared_right));mx.eval(a,b,maps,gathered,prepared_right,positive.status);live()
            def invoke(scope,arm):
                if scope=='complete':
                    staged=mx.array(case['map_cpu']);roots.append(staged)
                    if arm=='ordinary_gather':
                        gathered_now=table[staged];rhs=mx.swapaxes(gathered_now,-1,-2)if nt else gathered_now
                        value=ordinary.private_matmul(left,rhs,transpose_b=nt,check_live=live);roots.extend((gathered_now,rhs,value));mx.eval(value);return value
                    value=mapped.private_matmul(left,table,staged,transpose_b=nt,check_live=live)
                    roots.append(value)
                    mx.eval(value.output,value.status)
                    if np.any(np.asarray(value.status)):raise RuntimeError('Timed mapped validation refused a positive map')
                    return value.output
                if arm=='ordinary_gather':
                    value=ordinary.private_matmul(left,prepared_right,transpose_b=nt,check_live=live);roots.append(value);mx.eval(value);return value
                value=_prepared_mapped(mx,mapped,a,b,maps,positive.status,positive.descriptor,live);roots.append(value);mx.eval(value);return value
            timings=[]
            for scope in('complete','prepared_consumer_lower_bound'):
                for arm in('ordinary_gather','mapped'):
                    drained=False;value=invoke(scope,arm);drained=True
                    if not _compare(value,expected)['equal']:raise RuntimeError('Mapped timing warm changed arithmetic')
                for sequence,arm in enumerate(('ordinary_gather','mapped','mapped','ordinary_gather')):
                    live();drained=False;started=time.perf_counter_ns();value=invoke(scope,arm);elapsed=time.perf_counter_ns()-started;drained=True
                    if not _compare(value,expected)['equal']:raise RuntimeError('Timed mapped output changed arithmetic')
                    timings.append(dict(scope=scope,sequence=sequence,arm=arm,elapsed_ns=elapsed))
            row['timings']=timings;result.update(cases=rows);_write(path,result)
        delta={key:value-before.get(key,0)for key,value in _counters(adapter).items()if type(value)in(int,float)}
        result.update(status='passed'if all(row['passed']for row in rows)else'raw_mismatch',completed=True,
            passed=len(rows)==32 and all(row['passed']for row in rows),implementation_identity=mapped.implementation_identity(),
            component_counter_delta=delta,no_component_expert_or_engram_io=all(delta.get(key)==0 for key in('expert_bytes_read','expert_uploads','host_route_materializations','engram_bytes_read')),
            timing_contract=dict(order=['ordinary_gather','mapped','mapped','ordinary_gather'],samples_per_arm_per_scope=2,
                independent_ABBA_repetitions=1,common_table_CPU_dedup_outside_timer=True,common_table_GPU_setup_outside_timer=True,
                complete='CPUmapstaging+normalgather+pinnedconsumer versus CPUmapstaging+validation+mappedconsumer+statuscheck, includes GPUcompletion',
                prepared_consumer_lower_bound='PregatherednormalKV versus immutableprevalidatedmappedtable/map/status; excludes staging/gather/validation, not serving path',
                controlled_model_performance=False,explicit_cross_query_threadgroup_reuse=False),selected_or_defaulted=False)
        return result
    except BaseException as error:
        result.update(status='failed',completed=False,error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException:_QUARANTINE.append((cases,roots))
        _write(path,result)
