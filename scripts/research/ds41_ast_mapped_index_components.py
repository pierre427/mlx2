# SPDX-License-Identifier: MIT
"""Owned mapped-index replay over persisted actual H32/K128 classes.

CPU byte-deduplication constructs fixture-only tables/maps. Complete matrix
scopes include staging/gather or validation, not weighted/topk selection.
"""
from pathlib import Path
import hashlib
import json
import time

_QUARANTINE=[]


def run_replay(adapter,unit,check_live,fixture_path,fixture_sha):
    if not callable(check_live):raise ValueError('Mapped index replay needs owned live callback')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Mapped index replay requires parent-owned GPU')
    import numpy as np
    from ds41_ast_index_components import run_replay as ordinary_replay,groups
    from ds41_ast_mapped_kv_components import deduplicate,_write
    from ds41_ast_attention_matmul_components import _compare
    from ds41_resident_prefill_model_gate import _checkpoint,_counters
    from mlx2.runtime.gguf import source_ast_batched_index_matmul as ordinary
    from mlx2.runtime.models.deepseek_v4 import stable_pool_topk
    def live():
        if check_live()is False:raise RuntimeError('Mapped index replay ownership lost')
        return True
    unit=Path(unit);path=unit/'ast-mapped-index-components.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Mapped index evidence exists')
    spec=json.loads((unit/'spec.json').read_text());result=dict(schema='mlx2.ds41-ast-mapped-index-components.v1',
        checkpoint=_checkpoint(adapter),source_revision=spec['source_revision'],source_snapshot_sha256=spec['source_snapshot_sha256'],
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='owning_baseline',
        completed=False,passed=False,qualified=False,rate_eligible=False,APCv2_publication=False,model_capture_performed=False,
        CPU_fixture_byte_dedup=True,runtime_CPU_rowmaps=False)
    cases=[];roots=[];drained=True
    try:
        live();_write(path,result)
        # Reuse the strict source/header/library/actual-owning baseline gate.
        # No PP or DSpark/model tensor capture is repeated.
        baseline=ordinary_replay(adapter,unit,live,fixture_path,fixture_sha)
        if not baseline['passed']or set(baseline['passing_families'])!={'native_batch_v1','pinned_regular_s64_v1'}:
            raise ValueError('Actual native and S64 index baselines must both pass')
        result['owning_baseline_receipt']=dict(path=str(unit/'ast-index-components.json'),
            sha256=hashlib.sha256((unit/'ast-index-components.json').read_bytes()).hexdigest(),
            class_count=len(baseline['classes']),passing_families=baseline['passing_families'])
        original=Path(fixture_path).resolve();fixture=json.loads(original.read_text());archive=Path(fixture['archive']);metadata=fixture['cases']
        plans=groups(metadata);derived={};case_meta=[];total=0
        with np.load(archive,allow_pickle=False)as values:
            for index,(layer,count,ids)in enumerate(plans):
                left=np.concatenate([values[f'case{i}_query'].reshape(1,32,128)for i in ids],axis=0)
                keys=np.concatenate([values[f'case{i}_keys']for i in ids],axis=0)
                table,mapping=deduplicate(np.swapaxes(keys,-1,-2),True)
                raw=np.concatenate([values[f'case{i}_raw'][:,0]for i in ids],axis=0)
                weights=np.concatenate([values[f'case{i}_raw_weight']for i in ids],axis=0)
                score=np.concatenate([values[f'case{i}_score']for i in ids],axis=0)
                top=np.concatenate([values[f'case{i}_indices']for i in ids],axis=0)
                case=dict(index=index,layer=layer,count=count,nodes=[metadata[i]['node']for i in ids],
                    left_cpu=left,keys_cpu=keys,table_cpu=table,map_cpu=mapping,raw_cpu=raw,weights_cpu=weights,score_cpu=score,top_cpu=top)
                cases.append(case)
                for name,array in(('left',left),('table',table),('row_map',mapping),('raw',raw),('weights',weights),('score',score),('top',top)):
                    derived[f'class{index}_{name}']=array.copy();total+=array.nbytes
                case_meta.append(dict(index=index,layer=layer,count=count,nodes=case['nodes'],batches=len(ids),
                    table_rows=table.shape[0],logical_key_rows=int(mapping.size),logical_key_bytes=keys.nbytes,
                    common_table_bytes=table.nbytes,map_bytes=mapping.nbytes))
        if total>128<<20:raise ValueError('Mapped index derived fixture exceeds128MiB')
        output_archive=unit/'ast-mapped-index-fixture.npz'
        with output_archive.open('xb')as stream:np.savez(stream,**derived)
        manifest=dict(schema='mlx2.ds41-ast-mapped-index-fixture.v1',checkpoint=result['checkpoint'],
            source_revision=result['source_revision'],source_snapshot_sha256=result['source_snapshot_sha256'],
            original_fixture=str(original),original_fixture_sha256=fixture_sha,archive=str(output_archive.resolve()),
            archive_sha256=hashlib.sha256(output_archive.read_bytes()).hexdigest(),array_bytes=total,cases=case_meta,
            persisted_before_mapped_index_compile=True,CPU_byte_deduplication_scope='Fixture-only; no runtime CPUindices or persistent cross-owner sharing')
        manifest_path=unit/'ast-mapped-index-fixture.json';_write(manifest_path,manifest)
        result.update(status='derived_fixture_saved',fixture=dict(path=str(manifest_path),sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),array_bytes=total));_write(path,result)
        from mlx2.runtime.gguf import source_ast_mapped_index_matmul as mapped
        before=_counters(adapter);rows=[];scale=adapter.model.args.index_head_dim**-.5*adapter.model.args.index_n_heads**-.5
        for case,meta in zip(cases,case_meta):
            left=mx.array(case['left_cpu']);table=mx.array(case['table_cpu']);maps=mx.array(case['map_cpu']);keys=mx.array(case['keys_cpu'])
            expected=mx.array(case['raw_cpu']);weight=mx.array(case['weights_cpu']);expected_score=mx.array(case['score_cpu']);expected_top=mx.array(case['top_cpu'])
            roots=[left,table,maps,keys,expected,weight,expected_score,expected_top]
            right=mx.swapaxes(keys,-1,-2);g=left.shape[0];n=case['count']
            native=ordinary.private_matmul(left,right,check_live=live,implementation='native_batch_v1')
            pinned=ordinary.private_matmul(left,right,check_live=live,implementation='pinned_regular_s64_v1')
            roots.extend((native,pinned));drained=False;positive=mapped.private_matmul(left,table,maps,check_live=live);roots.extend((positive.output,positive.status))
            weighted=mx.sum(mx.maximum(positive.output,0)*(weight*scale).transpose(0,2,1),axis=1)[:,None]
            top=stable_pool_topk(weighted,512);roots.extend((weighted,top));mx.eval(native,pinned,positive.output,positive.status,weighted,top);drained=True;live()
            flags=np.asarray(positive.status)
            checks=dict(native_raw=_compare(native,expected),pinned_raw=_compare(pinned,expected),mapped_raw=_compare(positive.output,expected),
                mapped_weighted=_compare(weighted,expected_score),mapped_topk=_compare(top,expected_top))
            controls=[]
            for label,value in(('negative_one',-1),('equal_table_rows',table.shape[0])):
                bad=case['map_cpu'].astype(np.int32);bad[0,-1]=value;bad_gpu=mx.array(bad);roots.append(bad_gpu)
                drained=False;invalid=mapped.private_matmul(left,table,bad_gpu,check_live=live);roots.extend((invalid.output,invalid.status));mx.eval(invalid.output,invalid.status);drained=True
                status=np.asarray(invalid.status);output=np.asarray(invalid.output);expected_flags=np.zeros(g,np.uint32);expected_flags[0]=1
                other=_compare(output[1:],case['raw_cpu'][1:]);zero=np.zeros_like(output[0])
                controls.append(dict(label=label,map_position='lastcolumn_badbatch0',status_exact=bool(np.array_equal(status,expected_flags)),
                    defined_zero_batch=output[0].tobytes()==zero.tobytes(),other_batches=other,
                    passed=bool(np.array_equal(status,expected_flags))and output[0].tobytes()==zero.tobytes()and other['equal']))
            row={**meta,'descriptor':positive.descriptor,'checks':checks,'positive_status_zero':bool(np.array_equal(flags,np.zeros(g,np.uint32))),
                'invalid_controls':controls,'passed':all(v['equal']and v['finite']for v in checks.values())and not flags.any()and all(v['passed']for v in controls)}
            rows.append(row);result.update(status='raw_candidate_classes',cases=rows);_write(path,result)
            if not row['passed']:continue
            prepared=table[maps];prepared_right=mx.swapaxes(prepared,-1,-2);a=mx.contiguous(left);b=mx.contiguous(table)
            roots.extend((prepared,prepared_right,a,b));mx.eval(prepared,prepared_right,a,b,maps,positive.status);live()
            def invoke(scope,arm):
                if arm=='mapped':
                    if scope=='complete_matrix':
                        staged=mx.array(case['map_cpu']);roots.append(staged);value=mapped.private_matmul(left,table,staged,check_live=live)
                        roots.extend((value.output,value.status));mx.eval(value.output,value.status)
                        if np.any(np.asarray(value.status)):raise RuntimeError('Timed mapped index map refused')
                        return value.output
                    live();_,matrix=mapped._kernels(g,n,table.shape[0])
                    value=matrix(inputs=[a,b,maps,positive.status],grid=positive.descriptor['grid'],threadgroup=positive.descriptor['threadgroup'],
                        output_shapes=[positive.descriptor['output_shape']],output_dtypes=[mx.float32])[0]
                    roots.append(value);mx.eval(value);return value
                implementation='native_batch_v1'if arm=='native_gather'else'pinned_regular_s64_v1'
                if scope=='complete_matrix':
                    staged=mx.array(case['map_cpu']);gathered=table[staged];rhs=mx.swapaxes(gathered,-1,-2);roots.extend((staged,gathered,rhs))
                else:rhs=prepared_right
                value=ordinary.private_matmul(left,rhs,check_live=live,implementation=implementation);roots.append(value);mx.eval(value);return value
            samples=[]
            for baseline_arm in('native_gather','pinned_gather'):
                for scope in('complete_matrix','prepared_consumer_lower_bound'):
                    for arm in(baseline_arm,'mapped'):
                        drained=False;value=invoke(scope,arm);drained=True
                        if not _compare(value,expected)['equal']:raise RuntimeError('Mapped index warm changed arithmetic')
                    for sequence,arm in enumerate((baseline_arm,'mapped','mapped',baseline_arm)):
                        live();drained=False;started=time.perf_counter_ns();value=invoke(scope,arm);elapsed=time.perf_counter_ns()-started;drained=True
                        if not _compare(value,expected)['equal']:raise RuntimeError('Mapped index timed output changed arithmetic')
                        samples.append(dict(comparison_baseline=baseline_arm,scope=scope,sequence=sequence,arm=arm,elapsed_ns=elapsed))
            row['timings']=samples;result['cases']=rows;_write(path,result)
        delta={k:v-before.get(k,0)for k,v in _counters(adapter).items()if type(v)in(int,float)}
        result.update(status='passed'if all(v['passed']for v in rows)else'raw_mismatch',completed=True,
            passed=len(rows)==9 and all(v['passed']for v in rows),implementation_identity=mapped.implementation_identity(),component_counter_delta=delta,
            no_component_expert_or_engram_io=all(delta.get(k)==0 for k in('expert_bytes_read','expert_uploads','host_route_materializations','engram_bytes_read')),
            selected_or_defaulted=False,timing_contract=dict(comparison_blocks=['native_gather_vs_mapped','pinned_gather_vs_mapped'],
                samples_per_arm_per_scope_per_block=2,independent_ABBA_repetitions_per_block=1,common_table_GPU_setup_outside_timer=True,
                complete_matrix='CPUmapstaging+normalgather+native/pinnedconsumer versus CPUmapstaging+validation+mappedconsumer+statuscheck+GPUcompletion; weighted/topk outside timer',
                prepared_consumer_lower_bound='PregatherednormalKV versus immutableprevalidatedsharedtable/maps/status; no staging/gather/validation, not serving path',
                controlled_model_performance=False,explicit_cross_query_threadgroup_reuse=False))
        return result
    except BaseException as error:
        result.update(status='failed',completed=False,error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException:_QUARANTINE.append((cases,roots))
        _write(path,result)
