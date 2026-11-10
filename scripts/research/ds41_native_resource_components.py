"""Bounded native indirect-table functional suite; parent owns GPU lifecycle."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

def run(mx,check_live=None):
    from mlx2.adapters.deepseek_v41_source_native_resource import _owned_gpu,component_lookup,PROFILE
    _owned_gpu(mx,check_live)  # refuse before native import, shader creation or allocations
    import numpy as np
    import _ds41_resident_pilot as native
    record=dict(profile=PROFILE,qualified=False,performance_claim=False,serving_backend=False,cases=[])
    keys=list(range(20,35));kinds=[1+(i%2)for i in range(15)];serials=[(1<<34)+100+i for i in range(15)]
    # Delayed producer operations plus offset views: graph-input upload/order is exercised.
    def make_backings():
        return [((mx.arange(7+127+i,dtype=mx.uint32)+i*17)%251).astype(mx.uint8)[7:]for i in range(15)]
    backings=make_backings()
    expected=[((np.arange(7,7+127+i,dtype=np.uint32)+i*17)%251).astype(np.uint32)for i in range(15)]
    mx.eval(*backings)
    record['actual_backing_capability']=native.inspect_backings(backings,True)
    modes=['tracked_staging_v1']
    if all(row['direct_supported']for row in record['actual_backing_capability']):modes.append('direct_array_v1')
    else:record['direct_gate']='unsupported_actual_MLX_untracked_backing; no serving capability claimed'
    owners=[]
    for mode in modes:
        for batch in range(3):
            if check_live()is not True:raise RuntimeError('Native component ownership lost')
            indexes=list(range(batch*5,batch*5+5))+[batch*5]
            ids=[keys[i]for i in indexes];wanted=[serials[i]for i in indexes]
            item=component_lookup(make_backings(),mx.array(ids,mx.uint32),mx.array(wanted,mx.uint64),keys=keys,kinds=kinds,
                serials=serials,generation=(1<<33)+71,expected_generation=(1<<33)+71,check_live=check_live,mode=mode)
            owners.append(item)
            # Independent later MLX consumer must see the post-indirect output fence.
            consumer=item.output*mx.array(3,mx.uint32)+mx.array(7,mx.uint32)
            mx.eval(consumer);mx.synchronize()
            actual=np.asarray(item.output)
            want=np.array([[0,keys[i],kinds[i],int(expected[i][0]),int(expected[i][-1]),serials[i]&0xffffffff,serials[i]>>32,71]for i in indexes],np.uint32)
            np.testing.assert_array_equal(actual,want)
            np.testing.assert_array_equal(np.asarray(consumer),want*3+7)
            receipt=item.receipt()
            if not receipt['completed']or not receipt['succeeded']or receipt['retained_arrays']or receipt['retained_native_buffers']:raise RuntimeError('Native terminal callback incomplete')
            record['cases'].append(dict(mode=mode,batch=batch,passed=True,raw_output=actual.tolist(),expected=want.tolist(),receipt=receipt))
        for label,ids,wanted,expected_generation,status in [
            ('missing',[99]*6,[0]*6,(1<<33)+71,1),('invalid',[384]*6,[0]*6,(1<<33)+71,2),
            ('stale_generation',[keys[0]]*6,[serials[0]]*6,0,3),('stale_serial',[keys[0]]*6,[serials[0]+1]*6,(1<<33)+71,5)]:
            if check_live()is not True:raise RuntimeError('Native component ownership lost')
            item=component_lookup(make_backings(),mx.array(ids,mx.uint32),mx.array(wanted,mx.uint64),keys=keys,kinds=kinds,
                serials=serials,generation=(1<<33)+71,expected_generation=expected_generation,check_live=check_live,mode=mode)
            owners.append(item);mx.eval(item.output);mx.synchronize();out=np.asarray(item.output)
            np.testing.assert_array_equal(out[:,0],np.full(6,status,np.uint32))
            record['cases'].append(dict(mode=mode,label=label,passed=True,raw_output=out.tolist(),receipt=item.receipt()))
    # Two immutable generations coexist before either output is evaluated.
    pending=[]
    for generation in (81,82):
        item=component_lookup(make_backings(),mx.array([keys[0]]*6,mx.uint32),mx.array([serials[0]]*6,mx.uint64),
            keys=keys,kinds=kinds,serials=serials,generation=generation,expected_generation=generation,
            check_live=check_live,mode='tracked_staging_v1')
        pending.append((generation,item,item.output*3+7));owners.append(item)
        if item.receipt()['retained_arrays']!=17:raise RuntimeError('Lazy generation lost original input owners')
    if check_live()is not True:raise RuntimeError('Native component ownership lost')
    mx.async_eval(*(consumer for _,_,consumer in pending));mx.eval(*(consumer for _,_,consumer in pending));mx.synchronize()
    for generation,item,consumer in pending:
        out=np.asarray(item.output);np.testing.assert_array_equal(out[:,7],np.full(6,generation,np.uint32))
        np.testing.assert_array_equal(np.asarray(consumer),out*3+7)
        receipt=item.receipt()
        if not receipt['succeeded']or receipt['retained_arrays']or receipt['retained_native_buffers']:
            raise RuntimeError('Overlapping generation failed terminal ownership release')
        record['cases'].append(dict(label='overlapping_generation',generation=generation,raw_output=out.tolist(),passed=True,receipt=receipt))
    record.update(passed=True,case_count=len(record['cases']),scope='15 tiny resources; dynamic GPU address table/status/lifetime/final MLX consumer only; no expert math')
    return record
