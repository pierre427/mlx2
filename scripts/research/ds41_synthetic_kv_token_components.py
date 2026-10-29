# SPDX-License-Identifier: MIT
"""Forty-local/four-source KV-only token graphs, synthetic storage."""
import time,statistics
MODES=('ordinary_kv','skip_owned_local_copy','no_kv_writes')

def run(mx,check_live=None):
 from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
 check=check_live or (lambda:None);rows=[]
 def reset(frontier):
  states=[]
  for i in range(40):
   source=i<4;ratio=(1,1,2,2)[i]if source else 0
   s=DeepSeekV41LayerCache(ratio,xp=mx,source=source,identity='synthetic40layerKVonly');s.offset=frontier-1
   s.keys=mx.full((1,1,min(frontier-1,128),512),.125,dtype=mx.float32)
   if source:
    count=(frontier-1)//ratio;cap=((count+1+255)//256)*256;s._pool_count=count;s._capacity=cap
    s._pooled=mx.full((1,cap,512),.125,dtype=mx.float32);s._index_keys=mx.full((1,cap,128),.25,dtype=mx.float32)
   states.append(s)
  key=mx.full((1,1,1,512),.125,dtype=mx.float32);pooled=mx.full((1,1,512),.125,dtype=mx.float32);index=mx.full((1,1,128),.25,dtype=mx.float32)
  mx.eval(key,pooled,index,[s.keys for s in states],[s._pooled for s in states],[s._index_keys for s in states]);mx.synchronize()
  return states,key,pooled,index
 def trial(frontier,mode):
  states,key,pool,index=reset(frontier)
  t=time.perf_counter()
  if mode!='no_kv_writes':
   for s in states:
    if mode=='ordinary_kv':s.update_local(key)
    else:
     band=mx.concatenate((s.keys[:,:,-127:,:],key),axis=2);s.keys=band[:,:,-128:,:];s.offset+=1
   for s in states[:4]:s.append_pools(pool,index)
  submitted=time.perf_counter();mx.eval([s.keys for s in states],[s._pooled for s in states],[s._index_keys for s in states]);mx.synchronize();end=time.perf_counter()
  return dict(host_graph_submit_seconds=submitted-t,completion_wait_seconds=end-submitted,total_seconds=end-t)
 cells=[(n,m)for n in (100,16384)for m in MODES];data={cell:[]for cell in cells}
 for cell in cells:check();trial(*cell)
 for rep in range(7):
  for cell in cells[rep%len(cells):]+cells[:rep%len(cells)]:check();data[cell].append(trial(*cell))
 for (n,mode),samples in data.items():rows.append(dict(context=n,mode=mode,samples=samples,medians={k:statistics.median(s[k]for s in samples)for k in samples[0]}))
 return dict(schema='mlx2.ds41-synthetic40KVtoken.v1',complete=True,checkpoint_loaded=False,model_constructed=False,cases=rows,repetitions=7,local_layers=40,pool_source_writes=4,source_ratios=[1,1,2,2],scope='Storage-only40local+4sourcewrites inonecompletedgraph; reset outsideclocks; sourcecompressed vectors supplied; excludes sourceproducer/attention/partialgroupmath; no outputsqualityassessment')
