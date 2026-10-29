"""Resumable CPU-only conversion of all routed Source experts; preserves originals."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, ctypes, hashlib, json, os, time
import numpy as np
from safetensors.numpy import save_file
from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, read_tensors
from mlx2.adapters.deepseek_v41_source_weights import expert_names, PROJECTIONS

REVISION='source-q1-affine-g64-lloyd4-f16-v1'
ARTIFACT=DEST=LIB=None
def initialize(model,destination,library):
    global ARTIFACT,DEST,LIB
    ARTIFACT=inspect_artifact(model);DEST=Path(destination);LIB=ctypes.CDLL(library)
    LIB.source_q1.argtypes=[ctypes.c_void_p]*5+[ctypes.c_size_t,ctypes.c_void_p]
    LIB.source_q1.restype=ctypes.c_int

def convert(item):
    layer,expert=item;start=time.perf_counter();name=f'l{layer:02d}-e{expert:03d}.safetensors';target=DEST/name;receipt=target.with_suffix('.json')
    if target.exists() and receipt.exists():
        record=json.loads(receipt.read_text())
        if record['source']!=ARTIFACT['identity']['fingerprint'] or record['quantization']!=REVISION:raise ValueError('existing incompatible expert')
        h=hashlib.sha256(target.read_bytes()).hexdigest()
        if h!=record['sha256']:raise ValueError('existing expert hash changed')
        return record
    if target.exists() or receipt.exists():raise ValueError('partial expert publication: '+name)
    raw=read_tensors(ARTIFACT,expert_names(layer,expert));arrays={};errors={};original=[]
    for kind,part in PROJECTIONS.items():
        prefix=f'layers.{layer}.ffn.experts.{expert}.{part}.';w=raw[prefix+'weight'];s=raw[prefix+'scale'];n,k=w.shape[0],w.shape[1]*2
        q=np.empty((n,k//32),dtype='<u4');scale=np.empty((n,k//64),dtype='<f2');bias=np.empty_like(scale);metrics=np.zeros(2,np.float64)
        code=LIB.source_q1(w.ctypes.data,s.ctypes.data,q.ctypes.data,scale.ctypes.data,bias.ctypes.data,scale.size,metrics.ctypes.data)
        if code:raise ValueError('nonfinite source or Q1 parameter '+str(code))
        arrays.update({kind+'.qweight':q,kind+'.scales':scale,kind+'.biases':bias});errors[kind]={'squared_error':float(metrics[0]),'source_squared_norm':float(metrics[1]),'relative_rmse':float(np.sqrt(metrics[0]/metrics[1])) if metrics[1] else 0}
        original.extend([{**ARTIFACT['tensors'][prefix+field],'tensor':prefix+field} for field in ('weight','scale')])
    temp=target.with_suffix('.tmp');save_file(arrays,str(temp),metadata={'source':ARTIFACT['identity']['fingerprint'],'quantization':REVISION,'layer':str(layer),'expert':str(expert)})
    h=hashlib.sha256(temp.read_bytes()).hexdigest();os.replace(temp,target)
    record={'layer':layer,'expert':expert,'file':name,'source':ARTIFACT['identity']['fingerprint'],'quantization':REVISION,'sha256':h,'size':target.stat().st_size,'payload_bytes':sum(v.nbytes for v in arrays.values()),'original_extents':original,'errors':errors,'seconds':time.perf_counter()-start}
    temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(record)+'\n');os.replace(temporary,receipt);return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--output',required=True);parser.add_argument('--library',required=True);parser.add_argument('--workers',type=int,default=6);parser.add_argument('--limit',type=int,default=15360);args=parser.parse_args()
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True);artifact=inspect_artifact(args.model);items=[(l,e) for l in range(40) for e in range(384)][:args.limit];records=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.model,args.output,args.library)) as pool:
        for record in pool.map(convert,items,chunksize=1):
            records.append(record)
            if len(records)%128==0 or len(records)==len(items):print(json.dumps({'completed':len(records),'total':len(items),'seconds':time.perf_counter()-start,'payload_bytes':sum(r['payload_bytes'] for r in records)}),flush=True)
    if len(records)==15360:
        manifest={'schema':'mlx2.source-expert-q1.v1','source':artifact['identity']['fingerprint'],'source_revision':'1bae0e1088993b9b82de08f59de5be5a34e42170','quantization':REVISION,'group_size':64,'weight_bits':1,'effective_bits_with_parameters':1.5,'dense_modified':False,'original_objects':'immutable six-tensor extents in original Source shards','experts':records,'payload_bytes':sum(r['payload_bytes'] for r in records),'seconds':time.perf_counter()-start}
        manifest['fingerprint']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest();temp=destination/'manifest.tmp';temp.write_text(json.dumps(manifest)+'\n');os.replace(temp,destination/'manifest.json')
