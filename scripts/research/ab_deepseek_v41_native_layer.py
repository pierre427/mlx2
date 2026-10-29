"""Research-only layer2 native MLX affine A/B; preserves the original model.

Conversion algebra derives from this project's GGUF block oracle. A converted
layer is a derived artifact, not a serving route or numerical qualification.
"""
from contextlib import contextmanager
from copy import copy
from dataclasses import asdict
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import time

import numpy as np
from mlx2.desktop_campaign_worker import atomic

LAYER = 2
PREFIX = 'blk.2.'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(16 << 20): h.update(chunk)
    return h.hexdigest()


def prepare(model_path, destination, *, phase=print):
    """CPU-only bounded row conversion, then safetensors serialization."""
    from safetensors.numpy import save_file
    from mlx2.adapters.deepseek_v41_artifact import FILENAME, inspect_artifact
    from mlx2.runtime.gguf.reader import GGUFFile, FORMATS
    from deepseek_v41_native_layer import affine_spec, convert_affine, validate_affine

    artifact = inspect_artifact(model_path)
    source = Path(model_path) / FILENAME
    source_stamp=source.stat()
    destination = Path(destination)
    if destination.exists():
        raise ValueError('refusing existing native-layer artifact directory')
    destination.mkdir(parents=True)
    manifest = dict(schema='mlx2.v41.native-affine-layer.v1', checkpoint=FILENAME+'@'+artifact['identity']['fingerprint'],
                    source_artifact=artifact['identity'],layer=LAYER,converted=False,tensors={},
                    algebra='IQ1_S->affine2/group32; Q8_0->affine8/group32; FP32 scales/biases; original native scalar tensors',
                    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    root=Path(__file__).resolve().parents[2]
    manifest['conversion_sources']={name:file_hash(root/name) for name in ('scripts/research/ab_deepseek_v41_native_layer.py','scripts/research/deepseek_v41_native_layer.py','src/mlx2/runtime/gguf/blocks.py','src/mlx2/runtime/gguf/iq1_codebook.py','src/mlx2/runtime/gguf/reader.py')}
    arrays = {}
    fd = os.open(source, os.O_RDONLY)
    file = GGUFFile(source)
    started = time.perf_counter()
    try:
        for name,tensor in file.tensors.items():
            if not name.startswith(PREFIX):continue
            info = dict(source_type=tensor.ggml_type,source_shape=tensor.numpy_shape,source_bytes=tensor.nbytes,validations=[])
            if tensor.ggml_type in (8,19):
                shape = tensor.numpy_shape; width = shape[-1]; rows = math.prod(shape[:-1]);bits=2 if tensor.ggml_type==19 else 8
                output = []
                for suffix,dtype,last in [('qweight',np.uint32,width*bits//32),('scales',np.float32,width//32),('biases',np.float32,width//32)]:
                    path=destination/(name+'.'+suffix+'.npy')
                    mapped=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=(*shape[:-1],last))
                    arrays[name+'.'+suffix]=mapped;output.append(mapped.reshape(rows,last))
                _,block_size,block_bytes=FORMATS[tensor.ggml_type]
                row_bytes=width//block_size*block_bytes
                for first in range(0,rows,64):
                    last=min(rows,first+64)
                    raw=os.pread(fd,(last-first)*row_bytes,file.data_offset+tensor.offset+first*row_bytes)
                    if len(raw)!=(last-first)*row_bytes:raise EOFError(name)
                    chunk=convert_affine(raw,tensor.ggml_type,(last-first,width),max_elements=1<<20)
                    for target,value in zip(output,chunk):target[first:last]=value
                    if first==0 or last==rows or first<=rows//2<last:
                        # Independent scalar decoder on a bounded actual-source row.
                        info['validations'].append(validate_affine(raw[:row_bytes],tensor.ggml_type,(1,width),max_elements=1<<20))
                for target in output:target.flush()
                info.update(bits=bits,group_size=32,native_bytes=sum(target.nbytes for target in output))
            else:
                if tensor.ggml_type not in (0,1):raise ValueError('unexpected layer scalar type')
                raw=os.pread(fd,tensor.nbytes,file.data_offset+tensor.offset)
                if len(raw)!=tensor.nbytes:raise EOFError(name)
                arrays[name]=np.frombuffer(raw,dtype='<f4' if tensor.ggml_type==0 else '<f2').reshape(tensor.numpy_shape)
                info['native_bytes']=arrays[name].nbytes
            manifest['tensors'][name]=info
            atomic(destination/'conversion.json',manifest)
            phase('native_layer_tensor_converted',tensor=name,native_bytes=info['native_bytes'])
        after=os.fstat(fd)
        if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)!=(source_stamp.st_dev,source_stamp.st_ino,source_stamp.st_size,source_stamp.st_mtime_ns) or source.stat()!=source_stamp:
            raise ValueError('original GGUF changed during conversion')
        if len(manifest['tensors'])!=32:raise RuntimeError('layer2 inventory changed')
        path=destination/'layer-2.safetensors'
        save_file(arrays,str(path),metadata={'format':'mlx','source_checkpoint':manifest['checkpoint'],'layer':'2','qualification':'unqualified','quantization':'affine2/group32 experts; affine8/group32 dense; fp32 scale/bias'})
        manifest.update(converted=True,native_bytes=sum(v['native_bytes'] for v in manifest['tensors'].values()),source_bytes=sum(v['source_bytes'] for v in manifest['tensors'].values()),
                        safetensors=str(path.resolve()),safetensors_bytes=path.stat().st_size,safetensors_sha256=file_hash(path),conversion_seconds=time.perf_counter()-started)
        atomic(destination/'conversion.json',manifest)
    finally:
        arrays.clear();gc.collect();file.close();os.close(fd)
    # Row memmaps were temporary bounded construction files, no longer needed.
    for path in destination.glob('*.npy'):path.unlink()
    return manifest


class NativeWeights:
    """Isolated adapter-owned layer facade; no selected serving policy changes."""
    def __init__(self, original, arrays, *, dense, experts, dtype, mx):
        self.original,self.data,self.native_dense,self.native_experts,self.dtype,self.mx=original,arrays,dense,experts,dtype,mx
        self.expert_stream=None if experts else original.expert_stream
        self.params={}
        for name,tensor in original.tensors.items():
            if name.startswith(PREFIX) and tensor.ggml_type in (8,19):
                self.params[name]=(arrays[name+'.qweight'],arrays[name+'.scales'].astype(dtype),arrays[name+'.biases'].astype(dtype))
        mx.eval(self.params)

    def __getattr__(self,name):return getattr(self.original,name)

    def array(self,name):
        return self.data[name] if name in self.data else self.original.array(name)

    def can_fuse_dense(self,x,*names):
        return False if self.native_dense else self.original.can_fuse_dense(x,*names)

    def linear(self,name):
        if not self.native_dense:return self.original.linear(name)
        mx=self.mx
        if name in self.params:
            q,s,b=self.params[name]
            return lambda x:mx.quantized_matmul(x.astype(self.dtype),q,s,b,group_size=32,bits=8).astype(mx.float32)
        weight=self.array(name).T.astype(mx.float32)
        return lambda x:x.astype(mx.float32)@weight

    def gate_up(self,x,gate,up):
        if not self.native_dense:return self.original.gate_up(x,gate,up)
        return self.linear(gate)(x),self.linear(up)(x)

    def grouped(self,x,name,*,groups=8):
        if not self.native_dense:return self.original.grouped(x,name,groups=groups)
        mx=self.mx;q,s,b=self.params[name];n,k=self.original.tensors[name].numpy_shape
        shape=(groups,n//groups)
        xx=x.transpose(0,2,1,3).astype(self.dtype)
        result=mx.quantized_matmul(xx,q.reshape(*shape,q.shape[-1]),s.reshape(*shape,s.shape[-1]),b.reshape(*shape,b.shape[-1]),group_size=32,bits=8)
        return result.transpose(0,2,1,3).astype(mx.float32)

    def moe(self,x,ids,scores,prefix,**kwargs):
        if not self.native_experts:return self.original.moe(x,ids,scores,prefix,**kwargs)
        from mlx2.runtime.models.deepseek_v4 import limited_swiglu
        mx=self.mx;t,topk=ids.shape
        # Sort routed assignments without a host readback; restore slot order
        # before the original left-to-right reduction.
        flat=ids.reshape(-1);order=mx.argsort(flat);rhs=flat[order];lhs=(order//topk).astype(mx.uint32)
        def gather(value,kind):
            q,s,b=self.params[prefix+'.ffn_'+kind+'_exps.weight']
            return mx.gather_qmm(value.astype(self.dtype),q,s,b,rhs_indices=rhs,
                                 group_size=32,bits=2,sorted_indices=True).astype(mx.float32)
        rounded=lambda v:v.astype(mx.bfloat16).astype(mx.float32)
        sorted_input=x[lhs,None,:]
        gate=rounded(gather(sorted_input,'gate')).squeeze(1)
        up=rounded(gather(sorted_input,'up')).squeeze(1)
        middle=rounded(limited_swiglu(gate,up)*scores.reshape(-1)[order,None])
        down=rounded(gather(middle[:,None,:],'down')).squeeze(1)
        parts=down[mx.argsort(order)].reshape(t,topk,x.shape[-1])
        result=mx.zeros((t,x.shape[-1]),mx.float32)
        for slot in range(topk):result=result+parts[:,slot]
        return result


def numeric(reference,candidate):
    a,b=np.asarray(reference).astype(np.float64),np.asarray(candidate).astype(np.float64)
    if a.shape!=b.shape:raise ValueError('numerical shape mismatch')
    if not a.size:return dict(shape=a.shape,finite=True,max_abs=0.,rmse=0.,relative_rms=0.,equal_fraction=1.,reference_rms=0.)
    difference=b-a
    rms=float(np.sqrt(np.mean(a*a)))
    return dict(shape=a.shape,finite=bool(np.isfinite(b).all()),max_abs=float(np.max(np.abs(difference))),
                rmse=float(np.sqrt(np.mean(difference*difference))),relative_rms=float(np.sqrt(np.mean(difference*difference)))/max(rms,1e-30),
                equal_fraction=float(np.mean(a==b)),reference_rms=rms)


@contextmanager
def scopes(model,rows):
    from mlx2.runtime.gguf.prefill_scope import dense_prefill_context
    from mlx2.runtime.gguf.segmented_prefill import expert_prefill_context
    if rows>1:
        with dense_prefill_context(model.weights.dense_prefill_policy,tokens=rows),expert_prefill_context(model.weights.expert_prefill_policy,tokens=rows):yield
    else:yield


def _run_impl(adapter,mx,out,phase,*,guard):
    from mlx2.runtime.models import deepseek_v41 as module
    from bench_deepseek_v41_1k import prepare as prepare_request
    from profile_deepseek_v41 import arrays_numpy

    out=Path(out);model=adapter.model;w=adapter.weights
    path=Path(json.loads((out/'spec.json').read_text())['native_layer_path'])
    converted=json.loads((path/'conversion.json').read_text())
    if not converted['converted'] or converted['checkpoint']!=json.loads((out/'spec.json').read_text())['checkpoint']:raise ValueError('native artifact binding mismatch')
    if Path(converted['safetensors']).stat().st_size!=converted['safetensors_bytes'] or file_hash(converted['safetensors'])!=converted['safetensors_sha256']:
        raise ValueError('native safetensors content changed')
    guard();started=time.perf_counter();arrays=mx.load(converted['safetensors']);mx.eval(arrays)
    result=dict(schema='mlx2.v41.native-layer-ab.v1',layer=LAYER,conversion=converted,load_seconds=time.perf_counter()-started,
                scope='same captured real layer2 inputs/state; same original model math except native MLX affine matrix dispatch and its accumulation; no APC reuse; weights preloaded; thermally uncontrolled',
                interpretation='Numerical differences are measured, never promoted to exact parity or qualification. FP16 arm adds scale/bias/output precision changes.',
                cases=[],passed=False)
    atomic(out/'native-layer-ab.json',result)
    prepare_request(adapter.model_path if hasattr(adapter,'model_path') else json.loads((out/'spec.json').read_text())['model'],out)
    request=json.loads((out/'request-1024.json').read_text())['request'];tokens=list(adapter.prompt_tokens(request));assert len(tokens)==960
    frames=[];original=module.Block.__call__;original_moe=w.moe
    def capture_moe(x,ids,scores,prefix,**kwargs):
        if prefix=='blk.2' and frames:
            mx.eval(x,ids,scores);frames[-1].update(ffn_x=x,ids=ids,scores=scores)
        return original_moe(x,ids,scores,prefix,**kwargs)
    def capture(block,h,pre,state,cache,shared):
        if block.layer==LAYER:
            mx.eval(h,pre,state.state,shared)
            # This is a private operand capture within an unfinished forward,
            # not an APCv2 publication. Preserve the runtime publication gate.
            from mlx2.runtime.models.deepseek_v4_cache import DeepSeekV4Snapshot, _copy
            state.validate_state()
            snapshot=DeepSeekV4Snapshot(state.meta_state,_copy(mx,state.state));mx.eval(snapshot.arrays)
            frames.append(dict(rows=h.shape[1],start=state.offset,h=h,pre=pre,snapshot=snapshot,cache=cache,shared=dict(shared)))
        output=original(block,h,pre,state,cache,shared)
        if block.layer==LAYER:
            mx.eval(output,state.state)
            frames[-1]['original_output']=output
            frames[-1]['original_state']=arrays_numpy(state.state)
            frames[-1]['original_meta']=state.meta_state
        return output
    module.Block.__call__=capture;w.moe=capture_moe
    cache=model.make_cache()
    try:
        for first,last in [(0,512),(512,960)]:
            guard();value=model.prefill_forward(mx.array([tokens[first:last]],mx.uint32),cache);mx.eval(value,[s.state for s in cache])
            phase('native_layer_inputs_captured',rows=last-first,context=last)
        selected=int(mx.argmax(value[0,-1]).item())
        value=model(mx.array([[selected]],mx.uint32),cache);mx.eval(value,[s.state for s in cache])
        phase('native_layer_decode_captured',context=960,token=selected)
    finally:
        module.Block.__call__=original;w.moe=original_moe
    if len(frames)!=3:raise RuntimeError('layer input capture shape changed')
    views={'gguf':model};blocks={'gguf':model.layers[LAYER]}
    control=copy(model);control.weights=copy(w)
    control.weights.dense_prefill_policy='reference';control.weights.expert_prefill_policy='reference'
    views['gguf_f32_control']=control;blocks['gguf_f32_control']=module.Block(control,LAYER)
    configs=[('native_dense_f32',True,False,mx.float32),('native_experts_f32',False,True,mx.float32),('native_all_f32',True,True,mx.float32),('native_all_f16',True,True,mx.float16)]
    for name,dense,experts,dtype in configs:
        guard();view=copy(model);view.weights=NativeWeights(w,arrays,dense=dense,experts=experts,dtype=dtype,mx=mx);block=module.Block(view,LAYER)
        views[name]=view;blocks[name]=block
    result['resident_memory']=dict(active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory())
    for index,frame in enumerate(frames):
        guard();rows=frame['rows'];state=frame['cache'][LAYER]
        case=dict(rows=rows,start_context=frame['start'],variants={},moe={},pairs=[],routing_topk=frame['ids'].tolist())
        result['cases'].append(case)
        reference=None
        reference_state=None
        reference_meta=None
        def restore():
            state.restore(frame['snapshot']);mx.eval(state.state)
        def execute(name):
            with scopes(views[name],rows):
                output=blocks[name](frame['h'],frame['pre'],state,frame['cache'],dict(frame['shared']))
                mx.eval(output,state.state)
            return output
        for name in views:
            guard();restore();output=execute(name);restore();output=execute(name)
            values=(np.asarray(output[0]),np.asarray(output[1]))
            if name=='gguf':
                reference=values
                reference_state=arrays_numpy(state.state);reference_meta=state.meta_state
                case['original_capture_replay']=[numeric(a,b) for a,b in zip(frame['original_output'],values)]
                if not all(v['equal_fraction']==1 for v in case['original_capture_replay']):raise RuntimeError('original layer snapshot replay changed values')
            case['variants'][name]=dict(output=numeric(reference[0],values[0]),next_hc_pre=numeric(reference[1],values[1]),samples_seconds=[],thread_cpu_seconds=[])
            state_rows=[]
            for a,b in zip(reference_state,arrays_numpy(state.state)):
                state_rows.append(dict(both_absent=True) if a is None and b is None else numeric(a,b))
            case['variants'][name].update(state_numerics=state_rows,state_metadata_equal=state.meta_state==reference_meta)
            phase('native_layer_variant_warmed',rows=rows,variant=name,error=case['variants'][name]['output']['relative_rms'])
            atomic(out/'native-layer-ab.json',result)
            if not case['variants'][name]['output']['finite']:raise RuntimeError('nonfinite native layer output')
        for repeat in range(6):
            names=list(views);names=names[repeat%len(names):]+names[:repeat%len(names)]
            for name in names:
                guard();restore();before=w.expert_stream.counters();cpu=time.thread_time();started=time.perf_counter();output=execute(name);elapsed=time.perf_counter()-started;cpu_elapsed=time.thread_time()-cpu
                after=w.expert_stream.counters();row=case['variants'][name];row['samples_seconds'].append(elapsed);row['thread_cpu_seconds'].append(cpu_elapsed)
                row.setdefault('expert_source_bytes',[]).append(after['v41_expert_bytes_read']-before['v41_expert_bytes_read'])
            phase('native_layer_pair',case=index,rows=rows,repeat=repeat,seconds={name:case['variants'][name]['samples_seconds'][-1] for name in views})
            atomic(out/'native-layer-ab.json',result)
        # Isolated MoE uses identical original expert IDs/scores in every arm.
        for name in ['gguf','native_experts_f32','native_all_f16']:
            target=w if name=='gguf' else views[name].weights
            def moe():
                with scopes(model,rows):
                    y=target.moe(frame['ffn_x'],frame['ids'],frame['scores'],'blk.2');mx.eval(y)
                return y
            expected=moe() if name=='gguf' else expected
            y=moe();row=dict(output=numeric(np.asarray(expected),np.asarray(y)),samples_seconds=[])
            for repeat in range(6):
                guard();start=time.perf_counter();y=moe();row['samples_seconds'].append(time.perf_counter()-start)
            row['median_seconds']=statistics.median(row['samples_seconds']);case['moe'][name]=row
        for name,row in case['variants'].items():
            row['median_seconds']=statistics.median(row['samples_seconds']);row['mean_seconds']=statistics.mean(row['samples_seconds']);row['speedup_vs_gguf']=statistics.median(case['variants']['gguf']['samples_seconds'])/row['median_seconds']
        atomic(out/'native-layer-ab.json',result)
        phase('native_layer_case_complete',rows=rows,medians={k:v['median_seconds'] for k,v in case['variants'].items()})
    result['passed']=True;atomic(out/'native-layer-ab.json',result)
    blocks.clear();views.clear();arrays.clear();frames.clear();gc.collect();mx.clear_cache()
    return result


def run(adapter,mx,out,phase,*,guard):
    # On success the implementation stack (including its last local block,
    # operands and facade) unwinds before the allocator cleanup boundary.
    try:
        return _run_impl(adapter,mx,out,phase,guard=guard)
    finally:
        gc.collect()
        mx.clear_cache()
