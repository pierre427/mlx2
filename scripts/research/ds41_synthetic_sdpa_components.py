# SPDX-License-Identifier: MIT
"""Actual composed attention versus installed fused SDPA; synthetic M1 only."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

def run(mx,check_live=None):
 from mlx2.runtime.models.deepseek_v41 import selected_attention
 from ds41_synthetic_attention_components import _timed
 mx.random.seed(41);rows=[]
 for length in (100,16384):
  local_n=min(length,128);count=min(length,512)
  q=mx.random.normal((1,64,1,512),dtype=mx.float32)*.1
  local=mx.random.normal((1,1,local_n,512),dtype=mx.float32)*.1
  pool=mx.random.normal((1,length,512),dtype=mx.float32)*.1
  ids=mx.arange(length-count,length,dtype=mx.uint32)[None,None,:];valid=mx.ones((1,1,count),mx.bool_)
  mask=mx.ones((1,1,1,local_n),mx.bool_);sinks=mx.zeros((64,),mx.float32)
  selected=pool[0,ids[0]][None];joined=mx.concatenate((local,selected),axis=2);joined_mask=mx.ones((1,1,1,local_n+count),mx.bool_)
  qb=q.astype(mx.bfloat16);kb=joined.astype(mx.bfloat16);sinkb=sinks.astype(mx.bfloat16)
  mx.eval(q,local,pool,ids,valid,mask,sinks,sinkb,selected,joined,joined_mask,qb,kb)
  funcs={
   'actual_source_pre_gathered':lambda:selected_attention(q,local,mask,pool,ids,valid,sinks,gathered=selected),
   'actual_source_full_gather':lambda:selected_attention(q,local,mask,pool,ids,valid,sinks),
   'sdpa_auto_F32_prejoined':lambda:mx.fast.scaled_dot_product_attention(q,joined,joined,scale=512**-.5,mask=joined_mask,sinks=sinks),
   'sdpa_force_F32_prejoined':lambda:mx.fast.scaled_dot_product_attention(q,joined,joined,scale=512**-.5,mask=joined_mask,sinks=sinks,force_fused=True),
   'sdpa_force_BF16_prejoined':lambda:mx.fast.scaled_dot_product_attention(qb,kb,kb,scale=512**-.5,mask=joined_mask,sinks=sinkb,force_fused=True),
  }
  def full_sdpa():
   gathered=pool[0,ids[0]][None];kv=mx.concatenate((local,gathered),axis=2)
   return mx.fast.scaled_dot_product_attention(q,kv,kv,scale=512**-.5,mask=joined_mask,sinks=sinks,force_fused=True)
  funcs['sdpa_force_F32_fresh_gather_join']=full_sdpa
  def split(query,keys,sink,width):
   return mx.concatenate([mx.fast.scaled_dot_product_attention(query[:,first:first+width],keys,keys,scale=512**-.5,mask=joined_mask,sinks=sink[first:first+width],force_fused=True)for first in range(0,64,width)],axis=1)
  funcs['sdpa_F32_2x32_heads']=lambda:split(q,joined,sinks,32)
  funcs['sdpa_F32_4x16_heads']=lambda:split(q,joined,sinks,16)
  funcs['sdpa_BF16_2x32_heads']=lambda:split(qb,kb,sinkb,32)
  funcs['sdpa_BF16_4x16_heads']=lambda:split(qb,kb,sinkb,16)
  def fresh_split():
   gathered=pool[0,ids[0]][None];kv=mx.concatenate((local,gathered),axis=2)
   return split(q,kv,sinks,32)
  funcs['sdpa_F32_fresh_gather_join_2x32']=fresh_split
  funcs['sdpa_F32_8x8_heads']=lambda:split(q,joined,sinks,8)
  funcs['sdpa_BF16_8x8_heads']=lambda:split(qb,kb,sinkb,8)
  aliased=mx.broadcast_to(joined,(1,8,joined.shape[2],512))
  aliased_bf16=mx.broadcast_to(kb,(1,8,kb.shape[2],512))
  mx.eval(aliased,aliased_bf16)
  funcs['sdpa_F32_8_logical_KV_heads']=lambda:mx.fast.scaled_dot_product_attention(q,aliased,aliased,scale=512**-.5,mask=joined_mask,sinks=sinks,force_fused=True)
  funcs['sdpa_BF16_8_logical_KV_heads']=lambda:mx.fast.scaled_dot_product_attention(qb,aliased_bf16,aliased_bf16,scale=512**-.5,mask=joined_mask,sinks=sinkb,force_fused=True)
  def fresh_alias():
   gathered=pool[0,ids[0]][None];kv=mx.concatenate((local,gathered),axis=2)
   heads=mx.broadcast_to(kv,(1,8,kv.shape[2],512))
   return mx.fast.scaled_dot_product_attention(q,heads,heads,scale=512**-.5,mask=joined_mask,sinks=sinks,force_fused=True)
  funcs['sdpa_F32_fresh_gather_join_8_KV_heads']=fresh_alias
  funcs['sdpa_F32_8KV_no_mask']=lambda:mx.fast.scaled_dot_product_attention(q,aliased,aliased,scale=512**-.5,sinks=sinks,force_fused=True)
  funcs['sdpa_BF16_8KV_no_mask']=lambda:mx.fast.scaled_dot_product_attention(qb,aliased_bf16,aliased_bf16,scale=512**-.5,sinks=sinkb,force_fused=True)
  funcs['sdpa_F32_8KV_no_mask_no_sinks_control']=lambda:mx.fast.scaled_dot_product_attention(q,aliased,aliased,scale=512**-.5,force_fused=True)
  funcs['sdpa_BF16_8KV_no_mask_no_sinks_control']=lambda:mx.fast.scaled_dot_product_attention(qb,aliased_bf16,aliased_bf16,scale=512**-.5,force_fused=True)
  def split_unmasked(query,keys,sink):
   return mx.concatenate([mx.fast.scaled_dot_product_attention(query[:,first:first+8],keys,keys,scale=512**-.5,sinks=sink[first:first+8],force_fused=True)for first in range(0,64,8)],axis=1)
  funcs['sdpa_F32_8x8_no_mask']=lambda:split_unmasked(q,joined,sinks)
  funcs['sdpa_BF16_8x8_no_mask']=lambda:split_unmasked(qb,kb,sinkb)
  def fresh_alias_nomask():
   gathered=pool[0,ids[0]][None];kv=mx.concatenate((local,gathered),axis=2);heads=mx.broadcast_to(kv,(1,8,kv.shape[2],512))
   return mx.fast.scaled_dot_product_attention(q,heads,heads,scale=512**-.5,sinks=sinks,force_fused=True)
  funcs['sdpa_F32_fresh_gather_join_8KV_no_mask']=fresh_alias_nomask
  def joint_math(query,keys,sink):
   qr=query.reshape(64,512);kr=keys.reshape(-1,512)
   scores=(qr@kr.T)*512**-.5
   probabilities=mx.softmax(mx.concatenate((scores,sink[:,None].astype(scores.dtype)),axis=-1),axis=-1)
   return (probabilities[:,:-1]@kr).reshape(1,64,1,512)
  compiled_joint=mx.compile(joint_math)
  funcs['joint_headrows_F32_prejoined']=lambda:joint_math(q,joined,sinks)
  funcs['joint_headrows_F32_compiled']=lambda:compiled_joint(q,joined,sinks)
  funcs['joint_headrows_BF16_diagnostic']=lambda:joint_math(qb,kb,sinkb)
  def fresh_joint():
   rows=pool[0,ids[0]][None];kv=mx.concatenate((local,rows),axis=2)
   return compiled_joint(q,kv,sinks)
  funcs['joint_headrows_F32_compiled_fresh_gather_join']=fresh_joint
  unsupported={};supported={}
  for name,fn in funcs.items():
   try:out=fn();mx.eval(out);mx.synchronize();supported[name]=fn
   except ValueError as e:unsupported[name]=str(e)
  timing=_timed(mx,supported,check_live)
  finite={}
  for name,fn in supported.items():out=fn();mx.eval(out);finite[name]=bool(mx.all(mx.isfinite(out)).item())
  rows.append(dict(context=length,heads=64,head_dim=512,local_tokens=local_n,selected_tokens=count,kv_heads=1,timing=timing,unsupported=unsupported,finite=finite))
 return dict(schema='mlx2.synthetic-ds41-sdpa-comparison.v1',complete=True,checkpoint_loaded=False,model_constructed=False,cases=rows,seed=41,repeats=7,scope='Synthetic actual composedfunction vs installedSDPA GQA D512 including sinks; preparedmath andfreshgather+join separately; BF16 alternative is timing-only, no numericalquality/parityassessment')
