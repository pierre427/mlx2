"""Source-loaded stage/lifetime contracts; no MLX or native runtime import."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('lifetime',ROOT/'src/mlx2/runtime/packed_prefill_lifetime.py')
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)

def method(path,klass,name,namespace):
    namespace.setdefault('__name__', 'mlx2.runtime.models._source_contract')
    namespace.setdefault('__package__', 'mlx2.runtime.models')
    tree=ast.parse((ROOT/path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==klass)
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

def candidate():
    args=NS(hidden_size=5120,intermediate_size=17408,num_attention_heads=24,num_key_value_heads=4,head_dim=256,
        linear_num_key_heads=16,linear_num_value_heads=48,linear_key_head_dim=128,
        linear_value_head_dim=128,linear_conv_kernel_dim=4,vocab_size=248320)
    layers=tuple(NS(linear_attn=NS(training=False,_prefill_scan_chunk=0,_gdn_state_dtype=None,
        mixed_materialized=lambda:None),mlp=NS(materialized=lambda:None)) for _ in range(64))
    return NS(args=args,_prefill_packed_n20=True,_prefill_layer_lifetime=True,_prefill_eval_block_size=1,
        native_layer_count=16,layer_map=NS(recurrent=tuple(range(48))),trunk=NS(layers=layers),native_dtype_preflight=lambda:'bfloat16')

class Tests(unittest.TestCase):
    def test_full_twenty_lane_charge_counts_persistent_successor_stage_and_arena(self):
        c=candidate();counts=(7000,)*20;caps=(192,)*20
        charge=L.n20_stagewise_charge_components(c,counts,caps)
        self.assertEqual(charge['arena_bytes'],2*16*20*((7192+63)//64+2)*4*64*256*2)
        self.assertEqual(charge['persistent_recurrent_bytes'],20*48*(48*128*128*4+3*10240*2))
        self.assertEqual(charge['recurrent_successor_overlap_bytes'],charge['persistent_recurrent_bytes'])
        self.assertEqual(charge['layer_activation_bytes'],140000*(2*5120+3*17408)*2)
        self.assertLess(sum(charge.values()),40<<30)
        self.assertGreater(sum(charge.values()),32<<30)
        # Admission must still add pinned weights/headroom to its separate RSS ceiling.
        self.assertGreater(sum(charge.values())+(16<<30),48<<30)
    def test_charge_refuses_unsupported_policy_before_allocation(self):
        for mutate in (lambda c:setattr(c,'_prefill_packed_n20',False),lambda c:setattr(c,'_prefill_eval_block_size',16),
                       lambda c:setattr(c.trunk.layers[0].linear_attn,'training',True),
                       lambda c:setattr(c.trunk.layers[0].linear_attn,'_gdn_state_dtype','bfloat16'),
                       lambda c:setattr(c.trunk.layers[0].mlp,'materialized',None)):
            c=candidate();mutate(c)
            with self.assertRaises(ValueError):L.n20_stagewise_charge_components(c,(7000,)*20,(192,)*20)
        for counts,caps in (((7000,)*21,(192,)*21),((8001,),(192,)),((255,),(2,)),((7000,),(True,))):
            with self.assertRaises(ValueError):L.n20_stagewise_charge_components(candidate(),counts,caps)
    def test_tensorfold_and_sharded_policies_refused_before_allocation(self):
        for target,attribute,value in (('gdn','_prefill_counts',(7000,)*20),('gdn','sharding_group',object()),('mlp','_prefill_counts',(7000,)*20)):
            c=candidate();module=c.trunk.layers[0].linear_attn if target=='gdn' else c.trunk.layers[0].mlp
            setattr(module,attribute,value)
            with self.assertRaises(ValueError):L.n20_stagewise_charge_components(c,(7000,)*20,(192,)*20)
    def test_materialized_gdn_refuses_tensorfold_before_projections(self):
        selected=method('src/mlx2/runtime/models/qwen3_5.py','GatedDeltaNet','mixed_materialized',{})
        c=NS(sharding_group=None,_prefill_counts=(7000,)*20)
        with self.assertRaisesRegex(ValueError,'projection ownership'):
            selected(c,np.ones((1,4,3)),[],materialize=lambda *v:None)
    def test_materialization_retains_exact_failure_roots_and_deduplicates(self):
        x=np.ones((8,),np.float32);calls=[]
        stage=L.StageMaterialization(NS(eval=lambda *v:calls.append(v)),x.nbytes)
        stage('same',x,x);self.assertEqual(stage.maximum_evaluated_bytes,x.nbytes);self.assertEqual(stage.failure_roots,())
        def fail(*v):raise RuntimeError('evaluation failed')
        stage.mx.eval=fail
        with self.assertRaises(RuntimeError):stage('failure',x)
        self.assertIs(stage.failure_roots[0],x);self.assertEqual(stage.evaluations,1)
        stage.activation_bound=0
        with self.assertRaises(MemoryError):stage('budget',x)
        self.assertIs(stage.failure_roots[0],x)
    def test_mlp_materialization_preserves_full_projection_shapes_and_arithmetic(self):
        shapes=[];stages=[]
        def projection(scale):
            def call(x):shapes.append(x.shape);return x*scale
            return call
        c=NS(gate_proj=projection(2),up_proj=projection(3),down_proj=projection(4))
        swiglu=lambda gate,up:gate/(1+np.exp(-gate))*up
        methodpath='src/mlx2/runtime/models/qwen3_next.py'
        ordinary=method(methodpath,'Qwen3NextMLP','__call__',{
            'swiglu':swiglu,'mx':NS(array=np.ndarray),'current_observer':lambda:None})
        selected=method(methodpath,'Qwen3NextMLP','materialized',{'swiglu':swiglu})
        x=np.arange(30,dtype=np.float32).reshape(1,10,3)/20
        dense=ModuleType('mlx2.runtime.models.varlen_dense_mlp')
        dense.compact_rows=lambda value:None
        with patch.dict(sys.modules,{'mlx2.runtime.models.varlen_dense_mlp':dense}):
            expected=ordinary(c,x)
        actual=selected(c,x,materialize=lambda s,*v:stages.append(s))
        np.testing.assert_array_equal(expected,actual);self.assertEqual(shapes,[(1,10,3)]*6)
        self.assertEqual(stages,['mlp_projections','mlp_product','mlp_output'])
        c._prefill_counts=(5,5)
        with self.assertRaises(ValueError):selected(c,x,materialize=lambda *v:None)
    def test_mlp_single_eval_preserves_arithmetic_and_one_root_evaluation(self):
        def projection(scale):return lambda x:x*scale
        c=NS(gate_proj=projection(2),up_proj=projection(3),down_proj=projection(4),
             _prefill_materialization_mode='single_eval_qmm')
        swiglu=lambda gate,up:gate/(1+np.exp(-gate))*up
        selected=method('src/mlx2/runtime/models/qwen3_next.py','Qwen3NextMLP','materialized',
            {'swiglu':swiglu,'_prefill_bf16_projection':lambda module,value:module(value)})
        class Materialize:
            def __init__(self):self.stages=[];self.pending=[]
            def retain_pending_roots(self,*values):self.pending.append(len(values))
            def __call__(self,stage,*values):self.stages.append((stage,len(values)))
        x=np.arange(30,dtype=np.float32).reshape(1,10,3)/20;m=Materialize()
        actual=selected(c,x,materialize=m)
        expected=projection(4)(swiglu(projection(2)(x),projection(3)(x)))
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual(m.stages,[('mlp_graph',1)])
        self.assertEqual(m.pending,[1,3,4])
    def test_tiled_swiglu_uses_inferred_geometry_and_two_stage_roots(self):
        def projection(scale):return lambda x:x*scale
        geometry={'sha256':'a'*64}
        c=NS(gate_proj=projection(2),up_proj=projection(3),down_proj=projection(4),
             _prefill_materialization_mode='tiled_q4_swiglu',
             _prefill_mlp_semantics={'kind':'swiglu'},_prefill_mlp_geometry=geometry)
        tiled=ModuleType('mlx2.runtime.models.tensorfold_prefill')
        tiled.project_swiglu=lambda x,gate,up:gate(x)*up(x)
        inferred=ModuleType('mlx2.runtime.dense_mlp_geometry')
        inferred.infer_dense_glu_geometry=lambda layer,semantics:geometry
        selected=method('src/mlx2/runtime/models/qwen3_next.py','Qwen3NextMLP','materialized',
            {'swiglu':lambda gate,up:gate*up,'__package__':'mlx2.runtime.models'})
        stages=[];x=np.arange(30,dtype=np.float32).reshape(1,10,3)/20
        with patch.dict(sys.modules,{
            'mlx2.runtime.models.tensorfold_prefill':tiled,
            'mlx2.runtime.dense_mlp_geometry':inferred}):
            actual=selected(c,x,materialize=lambda stage,*values:stages.append(stage))
        np.testing.assert_allclose(actual,x*x*24,rtol=1e-6,atol=0)
        self.assertEqual(stages,['mlp_tiled_gate_up_swiglu','mlp_output'])
        c._prefill_mlp_geometry={'sha256':'b'*64}
        with patch.dict(sys.modules,{
            'mlx2.runtime.models.tensorfold_prefill':tiled,
            'mlx2.runtime.dense_mlp_geometry':inferred}),self.assertRaises(ValueError):
            selected(c,x,materialize=lambda *values:None)
    def test_packed_gate_up_uses_one_stock_projection_and_two_stage_roots(self):
        def projection(scale):return lambda x:x*scale
        geometry={'sha256':'a'*64}
        class Group:
            def matches(self,modules):return modules==(c.gate_proj,c.up_proj)
            def __call__(self,x,modules):return tuple(module(x) for module in modules)
        c=NS(gate_proj=projection(2),up_proj=projection(3),down_proj=projection(4),
             _prefill_materialization_mode='packed_gate_up_qmm',
             _prefill_mlp_semantics={'kind':'swiglu'},_prefill_mlp_geometry=geometry,
             _prefill_mlp_group=Group())
        packed=ModuleType('mlx2.runtime.models.tensorfold_prefill')
        packed.PackedProjectionGroup=Group
        inferred=ModuleType('mlx2.runtime.dense_mlp_geometry')
        inferred.infer_dense_glu_geometry=lambda layer,semantics:geometry
        swiglu=lambda gate,up:gate/(1+np.exp(-gate))*up
        selected=method('src/mlx2/runtime/models/qwen3_next.py','Qwen3NextMLP','materialized',
            {'swiglu':swiglu,'__package__':'mlx2.runtime.models'})
        stages=[];x=np.arange(30,dtype=np.float32).reshape(1,10,3)/20
        with patch.dict(sys.modules,{
            'mlx2.runtime.models.tensorfold_prefill':packed,
            'mlx2.runtime.dense_mlp_geometry':inferred}):
            actual=selected(c,x,materialize=lambda stage,*values:stages.append(stage))
        expected=projection(4)(swiglu(projection(2)(x),projection(3)(x)))
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(stages,['mlp_packed_gate_up_swiglu','mlp_output'])
    def test_bf16_mlp_modes_charge_three_dequantized_projection_tables(self):
        for mode in ('staged_bf16','single_eval_bf16'):
            c=candidate();c._prefill_mlp_materialization_mode=mode
            for layer in c.trunk.layers:layer.mlp._prefill_materialization_mode=mode
            charge=L.n20_stagewise_charge_components(c,(7000,)*3,(4,)*3)
            self.assertEqual(charge['mlp_dequantized_projection_bytes'],3*5120*17408*2)
        c=candidate();c._prefill_mlp_materialization_mode='invented'
        with self.assertRaises(ValueError):L.n20_stagewise_charge_components(c,(7000,)*3,(4,)*3)
    def test_geometry_bound_swiglu_modes_have_no_extra_weight_table(self):
        for mode in ('tiled_q4_swiglu','packed_gate_up_qmm'):
            c=candidate();c._prefill_mlp_materialization_mode=mode
            geometry={'sha256':'a'*64}
            c._prefill_mlp_geometry={'layer_count':64,'geometry':geometry}
            for layer in c.trunk.layers:
                layer.mlp._prefill_materialization_mode=mode
                layer.mlp._prefill_mlp_geometry=geometry
            charge=L.n20_stagewise_charge_components(c,(7000,)*3,(4,)*3)
            self.assertNotIn('mlp_dequantized_projection_bytes',charge)
            del c.trunk.layers[0].mlp._prefill_mlp_geometry
            with self.assertRaises(ValueError):
                L.n20_stagewise_charge_components(c,(7000,)*3,(4,)*3)
    def test_expanded_gdn_wave_is_explicitly_charged_for_long_n3(self):
        c=candidate();c._prefill_gdn_eval_wave_max_segments=4
        with self.assertRaises(MemoryError):
            L.n20_stagewise_charge_components(c,(6982,6961,6940),(4,4,4))
        c._prefill_gdn_eval_wave_expanded_charge=True
        for index in c.layer_map.recurrent:
            module=c.trunk.layers[index].linear_attn
            module.conv1d=NS(groups=10240,stride=1,dilation=1,padding=0,
                weight=NS(shape=(10240,4,1),dtype='bfloat16'))
            module.norm=NS(weight=NS(dtype='bfloat16'))
        guard=NS(assert_profile_applied=lambda *args,**kwargs:None)
        env={'MLX_GDN_PACKED':'1','MLX_GDN_CORE':'0','MLX2_PAGED_GDN_EVAL_WAVE_MAX_SEGMENTS':'4'}
        with patch.dict(os.environ,env,clear=True),patch.dict(sys.modules,{'mlx2.runtime.models.import_env':guard}):
            charge=L.n20_stagewise_charge_components(c,(6982,6961,6940),(4,4,4))
        self.assertGreater(charge['gdn_grouped_wave_extra_bytes'],0)
        bound=charge['layer_activation_bytes']+charge['gdn_grouped_wave_extra_bytes']
        plan=L.gdn_evaluation_wave_plan((6982,6961,6940),bound,4)
        self.assertEqual(plan['groups'],((0,1,2),))
    def test_gdn_materialization_preserves_segment_core_inputs_and_cache_outputs(self):
        path='src/mlx2/runtime/models/qwen3_5.py';ns={
            'mx':NS(concatenate=np.concatenate),'current_observer':lambda:None}
        ordinary=method(path,'GatedDeltaNet','mixed',dict(ns));selected=method(path,'GatedDeltaNet','mixed_materialized',dict(ns))
        def run(fn):
            calls=[];cache=(NS(cache=[None,None],speculating=False),NS(cache=[None,None],speculating=False))
            def core(q,z,b,a,mask,c,*,dtype):
                calls.append((q.copy(),z.copy(),b.copy(),a.copy(),mask,dtype));c.cache=[q[:,-1:].copy(),q.sum(1)];return q+z+b+a
            c=NS(sharding_group=None,_input_projections=lambda x:(x*2,x*3,x*4,x*5),_recurrent_core=core,out_proj=lambda x:x*6)
            x=np.arange(30,dtype=np.float32).reshape(1,10,3)
            parts=[(1,4,0,cache[0],None),(1,6,4,cache[1],None)]
            kwargs={'materialize':lambda *v:None} if fn is selected else {}
            return fn(c,x,parts,**kwargs),cache,calls
        a,ca,qa=run(ordinary);b,cb,qb=run(selected);np.testing.assert_array_equal(a,b)
        for left,right in zip(ca,cb):
            for x,y in zip(left.cache,right.cache):np.testing.assert_array_equal(x,y)
        for left,right in zip(qa,qb):
            for x,y in zip(left[:4],right[:4]):np.testing.assert_array_equal(x,y)
            self.assertEqual(left[4:],right[4:])
    def test_phase_boundary_checks_real_terminal_state_and_never_publishes(self):
        path='src/mlx2/runtime/qwen35_paged_graph_factory.py'
        phase=method(path,'HybridReservation','evaluated_phase',{})
        reservation=NS(completed=False,session=NS(closed=False),phase_boundaries=[])
        writer=NS(pending_epochs=(),ledger=NS(pending_count=0))
        event={'materialized':True,'native_terminals_drained':True,'public_state_published':False}
        seen=[];phase(reservation,event,writer=writer,orphaned_reads={},callback=seen.append)
        self.assertEqual(seen,[event]);self.assertIsNot(seen[0],event)
        for field,value in (('pending_epochs',(1,)),):
            setattr(writer,field,value)
            with self.assertRaises(RuntimeError):phase(reservation,event,writer=writer,orphaned_reads={})
            setattr(writer,field,())
        with self.assertRaises(RuntimeError):phase(reservation,dict(event,public_state_published=True),writer=writer,orphaned_reads={})
        with self.assertRaises(RuntimeError):phase(reservation,event,writer=writer,orphaned_reads={1:object()})
        def fail(v):raise ValueError('quantum cancelled')
        with self.assertRaises(ValueError):phase(reservation,event,writer=writer,orphaned_reads={},callback=fail)
        self.assertFalse(reservation.completed)

if __name__=='__main__':unittest.main()
