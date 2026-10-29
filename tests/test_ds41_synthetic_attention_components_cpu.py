"""CPU-only geometry and timing-contract checks; no MLX/model construction."""
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).parents[1]/'scripts/research/ds41_synthetic_attention_components.py'
SPEC=importlib.util.spec_from_file_location('synthetic_attention_cpu',PATH)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

@pytest.mark.parametrize('length',M.CONTEXTS)
@pytest.mark.parametrize('ratio',M.RATIOS)
def test_actual_geometry_bounded(length,ratio):
    g=M.geometry(length,ratio)
    assert g['query_shape']==(1,64,1,512)
    assert g['local_shape']==(1,1,min(128,length),512)
    assert g['pool_tokens']==length//ratio
    assert g['selected_tokens']==min(512,length//ratio)
    assert g['logical_input_bytes']<2*1024**3

@pytest.mark.parametrize('args',[(True,1),(100,True),(99,1),(100,4)])
def test_reject_unbounded_geometry(args):
    with pytest.raises(ValueError):M.geometry(*args)

def test_interleaved_warm_reset_and_completion_clock():
    events=[]
    class MX:
        @staticmethod
        def eval(value):events.append(('eval',value))
        @staticmethod
        def synchronize():events.append(('fence',))
    tick=iter(range(1000))
    def operation(name):
        def op():events.append(('call',name));return name
        return op
    summary=M._timed(MX,{'a':operation('a'),'b':operation('b')},
        lambda:events.append(('live',)),clock=lambda:next(tick),reset=lambda:events.append(('reset',)))
    assert [v[1]for v in events if v[0]=='call'][:6]==['a','b','a','b','b','a']
    assert sum(v[0]=='call'for v in events)==16 # 2 warm +14 measured
    assert sum(v[0]=='reset'for v in events)==16
    assert sum(v[0]=='fence'for v in events)==14
    assert sum(v[0]=='live'for v in events)==14
    for row in summary.values():
        assert len(row['samples'])==7
        assert row['median_submit_seconds']==1
        assert row['median_completion_seconds']==1
        assert row['median_total_seconds']==2

def test_import_does_not_load_mlx_or_construct_model():
    import ast
    tree=ast.parse(PATH.read_text())
    topimports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('mlx'not in ast.unparse(node)for node in topimports)
    text=PATH.read_text()
    assert 'DeepSeekV41Model('not in text
    assert 'load_artifact('not in text
    assert 'state.update_local(new_token)'in text
    assert 'return value,state.keys'in text
