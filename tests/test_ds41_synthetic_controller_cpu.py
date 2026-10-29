"""CPU-only controller contracts, no MLX import or model inference."""
import ast,importlib.util,os
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('synthetic_decode_controller_test',ROOT/'scripts/research/probe_ds41_synthetic_decode.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def spec(out):return dict(schema=m.SCHEMA,out=str(out),job_id=out.name,suites=list(m.MODULES),checkpoint_loaded=False,memory_limit_bytes=m.base.MEMORY_LIMIT,host_live_reserve_bytes=m.base.HOST_RESERVE,host_initial_floor_bytes=m.base.HOST_INITIAL,rate_eligible=False,ownership_poll_seconds=1,numerical_environment={'MLX_ENABLE_TF32':'0'})
def test_sealed_function_only_spec(tmp_path,monkeypatch):
 monkeypatch.setenv('MLX_ENABLE_TF32','0');m.validate(spec(tmp_path),tmp_path)
@pytest.mark.parametrize('key,value',[('checkpoint_loaded',True),('memory_limit_bytes',99<<30),('host_live_reserve_bytes',0),('rate_eligible',True),('suites',['unknown'])])
def test_wrong_shape_or_guards_refused(tmp_path,monkeypatch,key,value):
 monkeypatch.setenv('MLX_ENABLE_TF32','0');v=spec(tmp_path);v[key]=value
 with pytest.raises(ValueError):m.validate(v,tmp_path)
def test_no_checkpoint_or_server_loading_calls():
 t=ast.parse((ROOT/'scripts/research/probe_ds41_synthetic_decode.py').read_text());calls=[ast.unparse(n.func)for n in ast.walk(t)if isinstance(n,ast.Call)]
 assert not set(calls)&{'load','load_model','inspect_artifact','start_server','Tokenizer.from_file'}
