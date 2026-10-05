import importlib.util,json,pathlib,time,statistics,sys
from tokenizers import Tokenizer
root=pathlib.Path(__file__).resolve().parents[1]
manifest,inputs,output=sys.argv[1:]
model=pathlib.Path(json.load(open(manifest))["tokenizer"]["path"]).parent
spec=importlib.util.spec_from_file_location('worker',root/'src/mlx2/runtime/tokenizers_v1_worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
worker=m.TokenizersV1Worker(manifest,model_path=model)
base=Tokenizer.from_file(str(model/"tokenizer.json"))
text=json.load(open(inputs))['docs'][0]['text']
worker.encode('warmup',add_special_tokens=False)
rows=[]
for length in (32,128,512,1024,2048,4096,8192,16384,32768):
 docs=[(text*((length//len(text))+1))[:length]+f'\nnew-tail-{i:04d}' for i in range(200)]
 references=[base.encode(d,add_special_tokens=False).ids for d in docs]
 times=[]
 for mode in ('ordinary','worker'):
  samples=[]
  for _ in range(3):
   start=time.perf_counter()
   result=[base.encode(d,add_special_tokens=False).ids if mode=='ordinary' else worker.encode(d,add_special_tokens=False) for d in docs]
   samples.append(time.perf_counter()-start);assert result==references
  times.append(samples)
 rows.append({'prefix_chars':length,'documents':200,'baseline_seconds':times[0],'worker_seconds':times[1],'median_speedup':statistics.median(times[0])/statistics.median(times[1])})
result={'rows':rows,'status':worker.status(),'scope':'UTF-8 JSON IPC plus worker encode IDs materialization/validation; warm shared prefixes with new tails','gpu_used':False}
worker.close()
pathlib.Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
