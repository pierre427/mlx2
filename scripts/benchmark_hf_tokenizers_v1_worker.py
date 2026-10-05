#!/usr/bin/env python3
"""CPU-only complete chat-template/encode comparison; isolated child executable."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import types
import sys
import time

os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

def main():
    worker_manifest, output = sys.argv[1:]
    model = Path(json.loads(Path(worker_manifest).read_text())["tokenizer"]["path"]).parent
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = (model / 'chat_template.jinja').read_text()
    spec = importlib.util.spec_from_file_location('integrity_cpu', ROOT/'src/mlx2/runtime/tokenizer_integrity.py')
    integrity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integrity)
    repair = integrity.repair_loaded_tokenizer(tokenizer, model)
    source = ROOT/'scripts/run_spomin_20x20.py'
    names = {'needle_values','sentinel','filler_text','needle_text','token_ids','prepare_case'}
    defs = [n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace = {'hashlib':hashlib, 'Any':object, 'NEEDLE_TARGETS':{'early':.08,'middle':.45,'late':.72},
                 'digest':lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()}
    exec(compile(ast.Module(body=defs,type_ignores=[]),str(source),'exec'), namespace)
    corpus = json.loads((ROOT/'qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json').read_text())
    messages = [[{'role':'system','content':corpus['system']}]+namespace['prepare_case'](c,tokenizer,8192)['full_messages'] for c in corpus['cases']]
    def render(m):
        return tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    references = [tokenizer.encode(render(m),add_special_tokens=False) for m in messages]
    result = {'schema':'mlx2.tokenizers-v1.template-worker.cpu.v1','documents':len(messages),
              'tokens':sum(map(len,references)), 'repair':repair, 'gpu_used':False, 'mlx_imported':False,
              'scope':'current_transformers_chat_template_render_then_encode_and_materialize_ids',
              'serving_qualified':False, 'production_selected':False}
    times = []
    for _ in range(3):
        start=time.perf_counter()
        outputs=[tokenizer.encode(render(m),add_special_tokens=False) for m in messages]
        times.append(time.perf_counter()-start)
        assert outputs==references
    result['baseline_seconds']=times
    package = types.ModuleType('_mlx2_cpu_runtime')
    package.__path__ = [str(ROOT/'src/mlx2/runtime')]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__+'.tokenizer_utils', ROOT/'src/mlx2/runtime/tokenizer_utils.py')
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    os.environ.pop('MLX2_TOKENIZERS_V1_MANIFEST', None)
    wrapper = utils.TokenizerWrapper(tokenizer, detokenizer_class=utils.BPEStreamingDetokenizer)
    start = time.perf_counter()
    wrapper.enable_v1_encode(worker_manifest)
    result['worker_startup_seconds'] = time.perf_counter()-start
    try:
        times=[]
        for _ in range(3):
            start=time.perf_counter()
            outputs=[wrapper.apply_chat_template(m,add_generation_prompt=True,enable_thinking=False) for m in messages]
            times.append(time.perf_counter()-start)
            assert outputs==references
        result['worker_seconds']=times
        result['exact_ids_equal_all_1200_encodes']=True
        result['worker_status']=wrapper.tokenizer_v1_status()
    finally:
        wrapper.close()
    result['median_speedup']=statistics.median(result['baseline_seconds'])/statistics.median(result['worker_seconds'])
    result['first_use_speedup_including_startup']=statistics.median(result['baseline_seconds'])/(statistics.median(result['worker_seconds'])+result['worker_startup_seconds'])
    result['tokenizer_sha256']=hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest()
    result['chat_template_sha256']=hashlib.sha256((model/'chat_template.jinja').read_bytes()).hexdigest()
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
    Path(output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
