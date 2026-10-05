#!/usr/bin/env python3
"""CPU-only, isolated-environment HF tokenizer intake. Never imports mlx2/MLX.

build-inputs uses the existing Transformers stack and repairs its file contract.
run compares exact file-tokenizer IDs/decode and reports cold/warm timings.
worker is an encode-only JSONL process launched by Popen (never fork).
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, os, pathlib, statistics, subprocess, sys, time, platform
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def build(args):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    corpus_path = ROOT / 'qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json'
    corpus = json.loads(corpus_path.read_text())
    source = ROOT / 'scripts/run_spomin_20x20.py'
    tree = ast.parse(source.read_text())
    names = {'needle_values','sentinel','filler_text','needle_text','token_ids','prepare_case'}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'hashlib':hashlib, 'Any':object, 'NEEDLE_TARGETS':{'early':.08,'middle':.45,'late':.72},
          'digest':lambda v:sha(json.dumps(v,sort_keys=True).encode())}
    exec(compile(ast.Module(body=defs,type_ignores=[]),str(source),'exec'), ns)
    repair = load_file('integrity_cpu', ROOT/'src/mlx2/runtime/tokenizer_integrity.py')
    models=[]
    for model_path in args.models:
        path=pathlib.Path(model_path)
        tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        receipt=repair.repair_loaded_tokenizer(tok,path)
        template_file=path/'chat_template.jinja'
        if tok.chat_template is None and template_file.is_file():
            tok.chat_template=template_file.read_text()
        raw=Tokenizer.from_file(str(path/'tokenizer.json'))
        texts=['', 'Hello world\n1234567890', 'é e\u0301 Ελληνικά हिन्दी ไทย العربية بِسْمِ',
               '👩🏽‍💻 🤖 \u200d \x00\t\r\n', '<|im_start|>user\nHi<|im_end|>', '中文 日本語 한국어']
        # Every special-token string declared by the pinned file is tested independently.
        data=json.loads((path/'tokenizer.json').read_text())
        texts += [r['content'] for r in data.get('added_tokens',[]) if r.get('special')]
        chats=[]
        for thinking in ((False,True) if tok.chat_template is not None else ()):
            messages=[{'role':'system','content':'You are a precise assistant.'},
                      {'role':'user','content':texts[2]},{'role':'assistant','content':'Acknowledged.'},
                      {'role':'user','content':'Return 123 and 🤖.'}]
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
            ids=tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=thinking)
            chats.append({'text':text,'ids':ids})
        cases=[]
        for text in texts+[c['text'] for c in chats]:
            for add in (False,True):
                ids=raw.encode(text,add_special_tokens=add).ids
                cases.append({'text':text,'add':add,'ids':ids,
                              'decode':raw.decode(ids,skip_special_tokens=False),
                              'prefix_decode':[raw.decode(ids[:n],skip_special_tokens=False) for n in range(len(ids)+1)]})
        models.append({'path':str(path),'tokenizer_sha256':sha((path/'tokenizer.json').read_bytes()),
                       'config_sha256':sha((path/'tokenizer_config.json').read_bytes()),
                       'chat_template_sha256':sha(tok.chat_template.encode()) if isinstance(tok.chat_template,str) else None,
                       'repair':receipt,'cases':cases,'chats':chats})
        if len(models)==1:
            docs=[]
            for case in corpus['cases']:
                prepared=ns['prepare_case'](case,tok,args.capacity)
                messages=[{'role':'system','content':corpus['system']}]+prepared['full_messages']
                text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(text,add_special_tokens=False)
                assert ids==raw.encode(text,add_special_tokens=False).ids
                docs.append({'case_id':case['case_id'],'text':text,'ids_sha256':sha(json.dumps(ids).encode()),'tokens':len(ids)})
    payload={'schema':'mlx2.tokenizers-v1.cpu-probe.v1','corpus_sha256':sha(corpus_path.read_bytes()),
             'harness_sha256':sha(source.read_bytes()),'capacity_tokens':args.capacity,'models':models,'docs':docs}
    pathlib.Path(args.output).write_text(json.dumps(payload,ensure_ascii=False))
    print(json.dumps({'documents':len(docs),'tokens':sum(d['tokens'] for d in docs),'models':len(models)}))

def worker(args):
    from tokenizers import Tokenizer
    t=Tokenizer.from_file(args.model+'/tokenizer.json')
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        r=json.loads(line)
        if r.get('stop'): break
        ids=t.encode(r['text'],add_special_tokens=False).ids
        print(json.dumps({'ids':ids}),flush=True)

def run(args):
    import tokenizers
    from tokenizers import Tokenizer
    data=json.loads(pathlib.Path(args.inputs).read_text()); result={'version':tokenizers.__version__,'models':[],
        'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
        'gpu_used':False,'mlx_imported':False,'serving_qualified':False,'production_selected':False}
    for model in data['models']:
        t=Tokenizer.from_file(model['path']+'/tokenizer.json'); errors=[]; passed=0
        for i,c in enumerate(model['cases']):
            try:
                ids=t.encode(c['text'],add_special_tokens=c['add']).ids
                assert ids==c['ids'], 'ids mismatch'
                assert t.decode(ids,skip_special_tokens=False)==c['decode'], 'decode mismatch'
                assert [t.decode(ids[:n],skip_special_tokens=False) for n in range(len(ids)+1)]==c['prefix_decode'], 'stream prefix mismatch'
                passed+=1
            except Exception as e: errors.append({'case':i,'error':str(e)})
        chats=[t.encode(c['text'],add_special_tokens=False).ids==c['ids'] for c in model['chats']] if not errors else []
        result['models'].append({'path':model['path'],'passed':passed,'errors':errors,'chat_template_ids_equal':chats,
                                'missing_python_apis':[n for n in ('from_str','to_str','pre_tokenizer','normalizer','get_vocab','token_to_id','decode_stream') if not hasattr(t,n)]})
    if not result['models'][0]['errors']:
        model=data['models'][0]['path']; docs=data['docs']; samples={}
        for arm in ('cold_new_documents','warm_repeated_documents'):
            times=[]
            for _ in range(args.repeats):
                t=Tokenizer.from_file(model+'/tokenizer.json')
                if arm.startswith('warm'):
                    for d in docs: t.encode(d['text'],add_special_tokens=False).ids
                start=time.perf_counter(); n=0
                for d in docs:
                    ids=t.encode(d['text'],add_special_tokens=False).ids
                    n+=len(ids)
                times.append(time.perf_counter()-start)
            samples[arm]={'seconds':times,'median_seconds':statistics.median(times),'tokens':n,
                          'tokens_per_second':n/statistics.median(times)}
        # Untimed exact corpus IDs check; comparison is against repaired Transformers rendered inputs.
        t=Tokenizer.from_file(model+'/tokenizer.json')
        mismatch=[d['case_id'] for d in docs if sha(json.dumps(t.encode(d['text'],add_special_tokens=False).ids).encode())!=d['ids_sha256']]
        result.update(benchmark=samples,corpus_exact_id_mismatches=mismatch,documents=len(docs))
        if args.worker_python:
            command=[args.worker_python,str(pathlib.Path(__file__).resolve()),'worker','--model',model]
            start=time.perf_counter(); p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            assert json.loads(p.stdout.readline())['ready']; startup=time.perf_counter()-start
            try:
                start=time.perf_counter(); wrong=[]
                for d in docs:
                    p.stdin.write(json.dumps({'text':d['text']})+'\n'); p.stdin.flush()
                    ids=json.loads(p.stdout.readline())['ids']
                    if sha(json.dumps(ids).encode())!=d['ids_sha256']: wrong.append(d['case_id'])
                elapsed=time.perf_counter()-start
                result['spawned_encode_worker']={'startup_seconds':startup,'encode_jsonl_seconds':elapsed,'exact_id_mismatches':wrong,
                                                'tokens_per_second':sum(d['tokens'] for d in docs)/elapsed,'command':command}
            finally:
                p.stdin.write('{"stop":true}\n');p.stdin.flush();p.wait(timeout=20)
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules), 'MLX import forbidden'
    result['inputs_sha256']=sha(pathlib.Path(args.inputs).read_bytes())
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build-inputs');b.add_argument('--models',nargs='+',required=True);b.add_argument('--capacity',type=int,default=8192);b.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--inputs',required=True);r.add_argument('--output',required=True);r.add_argument('--repeats',type=int,default=3);r.add_argument('--worker-python')
    w=sub.add_parser('worker');w.add_argument('--model',required=True)
    a=p.parse_args(); {'build-inputs':build,'run':run,'worker':worker}[a.mode](a)
if __name__=='__main__': main()
