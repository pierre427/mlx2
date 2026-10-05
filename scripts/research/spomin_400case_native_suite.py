"""Actual frozen20domains×20cases input preparation and bounded domain HTTP gates.

Client concurrency20, original output cap192. No synthetic/two-prompt replacement.
Preparation is CPU-only. Execution requires root GPUQ ownership and a separately
capable ordinary/N20 service; a missing native route fails closed, not to B2.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,threading,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from run_spomin_20x20 import prepare_case,make_body,FROZEN_CORPUS_SHA256,CORPUS_SCHEMA
CORPUS=ROOT/'qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json'
SCHEMA='mlx2.spomin-400case-native-inputs.v1'
MAX_TOKENS=192;CONCURRENCY=20;CAPACITY=8192


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def corpus_rows(path=CORPUS):
    if file_sha(path)!=FROZEN_CORPUS_SHA256:raise ValueError('frozen400case corpus hash differs')
    data=json.loads(Path(path).read_text());domains=data.get('domain_order');rows=data.get('cases')
    if data.get('schema')!=CORPUS_SCHEMA or type(domains) is not list or len(domains)!=20 or len(set(domains))!=20 or type(rows) is not list or len(rows)!=400:
        raise ValueError('twenty frozen domains/fourhundred cases required')
    ids=[row.get('case_id') for row in rows]
    if len(set(ids))!=400 or any(not isinstance(i,str) or not i for i in ids) or Counter(row.get('domain') for row in rows)!=Counter({name:20 for name in domains}):
        raise ValueError('400unique cases and20cases per actual domain required')
    return data


def prepare_inputs(tokenizer,tokenizer_root,*,corpus_path=CORPUS,transcript_arm='full',nonce='spomin-nativeN-20261004',prepare=prepare_case):
    if transcript_arm not in ('full','compacted') or not isinstance(nonce,str) or not nonce:raise ValueError('explicit transcript and frozen nonce required')
    corpus=corpus_rows(corpus_path);root=Path(tokenizer_root).resolve()
    source=ROOT/'scripts/run_spomin_20x20.py'
    system=corpus['system']+' Reproduce all three audit tokens exactly and do not omit the final code line. Campaign nonce: '+nonce
    rows=[]
    for case in corpus['cases']:
        prepared=prepare(case,tokenizer,CAPACITY)
        if prepared['receipt']['shortfall_tokens']!=0:raise ValueError('original transcript preparation shortfall')
        body=make_body(system,prepared,transcript_arm,MAX_TOKENS)
        ids=tuple(tokenizer.apply_chat_template(body['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False))
        if not ids or any(type(t) is not int or t<0 for t in ids) or len(ids)+MAX_TOKENS>CAPACITY:
            raise ValueError('actual prepared chat input does not fit original8192 context: '+case['case_id'])
        rows.append({'case_id':case['case_id'],'domain':case['domain'],'body':body,'body_sha256':digest(body),
            'prompt_token_ids':list(ids),'prompt_tokens':len(ids),'preparation_receipt':prepared['receipt'],
            'expected_concepts':case['expected_concepts'],'sentinel':prepared['sentinel'],'needles':prepared['needles']})
    files={name:file_sha(root/name) for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json') if (root/name).is_file()}
    if not {'tokenizer.json','tokenizer_config.json'}<=set(files):raise ValueError('tokenizer byte binding incomplete')
    result={'schema':SCHEMA,'corpus_sha256':FROZEN_CORPUS_SHA256,'preparation_source_sha256':file_sha(source),
        'domain_order':corpus['domain_order'],'case_count':400,'cases_per_domain':20,'client_concurrency':20,
        'generation_max_tokens':192,'capacity_tokens':8192,'transcript_arm':transcript_arm,'nonce':nonce,
        'tokenizer_root':str(root),'tokenizer_files_sha256':files,'rows':rows,'qualified':False,'price_usable':False,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    result['inputs_sha256']=digest({k:v for k,v in result.items() if k!='inputs_sha256'})
    return result


def validate_inputs(data):
    if type(data) is not dict or data.get('schema')!=SCHEMA or data.get('corpus_sha256')!=FROZEN_CORPUS_SHA256 or data.get('case_count')!=400 or data.get('cases_per_domain')!=20 or data.get('client_concurrency')!=20 or data.get('generation_max_tokens')!=192 or data.get('capacity_tokens')!=8192:
        raise ValueError('actual suite scope differs')
    if digest({k:v for k,v in data.items() if k!='inputs_sha256'})!=data.get('inputs_sha256'):raise ValueError('prepared suite identity differs')
    corpus=corpus_rows();expected=[(r['case_id'],r['domain']) for r in corpus['cases']]
    if [(r['case_id'],r['domain']) for r in data['rows']]!=expected or data['domain_order']!=corpus['domain_order']:raise ValueError('frozen case/domain order differs')
    for row in data['rows']:
        ids=row['prompt_token_ids'];body=row['body']
        if digest(body)!=row['body_sha256'] or row['prompt_tokens']!=len(ids) or not 256<=len(ids)<=8192-192 or body.get('max_tokens')!=192 or body.get('temperature')!=0 or body.get('enable_thinking') is not False:
            raise ValueError('prepared original body/token/context differs')
    return data


def domain_rows(inputs,index):
    validate_inputs(inputs)
    if type(index) is not int or not 0<=index<20:raise ValueError('domain index0..19 required')
    domain=inputs['domain_order'][index];rows=[r for r in inputs['rows'] if r['domain']==domain]
    if len(rows)!=20:raise ValueError('actual domain must contain20distinct cases')
    return rows


def request_body(row,*,model,native,cohort_id,inputs_sha256=None):
    body={**row['body'],'model':model,'skip_writing_prefix_cache':True}
    if native:
        if not isinstance(inputs_sha256,str) or len(inputs_sha256)!=64:raise ValueError('actual input manifest hash required')
        body['batch_cohort']={'id':cohort_id,'size':20}
        body.update(paged_native_packed_n20_research=True,native_research_input_id=row['case_id'],native_research_inputs_sha256=inputs_sha256)
    return body


def run_domain_http(inputs,index,base,model,*,native=False,timeout=100,post=None):
    from varlen_pack_price_bench import _gpuq_owner
    before=_gpuq_owner();rows=domain_rows(inputs,index);barrier=threading.Barrier(21);starts={};lock=threading.Lock()
    cohort='spomin400-'+str(index)+('-native' if native else '-ordinary')
    def default_post(body):
        request=Request(base.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=timeout) as response:return response.status,json.load(response)
    post=post or default_post
    def one(row):
        barrier.wait(timeout=30);tick=time.monotonic()
        with lock:starts[row['case_id']]=tick
        status,body=post(request_body(row,model=model,native=native,cohort_id=cohort,inputs_sha256=inputs['inputs_sha256']))
        ended=time.monotonic()
        if status!=200:raise RuntimeError('HTTP suite failed status'+str(status))
        usage=body.get('usage',{});choices=body.get('choices',[]);details=body.get('mlx2',{});receipt=details.get('route_receipt',{})
        actual=usage.get('completion_tokens')
        if type(actual) is not int or not 1<=actual<=192 or len(choices)!=1 or choices[0].get('finish_reason') not in ('length','stop') or (choices[0]['finish_reason']=='length' and actual!=192):
            raise RuntimeError('actual original max192 finish/usage differs')
        if usage.get('prompt_tokens')!=row['prompt_tokens'] or usage.get('prompt_tokens_details',{}).get('cached_tokens',0)!=0:
            raise RuntimeError('actual frozen prepared chat/cold input differs')
        if native:
            if receipt.get('route')!='native_hybrid_packed_n20_research' or receipt.get('qualified') is not False or receipt.get('price_usable') is not False or receipt.get('observed_used') is not True or receipt.get('prefill_cohort_width')!=20:
                raise RuntimeError('genuine nativeN20 route was not observed; no B2 substitute')
            ids=receipt.get('output_token_ids')
            if not isinstance(ids,list) or len(ids)!=actual:raise RuntimeError('actual native sampled IDs missing')
        elif receipt.get('route')=='native_hybrid_packed_n20_research':raise RuntimeError('ordinary control selected native')
        return {'case_id':row['case_id'],'domain':row['domain'],'body_sha256':row['body_sha256'],'start_monotonic':tick,'end_monotonic':ended,
            'http_seconds':ended-tick,'actual_completion_tokens':actual,'usage':usage,'finish_reason':choices[0]['finish_reason'],
            'text':choices[0].get('message',{}).get('content',''),'route_receipt':receipt,'mlx2':details}
    pool=ThreadPoolExecutor(max_workers=20)
    try:
        futures=[pool.submit(one,row) for row in rows];barrier.wait(timeout=30)
        output=[future.result(timeout=timeout) for future in as_completed(futures,timeout=timeout)]
    finally:pool.shutdown(wait=False,cancel_futures=True)
    if _gpuq_owner()!=before:raise RuntimeError('suite command lease changed')
    began=min(r['start_monotonic'] for r in output);ended=max(r['end_monotonic'] for r in output)
    return {'domain_index':index,'domain':rows[0]['domain'],'native':native,'gpuq_owner':before,'client_concurrency':20,
        'cohort_width_requested':20,'compute_width':'see actual route/sample receipts; client barrier is not kernel proof',
        'rows':sorted(output,key=lambda r:r['case_id']),'domain_http_wall_seconds':ended-began,
        'actual_completion_tokens':sum(r['actual_completion_tokens'] for r in output),'inputs_sha256':inputs['inputs_sha256'],
        'qualified':False,'price_usable':False,'numeric_tensor_parity':'not_tested'}


def compare_domain(native,ordinary):
    if native['inputs_sha256']!=ordinary['inputs_sha256'] or native['domain_index']!=ordinary['domain_index']:raise ValueError('domain control inputs differ')
    left={r['case_id']:r for r in native['rows']};right={r['case_id']:r for r in ordinary['rows']}
    if left.keys()!=right.keys() or len(left)!=20:raise ValueError('twenty matched actual cases required')
    drift=[case for case in left if any(left[case][key]!=right[case][key] for key in ('body_sha256','actual_completion_tokens','finish_reason','text'))]
    if drift:raise RuntimeError('sameinput domain exact HTTP control differs: '+','.join(drift))
    return {'domain_index':native['domain_index'],'actual_cases':20,'exact_text_usage_finish':True,
        'sampled_id_parity':'must be checked by instrumented serving worker; ordinaryHTTPtext alone is insufficient',
        'paired_http_ratio':ordinary['domain_http_wall_seconds']/native['domain_http_wall_seconds']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--tokenizer-root',type=Path)
    parser.add_argument('--transcript-arm',choices=('full','compacted'),default='full');parser.add_argument('--nonce',default='spomin-nativeN-20261004')
    parser.add_argument('--inputs',type=Path);parser.add_argument('--execute-domain',type=int);parser.add_argument('--base-url');parser.add_argument('--model');parser.add_argument('--native',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        if not args.tokenizer_root:parser.error('local tokenizer root required')
        # Tokenizer-only CPU preparation must not activate Transformers MLX tensor support.
        import importlib.util
        original_find_spec=importlib.util.find_spec
        try:
            importlib.util.find_spec=lambda name,*a,**kw:None if name=='mlx' or name.startswith('mlx.') else original_find_spec(name,*a,**kw)
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_root,trust_remote_code=False,local_files_only=True)
        finally:importlib.util.find_spec=original_find_spec
        save(args.output,prepare_inputs(tokenizer,args.tokenizer_root,transcript_arm=args.transcript_arm,nonce=args.nonce));return 0
    if args.execute_domain is not None:
        if not all((args.inputs,args.base_url,args.model)):parser.error('frozen inputs/baseURL/model required')
        result=run_domain_http(json.loads(args.inputs.read_text()),args.execute_domain,args.base_url,args.model,native=args.native);save(args.output,result);return 0
    save(args.output,{'schema':SCHEMA,'status':'planned','corpus_sha256':FROZEN_CORPUS_SHA256,'cases':400,'domains':20,'client_concurrency':20,'max_tokens':192,'gpu_executed':False,'native_route':'requires distinct nativeN20 capability/admission; never substitute B2'});return 0
if __name__=='__main__':raise SystemExit(main())
