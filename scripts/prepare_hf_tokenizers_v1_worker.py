#!/usr/bin/env python3
"""Create a host-local, content-bound encode-worker manifest after CPU parity.

Run in the isolated baseline environment. This command does not install packages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    for name in ('python','wheel','model','cpu-receipt','patch','inputs','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
    model=Path(args.model).resolve()
    receipt=json.loads(Path(args.cpu_receipt).read_text())
    if (receipt['version']!='1.0.0-rc.2' or receipt.get('corpus_exact_id_mismatches')!=[]
            or any(m['errors'] for m in receipt['models']) or receipt.get('documents')!=400):
        raise ValueError('CPU exact encode receipt is incomplete')
    inputs=json.loads(Path(args.inputs).read_text())
    if hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest()!=receipt['inputs_sha256']:
        raise ValueError('CPU receipt/input identity mismatch')
    selected=[m for m in inputs['models'] if Path(m['path']).resolve()==model]
    if len(selected)!=1:
        raise ValueError('model lacks pinned CPU input contract')
    probe='import tokenizers,pathlib,json; print(json.dumps({"version":tokenizers.__version__,"extension":str(pathlib.Path(tokenizers.__file__).parent/"tokenizers.abi3.so")}))'
    identity=json.loads(subprocess.check_output([args.python,'-I','-c',probe],text=True,timeout=10))
    if identity['version']!='1.0.0-rc.2':
        raise ValueError('unexpected isolated tokenizers version')
    files={'python':args.python,'extension':identity['extension'],'wheel':args.wheel,
           'tokenizer':str(model/'tokenizer.json'),'config':str(model/'tokenizer_config.json'),
           'chat_template':str(model/'chat_template.jinja'),'cpu_receipt':args.cpu_receipt}
    manifest={'schema':'mlx2.tokenizers-v1-worker.v1','tokenizers_version':identity['version'],
              'qualification':'cpu_exact_encode_candidate','minimum_chars':8192,'timeout_seconds':5.0,
              'source_revision':'7616272a1b0b580fb43e3a06701fd11db9a2706a',
              'source_patch_sha256':hashlib.sha256(Path(args.patch).read_bytes()).hexdigest(),
              'canaries':[{'text':c['text'],'add_special_tokens':c['add'],'ids':c['ids']} for c in selected[0]['cases']]}
    manifest.update({k:{'path':str(Path(p).absolute()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for k,p in files.items()})
    for field in ('tokenizer','config'):
        if manifest[field]['sha256']!=selected[0][field+'_sha256']:
            raise ValueError('model files changed since CPU parity')
    if manifest['chat_template']['sha256']!=selected[0]['chat_template_sha256']:
        raise ValueError('chat template changed since CPU parity')
    Path(args.output).write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    main()
