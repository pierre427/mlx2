#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Terms and scope: provenance/LICENSE-calculator.txt; provenance/LICENSE-calculator-scope.json
"""Prepare Qwen resident-calculator manifests/policies without loading weights or using GPU."""
from __future__ import annotations
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse,hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def prepare(model,drafter,out):
    from mlx2.adapters.qwen38_27b import inspect_artifact
    from mlx2.runtime.amoeba_det_tiles import SCHEMA,PARSER_V5,load_manifest
    model=Path(model).resolve();drafter=Path(drafter).resolve();out=Path(out).resolve()
    artifact=inspect_artifact(model);fp=artifact['identity']['fingerprint']
    config=json.loads((model/'config.json').read_text());text=config.get('text_config',config)
    layers=int(text['num_hidden_layers'])
    if not (drafter/'config.json').is_file():raise ValueError('local drafter config missing')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    out.mkdir(parents=True,exist_ok=False)
    def write(name,data):
        (out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    write('resident.json',dict(schema='mlx2.calculator-resident-binding.v1',parser=PARSER_V5,
        gate='complete_syntax_and_exact_result',source_revision=revision,
        license='LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0'))
    manifest=out/'manifest.json';checkpoint=model.name+'@'+fp[:8]
    write('manifest.json',dict(schema=SCHEMA,source=dict(fingerprint=fp,checkpoint=checkpoint,path=str(model)),
        layers=layers,tiles=[dict(id='calculator',kind='calculator',write_mode='force',
            read_layer=layers-1,write_layer=layers-1,file='resident.json',parser=PARSER_V5,
            arithmetic_backend='gpu',gate=dict(kind='validated_syntax'),complete_literal_requests=True)],
        files={'resident.json':hashlib.sha256((out/'resident.json').read_bytes()).hexdigest()},
        qualification='unqualified',host_tiles_manifest_sha256=None,
        calculator_license='LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0'))
    load_manifest(manifest)
    # Explicit null defeats pair-default tree selection for the chain control.
    chain=dict(draft_model=str(drafter),num_draft=7,batch_size_route=None,
               pairwise_selection='host',amoeba_det_tiles=str(manifest))
    write('chain-policy.json',chain)
    write('ast-policy.json',dict(amoeba_det_tiles=str(manifest)))
    common=['python','-m','mlx2.server','--model',str(model),'--port','8393','--max-lanes','4']
    write('setup.json',dict(checkpoint=checkpoint,fingerprint=fp,source_revision=revision,
        model_loaded=False,gpu_used=False,qualification='unqualified',manifest=str(manifest),
        drafter=str(drafter),payload_revision_pins='computed and checked by the external loader on startup',
        chain_command=common+['--external-draft','--execution-policy',str(out/'chain-policy.json')],
        tree15_command=common+['--ast','--ast-drafter',str(drafter),'--execution-policy',str(out/'ast-policy.json')],
        required_environment=dict(PYTHONPATH=str(ROOT/'src'),MLX_METAL_FAST_SYNCH='0',HF_HUB_OFFLINE='1'),
        ownership='Run server commands only inside an owned CPG/paired-lock GPU window.'))
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,default=Path.home()/'mlx-models/Qwen3.8-27B-BF16-q8-mlx2')
    ap.add_argument('--drafter',type=Path,default=Path.home()/'mlx-models/Qwen3.8-27B-DFlash2')
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    print(prepare(a.model,a.drafter,a.out))


if __name__=='__main__':main()
