#!/usr/bin/env python3
"""Prepare a source-bound DS4.1 campaign without loading MLX or using the GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from mlx2.experimental.ds41_performance import Binding,campaign,runtime_source_paths,source_snapshot,write_new_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',required=True,type=Path)
    parser.add_argument('--candidate-policy',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--reference-policy',type=Path)
    parser.add_argument('--candidate-numerical-profile')
    parser.add_argument('--runtime-input',action='append',default=[])
    args=parser.parse_args(argv)
    binding=Binding(**json.loads(args.binding.read_text()))
    policy=json.loads(args.candidate_policy.read_text())
    spec=campaign(binding,policy,seed=args.seed,
        reference_policy=json.loads(args.reference_policy.read_text()) if args.reference_policy else None,
        candidate_numerical_profile=args.candidate_numerical_profile)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('campaign directory is nonempty; preserve the earlier attempt')
    paths=runtime_source_paths(ROOT,additional=args.runtime_input)
    snapshot=source_snapshot(ROOT,paths)
    if snapshot['sha256']!=binding.source_snapshot:
        raise ValueError('binding source snapshot differs from current source; reseal before preparing')
    write_new_json(args.output/'source.json',snapshot)
    write_new_json(args.output/'campaign.json',spec)
    print(json.dumps({'prepared':str(args.output),'executed':False,'gpu_authorization':'pending'}))


if __name__=='__main__':main()
