#!/bin/zsh
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
echo "$(date +%T) start kvmix"
/usr/bin/perl -e 'alarm 600; exec @ARGV' ~/Desktop/mlx-uag/.venv/bin/python kv_mix.py --model ~/mlx-models/Qwen3-0.6B-BF16-microscaling-source --paper1 paper_cb0240527.txt --paper2 paper_cb0240685.txt --out kv_mix_answers.json > kv_mix.log 2>&1
echo "$(date +%T) done rc=$?"
