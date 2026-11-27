#!/bin/zsh
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
echo "$(date +%T) start reason probe 2"
/usr/bin/perl -e 'alarm 1380; exec @ARGV' ~/Desktop/mlx-uag/.venv/bin/python reason_probe2.py --model ~/mlx-models/Qwen3-0.6B-BF16-microscaling-source --gsm ~/Desktop/mlx2/benchmarks/gsm8k/test.jsonl --guard ~/Desktop/mlx2/src/mlx2/thinking_guard.py --sets neuron_sets.json --out reason2_q06 --n 64 --budget 2400 --caps 540,360,180,240 --d-problems 20 > reason2_q06.log 2>&1
echo "$(date +%T) done rc=$?"
