#!/bin/zsh
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
echo "$(date +%T) start reason probe"
/usr/bin/perl -e 'alarm 1140; exec @ARGV' ~/Desktop/mlx-uag/.venv/bin/python reason_probe.py --model ~/mlx-models/Qwen3-0.6B-BF16-microscaling-source --gsm ~/Desktop/mlx2/benchmarks/gsm8k/test.jsonl --out reason_q06f --n 80 --gen-seconds 540 --reuse reason_q06.phaseA.thinking-traces.json > reason_q06f.log 2>&1
echo "$(date +%T) done rc=$?"
