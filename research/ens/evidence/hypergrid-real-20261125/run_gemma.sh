#!/bin/zsh
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
echo $$ > ~/amoeba-runs/gate-27b-ensdonor.pid
PY=~/Desktop/mlx-uag/.venv/bin/python
M=~/mlx-models/gemma-4-26B-A4B-it-MLX-8bit-staged
G=~/Desktop/mlx2/benchmarks/gsm8k/test.jsonl
GU=~/Desktop/mlx2/src/mlx2/thinking_guard.py
echo "$(date +%T) smoke"
/usr/bin/perl -e 'alarm 600; exec @ARGV' $PY gemma_probe.py --model $M --gsm $G --guard $GU --out gemma_smoke --n 4 --budget 60 --caps 60,30,60,30 > gemma_smoke.log 2>&1
rc=$?; echo "$(date +%T) smoke rc=$rc"
if [ $rc -ne 0 ]; then rm -f ~/amoeba-runs/gate-27b-ensdonor.pid; exit $rc; fi
echo "$(date +%T) full"
/usr/bin/perl -e 'alarm 1800; exec @ARGV' $PY gemma_probe.py --model $M --gsm $G --guard $GU --out gemma_g26 --n 48 --budget 1600 --caps 600,90,480,150 > gemma_g26.log 2>&1
echo "$(date +%T) full rc=$?"
rm -f ~/amoeba-runs/gate-27b-ensdonor.pid
