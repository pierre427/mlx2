#!/bin/zsh
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
echo $$ > ~/amoeba-runs/gate-27b-ensdonor.pid
PY=~/Desktop/mlx-uag/.venv/bin/python
M=~/mlx-models/gemma-4-26B-A4B-it-MLX-8bit-staged
G=~/Desktop/mlx2/benchmarks/gsm8k/test.jsonl
GU=~/Desktop/mlx2/src/mlx2/thinking_guard.py
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1 }
S0=$(swapmb); echo "$(date +%T) swap used at start ${S0}M"
( while sleep 10; do s=$(swapmb); if [ $((s - S0)) -gt 3072 ]; then echo "$(date +%T) SWAP GUARD: used ${s}M (start ${S0}M), stopping probe"; pkill -f "gemma_probe.py --model"; touch swapguard.tripped; break; fi; done ) &
WP=$!
echo "$(date +%T) smoke"
/usr/bin/perl -e 'alarm 600; exec @ARGV' $PY gemma_probe.py --model $M --gsm gsm8k_hard64.jsonl --guard $GU --out gemma_hsmoke --n 4 --budget 60 --caps 60,30,60,30 > gemma_hsmoke.log 2>&1
rc=$?; echo "$(date +%T) smoke rc=$rc"
if [ $rc -ne 0 ]; then kill $WP; echo 999999 > ~/amoeba-runs/gate-27b-ensdonor.pid; exit $rc; fi
echo "$(date +%T) full"
/usr/bin/perl -e 'alarm 2400; exec @ARGV' $PY gemma_probe.py --model $M --gsm gsm8k_hard64.jsonl --guard $GU --out gemma_hard2 --n 64 --budget 2400 --caps 1080,120,660,180 > gemma_hard2.log 2>&1
echo "$(date +%T) full rc=$?"
echo 999999 > ~/amoeba-runs/gate-27b-ensdonor.pid
kill $WP 2>/dev/null; echo "$(date +%T) swap used at end $(swapmb)M"
