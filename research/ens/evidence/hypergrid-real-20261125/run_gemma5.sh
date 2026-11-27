#!/bin/zsh
# Night unit 2026-11-26: (1) no-think ablation control on the EASY set (first 48 GSM8K, sets from gemma_g26.json),
# (2) hard-set readouts reusing gemma_hard2 traces. Each step saves per problem; swap guard +3 GB stops the probe.
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
GF=~/amoeba-runs/gate-27b-ensdonor.pid
echo $$ > $GF
PY=~/Desktop/mlx-uag/.venv/bin/python
M=~/mlx-models/gemma-4-26B-A4B-it-MLX-8bit-staged
G=~/Desktop/mlx2/benchmarks/gsm8k/test.jsonl
GU=~/Desktop/mlx2/src/mlx2/thinking_guard.py
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1 }
S0=$(swapmb); echo "$(date +%T) swap used at start ${S0}M"; vm_stat | head -6
rm -f swapguard.tripped
( while sleep 10; do s=$(swapmb); if [ $((s - S0)) -gt 3072 ]; then echo "$(date +%T) SWAP GUARD: used ${s}M (start ${S0}M), stopping probe"; pkill -f "gemma_probe.py --model"; touch swapguard.tripped; break; fi; done ) &
WP=$!
( while sleep 10; do echo "$(date +%T) swap $(swapmb)M"; done ) > swap_trace5.log &
TP=$!
finish() { echo 999999 > $GF; kill $WP $TP 2>/dev/null; echo "$(date +%T) swap used at end $(swapmb)M"; }
echo "$(date +%T) control smoke"
/usr/bin/perl -e 'alarm 300; exec @ARGV' $PY gemma_probe.py --model $M --gsm $G --guard $GU --out gemma_ntctl_smoke --n 2 --nothink-control gemma_g26.json > gemma_ntctl_smoke.log 2>&1
rc=$?; echo "$(date +%T) control smoke rc=$rc"
if [ $rc -ne 0 ]; then finish; exit $rc; fi
echo "$(date +%T) control"
/usr/bin/perl -e 'alarm 1200; exec @ARGV' $PY gemma_probe.py --model $M --gsm $G --guard $GU --out gemma_ntctl --n 48 --nothink-control gemma_g26.json > gemma_ntctl.log 2>&1
echo "$(date +%T) control rc=$?"
if [ -f swapguard.tripped ]; then echo "$(date +%T) guard tripped, skipping hard readouts"; finish; exit 3; fi
echo "$(date +%T) hard readouts"
/usr/bin/perl -e 'alarm 2400; exec @ARGV' $PY gemma_probe.py --model $M --gsm gsm8k_hard64.jsonl --guard $GU --out gemma_hard4 --reuse gemma_hard2.traces.json --n 64 --budget 2400 --caps 0,120,1500,300 > gemma_hard4.log 2>&1
echo "$(date +%T) hard rc=$?"
finish
