#!/bin/zsh
# Orchestrator 2026-11-26: run behind the Nemotron parity unit (gate-27b-techparity4.pid), start early (Pierre via orchestrator, 22:05).
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
G=~/amoeba-runs/gate-27b-techparity4.pid
until [ -f $G ]; do sleep 20; done
echo "$(date +%T) gate file seen: $(cat $G)"
while true; do p=$(cat $G); if [ "$p" = "999999" ] || ! kill -0 "$p" 2>/dev/null; then break; fi; sleep 20; done
echo "$(date +%T) Nemotron parity done (pid file: $(cat $G)); queueing for the GPU lease"
GPUQ_SESSION=ens-donor-20261125 /Users/Shared/mlxuag/gpuq-reference.sh ens-gemma-night-20261126 ./run_gemma5.sh
echo "$(date +%T) all done"
