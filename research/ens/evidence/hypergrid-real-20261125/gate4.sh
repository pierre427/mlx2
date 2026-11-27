#!/bin/zsh
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
G=~/amoeba-runs/gate-27b-specround-raon-p1a2.pid
until [ -f $G ]; do sleep 20; done
echo "$(date +%T) gate file seen: $(cat $G)"
while true; do p=$(cat $G 2>/dev/null); if [ "$p" = "999999" ] || ! kill -0 "$p" 2>/dev/null; then break; fi; sleep 20; done
echo "$(date +%T) raon unit done (pid file: $(cat $G)); starting"
GPUQ_SESSION=ens-donor-20261125 /Users/Shared/mlxuag/gpuq-reference.sh ens-gemma-hard3-20261126 ./run_gemma4.sh
echo "$(date +%T) all done"
