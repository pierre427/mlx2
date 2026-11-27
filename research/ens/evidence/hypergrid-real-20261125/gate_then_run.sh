#!/bin/zsh
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
while kill -0 48796 48772 2>/dev/null; do sleep 15; done
echo "$(date +%T) preflight gone"
t=0; while [ $t -lt 300 ]; do u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1); [ $u -lt 14336 ] && break; sleep 15; t=$((t+15)); done
echo "$(date +%T) swap ${u}M, starting"
GPUQ_SESSION=ens-donor-20261125 /Users/Shared/mlxuag/gpuq-reference.sh ens-gemma-hard2-20261126 ./run_gemma3.sh
