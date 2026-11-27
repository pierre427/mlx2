#!/bin/zsh
cd ~/Desktop/mlx2/output/ens-hypergrid-real-20261125
echo "$(date +%T) start hidden dump swap $(sysctl -n vm.swapusage | awk '{print $6}')"
~/Desktop/mlx-uag/.venv/bin/python donor_hidden.py --model ~/mlx-models/Qwen3-0.6B-BF16-microscaling-source --corpus fluids.jsonl --out donor_h06 --max-seconds 780 > donor_h06.log 2>&1
echo "$(date +%T) done rc=$?"
