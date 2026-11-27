#!/bin/zsh
# Three donor dumps, one model per process, hard 60-minute window (run under gpuq-reference.sh).
set -u
D=~/Desktop/mlx2/output/ens-hypergrid-real-20261125
PY=~/Desktop/mlx-uag/.venv/bin/python
end=$(( $(date +%s) + 3420 ))   # 57 min of work, leaves margin inside the hour
cd $D
for spec in "q06:Qwen3-0.6B-BF16-microscaling-source:600" "q4b:Qwen3-4B:900" "nem30:NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit:1500"; do
  tag=${spec%%:*}; rest=${spec#*:}; model=${rest%%:*}; cap=${rest##*:}
  left=$(( end - $(date +%s) - 120 ))
  (( left < 120 )) && { echo "$(date +%T) skip $tag: window exhausted"; continue; }
  (( cap > left )) && cap=$left
  echo "$(date +%T) start $tag $model cap ${cap}s swap $(sysctl -n vm.swapusage | awk '{print $6}')"
  $PY donor_dump.py --model ~/mlx-models/$model --corpus fluids.jsonl \
      --out donor_$tag --max-seconds $cap > donor_$tag.log 2>&1
  echo "$(date +%T) done $tag rc=$? $(tail -1 donor_$tag.log | cut -c1-200)"
done
echo "$(date +%T) all done"
