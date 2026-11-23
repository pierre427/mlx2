#!/bin/zsh
# Stop the queued H25 part as soon as H26 is DONE (the peer review showed H25 cannot answer the two-op question)
until test -e ~/ens-lab/evidence/h26-20261122/DONE; do sleep 20; done
sleep 5
pkill -TERM -f "queue-h26-h25.sh" ; sleep 2
pkill -TERM -f "capacity24k-20261122" ; sleep 2
pkill -TERM -f "research.ens.m3 run --job ens-a0-h26-h25" ; true
echo "stopped H25 at $(date)" >> ~/ens-lab/evidence/h26-20261122/STOPPED-H25.txt
