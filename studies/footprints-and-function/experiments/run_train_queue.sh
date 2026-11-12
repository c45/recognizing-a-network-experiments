#!/bin/zsh
# Runs every line of train_queue.txt (args to train_variant.py), 5 at a time, 3 threads each; skips finished runs.
cd "${0:A:h}"
PY=${WR_PYTHON:-python3}
one() { local out=$(echo "$1" | awk '{print $2}'); [ -s $out/actor.npz ] && return; mkdir -p $out; eval "$PY -I train_variant.py $1 --threads 3" > $out/train.log 2>&1; echo "done $out $?"; }
while read -r line; do one "$line" & while [ $(jobs -r | wc -l) -ge 5 ]; do sleep 10; done; done < train_queue.txt
wait
echo all done
