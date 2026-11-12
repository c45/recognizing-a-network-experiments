#!/bin/zsh
# One Astra tool-free reply of the hint control: job_hint.sh <suite> <network id>. Up to 3 attempts; skips answered ones.
cd "${0:A:h}"
S=$1 n=$2 d=$1/toolfree/astra_$2
for i in 1 2 3; do
  [ -s $d/answer.md ] && break
  rm -rf $d
  ./run_astra.sh $S/prompts/$n.txt $d >/dev/null 2>&1
  [ -s $d/answer.md ] || sleep 90
done
echo "$(date +%H:%M) $S astra $n $([ -s $d/answer.md ] && echo ok || echo FAILED)"
