#!/bin/zsh
# One tool-free reply. Usage: one_reply.sh <suite dir> <astra|opus> <prompt id>. Skips if answered.
# A reply that is a usage-limit notice is not an answer: it is moved to <suite>/invalid_limit/ and retried after 10 min
# (up to 60 times). Other empty or error replies are retried once.
cd "${0:A:h}"
S=$1 m=$2 n=$3 d=$1/toolfree/$2_$3
tries=0 waits=0
while true; do
  [ -s $d/answer.md ] && ! grep -q "hit your .*limit" $d/answer.md && break
  if [ -s $d/answer.md ] && grep -q "hit your .*limit" $d/answer.md; then
    mkdir -p $S/invalid_limit; mv $d $S/invalid_limit/$2_$3_$(date +%H%M%S); waits=$((waits+1))
    [ $waits -gt 60 ] && break; tries=$((tries-1)); [ $tries -lt 0 ] && tries=0; sleep 600; continue
  fi
  [ $tries -ge 2 ] && break
  tries=$((tries+1))
  if [ $m = astra ]; then ./run_astra.sh $S/prompts/$n.txt $d
  else ./run_claude.sh $S/prompts/$n.txt $d claude-opus-5-5 max; fi >/dev/null 2>&1
done
echo "$(date +%H:%M) $m $n $([ -s $d/answer.md ] && ! grep -q 'hit your .*limit' $d/answer.md && echo ok || echo EMPTY)"
