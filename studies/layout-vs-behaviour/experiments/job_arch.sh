#!/bin/zsh
# One reader job: job_arch.sh tf|an|as <suite> <astra|opus> <network id>
# opus = claude-opus-5-5; every Opus reply in this study ran on Claude Code 2.1.314 (set OPUS_CLI to that binary to
# match; default: the claude on PATH), at effort max with the output-cap ladder below.
# tf: one tool-free reply, retried on a service error. an / as: one analyst conversation.
cd "${0:A:h}"
LIM='hit your (5-hour |weekly )?limit|usage limit|rate limit|limit (will )?reset'
maxout() { grep -q "output token maximum" $1/answer.md 2>/dev/null; }
limited() { maxout $1 && return 1; grep -qiE "$LIM" $1/answer.md $1/stderr.log 2>/dev/null || grep -q '"is_error": true' $1/run.json 2>/dev/null; }
kind=$1 S=$2 m=$3 n=$4
is_claude() { [ $m = opus ]; }
# One runner per job; a lock whose process is gone is taken over.
LOCK=$S/.lock_${kind}_${m}_$n
if ! mkdir $LOCK 2>/dev/null; then
  [ -s $LOCK/pid ] || sleep 2
  op=$(cat $LOCK/pid 2>/dev/null)
  if [ -n "$op" ] && kill -0 $op 2>/dev/null; then echo "$(date +%H:%M) $S $m $kind $n already running, skipped"; exit 0; fi
  rm -rf $LOCK; mkdir $LOCK 2>/dev/null || exit 0
fi
echo $$ > $LOCK/pid
trap 'rm -rf $LOCK' EXIT
case $m in opus) MODEL=claude-opus-5-5; export CLAUDE_BIN="${OPUS_CLI:-claude}";; esac
if [ $kind = tf ]; then
  d=$S/toolfree/${m}_$n; rc=0
  # Opus effort ladder for the 128k output-token cap: max -> xhigh -> high, then unanswered (LADDER_DONE).
  ladder_next() { case $1 in max) echo xhigh;; xhigh) echo high;; *) echo none;; esac; }
  E=${OPUS_EFFORT:-max}
  # Resume the ladder from archived attempts: an interrupted rerun leaves no folder, and would otherwise start again
  # at max.
  if is_claude; then
    ch=($d.capped_high_*(N)) cx=($d.capped_xhigh_*(N)) cm=($d.capped_max_*(N))
    if [ ${#ch} -gt 0 ]; then E=none; elif [ ${#cx} -gt 0 ]; then E=high; elif [ ${#cm} -gt 0 ]; then E=xhigh; fi
    [ $E = none ] && { echo "$(date +%H:%M) $S $m $n ladder already exhausted"; exit 0; }
  fi
  if is_claude && [ -s $d/answer.md ] && [ ! -e $d/LADDER_DONE ] && { maxout $d || grep -q "^UNANSWERED" $d/answer.md; }; then
    prev=$(cat $d/effort.txt 2>/dev/null || echo max); E=$(ladder_next $prev)
    mv $d $d.capped_${prev}_$(date +%s)
    [ $E = none ] && { echo "$(date +%H:%M) $S $m $n ladder already exhausted"; exit 0; }
  fi
  for i in {1..60}; do
    if [ -s $d/answer.md ] && maxout $d; then
      nxt=$(ladder_next $E)
      if [ $nxt = none ]; then echo "$(date +%H:%M) $S $m $n unanswered: output token cap at max, xhigh and high"; touch $d/LADDER_DONE; break; fi
      echo "$(date +%H:%M) $S $m $n output cap at $E; retrying at $nxt"
      mv $d $d.capped_${E}_$(date +%s); E=$nxt; continue
    fi
    if [ -s $d/answer.md ] && ! limited $d; then break; fi
    # keep the text of a failed attempt (service error) before the folder is cleared for the retry
    if [ -s $d/answer.md ]; then mkdir -p $S/toolfree_errors
      { date; cat $d/answer.md; echo; tail -c 1500 $d/stderr.log; grep rate_limit_info $d/events.jsonl | tail -1 | cut -c1-700; } > $S/toolfree_errors/${m}_${n}_$(date +%s).txt 2>/dev/null; fi
    rm -rf $d
    if [ $m = astra ]; then ${ASTRA_SCRIPT:-./run_astra.sh} $S/prompts/$n.txt $d; else ./run_claude.sh $S/prompts/$n.txt $d "$MODEL" $E; fi >/dev/null 2>&1
    rc=$?
    is_claude && echo $E > $d/effort.txt
    if [ -s $d/answer.md ] && ! limited $d; then break; fi
    if limited $d; then echo "$(date +%H:%M) $S $m $n limited, waiting"; sleep 600; else sleep 60; [ $i -ge 3 ] && break; fi
  done
  echo "$(date +%H:%M) $S $m toolfree $n rc=$rc $([ -s $d/answer.md ] && echo ok || echo FAILED)"
else
  # an: the open question with Python (analyst/). as: the behaviour question with Python (analyst_stance_max/).
  if [ $kind = as ]; then AD=analyst_stance_max; SCRIPT=analyst_loop_stance_max.py; else AD=analyst; SCRIPT=analyst_loop.py; fi
  for i in 1 2; do
    [ -s $S/$AD/${m}_$n/final_answer.md ] && break
    rm -rf $S/$AD/${m}_$n
    (cd analyst && ${WR_PYTHON:-python3} -I $SCRIPT $m $n ../$S > ../$S/${AD}_${m}_${n}.log 2>&1)
  done
  echo "$(date +%H:%M) $S $m $AD $n $([ -s $S/$AD/${m}_$n/final_answer.md ] && echo ok || echo FAILED)"
fi
