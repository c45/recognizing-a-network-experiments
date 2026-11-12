#!/bin/zsh
# One analyst conversation. Usage: one_analyst.sh <suite dir> <astra|opus> <network id>. Skips if already answered.
cd "${0:A:h}"
S=$1 b=$2 n=$3
mkdir -p $S/analyst
[ -s $S/analyst/${b}_$n/final_answer.md ] || ${WR_PYTHON:-python3} -I analyst_loop_r.py $b $n $S > $S/analyst_${b}_$n.log 2>&1
echo "$(date +%H:%M) analyst $b $n $([ -s $S/analyst/${b}_$n/final_answer.md ] && echo ok || echo FAIL)"
