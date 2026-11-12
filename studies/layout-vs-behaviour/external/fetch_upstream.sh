#!/bin/sh
# Fetch the post's released Exp 1 material (third-party; not redistributed): the prompt template, the network
# description used by the analyst arm, and the released networks.
set -e
cd "$(dirname "$0")"
if [ ! -d lm_understand-logs ]; then
  git clone --filter=blob:none --no-checkout https://github.com/RikiyaT/lm_understand-logs.git lm_understand-logs
fi
cd lm_understand-logs
git sparse-checkout init --cone
git sparse-checkout set exp1_which_game/networks exp1_which_game/runs/pong
git checkout 4d09ba6
echo "fetched: external/lm_understand-logs/exp1_which_game/{networks,runs/pong}"
