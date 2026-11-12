#!/bin/sh
# Fetch the post's released material (third-party, CC BY 4.0; not redistributed here): Exp 1 networks and prompts
# (the four games' released policies and prompt texts) and the Exp 2 diffusion networks.
set -e
cd "$(dirname "$0")"
if [ ! -d lm_understand-logs ]; then
  git clone --filter=blob:none --no-checkout https://github.com/RikiyaT/lm_understand-logs.git lm_understand-logs
fi
cd lm_understand-logs
git sparse-checkout init --cone
git sparse-checkout set exp1_which_game/networks exp1_which_game/runs exp2_diffusion/networks
git checkout 4d09ba6
echo "fetched: external/lm_understand-logs/{exp1_which_game/networks,exp1_which_game/runs,exp2_diffusion/networks}"
