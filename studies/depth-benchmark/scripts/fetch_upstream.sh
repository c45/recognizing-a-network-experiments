#!/bin/sh
# Fetch the released "which game" networks from the original post (third-party; not redistributed in this repo).
set -e
cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/lm_understand-logs ]; then
  git clone --filter=blob:none --no-checkout https://github.com/RikiyaT/lm_understand-logs.git external/lm_understand-logs
fi
cd external/lm_understand-logs
git sparse-checkout init --cone
git sparse-checkout set exp1_which_game/networks
git checkout 4d09ba6
echo "fetched: external/lm_understand-logs/exp1_which_game/networks"
