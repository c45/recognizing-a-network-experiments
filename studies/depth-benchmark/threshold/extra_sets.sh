#!/bin/zsh
# Amendment 3: spare seeds for depths 6, 7, 8, 10; extra blinded sets (excluding all networks already used at that
# depth); key hashes appended to PREREG.md before any call.
cd "${0:A:h}/../ladder/scale"
PY=${PY:-python3}
SEEDS=200 EPOCHS=10 $PY -I train_variants.py ../l2/data_d/states.npz d6:6:32 d7:7:32 d8:8:32 d10:10:32 || exit 1
build() {  # depth, suffix list
  local d=$1; shift
  local excl=readers_d$d/key.json
  [[ -f readers_d${d}b/key.json ]] && excl=$excl,readers_d${d}b/key.json
  local seed=$((20261200 + d * 10))
  for sfx in "$@"; do
    seed=$((seed + 1))
    $PY -I scale.py prompts nets_d$d readers_d$d$sfx --depth $d --width 32 --seed $seed --exclude $excl || exit 1
    printf "\n## %s key hash (recorded before any call)\n\n\`%s\`\n" "scale/readers_d$d$sfx" \
      "$(cut -d' ' -f1 readers_d$d$sfx/PREREG_KEY_HASH.txt)" >> ../../threshold/PREREG.md
    excl=$excl,readers_d$d$sfx/key.json
  done
}
build 6 c d e
build 7 c d e
build 8 c d
build 10 c d e
SETS=(readers_d6c readers_d7c readers_d8c readers_d10c readers_d6d readers_d7d readers_d8d readers_d10d readers_d6e readers_d7e readers_d10e)
# Replies: one tool-free reply per prompt from each model at effort xhigh (ladder/scale/readers_*/run_astra.sh,
# run_claude.sh), written to replies/astra_xhigh/<id> and replies/opus_xhigh/<id>.
echo "extra sets built"
