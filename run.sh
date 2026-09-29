#!/usr/bin/env bash
#: Usage: ./run.sh <command> [run] [config]
#:
#:   setup    install the Python environment
#:   toy      train a tiny model on toy data (CPU, about 15 seconds)
#:   data     download the training data
#:   train    train a run and score it on GEP-Val (default config: configs/baseline.yaml)
#:   test     score a run on GEP-Test
#:   eval     score a run on GIFT-Eval
#:   submit   package a run as submissions/<team>/: ./run.sh submit <run> <team>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .env ]]; then
  while IFS='=' read -r key value || [[ -n $key ]]; do
    value=${value%\"}; value=${value#\"}; value=${value%\'}; value=${value#\'}
    [[ $key =~ ^[A-Z_][A-Z0-9_]*$ && -z "${!key+x}" ]] && export "$key=$value"
  done < .env
fi

GIFT_CODE_REVISION=9a014e9e8ea130ba39c100c60d5dcbab7db57ac9
RUNS=${RUNS:-runs}
extras=(--extra gift)
[[ -n "${TORCH:-}" ]] && extras+=(--extra "$TORCH")

py() { uv run --locked "${extras[@]}" python "$@"; }
usage() { sed -n 's/^#: \{0,1\}//p' "$0"; }
need_run() { [[ -f "$RUNS/$1/checkpoint.pt" ]] || { echo "No checkpoint in $RUNS/$1" >&2; exit 1; }; }

case "${1:-help}" in
setup)
  uv sync --locked "${extras[@]}"
  ;;
toy)
  dir=$(mktemp -d "${TMPDIR:-/tmp}/nanotsfm-toy.XXXXXX")
  py -m nanotsfm.train --config configs/toy.yaml --output "$dir"
  echo "Toy run finished: $dir"
  ;;
data)
  py -m nanotsfm.data download
  ;;
train)
  name=${2:?Usage: ./run.sh train <run> [config]}
  py -m nanotsfm.train --config "${3:-configs/baseline.yaml}" --output "$RUNS/$name"
  py -m nanotsfm.evaluation gep --checkpoint "$RUNS/$name/checkpoint.pt" --split validation \
    --output "$RUNS/$name/gep-val.json" --device auto
  ;;
test)
  name=${2:?Usage: ./run.sh test <run>}
  need_run "$name"
  py -m nanotsfm.evaluation gep --checkpoint "$RUNS/$name/checkpoint.pt" --split test \
    --output "$RUNS/$name/gep-test.json" --device auto
  ;;
eval)
  name=${2:?Usage: ./run.sh eval <run>}
  need_run "$name"
  if [[ ! -d external/gift-eval ]]; then
    git clone -q https://github.com/SalesforceAIResearch/gift-eval.git external/gift-eval
    git -C external/gift-eval checkout -q "$GIFT_CODE_REVISION"
  fi
  cores=$(getconf _NPROCESSORS_ONLN)
  workers=$(( cores < 16 ? cores : 16 ))
  py -m nanotsfm.evaluation gift --checkpoint "$RUNS/$name/checkpoint.pt" \
    --upstream external/gift-eval --tasks configs/gift-full.json \
    --output "$RUNS/$name/gift.json" --device auto --workers "$workers"
  ;;
submit)
  name=${2:?Usage: ./run.sh submit <run> <team>}
  team=${3:?Usage: ./run.sh submit <run> <team>}
  need_run "$name"
  py -m scripts.submission package "$RUNS/$name" "submissions/$team" "${@:4}"
  py -m scripts.submission check "submissions/$team" --checkpoint "$RUNS/$name/checkpoint.pt"
  ;;
*)
  usage
  ;;
esac
