#!/usr/bin/env bash
#: Usage: ./run.sh <command> [run] [config] [seed]
#:
#:   setup    install the Python environment
#:   toy      train a tiny model on toy data (CPU, about 15 seconds)
#:   data     download the training data
#:   train    train a run and score it on GEP-Val (default config: configs/baseline.yaml)
#:   test     score a run on GEP-Test
#:   eval     score a run on GIFT-Eval
#:   submit   package three or more runs as records/<name>/: ./run.sh submit <name> <run>...
#:   table    print the README's record table
#:   site     build the record page into site/
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
  name=${2:?Usage: ./run.sh train <run> [config] [seed]}
  py -m nanotsfm.train --config "${3:-configs/baseline.yaml}" --output "$RUNS/$name" \
    ${4:+--seed "$4"}
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
    --output "$RUNS/$name/gift-forecasts" --device auto --workers "$workers"
  # Scoring runs in its own process, which loads none of nanotsfm's code.
  py -m scripts.score --forecasts "$RUNS/$name/gift-forecasts" --upstream external/gift-eval \
    --tasks configs/gift-full.json --output "$RUNS/$name/gift.json" --workers "$workers"
  rm -r "$RUNS/$name/gift-forecasts"  # 1.4 GB, no longer needed
  ;;
submit)
  name=${2:?Usage: ./run.sh submit <name> <run> <run> <run>...}
  runs=()
  for run in "${@:3}"; do need_run "$run"; runs+=("$RUNS/$run"); done
  py -m scripts.submission package "records/$name" "${runs[@]}"
  py -m scripts.submission check "records/$name" --runs "${runs[@]}"
  ;;
table)
  uv run --no-project --with pyyaml python scripts/records.py --table
  ;;
site)
  uv run --no-project --with pyyaml python scripts/records.py site
  ;;
*)
  usage
  ;;
esac
