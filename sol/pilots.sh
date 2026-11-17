#!/bin/bash
# Submit Track A jobs to Sol. Usage:
#   sol/pilots.sh reproduce   # the current record (configs/series.yaml), seeds 7, 1, 2
#   sol/pilots.sh pilots      # one seed (7) of every Track A config
#   sol/pilots.sh seeds <config-name>   # seeds 1 and 2 of one Track A config
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
submit() { sbatch --job-name="$1" sol/run.sbatch "$1" "$2" "$3"; }

case "${1:-}" in
reproduce)
  for seed in 7 1 2; do submit "series-s$seed" configs/series.yaml "$seed"; done
  ;;
pilots)
  for config in configs/track-a/*.yaml; do
    name=$(basename "$config" .yaml)
    submit "$name-s7" "$config" 7
  done
  ;;
seeds)
  name=${2:?config name, e.g. sp1.5}
  for seed in 1 2; do submit "$name-s$seed" "configs/track-a/$name.yaml" "$seed"; done
  ;;
*)
  sed -n '2,6p' "$0"; exit 1
  ;;
esac
