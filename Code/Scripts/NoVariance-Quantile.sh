#!/usr/bin/env bash
set -euo pipefail

# Inherit or set defaults
: "${omega:=0.05}"
: "${num_episodes:=1000}"

# If not provided by runall, seed a default array
if [[ -z "${Final_deadline+x}" ]]; then
  Final_deadline=(20 25 30 35 40)
fi

for deadline in "${Final_deadline[@]}"; do
  dir="Results/Quantile/NoVariance/$deadline"
  mkdir -p "$dir"
  export deadline
  export resultsfile="$dir/"
  export omega num_episodes
  echo "[NoVariance-Quantile] deadline=$deadline episodes=$num_episodes omega=$omega -> $resultsfile"
  python3 Src/novariance-quantile.py
done
