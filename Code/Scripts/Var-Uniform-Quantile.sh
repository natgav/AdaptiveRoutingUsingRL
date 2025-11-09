#!/usr/bin/env bash
set -euo pipefail

: "${omega:=0.05}"
: "${num_episodes:=1000}"

if [[ -z "${Final_deadline+x}" ]]; then
  Final_deadline=(20 25 30 35 40)
fi
if [[ -z "${variance+x}" ]]; then
  variance=(1 2 3 4 5)
fi

for deadline in "${Final_deadline[@]}"; do
  for var in "${variance[@]}"; do
    dir="Results/Quantile/Var/Uniform/${deadline}/${var}"
    mkdir -p "$dir"
    export deadline var
    export resultsfile="$dir/"
    export omega num_episodes
    echo "[Var-Uniform-Quantile] deadline=$deadline var=$var episodes=$num_episodes omega=$omega -> $resultsfile"
    python3 Src/meandelay-uniform-quantile.py
  done
done
