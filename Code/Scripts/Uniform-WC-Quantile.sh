#!/usr/bin/env bash
set -euo pipefail

: "${omega:=0.05}"
: "${num_episodes:=1000}"

if [[ -z "${Final_deadline+x}" ]]; then
  Final_deadline=(20 25 30 35 40)
fi

for deadline in "${Final_deadline[@]}"; do
  dir="Results/Quantile/Uniform-WC/$deadline"
  mkdir -p "$dir"
  export deadline
  export resultsfile="$dir/"
  export omega num_episodes
  echo "[Uniform-WC-Quantile] deadline=$deadline episodes=$num_episodes omega=$omega -> $resultsfile"
  python3 Src/meandelay-uniform-wc-quantile.py
done
