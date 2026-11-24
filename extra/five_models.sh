#!/bin/bash
# Robustness check from the review period: MPC + learned terminal cost with
# H = 3 in the stochastic setting, for all five DQN training runs
# (the paper's model plus models/other_runs/). Runs the five in parallel.
#   bash extra/five_models.sh
set -e
cd "$(dirname "$0")/.."
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # five processes share the cores
OUT=results/extra/five_models
mkdir -p "$OUT" logs
for m in models/model_t16100000.zip models/other_runs/*.zip; do
  name=$(basename "$m" .zip)
  python eval_batch.py --terminal_cost rl --horizon 3 --checkpoint "$m" \
      --seeds 20000-20099 --output "$OUT/hybrid_H3_${name}_per_seed.csv" \
      > "logs/five_models_${name}.log" 2>&1 &
done
wait
python scripts/summarize.py "$OUT"
