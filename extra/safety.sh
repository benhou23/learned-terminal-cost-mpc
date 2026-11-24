#!/bin/bash
# Safety metrics: runs MPC + learned terminal cost (H = 3) and pure MPC
# (H = 5, 7, 10) in the stochastic setting with per-step logs, then computes
# gap and braking statistics from the true SUMO gaps.
#   bash extra/safety.sh
set -e
cd "$(dirname "$0")/.."
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/extra/safety
mkdir -p "$OUT/steplogs" logs
STEP_LOG="$OUT/steplogs/hybrid_H3.csv" python eval_batch.py --terminal_cost rl --horizon 3 \
    --checkpoint models/model_t16100000.zip --seeds 20000-20099 \
    --output "$OUT/hybrid_H3_per_seed.csv" > logs/safety_hybrid_H3.log 2>&1 &
for H in 5 7 10; do
  STEP_LOG="$OUT/steplogs/mpc_H${H}.csv" python eval_batch.py --terminal_cost zero --horizon $H \
      --seeds 20000-20099 --output "$OUT/mpc_H${H}_per_seed.csv" > "logs/safety_mpc_H${H}.log" 2>&1 &
done
wait
python extra/safety_metrics.py "$OUT/steplogs"
