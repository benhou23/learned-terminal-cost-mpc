#!/bin/bash
# Pure MPC with longer horizons (H = 5, 7, 10) in the stochastic setting,
# to compare with the learned terminal cost at H = 3. The prediction-error
# std tables cover horizons up to 10.
#   bash extra/longer_horizons.sh
set -e
cd "$(dirname "$0")/.."
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/extra/longer_horizons
mkdir -p "$OUT" logs
for H in 5 7 10; do
  python eval_batch.py --terminal_cost zero --horizon $H --seeds 20000-20099 \
      --output "$OUT/mpc_H${H}_per_seed.csv" > "logs/longer_mpc_H${H}.log" 2>&1 &
done
wait
python scripts/summarize.py "$OUT"
