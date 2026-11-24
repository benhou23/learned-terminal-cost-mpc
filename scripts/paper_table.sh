#!/bin/bash
# Reproduce the paper's results table for one setting.
#
#   bash scripts/paper_table.sh                 # stochastic setting (main table)
#   bash scripts/paper_table.sh deterministic   # deterministic setting (appendix)
#
# Runs pure MPC, MPC + learned terminal cost and MPC + rollout terminal cost
# for horizons 1-5, plus the DQN on its own, on the 100 evaluation seeds
# 20000-20099. Results go to results/<setting>/; summarise them with
#   python scripts/summarize.py results/<setting>
# Expect several hours on a laptop; the runs are independent, so they can be
# split across machines.
set -e
cd "$(dirname "$0")/.."
SETTING="${1:-stochastic}"
SEEDS="${SEEDS:-20000-20099}"
CKPT="models/model_t16100000.zip"
OUT="results/${SETTING}"
mkdir -p "$OUT" logs

run () {  # name, then eval_batch arguments
  local name="$1"; shift
  echo "[$(date +%H:%M)] $name"
  python eval_batch.py --setting "$SETTING" --seeds "$SEEDS" \
      --output "$OUT/${name}_per_seed.csv" "$@" > "logs/${SETTING}_${name}.log" 2>&1
}

for H in 1 2 3 4 5; do
  run "mpc_H${H}"    --terminal_cost zero --horizon $H
  run "hybrid_H${H}" --terminal_cost rl   --horizon $H --checkpoint "$CKPT"
  for HR in 10 20; do
    run "rollout_H${H}_Hr${HR}" --terminal_cost rollout --horizon $H --rollout_horizon $HR
  done
done
run "pure_rl" --terminal_cost rl_only --checkpoint "$CKPT"

python scripts/summarize.py "$OUT"
