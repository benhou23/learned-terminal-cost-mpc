"""
Summarise the per-seed CSVs written by eval_batch.py.

    python scripts/summarize.py results/stochastic

Prints, per controller: episodes, episodes with a collision, cost per step
(total cost over total steps, as in the paper) and the median per-episode
cost per step. Cost is recovered from the scaled reward: cost = -reward / 1e-5.
"""
import glob
import os
import sys

import pandas as pd

REWARD_SCALE = 1e-5


def main(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*_per_seed.csv"))):
        d = pd.read_csv(f)
        cost = -d["cumulative_reward"] / REWARD_SCALE
        rows.append(dict(
            controller=os.path.basename(f).replace("_per_seed.csv", ""),
            episodes=len(d),
            collisions=int(d["collision"].sum()),
            cost_per_step=round(cost.sum() / d["steps"].sum(), 1),
            median_cost_per_step=round((cost / d["steps"]).median(), 1),
        ))
    if not rows:
        sys.exit(f"no *_per_seed.csv files in {folder}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/stochastic")
