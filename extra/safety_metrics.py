"""
Safety metrics from the per-step logs written with STEP_LOG.

    python extra/safety_metrics.py results/extra/safety/steplogs

Uses the true bumper-to-bumper gap to the leader taken from SUMO (not the
noisy measurement). Safe distance = D0 + T_HEAD * speed, without the
chance-constraint margin. Hard braking = the -2 m/s^2 action.
"""
import glob
import os
import sys

import pandas as pd

D0, T_HEAD, A_MIN = 5.0, 2.0, -2.0


def main(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        d = pd.read_csv(f)
        gap = d["true_gap_front"]
        below = gap < (D0 + T_HEAD * d["v"])
        rows.append(dict(
            controller=os.path.basename(f).replace(".csv", ""),
            steps=len(d),
            min_gap_m=round(gap.min(), 1),
            median_episode_min_gap_m=round(d.groupby("scenario")["true_gap_front"].min().median(), 1),
            pct_steps_below_safe_distance=round(100 * below.mean(), 1),
            pct_steps_hard_braking=round(100 * (d["a"] <= A_MIN).mean(), 1),
        ))
    if not rows:
        sys.exit(f"no step logs in {folder}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/extra/safety/steplogs")
