"""
Is the traffic dense and varied enough to learn from?

Drives the ego with random actions for a few episodes (needs SUMO, no model)
and looks at the observations: are vehicles spawned in all lanes, how often
the front gap is below the safe distance, how spread out the gaps and speeds
are, and so on. Prints the numbers and a pass/fail per check.

    python tests/test_traffic_quality.py --density 50 --episodes 5 --steps 300
"""

from __future__ import annotations

import os
import sys
import copy
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# params first, its values are used as CLI defaults
try:
    from params import SIM_PARAMS, ROAD_PARAMS
except ImportError:
    print("ERROR: params.py not found. Run from the project root.")
    sys.exit(1)

# CLI
_default_density = float(SIM_PARAMS.get("traffic_density_veh_per_km", 30.0))
p = argparse.ArgumentParser()
p.add_argument("--density",  type=float, default=_default_density,
               help=f"Traffic density veh/km (default: {_default_density} from params.py)")
p.add_argument("--episodes", type=int,   default=5,
               help="Number of episodes to sample (default 5)")
p.add_argument("--steps",    type=int,   default=300,
               help="Max steps per episode (default 300 = 60 s)")
p.add_argument("--seed0",    type=int,   default=0,
               help="First seed; episodes use seed0, seed0+1, ... (default 0)")
args = p.parse_args()

# Imports
try:
    from highway_env import Highway
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure highway_env.py is on the path and SUMO is available.")
    sys.exit(1)

# Helpers
PASS = FAIL = 0

def check(name, ok, detail="", warn=False):
    global PASS, FAIL
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    if ok:
        PASS += 1
    elif not warn:
        FAIL += 1
    detail_str = f"  [{detail}]" if detail else ""
    print(f"  {tag:<4}  {name}{detail_str}")

def make_sp(density, seed):
    sp = copy.deepcopy(SIM_PARAMS)
    sp["traffic_density_veh_per_km"] = density
    sp["warmup_s"]                   = 3.0
    sp["traffic_settle_s"]           = 2.0
    sp["init_steps"]                 = 5
    sp["duration_s"]                 = 999999
    sp["terminate_on_collision"]     = False   # don't cut episodes short
    sp["terminate_on_cost"]          = False
    sp["seed_spawn"]                 = seed
    return sp

def make_rp(label):
    rp = copy.deepcopy(ROAD_PARAMS)
    rp["scenario_dir"] = f"scenario_out_tq_{label}"
    return rp

# obs layout:
# [0]  v_ego
# [1]  lane
# [2]  gF   [3]  dvF
# [4]  gFL  [5]  dvFL
# [6]  gFR  [7]  dvFR
# [8]  gBL  [9]  dvBL
# [10] gBR  [11] dvBR
# [12] lc_prog
# [13] cd_prog

SR = float(SIM_PARAMS.get("sensor_range_m", 100.0))
D0     = float(SIM_PARAMS.get("D0",     10.0))
T_HEAD = float(SIM_PARAMS.get("T_HEAD",  2.0))
V_REF  = float(SIM_PARAMS.get("v_ref",  25.0))

print(f"\n{'='*60}")
print(f"  Traffic Quality Diagnostic")
print(f"  density={args.density} veh/km  |  {args.episodes} episodes  |  "
      f"{args.steps} steps/ep")
print(f"  sensor_range={SR:.0f} m  |  D0={D0}  T_HEAD={T_HEAD}  v_ref={V_REF}")
print(f"{'='*60}\n")

# Data collection

all_gF   = []    # front gap in own lane
all_gFL  = []    # front gap left lane
all_gFR  = []    # front gap right lane
all_gBL  = []    # rear gap left
all_gBR  = []    # rear gap right
all_v    = []    # ego speed
all_lane = []    # ego lane index
all_c    = []    # cooldown counter
close_calls   = 0   # gF < d_safe
danger_calls  = 0   # gF < 0.5 * d_safe
lc_ready      = 0   # c_ego == 0
total_steps   = 0
ep_lengths    = []
ep_nb_surr    = []
lane_seen     = set()

for ep in range(args.episodes):
    seed = args.seed0 + ep
    env = Highway(make_sp(args.density, seed),
                  make_rp(f"ep{ep}"),
                  use_gui=False,
                  env_label=f"tq_ep{ep}")
    try:
        obs, info = env.reset(seed=seed)
        ep_nb_surr.append(info.get("nb_surr", 0))

        done  = False
        steps = 0
        while not done and steps < args.steps:
            action = env.action_space.sample()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

            v    = float(obs[0])
            lane = int(obs[1])
            gF   = float(obs[2])
            gFL  = float(obs[4])
            gFR  = float(obs[6])
            gBL  = float(obs[8])
            gBR  = float(obs[10])
            c    = int(info.get("c_ego", 0))

            d_safe = D0 + T_HEAD * v

            all_gF.append(gF)
            all_gFL.append(gFL)
            all_gFR.append(gFR)
            all_gBL.append(gBL)
            all_gBR.append(gBR)
            all_v.append(v)
            all_lane.append(lane)
            all_c.append(c)
            lane_seen.add(lane)

            if gF < d_safe:       close_calls  += 1
            if gF < 0.5 * d_safe: danger_calls += 1
            if c == 0:            lc_ready     += 1

            steps += 1

        ep_lengths.append(steps)
        print(f"  ep={ep}  seed={seed}  steps={steps}  "
              f"nb_surr={ep_nb_surr[-1]}  "
              f"mean_gF={np.mean(all_gF[-steps:] if steps else [SR]):.1f}m  "
              f"lanes_visited={sorted(set(all_lane[-steps:]))}")
    finally:
        env.close()

    total_steps += steps

if total_steps == 0:
    print("\nERROR: No steps collected: environment may be crashing immediately.")
    sys.exit(1)

# Derived metrics
all_gF  = np.array(all_gF)
all_gFL = np.array(all_gFL)
all_gFR = np.array(all_gFR)
all_gBL = np.array(all_gBL)
all_gBR = np.array(all_gBR)
all_v   = np.array(all_v)
all_lane= np.array(all_lane)

gF_mean        = float(np.mean(all_gF))
gF_std         = float(np.std(all_gF))
gF_median      = float(np.median(all_gF))
gF_p10         = float(np.percentile(all_gF, 10))
gF_at_sr       = float(np.mean(all_gF >= SR - 0.5))   # fraction pegged at sensor range
close_rate     = close_calls  / total_steps
danger_rate    = danger_calls / total_steps
lc_ready_rate  = lc_ready     / total_steps
v_std          = float(np.std(all_v))
mean_ep_len    = float(np.mean(ep_lengths))
adj_gF_mean    = float(np.mean(np.minimum(all_gFL, all_gFR)))   # best adjacent front gap

# Lane distribution
lane_counts = {ln: int(np.sum(all_lane == ln)) for ln in range(3)}
lane_frac   = {ln: lane_counts[ln] / total_steps for ln in range(3)}

# Gap bucket dominance (bucket size = SR/5)
bucket_size  = SR / 5.0
bucket_idx   = np.clip((all_gF / bucket_size).astype(int), 0, 4)
bucket_counts= np.bincount(bucket_idx, minlength=5)
dominant_frac= float(bucket_counts.max() / total_steps)

nb_surr_mean = float(np.mean(ep_nb_surr))
target_surr  = int(round((args.density / 1000.0) *
                         float(SIM_PARAMS.get("moving_window_m", 1100.0))))

# LC opportunity: fraction of steps where BOTH front AND rear gap are safe
# in at least one adjacent lane, i.e. a lane change is physically viable.
# Uses d_safe = D0 + T_HEAD * v at each step.
d_safe_arr  = D0 + T_HEAD * all_v
left_ok     = (all_gFL > d_safe_arr) & (all_gBL > d_safe_arr)
right_ok    = (all_gFR > d_safe_arr) & (all_gBR > d_safe_arr)
lc_opp_rate = float(np.mean(left_ok | right_ok))

# Print metrics
print(f"\n{'─'*60}")
print(f"  Raw metrics  ({total_steps} total steps across {args.episodes} episodes)")
print(f"{'─'*60}")
print(f"  Surrounding vehicles spawned:  mean={nb_surr_mean:.0f}  target={target_surr}")
print(f"  Episode length:                mean={mean_ep_len:.0f}  min={min(ep_lengths)}  max={max(ep_lengths)}")
print(f"  Lanes visited by ego:          {sorted(lane_seen)}")
print(f"  Lane time fractions:           " +
      "  ".join(f"L{l}={lane_frac[l]*100:.0f}%" for l in range(3)))
print(f"")
print(f"  Front gap (own lane):")
print(f"    mean   = {gF_mean:.1f} m   (target: well below {SR:.0f} m)")
print(f"    std    = {gF_std:.1f} m   (target: > 15 m for variety)")
print(f"    median = {gF_median:.1f} m")
print(f"    p10    = {gF_p10:.1f} m   (low percentile: how tight it gets)")
print(f"    % at sensor range = {gF_at_sr*100:.1f}%  (target: < 50%)")
print(f"")
print(f"  Situation rates (per step):")
print(f"    Close calls  (gF < d_safe):         {close_rate*100:.1f}%  (target: > 20%)")
print(f"    Danger zone  (gF < 0.5*d_safe):     {danger_rate*100:.1f}%  (target: > 5%)")
print(f"    LC ready     (c_ego == 0):           {lc_ready_rate*100:.1f}%  (informational only)")
print(f"")
print(f"  Adjacent lane front gap mean:          {adj_gF_mean:.1f} m   (target: < {SR*0.75:.0f} m)")
print(f"  LC opportunity rate (both gaps safe):  {lc_opp_rate*100:.1f}%  (target: > 20%)")
print(f"  Ego speed std:                         {v_std:.2f} m/s (target: > 1 m/s)")
print(f"  Dominant gap bucket (20 m wide):       {dominant_frac*100:.1f}%  (target: < 80%)")
print(f"{'─'*60}\n")

# Checks
print("  Checks:")

check("Enough vehicles spawned (>= 80% of target)",
      nb_surr_mean >= 0.8 * target_surr,
      f"spawned={nb_surr_mean:.0f} target={target_surr}")

check("All 3 lanes have traffic (ego forced into all lanes occasionally)",
      len(lane_seen) == 3,
      f"lanes seen: {sorted(lane_seen)}")

check("Front gap mean < 75% of sensor range (traffic is close enough)",
      gF_mean < 0.75 * SR,
      f"mean={gF_mean:.1f} m  threshold={0.75*SR:.0f} m")

check("Front gap std > 15 m (situation variety)",
      gF_std > 15.0,
      f"std={gF_std:.1f} m")

check("< 50% of steps pegged at sensor range (gaps are real, not vacuum)",
      gF_at_sr < 0.50,
      f"{gF_at_sr*100:.1f}% at sensor range")

check("Close-call rate > 20% (safety cost fires regularly)",
      close_rate > 0.20,
      f"{close_rate*100:.1f}%")

check("Danger rate > 5% (strong safety gradient experienced)",
      danger_rate > 0.05,
      f"{danger_rate*100:.1f}%")

# The fraction of steps where a lane change is possible is not checked:
# a random agent asks for a lane change 2/3 of the time and sits in the
# cooldown almost always, so it would fail no matter the traffic.
print(f"  INFO  LC-ready fraction = {lc_ready_rate*100:.1f}%"
      f"  (not checked: random agent inflates cooldown artificially)")

check("Adjacent lane mean gap < 75 m (LC has real consequences)",
      adj_gF_mean < 0.75 * SR,
      f"adj_mean={adj_gF_mean:.1f} m  threshold={0.75*SR:.0f} m")

check("LC opportunity > 20% (front AND rear gap safe in at least one adjacent lane)",
      lc_opp_rate > 0.20,
      f"{lc_opp_rate*100:.1f}%")

check("Ego speed std > 1 m/s (agent not locked at constant speed)",
      v_std > 1.0,
      f"std={v_std:.2f} m/s")

check("Episodes survive > 30 steps on average (env not instantly crashing)",
      mean_ep_len > 30,
      f"mean_ep_len={mean_ep_len:.0f}")

check("No single 20 m gap bucket > 80% of obs (distribution not degenerate)",
      dominant_frac < 0.80,
      f"dominant bucket={dominant_frac*100:.1f}%")

# Recommendations
total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  {PASS}/{total + PASS - PASS} ... {PASS} passed  {FAIL} failed")
print(f"{'='*60}")

if FAIL == 0:
    print("  Traffic configuration looks good for training.\n")
else:
    print(f"\n  {FAIL} check(s) failed. Suggestions:")
    if gF_at_sr > 0.50 or gF_mean > 0.75 * SR:
        new_d = int(args.density * 1.3)
        print(f"  -> Too many steps with no vehicle in sight: try --density {new_d}")
        print(f"    Or reduce traffic_v_min so slow vehicles close the gap faster.")
    if close_rate < 0.20:
        new_d = int(args.density * 1.3)
        print(f"  -> Safety cost rarely fires: try --density {new_d}")
    if len(lane_seen) < 3:
        print(f"  -> Ego stays in one lane: LEFT_LANE_PENALTY may be too high, "
              f"or traffic so dense the agent never dares to move")
    if dominant_frac >= 0.80:
        print(f"  -> Gap distribution is degenerate: vehicles may be bunching. "
              f"Check place_clearance_m and spawning_safe_dist_m.")
    if lc_opp_rate < 0.20:
        print(f"  -> Adjacent lanes blocked too often: reduce density or increase "
              f"window_ahead_m so vehicles spread out more.")
    print()

sys.exit(0 if FAIL == 0 else 1)
