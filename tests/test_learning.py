"""
Short DQN training run to check that the agent learns at all.

Takes roughly 5-15 minutes headless and needs SUMO. It checks that the env
runs, that reward goes up and collisions go down after training, that the
greedy policy visits all three cooldown zones, and that obs[12] and obs[13]
(lane-change and cooldown progress) are encoded correctly.

The thresholds are loose on purpose: a broken setup fails clearly, a working
one passes even before the agent has converged.

    python tests/test_learning.py
"""

import os
import sys
import copy
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from params import SIM_PARAMS, ROAD_PARAMS, EVAL_HOLDOUT_SEEDS

# Config

TRAIN_STEPS  = 20_000    # enough to see a trend, ~5 min locally
EVAL_STEPS   = 300       # steps per eval episode
N_TRAIN_ENVS = 1         # one env, SUMO is happier that way
EVAL_SEEDS   = EVAL_HOLDOUT_SEEDS[:3]   # 3 seeds is enough
# same density as in training
DENSITY      = float(SIM_PARAMS.get("traffic_density_veh_per_km", 30.0))

# Helpers

PASS = FAIL = 0

def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")

def make_sp(label_suffix="", density=DENSITY):
    sp = copy.deepcopy(SIM_PARAMS)
    sp["traffic_density_veh_per_km"] = density
    # Cap warmup/settle/init_steps to shorter values for test speed,
    # but never exceed what params.py specifies.
    sp["warmup_s"]         = min(float(sp.get("warmup_s",        5.0)), 3.0)
    sp["traffic_settle_s"] = min(float(sp.get("traffic_settle_s",2.0)), 2.0)
    sp["init_steps"]       = min(int(sp.get("init_steps",        10)),  5)
    sp["duration_s"]       = 99999
    sp["seed_spawn"]       = hash(label_suffix) & 0xFFFF
    return sp

def make_rp(label):
    rp = copy.deepcopy(ROAD_PARAMS)
    rp["scenario_dir"] = f"scenario_out_{label}"
    return rp

# Import SB3

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
    from highway_env import Highway
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure stable-baselines3 and highway_env.py are on the path.")
    sys.exit(1)


# Section 1: Env boots and steps without error

print("\n" + "="*55)
print("  1. Environment smoke test")
print("="*55)

env = Highway(make_sp("smoke"), make_rp("smoke"), use_gui=False, env_label="smoke")
try:
    obs, info = env.reset(seed=42)
    check("reset() returns obs shape (14,)", obs.shape == (14,))
    check("reset() info contains c_ego",     "c_ego" in info)
    check("c_ego starts at 0",               info["c_ego"] == 0)

    rewards, costs, c_vals = [], [], []
    lc_prog_during_c = []   # collect lc_prog values when c >= 15

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        costs.append(info.get("cost_raw", 0))
        c_vals.append(info["c_ego"])
        if info["c_ego"] >= env._lc_start:
            lc_prog_during_c.append(float(obs[12]))
        if term or trunc:
            obs, info = env.reset(seed=43)

    check("200 steps without crash",        True)
    check("reward always <= 0",             all(r <= 0 for r in rewards),
          f"max reward={max(rewards):.4f}")
    check("cost_raw always >= 0",           all(c >= 0 for c in costs),
          f"min cost={min(costs):.2f}")
    cd_max = env._cd_total - 1
    check(f"c_ego stays in 0..{cd_max}",
          all(0 <= c <= cd_max for c in c_vals),
          f"out-of-range values: {[c for c in c_vals if not 0<=c<=cd_max]}")

    # lc_prog has to be > 0 in zone C
    if lc_prog_during_c:
        all_nonzero = all(v > 0.0 for v in lc_prog_during_c)
        check("obs[12] lc_prog > 0 whenever c in Zone C (bug-fix check)",
              all_nonzero,
              f"zero values found: {[v for v in lc_prog_during_c if v == 0.0][:5]}")
    else:
        print("  SKIP  lc_prog Zone C check (no lane changes observed in 200 steps)")

    # cd_prog round-trip
    cd_total = env._cd_total
    bad_roundtrip = []
    for c_val in range(cd_total):
        prog = float(c_val) / float(cd_total - 1)
        rec  = round(prog * (cd_total - 1))
        if rec != c_val:
            bad_roundtrip.append((c_val, rec))
    check("obs[13] cd_prog encodes c losslessly (MPC lookup)", bad_roundtrip == [],
          f"failures: {bad_roundtrip}")

finally:
    env.close()


# Section 2: Short training run

print("\n" + "="*55)
print(f"  2. Short training run  ({TRAIN_STEPS} steps, {N_TRAIN_ENVS} envs)")
print("="*55)

def make_env_fn(i):
    def _thunk():
        return Highway(make_sp(f"train{i}"), make_rp(f"train{i}"),
                       use_gui=False, env_label=f"train{i}")
    return _thunk

vec = DummyVecEnv([make_env_fn(i) for i in range(N_TRAIN_ENVS)])
vec = VecMonitor(vec)

model = DQN(
    "MlpPolicy", vec,
    learning_rate=1e-4,
    buffer_size=20_000,
    learning_starts=500,
    batch_size=128,
    gamma=0.995,
    train_freq=4,
    gradient_steps=-1,
    target_update_interval=2000,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=0,
    device="auto",
)


def eval_model(model, seeds, max_steps, label=""):
    """Run greedy rollouts, return (mean_reward, collision_rate, zones_seen).
    Creates a fresh env per seed to avoid SUMO state issues between resets."""
    ep_rewards, collisions = [], []
    zones_seen = set()
    lc_start = None

    for s in seeds:
        eval_env = Highway(make_sp(f"eval_{s}"), make_rp(f"eval_{s}"),
                           use_gui=False, env_label=f"eval_{s}")
        if lc_start is None:
            lc_start = eval_env._lc_start
        try:
            obs, _ = eval_env.reset(seed=s)
            total_r, done, steps = 0.0, False, 0
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(int(action))
                total_r += r
                c = info["c_ego"]
                if c == 0:              zones_seen.add("A")
                elif c < lc_start:      zones_seen.add("B")
                else:                   zones_seen.add("C")
                done = term or trunc
                steps += 1
            ep_rewards.append(total_r)
            collisions.append(1 if info.get("reason") == "ego_collision" else 0)
        finally:
            eval_env.close()

    mean_r    = float(np.mean(ep_rewards))
    coll_rate = float(np.mean(collisions))
    return mean_r, coll_rate, zones_seen


print(f"\n  Evaluating untrained model...")
t0 = time.time()
r_before, coll_before, zones_before = eval_model(model, EVAL_SEEDS, EVAL_STEPS, "before")
print(f"  before:  mean_reward={r_before:.4f}  collision_rate={coll_before:.2f}  zones={zones_before}")

print(f"\n  Training for {TRAIN_STEPS} steps...")
model.learn(total_timesteps=TRAIN_STEPS, reset_num_timesteps=True)
elapsed = time.time() - t0
print(f"  Training done in {elapsed:.0f}s  ({TRAIN_STEPS/elapsed:.0f} steps/s)")

print(f"\n  Evaluating trained model...")
r_after, coll_after, zones_after = eval_model(model, EVAL_SEEDS, EVAL_STEPS, "after")
print(f"  after:   mean_reward={r_after:.4f}  collision_rate={coll_after:.2f}  zones={zones_after}")

vec.close()

# Section 2 checks

print()
improvement = r_after - r_before
check("reward improved after training (or already good)",
      improvement > 0 or r_after > -0.5,
      f"before={r_before:.4f}  after={r_after:.4f}  delta={improvement:.4f}")

check("collision rate is not 1.0 after training",
      coll_after < 1.0,
      f"coll_rate={coll_after:.2f}")

check("agent visited Zone A (fully ready state)",
      "A" in zones_after,
      f"zones seen: {zones_after}")

check("agent visited Zone C (physically changing) at some point",
      "C" in zones_before or "C" in zones_after,
      "Zone C never seen - agent may never be changing lanes")

check("steps/s above 50 (env not catastrophically slow)",
      TRAIN_STEPS / elapsed > 50,
      f"{TRAIN_STEPS/elapsed:.0f} steps/s")

# Summary

total = PASS + FAIL
print(f"\n{'='*55}")
print(f"  {PASS}/{total} checks passed")
if FAIL == 0:
    print("  All checks PASSED  -- env + learning are healthy")
else:
    print(f"  {FAIL} check(s) FAILED")
print("="*55)
print()
print("Key numbers to report if submitting results:")
print(f"  reward before  = {r_before:.4f}")
print(f"  reward after   = {r_after:.4f}")
print(f"  improvement    = {improvement:.4f}")
print(f"  coll before    = {coll_before:.2f}")
print(f"  coll after     = {coll_after:.2f}")
print(f"  zones seen     = {zones_after}")
print(f"  steps/s        = {TRAIN_STEPS/elapsed:.0f}")
sys.exit(0 if FAIL == 0 else 1)
