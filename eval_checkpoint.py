"""
Evaluate a saved DQN checkpoint locally, the same way as during training.

    python eval_checkpoint.py --checkpoint models/model_t16100000.zip
    python eval_checkpoint.py --checkpoint models/model_t16100000.zip --seeds 20000-20009
"""

import argparse
import copy
import os
import sys
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from params import SIM_PARAMS, ROAD_PARAMS, EVAL_HOLDOUT_SEEDS
from stable_baselines3 import DQN
from highway_env import Highway


def evaluate(model, seeds, max_steps, sim_params, road_params):
    """Run the hold-out evaluation in the main thread."""
    ep_scaled, ep_raw, ep_len = [], [], []
    ego_collision = ego_missing = 0

    rp = copy.deepcopy(road_params)
    rp["scenario_dir"] = "scenario_out_eval_local"
    eval_env = Highway(sim_params, rp, use_gui=False, env_label="eval_local")

    try:
        for s in seeds:
            obs, _ = eval_env.reset(seed=s)
            done = False
            scaled = raw = steps = 0
            last_reason = ""

            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                done = terminated or truncated
                scaled += reward
                raw    += info.get("cost_raw", 0)
                last_reason = info.get("reason", "")
                steps += 1

            if last_reason == "ego_collision": ego_collision += 1
            if last_reason == "ego_missing":   ego_missing   += 1

            ep_scaled.append(scaled)
            ep_raw.append(raw)
            ep_len.append(steps)
            print(f"  [eval] seed={s}  steps={steps}  "
                  f"reward={scaled:.4f}  reason={last_reason or 'timeout'}", flush=True)
    finally:
        try:
            eval_env.close()
        except Exception:
            pass

    ep_scaled = np.array(ep_scaled)
    ep_raw    = np.array(ep_raw)
    ep_len    = np.array(ep_len)

    stats = dict(
        scaled_mean=float(ep_scaled.mean()),
        scaled_std=float(ep_scaled.std()),
        raw_mean=float(ep_raw.mean()),
        raw_std=float(ep_raw.std()),
        len_mean=float(ep_len.mean()),
        len_std=float(ep_len.std()),
        ego_collision=int(ego_collision),
        ego_missing=int(ego_missing),
        cumulative_reward=float(ep_scaled.sum()),
    )
    return stats


def parse_seeds(s):
    """Accepts '10000-10009' or '10000,10001,10002'."""
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint locally")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .zip checkpoint file")
    parser.add_argument("--seeds", default=None,
                        help="Seed range e.g. '10000-10009' (default: EVAL_HOLDOUT_SEEDS from params.py)")
    parser.add_argument("--max_steps", type=int, default=1500,
                        help="Max steps per episode (default: 1500)")
    parser.add_argument("--warmup_s", type=float, default=2.0)
    parser.add_argument("--traffic_settle_s", type=float, default=1.0)
    parser.add_argument("--init_steps", type=int, default=5)
    args = parser.parse_args()

    if args.seeds:
        seeds = parse_seeds(args.seeds)
    else:
        seeds = list(EVAL_HOLDOUT_SEEDS)
    print(f"Seeds: {seeds}", flush=True)

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = DQN.load(args.checkpoint, device="cpu")
    print(f"  exploration_rate: {model.exploration_rate:.4f}", flush=True)

    # same settings as the evaluation during training
    sim_params = copy.deepcopy(SIM_PARAMS)
    sim_params["warmup_s"]         = args.warmup_s
    sim_params["traffic_settle_s"] = args.traffic_settle_s
    sim_params["init_steps"]       = args.init_steps
    sim_params["terminate_on_cost"] = False

    print(f"\n=== EVAL: {args.checkpoint} ===", flush=True)
    stats = evaluate(model, seeds, args.max_steps, sim_params, ROAD_PARAMS)
    print(f"\nResults:", flush=True)
    print(stats)


if __name__ == "__main__":
    main()
