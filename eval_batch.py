"""
Evaluate a controller over many seeds and write the results to CSV.

Calls run() from run_controller.py in the same process, so the JIT warmup
and the DQN loading only happen once.

Measurement noise is seeded with the episode seed, so results are
reproducible.

Examples (stochastic setting):
    # MPC without terminal cost, H=5
    python eval_batch.py --terminal_cost zero --horizon 5 --seeds 20000-20099

    # MPC + learned terminal cost, H=3
    python eval_batch.py --terminal_cost rl --horizon 3 \
        --checkpoint models/model_t16100000.zip --seeds 20000-20099

    # DQN only
    python eval_batch.py --terminal_cost rl_only \
        --checkpoint models/model_t16100000.zip --seeds 20000-20099

    # MPC + rollout terminal cost
    python eval_batch.py --terminal_cost rollout --horizon 3 \
        --rollout_horizon 20 --seeds 20000-20099

Add --setting deterministic to run without noise and margins.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pred_error_logger import PredErrorLogger
except ImportError:
    PredErrorLogger = None

from params import EVAL_FINAL_SEEDS


def parse_seeds(s):
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal_cost", required=True,
                        choices=["zero", "rl", "rollout", "rl_only"])
    parser.add_argument("--horizon",          type=int,   default=5)
    parser.add_argument("--rollout_horizon",  type=int,   default=40)
    parser.add_argument("--checkpoint",       type=str,   default="")
    parser.add_argument("--rl_weight",        type=float, default=1.0)
    parser.add_argument("--setting",          type=str,   default="stochastic",
                        choices=["stochastic", "deterministic"])
    parser.add_argument("--seeds",            type=str,   default="20000-20099")
    parser.add_argument("--max_steps",        type=int,   default=1500)
    parser.add_argument("--density",          type=float, default=None)
    parser.add_argument("--warmup",           type=float, default=2.0)
    parser.add_argument("--output",           type=str,   default="",
                        help="Optional path to save results as .npy dict")
    parser.add_argument("--pred_errors",      type=str,   default="",
                        help="If set, log trajectory prediction errors to this CSV")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    print(f"\n=== BATCH EVAL: {args.terminal_cost.upper()}  H={args.horizon}  {args.setting}  "
          f"{len(seeds)} seeds ({seeds[0]}..{seeds[-1]}) ===\n", flush=True)

    from run_controller import run as run_episode, warmup_rollout
    from mpc_rl_utils import warmup_numba, load_dqn

    if args.terminal_cost != "rl_only":
        warmup_numba()
    if args.terminal_cost == "rollout":
        warmup_rollout()

    dqn_model = None
    if args.checkpoint:
        dqn_model = load_dqn(args.checkpoint)

    ep_rewards, ep_steps, ep_collisions = [], [], []
    cost_keys = ["cost_speed","cost_accel","cost_front","cost_back",
                 "cost_col_f","cost_col_b","cost_lc","cost_cd","cost_left"]
    cost_sums = {k: 0.0 for k in cost_keys}
    total_collisions = 0
    timing_keys = ["t_traj_ms","t_terminal_ms","t_dp_ms","t_forward_ms","t_rl_ms"]
    timing_lists = {k: [] for k in timing_keys}
    t0 = time.time()

    pred_logger = None
    if args.pred_errors and PredErrorLogger is not None:
        pred_logger = PredErrorLogger(args.horizon, 0.2, args.pred_errors)
        print(f"  Prediction error logging -> {args.pred_errors}", flush=True)

    for i, seed in enumerate(seeds):
        ep_args = argparse.Namespace(
            terminal_cost    = args.terminal_cost,
            horizon          = args.horizon,
            rollout_horizon  = args.rollout_horizon,
            checkpoint       = args.checkpoint,
            rl_weight        = args.rl_weight,
            setting          = args.setting,
            seed             = seed,
            steps            = args.max_steps,
            density          = args.density if args.density else 30.0,
            warmup           = args.warmup,
            gui              = False,
            verbose          = False,
        )

        np.random.seed(int(seed))  # same measurement-noise draws for this seed on every run
        try:
            result = run_episode(ep_args, pred_logger=pred_logger)
        except Exception as e:
            print(f"  [ERROR] seed={seed}: {e}", flush=True)
            continue

        ep_rewards.append(result["total_reward"])
        ep_steps.append(result["steps"])
        # in rl_only mode a crash only shows up as a -1000 reward
        seed_collided = result["collisions"] > 0 or result["total_reward"] <= -999.0
        ep_collisions.append(1 if seed_collided else 0)
        if seed_collided:
            total_collisions += 1
        for k in cost_keys:
            cost_sums[k] += result.get(k, 0.0)
        for k in timing_keys:
            v = result.get(k, 0.0)
            if v > 0:
                timing_lists[k].append(v)

        print(f"  seed={seed}  steps={result['steps']}  "
              f"reward={result['total_reward']:.4f}  "
              f"collisions={result['collisions']}  "
              f"reason={result['last_reason'] or 'timeout'}", flush=True)

    elapsed = time.time() - t0
    ep_rewards = np.array(ep_rewards)
    ep_steps   = np.array(ep_steps)
    n = max(len(seeds), 1)

    print(f"\n{'='*60}")
    print(f"=== RESULTS: {args.terminal_cost.upper()} H={args.horizon} "
          f"({len(seeds)} seeds, {elapsed:.0f}s) ===")
    print(dict(
        terminal_cost     = args.terminal_cost,
        horizon           = args.horizon,
        checkpoint        = os.path.basename(args.checkpoint) if args.checkpoint else "none",
        scaled_mean       = float(ep_rewards.mean()),
        scaled_std        = float(ep_rewards.std()),
        len_mean          = float(ep_steps.mean()),
        len_std           = float(ep_steps.std()),
        ego_collision     = total_collisions,
        n_seeds           = len(seeds),
        elapsed_s         = round(elapsed),
    ))
    print(f"\n  --- Mean cost/step breakdown ---")
    total_steps = ep_steps.sum()
    for k in cost_keys:
        print(f"  {k:<20} {cost_sums[k]/max(total_steps,1):.4f}")
    print(f"  --- Mean timing (ms/step) ---")
    for k in timing_keys:
        vals = timing_lists[k]
        if vals:
            arr = np.array(vals)
            print(f"  {k:<20} {arr.mean():.3f} ± {arr.std():.3f}")
    print(f"{'='*60}")

    ckpt_name = os.path.basename(args.checkpoint).replace(".zip","") if args.checkpoint else "none"
    total_steps = ep_steps.sum()

    # one row per seed
    seed_csv = args.output if args.output else (
        f"results_{args.terminal_cost}_H{args.horizon}_{ckpt_name}_per_seed.csv")
    with open(seed_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "cumulative_reward", "steps", "collision"])
        for i, s in enumerate(seeds[:len(ep_rewards)]):
            writer.writerow([s, round(float(ep_rewards[i]), 4),
                             int(ep_steps[i]), int(ep_collisions[i])])

    if pred_logger is not None:
        pred_logger.close()
        print(f"  Prediction errors saved -> {args.pred_errors}", flush=True)

    print(f"\nPer-seed CSV saved to {seed_csv}")

    # and one row with the averages
    summary_csv = seed_csv.replace("_per_seed.csv", "_summary.csv")
    summary_row = dict(
        terminal_cost    = args.terminal_cost,
        horizon          = args.horizon,
        checkpoint       = ckpt_name,
        n_seeds          = len(seeds),
        scaled_mean      = round(float(ep_rewards.mean()), 4),
        scaled_std       = round(float(ep_rewards.std()),  4),
        len_mean         = round(float(ep_steps.mean()),   2),
        len_std          = round(float(ep_steps.std()),    2),
        ego_collision    = total_collisions,
        collision_rate   = round(total_collisions / max(len(seeds), 1), 4),
        elapsed_s        = round(elapsed),
    )
    total_cost_per_step = sum(cost_sums[k] for k in cost_keys) / max(total_steps, 1)
    summary_row["total_cost_per_step"] = round(total_cost_per_step, 6)
    for k in cost_keys:
        summary_row[k + "_per_step"] = round(cost_sums[k] / max(total_steps, 1), 6)
    for k in timing_keys:
        vals = timing_lists[k]
        summary_row[k + "_mean"] = round(float(np.mean(vals)), 4) if vals else 0.0
        summary_row[k + "_std"]  = round(float(np.std(vals)),  4) if vals else 0.0

    write_header = not os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(f"Summary CSV saved to {summary_csv}")


if __name__ == "__main__":
    main()
