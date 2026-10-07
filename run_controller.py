"""
Run one episode with one of the controllers.

Controllers (--terminal_cost):
    zero     MPC with no terminal cost (J[H] = 0)
    rl       MPC with the DQN Q-values as terminal cost
    rollout  MPC with a rollout of a simple policy as terminal cost
    rl_only  the DQN policy on its own, no MPC

Examples:
    # plain MPC, H=5, with the SUMO GUI
    python run_controller.py --terminal_cost zero --horizon 5 --gui

    # MPC + learned terminal cost
    python run_controller.py --terminal_cost rl --horizon 5 \
        --checkpoint models/model_t16600000.zip

    # MPC + rollout terminal cost
    python run_controller.py --terminal_cost rollout --horizon 5 --rollout_horizon 40

    # DQN only
    python run_controller.py --terminal_cost rl_only --checkpoint models/model_t16600000.zip

    # other options
        --seed 10000      # random seed
        --steps 1500      # max episode steps
        --density 30      # traffic density veh/km
        --warmup 2.0      # SUMO warmup seconds
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import numpy as np
import csv

sys.path.insert(0, ".")

try:
    from pred_error_logger import PredErrorLogger
except ImportError:
    PredErrorLogger = None

from params import SIM_PARAMS, ROAD_PARAMS
from highway_env import Highway
from mpc_rl_utils import (
    backward_dp_numba, fill_terminal_rl, build_lane_trajs,
    nearest_leader, nearest_follower,
    initialize_terminal_cost, warmup_rollout,
    load_dqn, warmup_numba,
    dt, v_ref, v_min, v_max, a_min, a_max, D0, T_HEAD,
    W_SPEED, W_ACCEL, W_FRONT, W_BACK,
    LANE_CHANGE_PENALTY, COOLDOWN_PENALTY, LEFT_LANE_PENALTY,
    SENSOR_DIS, LANE_CHANGE_COOLDOWN, LANE_CHANGING_START,
    LANE_CHANGE_TIME, LC_START, accel_vals, v_vals, V_STEP, S_STEP,
    EGO_LENGTH, COLLISION_PENALTY, SUMO_MIN_GAP, a_break, all_traj_s_stds, all_traj_v_stds
)


def parse_args():
    p = argparse.ArgumentParser(description="Run any controller for one episode")
    p.add_argument("--terminal_cost", type=str, default="zero",
                   choices=["zero", "rl", "rollout", "rl_only"],
                   help="Terminal cost type: zero|rl|rollout|rl_only")
    p.add_argument("--horizon",         type=int,   default=5,
                   help="MPC horizon H in steps (ignored for rl_only)")
    p.add_argument("--rollout_horizon", type=int,   default=40,
                   help="Extra rollout steps beyond H (rollout only)")
    p.add_argument("--checkpoint",      type=str,   default="",
                   help="DQN checkpoint path (required for rl and rl_only)")
    p.add_argument("--rl_weight",       type=float, default=1.0,
                   help="Not used right now; the terminal cost is scaled by 0.98**H")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--steps",           type=int,   default=1500)
    p.add_argument("--density",         type=float,
                   default=float(SIM_PARAMS.get("traffic_density_veh_per_km", 30.0)))
    p.add_argument("--warmup",          type=float, default=2.0)
    p.add_argument("--verbose",         action="store_true",
                   help="Print every step (for collision analysis)")
    p.add_argument("--gui",             action="store_true",
                   help="Open SUMO GUI (local use only)")
    return p.parse_args()


def run(args=None, pred_logger=None):
    """
    Run one episode. Can be called directly or imported by a batch script.
    If args is None, parses from command line.
    Returns dict with results.
    """
    import traci
    import traci.constants as tc
    import time as _time

    if args is None:
        args = parse_args()

    tc_label = args.terminal_cost
    H        = args.horizon

    if tc_label in ("rl", "rl_only") and not args.checkpoint:
        print(f"ERROR: --checkpoint required for --terminal_cost {tc_label}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Controller:    {tc_label.upper()}")
    if tc_label != "rl_only":
        print(f"  MPC horizon:   H={H} ({H * dt:.1f}s)")
    if tc_label == "rollout":
        print(f"  Rollout steps: {args.rollout_horizon} ({args.rollout_horizon * dt:.1f}s)")
    if tc_label in ("rl", "rl_only"):
        print(f"  Checkpoint:    {args.checkpoint}")
    if tc_label == "rl":
        print(f"  RL weight:     {args.rl_weight}")
    print(f"  Seed={args.seed}  Steps={args.steps}  Density={args.density}")
    print(f"  GUI={args.gui}")
    print(f"{'='*60}\n")

    dqn_model = None
    if tc_label in ("rl", "rl_only"):
        dqn_model = load_dqn(args.checkpoint)

    # compile the numba code before the episode starts
    if tc_label != "rl_only":
        warmup_numba()
    if tc_label == "rollout":
        warmup_rollout()

    sp = copy.deepcopy(SIM_PARAMS)
    sp["traffic_density_veh_per_km"] = args.density
    sp["warmup_s"]         = args.warmup
    sp["traffic_settle_s"] = 1.0
    sp["terminate_on_cost"] = False

    rp = copy.deepcopy(ROAD_PARAMS)
    # separate scenario folder per checkpoint, so several runs can go in parallel
    ckpt_tag = ("_" + os.path.basename(args.checkpoint).replace(".zip","").replace("model_t",""))\
               if args.checkpoint else ""
    rp["scenario_dir"] = f"scenario_out_{tc_label}_H{args.horizon}{ckpt_tag}"

    env = Highway(sp, rp, use_gui=args.gui, env_label=tc_label)
    obs, info = env.reset(seed=args.seed)
    ego_id = env.ego_id

    print(f"Episode started  seed={args.seed}")
    if args.gui:
        print("  Press Play in SUMO-GUI to begin...\n")

    if tc_label != "rl_only":
        traci.switch(tc_label)

    c_ego      = 0
    l_ego_reg  = 0
    ego_initialized = False
    total_reward = 0.0
    total_cost   = 0.0
    cost_speed = cost_accel = cost_front = cost_back = 0.0
    cost_col_f = cost_col_b = cost_lc = cost_cd = cost_left = 0.0
    # timings per step, in seconds
    t_traj_list = []; t_terminal_list = []; t_dp_list = []
    t_forward_list = []; t_rl_list = []
    collisions   = 0
    steps_done   = 0
    last_reason  = ""
    csv_path = "data.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "scenario", "t",
                "s", "v", "lane",
                "a", "dl",
                "gap_front",
                "gap_front_left",
                "gap_front_right",
                "gap_back_left",
                "gap_back_right",
                "v_lead",
                "v_lead_left",
                "v_lead_right",
                "v_follow_left",
                "v_follow_right",
                "lane changing process",
                "v_cost",
                "a_cost",
                "lane_change_cost",
                "Cool_down_cost",
                "safety_cost",
                "back_safety_cost",
                "left_lane_cost",
                "collision_cost",
                "stage_cost"
            ])
        for step in range(args.steps):

            if tc_label == "rl_only":
                _t0_rl = _time.perf_counter()
                action, _ = dqn_model.predict(obs, deterministic=True)
                t_rl_list.append(_time.perf_counter() - _t0_rl)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                last_reason   = info.get("reason", "")
                if last_reason == "ego_collision":
                    collisions += 1
                cost_speed += info.get("cost_speed", 0.0)
                cost_accel += info.get("cost_accel", 0.0)
                cost_front += info.get("cost_front", 0.0)
                cost_back  += info.get("cost_back",  0.0)
                cost_col_f += info.get("cost_col_f", 0.0)
                cost_col_b += info.get("cost_col_b", 0.0)
                cost_lc    += info.get("cost_lc",    0.0)
                cost_cd    += info.get("cost_cd",    0.0)
                cost_left  += info.get("cost_left",  0.0)
                total_cost += info.get("cost_raw",   0.0)
                steps_done   += 1
                if args.verbose:
                    print(f"  step={step:4d}  v={obs[0]:.1f}  lane={int(obs[1])}  "
                        f"gF={obs[2]:.1f}  gFL={obs[4]:.1f}  gFR={obs[6]:.1f}  "
                        f"gBL={obs[8]:.1f}  gBR={obs[10]:.1f}  "
                        f"action={int(action)}  reward={reward:.3f}  "
                        f"reason={last_reason or 'ok'}", flush=True)
                elif step % 100 == 0:
                    print(f"  step={step:4d}  cumulative_reward={total_reward:.3f}", flush=True)
                if terminated or truncated:
                    break
                continue

            # MPC controllers from here on
            traci.switch(tc_label)

            if ego_id not in traci.vehicle.getIDList():
                print(f"  Ego missing at step {step}")
                break

            if not ego_initialized:
                ego_initialized = True

            s_ego   = traci.vehicle.getPosition(ego_id)[0]
            v_ego   = traci.vehicle.getSpeed(ego_id)
            l_ego   = traci.vehicle.getLaneIndex(ego_id)
            len_ego = traci.vehicle.getLength(ego_id)
            edge    = traci.vehicle.getLaneID(ego_id).split("_")[0]
            n_lanes = traci.edge.getLaneNumber(edge)
            lanes     = list(range(n_lanes))
            lanes_arr = np.array(lanes)

            # predict the other vehicles over the horizon
            traj_H = H
            _t0_traj = _time.perf_counter()
            # compare predictions from H steps ago with what actually happened
            if pred_logger is not None:
                pred_logger.flush_actuals(args.seed, step, ego_id)
            final_trajs = build_lane_trajs(ego_id, traj_H, dt, lanes)
            t_traj_list.append(_time.perf_counter() - _t0_traj)
            if pred_logger is not None:
                pred_logger.record_prediction(
                    args.seed, step, s_ego, v_ego, l_ego, len_ego, lanes, final_trajs
                )
            num_vehs    = len(final_trajs)

            n_arr         = max(1, num_vehs)
            all_traj_s    = np.zeros((n_arr, H + 1), dtype=np.float64)
            all_traj_v    = np.zeros((n_arr, H + 1), dtype=np.float64)
            all_traj_l    = np.zeros((n_arr, H + 1), dtype=np.int32)
            all_traj_len  = np.full(n_arr, EGO_LENGTH, dtype=np.float64)
            all_traj_amin = np.full(n_arr, -2.0,        dtype=np.float64)
            all_traj_vmax = np.full(n_arr, v_max,        dtype=np.float64)
            all_traj_amax = np.full(n_arr, a_max,        dtype=np.float64)

            for i, (vid, traj_tuple) in enumerate(final_trajs.items()):
                ts, tv, tl = traj_tuple[0], traj_tuple[1], traj_tuple[2]
                vlen, vamin = traj_tuple[3], traj_tuple[4]
                all_traj_s[i, :]   = ts
                all_traj_v[i, :]   = tv
                all_traj_l[i, :]   = tl
                all_traj_len[i]    = vlen
                all_traj_amin[i]   = vamin
                if len(traj_tuple) > 5:
                    all_traj_vmax[i] = traj_tuple[5]
                if len(traj_tuple) > 6:
                    all_traj_amax[i] = traj_tuple[6]


            s_vals = np.arange(s_ego, s_ego + v_max * H * dt + 2 * S_STEP, S_STEP)
            n_s = len(s_vals)
            n_v = len(v_vals)
            n_c = LANE_CHANGE_COOLDOWN

            J = np.full((H + 1, n_s, n_lanes, n_v, n_c), np.inf)

            # terminal cost J[H]
            _t0_terminal = _time.perf_counter()
            if tc_label == "zero":
                J[H, :, :, :, :] = 0.0

            elif tc_label == "rl":
                J[H, :, :, :, :] = 0.0
                fill_terminal_rl(J[H], s_vals, lanes, dqn_model, final_trajs,
                                H, s_ego, v_ego, rl_weight=0.98**H)

            elif tc_label == "rollout":
                # same discount (0.98 per step) as the stage costs
                terminal_cost_weight = 0.98 ** H
                initialize_terminal_cost(
                    J, s_vals, v_vals, lanes_arr, H, args.rollout_horizon,
                    all_traj_s[:, :H + 1],
                    all_traj_v[:, :H + 1],
                    all_traj_l[:, :H + 1],
                    all_traj_len, all_traj_amin,
                    all_traj_vmax, all_traj_amax,
                    s_ego, v_ego, S_STEP, V_STEP, c_ego, len_ego,
                    terminal_cost_weight,
                )
            t_terminal_list.append(_time.perf_counter() - _t0_terminal)

            _t0_dp = _time.perf_counter()
            J = backward_dp_numba(
                J, v_ref, s_vals, v_vals, lanes_arr, accel_vals,
                all_traj_s[:, :H + 1],
                all_traj_v[:, :H + 1],
                all_traj_l[:, :H + 1],
                all_traj_len,
                all_traj_s_stds, all_traj_v_stds,
                s_ego, v_ego, l_ego_reg, c_ego, len_ego, H, dt,
                a_min, a_max, v_min, v_max, V_STEP, S_STEP,
                D0, T_HEAD, W_FRONT, W_BACK, W_SPEED, W_ACCEL,
                LANE_CHANGE_PENALTY, LANE_CHANGING_START, LEFT_LANE_PENALTY,
                COOLDOWN_PENALTY, COLLISION_PENALTY,
            )
            t_dp_list.append(_time.perf_counter() - _t0_dp)

            # Forward step
            _t0_fwd = _time.perf_counter()
            if c_ego < LANE_CHANGING_START - 1:
                l_ego_reg = l_ego

            best_cost = np.inf; best_a = 0.0; best_dl = 0
            best_c_next = max(0, c_ego - 1); best_lane = l_ego_reg
            best_stage = 0.0; best_si_log = 0; best_vi_log = 0; best_ci_log = 0
            best_cs = 0.0; best_ca = 0.0; best_sf = 0.0; best_sb = 0.0
            best_col_f = 0.0; best_col_b = 0.0; best_lc = 0.0; best_cd = 0.0; best_cl = 0.0

            for dl in [0, -1, 1]:
                l_next = l_ego_reg + dl
                if l_next not in lanes:
                    continue
                if c_ego == 0:
                    lc_cost = LANE_CHANGE_PENALTY if dl != 0 else 0.0
                    cd_cost = 0.0
                    c_next  = LANE_CHANGE_COOLDOWN - 1 if dl != 0 else 0
                elif c_ego > LANE_CHANGING_START - 1:
                    if dl != 0: continue
                    lc_cost = cd_cost = 0.0; c_next = c_ego - 1
                else:
                    if dl != 0:
                        lc_cost = LANE_CHANGE_PENALTY; cd_cost = COOLDOWN_PENALTY
                        c_next  = LANE_CHANGE_COOLDOWN - 1
                    else:
                        lc_cost = cd_cost = 0.0; c_next = c_ego - 1

                for a in accel_vals:
                    v_next = float(np.clip(v_ego + a * dt, v_min, v_max))
                    s_next = s_ego + v_ego * dt + 0.5 * a * dt**2
                    si = int(np.clip(int((s_next - s_vals[0]) / S_STEP), 0, n_s - 1))
                    vi = int(np.clip(int((v_next - v_vals[0]) / V_STEP), 0, n_v - 1))
                    ci = int(np.clip(c_next, 0, n_c - 1))

                    gF, _  = nearest_leader(final_trajs, l_next, s_ego)
                    gB, vB = nearest_follower(final_trajs, l_next, s_ego, len_ego)

                    col_f = COLLISION_PENALTY if gF <= 0.0 else 0.0
                    d_sf  = D0 + T_HEAD * v_ego
                    sf    = W_FRONT * (d_sf - gF)**2 if gF < d_sf else 0.0

                    col_b = sb = 0.0
                    if dl != 0 and vB is not None:
                        col_b = COLLISION_PENALTY if gB <= max((vB- v_ego) ** 2 / (2 * a_break), 0) else 0.0
                        d_sb  = D0 + T_HEAD * vB
                        sb    = W_BACK * (d_sb - gB)**2 if gB < d_sb else 0.0

                    left_cost = LEFT_LANE_PENALTY * l_ego_reg if l_ego_reg > 0 else 0.0
                    stage = (W_SPEED * (v_ego - v_ref)**2 + W_ACCEL * a**2
                            + sf + sb + col_f + col_b + lc_cost + cd_cost + left_cost)
                    total = stage + J[1, si, l_next, vi, ci]

                    if total < best_cost:
                        best_cost   = total; best_a = a; best_dl = dl
                        best_c_next = c_next; best_lane = l_next
                        best_stage  = stage
                        best_si_log = si; best_vi_log = vi; best_ci_log = ci
                        best_cs = W_SPEED * (v_ego - v_ref)**2
                        best_ca = W_ACCEL * a**2
                        best_sf = sf; best_sb = sb
                        best_col_f = col_f; best_col_b = col_b
                        best_lc = lc_cost; best_cd = cd_cost
                        best_cl = left_cost

            t_forward_list.append(_time.perf_counter() - _t0_fwd)
            if best_dl != 0:
                traci.vehicle.changeLane(ego_id, l_ego_reg + best_dl, LANE_CHANGE_TIME)
            traci.vehicle.setAcceleration(ego_id, best_a, 1.0)
            c_ego     = best_c_next
            l_ego_reg = best_lane
            writer.writerow([
                    args.seed,
                    step,
                    s_ego,
                    v_ego,
                    l_ego,
                    best_a,
                    best_dl,
                    nearest_leader(final_trajs, l_ego, s_ego, 0)[0],
                    nearest_leader(final_trajs, l_ego+1, s_ego, 0)[0] if l_ego+1 in lanes else np.inf,
                    nearest_leader(final_trajs, l_ego-1, s_ego, 0)[0] if l_ego-1 in lanes else np.inf,
                    nearest_follower(final_trajs, l_ego+1, s_ego, len_ego, 0)[0] if l_ego+1 in lanes else np.inf,
                    nearest_follower(final_trajs, l_ego-1, s_ego, len_ego, 0)[0] if l_ego-1 in lanes else np.inf,
                    nearest_leader(final_trajs, l_ego, s_ego, 0)[1],
                    nearest_leader(final_trajs, l_ego+1, s_ego, 0)[1] if l_ego+1 in lanes else None,
                    nearest_leader(final_trajs, l_ego-1, s_ego, 0)[1] if l_ego-1 in lanes else None,
                    nearest_follower(final_trajs, l_ego+1, s_ego, len_ego, 0)[1] if l_ego+1 in lanes else None,
                    nearest_follower(final_trajs, l_ego-1, s_ego, len_ego, 0)[1] if l_ego-1 in lanes else None,
                    c_ego,
                    best_cs,
                    best_ca,
                    best_lc,
                    best_cd,
                    best_sf,
                    best_sb,
                    best_cl,
                    best_col_f + best_col_b,
                    best_stage
                    ])

            # bookkeeping, using the cost terms of the chosen action
            total_cost   += best_stage
            total_reward -= best_stage * 1e-5
            cost_speed   += best_cs
            cost_accel   += best_ca
            cost_front   += best_sf
            cost_back    += best_sb
            cost_col_f   += best_col_f
            cost_col_b   += best_col_b
            cost_lc      += best_lc
            cost_cd      += best_cd
            cost_left    += best_cl

            traci.simulationStep()
            env._step_count += 1

            if ego_id in traci.simulation.getCollidingVehiclesIDList():
                collisions += 1
                last_reason  = "ego_collision"
                total_cost  += COLLISION_PENALTY
                cost_col_f  += COLLISION_PENALTY
                if best_stage < COLLISION_PENALTY * 0.5:
                    total_reward += -1000.0
                print(f"\n  *** COLLISION at step {step} ***")
                break

            env.maintain_traffic_pool()  # same order as env.step()

            if traci.simulation.getMinExpectedNumber() == 0:
                break

            steps_done += 1
            if args.verbose:
                zone = "A" if c_ego == 0 else ("C" if c_ego >= LC_START else "B")
                gF_log, _ = nearest_leader(final_trajs, l_ego_reg, s_ego)
                gBL_log, _ = nearest_follower(final_trajs, l_ego_reg + 1, s_ego, len_ego) if l_ego_reg + 1 < 3 else (100.0, None)
                gBR_log, _ = nearest_follower(final_trajs, l_ego_reg - 1, s_ego, len_ego) if l_ego_reg - 1 >= 0 else (100.0, None)
                print(f"  step={step:4d}  v={v_ego:.1f}  lane={l_ego}  "
                    f"gF={gF_log:.1f}  gBL={gBL_log:.1f}  gBR={gBR_log:.1f}  "
                    f"a={best_a:+.1f}  dl={best_dl:+d}  c={c_ego}({zone})  "
                    f"stage={best_stage:.1f}", flush=True)
            elif step % 50 == 0:
                zone = "A" if c_ego == 0 else ("C" if c_ego >= LC_START else "B")
                print(f"  step={step:4d}  v={v_ego:.1f}m/s  lane={l_ego}"
                    f"  gF={gF:.0f}m  c={c_ego:2d}({zone})"
                    f"  a={best_a:+.1f}  dl={best_dl:+d}", flush=True)


    if pred_logger is not None:
        pred_logger.flush_remaining(args.seed)

    n = max(1, steps_done)
    print(f"\n{'='*60}")
    print(f"  Controller:          {tc_label.upper()}")
    print(f"  Steps:               {n}")
    print(f"  Cumul. reward:       {total_reward:.4f}   (RL scale, comparable across controllers)")
    if tc_label != "rl_only":
        print(f"  Mean cost/step:      {total_cost/n:.4f}")
        print(f"  --- Cost breakdown ---")
        print(f"  Speed cost/step:     {cost_speed/n:.4f}")
        print(f"  Accel cost/step:     {cost_accel/n:.4f}")
        print(f"  Front safety/step:   {cost_front/n:.4f}")
        print(f"  Back safety/step:    {cost_back/n:.4f}")
        print(f"  Front collision/step:{cost_col_f/n:.4f}")
        print(f"  Back collision/step: {cost_col_b/n:.4f}")
        print(f"  Lane change/step:    {cost_lc/n:.4f}")
        print(f"  Cooldown/step:       {cost_cd/n:.4f}")
        print(f"  Left lane/step:      {cost_left/n:.4f}")
    if tc_label == "rl_only":
        if t_rl_list:
            arr = np.array(t_rl_list) * 1000
            print(f"  --- Timing (ms/step) ---")
            print(f"  RL forward pass:     {arr.mean():.3f} ± {arr.std():.3f}")
    else:
        print(f"  --- Timing (ms/step) ---")
        def _fmt(lst):
            if not lst: return "n/a"
            a = np.array(lst) * 1000
            return f"{a.mean():.3f} ± {a.std():.3f}"
        print(f"  Traj prediction:     {_fmt(t_traj_list)}")
        print(f"  Terminal cost:       {_fmt(t_terminal_list)}")
        print(f"  Backward DP:         {_fmt(t_dp_list)}")
        print(f"  Forward step:        {_fmt(t_forward_list)}")
        total_ctrl = (np.array(t_traj_list) + np.array(t_terminal_list) +
                      np.array(t_dp_list)   + np.array(t_forward_list))
        print(f"  Total controller:    {total_ctrl.mean()*1000:.3f} ± {total_ctrl.std()*1000:.3f}")
    print(f"  Collisions:          {collisions}")
    print(f"{'='*60}")

    env.close()

    return dict(
        controller    = tc_label,
        horizon       = H,
        seed          = args.seed,
        steps         = n,
        total_reward  = total_reward,
        total_cost    = total_cost,
        collisions    = collisions,
        last_reason   = last_reason,
        cost_speed    = cost_speed,
        cost_accel    = cost_accel,
        cost_front    = cost_front,
        cost_back     = cost_back,
        cost_col_f    = cost_col_f,
        cost_col_b    = cost_col_b,
        cost_lc       = cost_lc,
        cost_cd       = cost_cd,
        cost_left     = cost_left,
        t_traj_ms     = float(np.mean(t_traj_list)     * 1000) if t_traj_list     else 0.0,
        t_terminal_ms = float(np.mean(t_terminal_list)  * 1000) if t_terminal_list  else 0.0,
        t_dp_ms       = float(np.mean(t_dp_list)        * 1000) if t_dp_list        else 0.0,
        t_forward_ms  = float(np.mean(t_forward_list)   * 1000) if t_forward_list   else 0.0,
        t_rl_ms       = float(np.mean(t_rl_list)        * 1000) if t_rl_list        else 0.0,
    )


if __name__ == "__main__":
    run()
