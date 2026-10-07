from __future__ import annotations

import numpy as np
import torch
from numba import njit, prange


dt                   = 0.2
v_ref                = 25.0
v_min, v_max         = 0.0, 33.33
a_min, a_max         = -2.0, 2.0
D0, T_HEAD           = 5.0, 2.0
SUMO_MIN_GAP         = 2.5
W_SPEED              = 1.0
W_ACCEL              = 1.0
W_FRONT              = 1000.0
W_BACK               = 1000.0
LANE_CHANGE_PENALTY  = 10.0
COOLDOWN_PENALTY     = 30.0
LEFT_LANE_PENALTY    = 2.0
SENSOR_DIS           = 150.0
EGO_LENGTH           = 12.0
COLLISION_PENALTY    = 1e8

LANE_CHANGE_COOLDOWN = int(5.0 / dt)          # 25
LANE_CHANGE_STEPS    = int(2.0 / dt)           # 10
LANE_CHANGING_START  = LANE_CHANGE_COOLDOWN - LANE_CHANGE_STEPS  # 15
LANE_CHANGE_TIME     = 2
a_break = 2
accel_vals = np.arange(a_min, a_max + 1e-9, 1.0)
v_vals     = np.arange(v_min, v_max + 1e-9, 1.0 * dt)
V_STEP     = 1.0 * dt
S_STEP     = 1.0 * dt


CD_TOTAL = 25
LC_TOTAL = 10
LC_START = 15

# Std of the position/speed prediction error per neighbour slot and step,
# measured with pred_error_logger.py. They are passed into the DP for an
# optional chance-constrained margin (s +- 2.33 sigma), but that margin is
# turned off in the version used for the paper.
all_traj_s_stds = np.array([[0.5, 0.509, 0.538, 0.585, 0.645, 0.719],  # front
                            [0.5, 0.511, 0.539, 0.584, 0.646, 0.719],  # front_left
                            [0.5, 0.512, 0.540, 0.582, 0.646, 0.722],  # front_right
                            [0.5, 0.510, 0.540, 0.586, 0.649, 0.722],  # back_left
                            [0.5, 0.510, 0.542, 0.584, 0.644, 0.717],  # back_right
                            ],dtype=np.float64)
all_traj_v_stds = np.array([
         [0.5, 0.500, 0.515, 0.524, 0.546, 0.570],  # front
         [0.5, 0.503, 0.513, 0.527, 0.553, 0.583],  # front_left
         [0.5, 0.505, 0.511, 0.524, 0.543, 0.572],  # front_right
         [0.5, 0.500, 0.514, 0.535, 0.560, 0.599],  # back_left
         [0.5, 0.499, 0.513, 0.529, 0.556, 0.578],  # back_right
         ],dtype=np.float64)


@njit(parallel=True)
def backward_dp_numba(
    J, v_ref, s_vals, v_vals, lanes, accel_vals,
    all_traj_s, all_traj_v, all_traj_l, all_traj_lengths, traj_s_stds, traj_v_stds,
    s_ego=0.0, v_ego=0.0, l_ego=0, c_ego=0, len_ego=12.0, H=1, dt=0.2,
    a_min=-2.0, a_max=2.0, v_min=0.0, v_max=33.33, V_STEP=0.2, S_STEP=0.2,
    D0=5.0, T_HEAD=2.0, W_FRONT=1000.0, W_BACK=1000.0, W_SPEED=1.0, W_ACCEL=1.0,
    LANE_CHANGE_PENALTY=10.0, LANE_CHANGING_START=15, LEFT_LANE_PENALTY=2.0,
    COOLDOWN_PENALTY=30.0, EGO_LENGTH=12.0, COLLISION_PENALTY=1e8
):
    """
    Backward dynamic programming over the grid (s, lane, v, cooldown).

    J[H] must already hold the terminal cost. Going from h = H-1 down to 0,
    each cell gets the best stage cost + cost-to-go over all accelerations
    and lane moves, using the predicted trajectories of the other vehicles
    at step h. Only cells the ego can actually reach are filled.
    """
    n_s    = len(s_vals)
    n_v    = len(v_vals)
    n_l    = len(lanes)
    n_vehs = all_traj_s.shape[0]

    for h in range(H - 1, -1, -1):
        s_max = s_ego + h * v_ego * dt + 0.5 * a_max * dt * dt * h * h + 2 * S_STEP
        s_min = s_ego + h * v_ego * dt + 0.5 * a_min * dt * dt * h * h - 2 * S_STEP
        J_next = J[h + 1]
        J_curr = J[h]

        for si in prange(n_s):
            s = s_vals[si]
            if s > s_max or s < s_min:
                continue
            if h == 0 and s != s_ego:
                continue

            for li in range(n_l):
                l = lanes[li]
                if h == 0 and l != l_ego:
                    continue

                v_max_reach = v_ego + h * a_max * dt + 2 * V_STEP
                v_min_reach = v_ego + h * a_min * dt - 2 * V_STEP

                for vi in range(n_v):
                    v = v_vals[vi]
                    if v > v_max_reach or v < v_min_reach:
                        continue
                    if h == 0 and v != v_ego:
                        continue

                    for ci in range(LANE_CHANGE_COOLDOWN):
                        if h == 0 and ci != c_ego:
                            continue
                        if ci < c_ego - h:
                            continue

                        best = np.inf

                        for dl in (-1, 0, 1):
                            l_next = l + dl
                            if l_next < 0 or l_next >= n_l:
                                continue

                            if ci == 0:
                                if dl != 0:
                                    c_next = LANE_CHANGE_COOLDOWN - 1
                                    lc_cost = LANE_CHANGE_PENALTY
                                    cd_cost = 0.0
                                else:
                                    c_next = 0
                                    lc_cost = 0.0
                                    cd_cost = 0.0
                            elif ci > LANE_CHANGING_START - 1:
                                if dl != 0:
                                    continue
                                c_next = ci - 1
                                lc_cost = 0.0
                                cd_cost = 0.0
                            else:
                                if dl != 0:
                                    c_next = LANE_CHANGE_COOLDOWN - 1
                                    lc_cost = LANE_CHANGE_PENALTY
                                    cd_cost = COOLDOWN_PENALTY
                                else:
                                    c_next = ci - 1
                                    lc_cost = 0.0
                                    cd_cost = 0.0

                            for a in accel_vals:
                                s_next = s + v * dt + 0.5 * a * dt * dt
                                v_next = v + a * dt
                                if v_next < v_min: v_next = v_min
                                if v_next > v_max: v_next = v_max

                                next_si = int((s_next - s_vals[0]) / S_STEP)
                                if next_si < 0:    next_si = 0
                                if next_si >= n_s: next_si = n_s - 1

                                next_vi = int((v_next - v_vals[0]) / V_STEP)
                                if next_vi < 0:    next_vi = 0
                                if next_vi >= n_v: next_vi = n_v - 1

                                min_front_gap = np.inf
                                min_back_gap  = np.inf
                                v_back = 0.0

                                # nearest vehicles ahead/behind in the target lane at step h
                                for i in range(n_vehs):
                                    if all_traj_l[i, h] == l_next:
                                        s_other  = all_traj_s[i, h]
                                        v_other  = all_traj_v[i, h]
                                        veh_len  = all_traj_lengths[i]
                                        s_rear   = s_other - veh_len
                                        if s_other > s:
                                            gap = s_rear - s
                                            if gap < min_front_gap:
                                                min_front_gap = gap
                                        elif s_other < s:
                                            gap = s - len_ego - s_other
                                            if gap < min_back_gap:
                                                min_back_gap = gap
                                                v_back = v_other

                                d_safe_front = D0 + T_HEAD * v
                                safety    = 0.0
                                collision = 0.0
                                if min_front_gap <= 0.0:
                                    collision += COLLISION_PENALTY
                                if min_front_gap < d_safe_front:
                                    safety += W_FRONT * (d_safe_front - min_front_gap)**2
                                if dl != 0:
                                    d_safe_back = D0 + T_HEAD * v_back
                                    if min_back_gap <= max((v_back- v) ** 2 / (2 * a_break), 0):
                                        collision += COLLISION_PENALTY
                                    if min_back_gap < d_safe_back:
                                        safety += W_BACK * (d_safe_back - min_back_gap)**2

                                left_lane_cost = LEFT_LANE_PENALTY * l if l > 0 else 0.0

                                cost = 0.98 ** h * (W_SPEED * (v - v_ref)**2
                                        + W_ACCEL * a * a
                                        + safety
                                        + collision
                                        + lc_cost + cd_cost
                                        + left_lane_cost) + J_next[next_si, li + dl, next_vi, c_next]  # li+dl works since lanes = [0, 1, 2]

                                if cost < best:
                                    best = cost

                        J_curr[si, li, vi, ci] = best

    return J


# Same scale as in highway_env (reward = -cost * REWARD_SCALE). Q-values are
# divided by this to bring them back to the units of the MPC stage cost.
REWARD_SCALE = 1e-5

#                next_c.add(c - 1)
#                next_c.add(LANE_CHANGE_COOLDOWN - 1)


def fill_terminal_rl(J_H, s_vals, lanes, dqn_model, final_trajs, H, s_ego, v_ego,
                     rl_weight: float = 1.0, skip_c: bool = False):
    """
    Fill the terminal cost table J_H from the DQN.

    Only (s, v) cells inside a box around what the ego can reach in H steps
    are evaluated, which keeps the DQN batch small. Cells outside the box
    keep whatever the caller initialised them to (zeros).
    """
    n_s, n_l, n_v, n_c = J_H.shape
    sr      = SENSOR_DIS
    n_lanes = len(lanes)

    # box of (s, v) values the ego can reach after H steps, plus a small margin
    s_max   = s_ego + H * v_ego * dt + 0.5 * a_max * dt**2 * H**2 + 2 * S_STEP
    s_min   = s_ego + H * v_ego * dt + 0.5 * a_min * dt**2 * H**2 - 2 * S_STEP
    v_max_r = min(v_max, v_ego + H * a_max * dt + 2 * V_STEP)
    v_min_r = max(v_min, v_ego + H * a_min * dt - 2 * V_STEP)

    s_mask  = (s_vals >= s_min) & (s_vals <= s_max)
    v_mask  = (v_vals >= v_min_r) & (v_vals <= v_max_r)
    s_arr   = s_vals[s_mask]
    v_arr   = v_vals[v_mask]
    s_idx   = np.where(s_mask)[0]
    v_idx   = np.where(v_mask)[0]

    if len(s_arr) == 0 or len(v_arr) == 0:
        return J_H

    ns_r = len(s_arr)
    c_reach = [0] if skip_c else list(range(n_c))

    # gaps to the surrounding vehicles at step H, per lane
    fg = np.full((n_lanes, ns_r), sr)
    fv = np.zeros((n_lanes, ns_r))
    bg = np.full((n_lanes, ns_r), sr)
    bv = np.zeros((n_lanes, ns_r))

    for li, lane in enumerate(lanes):
        for vid, (traj_s, traj_v, traj_l, veh_len, *_) in final_trajs.items():
            if len(traj_s) <= H:
                continue
            if traj_l[H] != lane:
                continue
            sv_h   = float(traj_s[H])
            v_at_H = float(traj_v[H])
            gaps_f   = np.where(sv_h > s_arr, sv_h - s_arr - veh_len, np.inf)
            better_f = gaps_f < fg[li]
            fg[li]   = np.where(better_f, gaps_f, fg[li])
            fv[li]   = np.where(better_f, v_at_H, fv[li])
            gaps_b   = np.where(sv_h < s_arr, s_arr - sv_h - EGO_LENGTH, np.inf)
            better_b = gaps_b < bg[li]
            bg[li]   = np.where(better_b, gaps_b, bg[li])
            bv[li]   = np.where(better_b, v_at_H, bv[li])

    # one DQN observation per grid cell
    obs_list = []
    idx_list = []

    for sj, si in enumerate(s_idx):
        for li, lane in enumerate(lanes):
            gF   = fg[li, sj]
            gFL  = fg[li+1, sj] if li+1 < n_lanes else sr
            gFR  = fg[li-1, sj] if li-1 >= 0      else sr
            gBL  = bg[li+1, sj] if li+1 < n_lanes else sr
            gBR  = bg[li-1, sj] if li-1 >= 0      else sr
            vFv  = fv[li, sj]
            vFL  = fv[li+1, sj] if li+1 < n_lanes else 0.0
            vFR  = fv[li-1, sj] if li-1 >= 0      else 0.0
            vBLv = bv[li+1, sj] if li+1 < n_lanes else 0.0
            vBRv = bv[li-1, sj] if li-1 >= 0      else 0.0

            for vj, vi in enumerate(v_idx):
                v_t  = v_arr[vj]
                dvF  = (vFv  - v_t) if gF  < sr else 0.0
                dvFL = (vFL  - v_t) if gFL < sr else 0.0
                dvFR = (vFR  - v_t) if gFR < sr else 0.0
                dvBL = (vBLv - v_t) if gBL < sr else 0.0
                dvBR = (vBRv - v_t) if gBR < sr else 0.0

                for ci in c_reach:
                    lc_prog = float(ci - LC_START + 1) / LC_TOTAL if ci >= LC_START else 0.0
                    cd_prog = float(ci) / max(1, CD_TOTAL - 1)
                    obs_list.append([v_t, float(lane),
                                     gF, dvF, gFL, dvFL, gFR, dvFR,
                                     gBL, dvBL, gBR, dvBR,
                                     lc_prog, cd_prog])
                    idx_list.append((si, li, vi, ci))

    if not obs_list:
        return J_H

    obs_t = torch.tensor(np.array(obs_list, dtype=np.float32))
    with torch.no_grad():
        q_vals = dqn_model.q_net(obs_t).numpy()
    costs = -np.max(q_vals, axis=1) / REWARD_SCALE * rl_weight

    if skip_c:
        for k, (si, li, vi, _) in enumerate(idx_list):
            J_H[si, li, vi, :] = costs[k]
    else:
        for k, (si, li, vi, ci) in enumerate(idx_list):
            J_H[si, li, vi, ci] = costs[k]

    return J_H


@njit
def v_safe_sumo(gap: float, v_lead: float, a_min_veh: float) -> float:
    """Krauss safe speed, the default car-following model in SUMO."""
    b   = abs(a_min_veh)
    tau = 1.0
    g   = max(0.0, gap - SUMO_MIN_GAP)
    return -b * tau + np.sqrt((b * tau) ** 2 + v_lead ** 2 + 2.0 * b * g)


@njit
def v_safe_ego(gap: float, v_lead: float, a_min_ego: float) -> float:
    """Safe speed for the ego with the D0 + T_HEAD * v spacing used in the cost."""
    b = abs(a_min_ego)
    g = max(0.0, gap - D0)
    return max(0.0, -b * T_HEAD + np.sqrt(max(0.0, (b * T_HEAD)**2 + v_lead**2 + 2.0 * b * g)))


def sumo_policy(s, v, l, a_max, a_min, v_max, v_ref, s_map, v_map, len_map, dt):
    """
    Simple prediction model for the other vehicles: keep the lane and
    follow the leader with the Krauss safe speed. Position is updated
    with the current speed, s_next = s + v * dt.
    """
    g_f_curr = 1e9
    v_f_curr = -1.0
    for s_other, v_other, len_other in zip(
        s_map.get(l, []), v_map.get(l, []), len_map.get(l, [])
    ):
        if s_other > s:
            gap = s_other - len_other - s
            if gap < g_f_curr:
                g_f_curr = gap
                v_f_curr = v_other

    v_allowed = v_ref if v_f_curr < 0 else v_safe_sumo(g_f_curr, v_f_curr, a_min)
    v_target  = min(v_ref, v_allowed)
    accel     = float(np.clip((v_target - v) / dt, a_min, a_max))
    v_next    = float(np.clip(v + accel * dt, 0.0, v_max))
    s_next    = s + v * dt
    return s_next, v_next, l     # lane unchanged


def build_lane_trajs(ego_id: str, H: int, dt_step: float, lanes: list) -> dict:
    """
    Predict the next H steps of the vehicles around the ego with sumo_policy.

    A vehicle that is in the middle of a lane change is counted in both
    lanes: we add a "ghost" copy of it in the target lane, so the ego
    keeps its distance in both until the manoeuvre is done.

    Returns a dict {vid: (traj_s, traj_v, traj_l, length, a_min, v_max, a_max)}.
    """
    import traci
    import traci.constants as tc
    import numpy as np

    LAT_THRESHOLD = 0.2

    results = traci.vehicle.getContextSubscriptionResults(ego_id)
    if not results:
        return {}

    vehicles = []

    for vid, data in results.items():
        try:
            lat_dev = float(data.get(tc.VAR_LANEPOSITION_LAT, 0.0))
            l = int(data[tc.VAR_LANE_INDEX])
            is_changing_left  = lat_dev > LAT_THRESHOLD
            is_changing_right = lat_dev < -LAT_THRESHOLD
            is_changing = is_changing_left or is_changing_right

            if is_changing_left:
                target_lane = l + 1
            elif is_changing_right:
                target_lane = l - 1
            else:
                target_lane = l

            # keep the target lane inside the road
            s_noisy = float(data[tc.VAR_POSITION][0])
            v_noisy = float(data[tc.VAR_SPEED])
            a_noisy = float(data[tc.VAR_ACCELERATION])
            vehicles.append({
                "id": vid,
                "s": s_noisy,
                "v": v_noisy,
                "l": int(data[tc.VAR_LANE_INDEX]),
                "a_max":  float(data[tc.VAR_ACCEL]),
                "a_min":  -abs(float(data[tc.VAR_DECEL])),
                "a_curr": a_noisy,

                "v_max": float(data.get(tc.VAR_MAXSPEED, v_max)),
                "v_ref": float(data.get(tc.VAR_MAXSPEED, v_max)),
                "length": float(data.get(tc.VAR_LENGTH, 4.8)),

                "traj_s": [s_noisy],
                "traj_v": [v_noisy],
                "traj_l": [int(data[tc.VAR_LANE_INDEX])],
                "is_ghost":     False,
                "is_changing":  is_changing,
                "target_lane":  target_lane,
            })
        except KeyError:
            continue
    # ghost copies for vehicles that are changing lane
    ghosts_to_add = []
    for veh in vehicles:
        if not veh["is_changing"]:
            continue
        tgt = veh["target_lane"]
        ghost = {
            "id":           veh["id"] + "_ghost",
            "s":            veh["s"],
            "v":            veh["v"],
            "l":            tgt,           # ghost starts in target lane
            "a_max":        veh["a_max"],
            "a_min":        veh["a_min"],
            "v_max":        veh["v_max"],
            "v_ref":        veh["v_ref"],
            "a_curr":       veh["a_curr"],
            "length":       veh["length"],
            "traj_s":       [veh["s"]],
            "traj_v":       [veh["v"]],
            "traj_l":       [tgt],
            "is_ghost":     True,
            "is_changing":  False,
            "target_lane":  tgt,
        }
        ghosts_to_add.append(ghost)

    vehicles.extend(ghosts_to_add)

    for _ in range(H):

        s_map   = {}
        v_map   = {}
        len_map = {}

        for veh in vehicles:
            s_map.setdefault(veh["l"],   []).append(veh["s"])
            v_map.setdefault(veh["l"],   []).append(veh["v"])
            len_map.setdefault(veh["l"], []).append(veh["length"])

        for veh in vehicles:
            s_n = veh["s"] + veh["v"] * dt_step
            if (veh["v"] + veh["a_curr"] * dt_step) < 0:
                v_n = 0
            elif (veh["v"] + veh["a_curr"] * dt_step) > veh["v_max"]:
                v_n = veh["v_max"]
            else:
                v_n = veh["v"] + veh["a_curr"] * dt_step
            l_n = veh["l"]

            veh["s"] = s_n
            veh["v"] = v_n
            veh["l"] = l_n
            veh["traj_s"].append(s_n)
            veh["traj_v"].append(v_n)
            veh["traj_l"].append(l_n)

    final_trajs = {}

    for veh in vehicles:
        if veh["id"] == ego_id or veh["id"] == ego_id + "_ghost":
            continue

        final_trajs[veh["id"]] = (
            np.array(veh["traj_s"], dtype=np.float64),
            np.array(veh["traj_v"], dtype=np.float64),
            np.array(veh["traj_l"], dtype=np.int32),
            float(veh["length"]),
            float(veh["a_min"]),
            float(veh["v_max"]),
            float(veh["a_max"]),
        )

    return final_trajs


@njit(parallel=True)
def initialize_terminal_cost(
    J, s_vals, v_vals, lanes, H, H_ROLL_OUT,
    all_traj_s, all_traj_v, all_traj_l,
    all_traj_lengths, all_traj_a_min, all_traj_v_max, all_traj_a_max,
    s_ego, v_ego, S_STEP, V_STEP, c_ego, len_ego,
    terminal_cost_weight,
):
    """
    Rollout terminal cost: for every reachable cell, simulate the ego and
    the traffic for H_ROLL_OUT more steps and use the accumulated cost.
    The result is scaled by terminal_cost_weight (H / H_ROLL_OUT by default).
    """
    n_s = len(s_vals)
    n_l = len(lanes)
    n_v = len(v_vals)
    n_vehs = all_traj_s.shape[0]

    s_max = s_ego + H * v_ego * dt + 0.5 * a_max * dt * dt * H * H + 2 * S_STEP
    s_min = s_ego + H * v_ego * dt + 0.5 * a_min * dt * dt * H * H - 2 * S_STEP

    for si in range(n_s):
        if s_vals[si] > s_max or s_vals[si] < s_min:
            continue
        for li in range(n_l):
            v_max_reach = v_ego + H * a_max * dt + 2 * V_STEP
            v_min_reach = v_ego + H * a_min * dt - 2 * V_STEP
            for vi in range(n_v):
                if v_vals[vi] > v_max_reach or v_vals[vi] < v_min_reach:
                    continue
                for ci in prange(LANE_CHANGE_COOLDOWN):
                    if ci < c_ego - H:
                        continue
                    J[H, si, li, vi, ci] = terminal_cost_weight * rollout_policy(
                        s_vals[si], v_vals[vi], lanes[li], ci, len_ego,
                        H_ROLL_OUT,
                        all_traj_s, all_traj_v, all_traj_l,
                        all_traj_lengths, all_traj_a_min,
                        all_traj_v_max, all_traj_a_max,
                        lanes,
                    )


@njit
def rollout_policy(
    s, v, l, c, length, H_total,
    all_traj_s, all_traj_v, all_traj_l,
    all_traj_lengths, all_traj_a_min,
    all_traj_v_max, all_traj_a_max,
    lanes,
):
    """
    Simulate ego and traffic together for H_total steps and return the
    summed stage cost. The ego uses a greedy rule (change lane when it is
    safe and faster), the others follow their leader with Krauss.
    """
    total_cost = 0.0
    n_vehs = all_traj_s.shape[0]
    n_l = len(lanes)
    ego_idx = n_vehs  # ego is stored last

    all_s    = np.zeros(n_vehs + 1)
    all_v    = np.zeros(n_vehs + 1)
    all_l    = np.zeros(n_vehs + 1, dtype=np.int64)
    all_len  = np.zeros(n_vehs + 1)
    all_amin = np.zeros(n_vehs + 1)
    all_amax = np.zeros(n_vehs + 1)
    all_vmax = np.zeros(n_vehs + 1)

    lane_leaders_s = np.full(n_l, -np.inf)
    is_front_edge = np.zeros(n_vehs, dtype = np.bool_)

    for i in range(n_vehs):
        all_s[i]    = all_traj_s[i, -1]
        all_v[i]    = all_traj_v[i, -1]
        all_l[i]    = all_traj_l[i, -1]
        all_len[i]  = all_traj_lengths[i]
        all_amin[i] = all_traj_a_min[i]
        all_amax[i] = all_traj_a_max[i]
        all_vmax[i] = all_traj_v_max[i]
        cur_lane = int(all_l[i])
        if all_s[i] > lane_leaders_s[cur_lane]:
            lane_leaders_s[cur_lane] = all_s[i]

    for i in range(n_vehs):
        if all_s[i] == lane_leaders_s[int(all_l[i])]:
            is_front_edge[i] = True

    all_s[ego_idx]    = s
    all_v[ego_idx]    = v
    all_l[ego_idx]    = l
    all_len[ego_idx]  = length
    all_amin[ego_idx] = a_min
    all_amax[ego_idx] = a_max
    all_vmax[ego_idx] = v_ref

    curr_c = c
    next_s = np.zeros(n_vehs + 1)
    next_v = np.zeros(n_vehs + 1)

    for h in range(H_total):
        step_cost = 0
        next_l_ego = all_l[ego_idx]

        # gap in the ego's own lane
        g_f_curr = 1e9
        v_f_curr = -1.0
        for i in range(n_vehs):
            if all_l[i] == all_l[ego_idx] and all_s[i] > all_s[ego_idx]:
                gap = all_s[i] - all_len[i] - all_s[ego_idx]
                if gap < g_f_curr:
                    g_f_curr = gap
                    v_f_curr = all_v[i]

        if v_f_curr < 0.0:
            v_allowed_curr = v_ref
        else:
            if g_f_curr < 0.0:
                step_cost += COLLISION_PENALTY
            d_safe = D0 + T_HEAD * all_v[ego_idx]
            if g_f_curr < d_safe:
                step_cost += W_FRONT * (d_safe - g_f_curr)**2
            v_allowed_curr = v_safe_ego(g_f_curr, v_f_curr, a_min)

        if all_l[ego_idx] > 0:
            step_cost += all_l[ego_idx] * LEFT_LANE_PENALTY

        v_allowed_next = np.inf

        # change lane if it is safe and lets the ego go faster
        if curr_c <= LANE_CHANGING_START - 1:
            for dl in (1, -1):   # try left before right
                l_next = all_l[ego_idx] + dl
                if l_next < 0 or l_next >= n_l:
                    continue

                g_f_n = 1e9; v_f_n = -1.0
                g_b_n = 1e9; v_b_n = -1.0
                a_min_b = a_min

                for i in range(n_vehs):
                    if all_l[i] == l_next:
                        if all_s[i] > all_s[ego_idx]:
                            gap = all_s[i] - all_len[i] - all_s[ego_idx]
                            if gap < g_f_n:
                                g_f_n = gap
                                v_f_n = all_v[i]
                        else:
                            gap = all_s[ego_idx] - all_len[ego_idx] - all_s[i]
                            if gap < g_b_n:
                                g_b_n = gap
                                v_b_n = all_v[i]
                                a_min_b = all_amin[i]

                if v_f_n < 0.0:
                    v_allowed_next = v_ref
                else:
                    v_allowed_next = v_safe_ego(g_f_n, v_f_n, a_min)

                is_better   = v_allowed_next > v_allowed_curr and v_allowed_curr < v_ref
                is_safe_b   = (v_b_n < 0.0 or
                               v_b_n <= v_safe_ego(g_b_n, all_v[ego_idx], a_min_b))
                is_safe_f   = all_v[ego_idx] <= v_allowed_next

                if is_safe_f and is_safe_b and is_better:
                    if curr_c > 0:
                        step_cost += COOLDOWN_PENALTY
                    if g_f_n < 0.0:
                        step_cost += COLLISION_PENALTY
                    if g_b_n < max((v_b_n - all_v[ego_idx]) ** 2 / (2 * a_break), 0):
                        step_cost += COLLISION_PENALTY
                    if v_b_n >= 0.0:
                        d_sb = D0 + T_HEAD * v_b_n
                        if g_b_n < d_sb:
                            step_cost += W_BACK * (d_sb - g_b_n)**2
                    curr_c    = LANE_CHANGE_COOLDOWN - 1
                    next_l_ego = l_next
                    step_cost += LANE_CHANGE_PENALTY
                    break
                else:
                    v_allowed_next = np.inf
        else:
            curr_c -= 1

        # move everyone one step
        for i in range(n_vehs + 1):
            if i == ego_idx:
                v_safe = min(v_allowed_curr, v_allowed_next)
                v_target = min(v_ref, v_safe)
            elif is_front_edge[i] == True:
                v_target = all_v[i]
            else:
                g_f = 1e9; v_f = -1.0
                for j in range(n_vehs + 1):
                    if i == j:
                        continue
                    if all_l[j] == all_l[i] and all_s[j] > all_s[i]:
                        gap = all_s[j] - all_len[j] - all_s[i]
                        if gap < g_f:
                            g_f = gap
                            v_f = all_v[j]
                v_safe   = all_vmax[i] if v_f < 0.0 else v_safe_sumo(g_f, v_f, all_amin[i])
                v_target = min(all_vmax[i], v_safe)

            accel_h = (v_target - all_v[i]) / dt
            if accel_h < all_amin[i]: accel_h = all_amin[i]
            if accel_h > all_amax[i]: accel_h = all_amax[i]
            next_v[i] = max(0.0, min(all_vmax[i], all_v[i] + accel_h * dt))
            next_s[i] = all_s[i] + all_v[i] * dt

            if i == ego_idx:
                step_cost += W_SPEED * (next_v[i] - v_ref)**2 + W_ACCEL * accel_h**2
                total_cost += step_cost * 0.98 ** h

        all_s[:] = next_s[:]
        all_v[:] = next_v[:]
        all_l[ego_idx] = next_l_ego

    return total_cost


def warmup_rollout():
    """Compile the rollout code once so the first real step is not slow."""
    print("  Warming up rollout JIT...", end="", flush=True)
    _J  = np.full((2, 5, 3, 10, 25), np.inf)
    _s  = np.linspace(0, 10, 5)
    _v  = np.linspace(0, 5, 10)
    _l  = np.array([0, 1, 2])
    _ts = np.zeros((2, 3))
    _tv = np.ones((2, 3)) * 5.0
    _tl = np.zeros((2, 3), dtype=np.int32)
    _tlen = np.full(2, 4.8)
    _tamin = np.full(2, -4.5)
    _tvmax = np.full(2, 33.33)
    _tamax = np.full(2, 2.6)
    initialize_terminal_cost(
        _J, _s, _v, _l, 1, 3,
        _ts, _tv, _tl, _tlen, _tamin, _tvmax, _tamax,
        0.0, 5.0, S_STEP, V_STEP, 0, 12.0, 1.0/3.0,
    )
    print(" done.", flush=True)


def nearest_leader(final_trajs: dict, lane: int, s: float, h: int = 0):
    """Gap and speed of the closest vehicle ahead in `lane` at step h."""
    best_gap, best_v = np.inf, None
    for vid, (traj_s, traj_v, traj_l, veh_len, *_) in final_trajs.items():
        if traj_l[h] == lane and traj_s[h] > s:
            gap = traj_s[h] - veh_len - s   # bumper to bumper, can be negative
            if gap < best_gap:
                best_gap = gap
                best_v   = float(traj_v[h])
    return best_gap, best_v


def nearest_follower(final_trajs: dict, lane: int, s: float,
                     len_ego: float, h: int = 0):
    """Gap and speed of the closest vehicle behind in `lane` at step h."""
    best_gap, best_v = np.inf, None
    s_ego_rear = s - len_ego
    for vid, (traj_s, traj_v, traj_l, _veh_len, *_) in final_trajs.items():
        if traj_l[h] == lane and traj_s[h] < s:
            gap = s_ego_rear - traj_s[h]   # bumper to bumper, can be negative
            if gap < best_gap:
                best_gap = gap
                best_v   = float(traj_v[h])
    return best_gap, best_v


def load_dqn(path: str):
    from stable_baselines3 import DQN
    if path.endswith(".zip"):
        path = path[:-4]
    model = DQN.load(path, device="cpu")
    print(f"  Loaded DQN: {path}.zip  (exploration_rate={model.exploration_rate:.4f})")
    return model


def warmup_numba():
    """Compile the DP kernel once at startup."""
    print("  Warming up Numba JIT...", end="", flush=True)
    _lanes    = np.array([0, 1, 2])
    _n_vehs   = 3
    _ns       = 5
    _J        = np.zeros((2, _ns, 3, len(v_vals), LANE_CHANGE_COOLDOWN))
    _s        = np.linspace(0.0, 4.0 * S_STEP, _ns)
    _traj_s   = np.zeros((_n_vehs, 2), dtype=np.float64)
    _traj_v   = np.zeros((_n_vehs, 2), dtype=np.float64)
    _traj_l   = np.zeros((_n_vehs, 2), dtype=np.int32)
    _traj_len = np.full(_n_vehs, 4.8)
    backward_dp_numba(_J, v_ref, _s, v_vals, _lanes, accel_vals,
                      _traj_s, _traj_v, _traj_l, _traj_len,all_traj_s_stds, all_traj_v_stds,
                      0.0, 25.0, 1, 0, EGO_LENGTH, 1, dt,
                      a_min, a_max, v_min, v_max, V_STEP, S_STEP,
                      D0, T_HEAD, W_FRONT, W_BACK, W_SPEED, W_ACCEL,
                      LANE_CHANGE_PENALTY, LANE_CHANGING_START, LEFT_LANE_PENALTY,
                      COOLDOWN_PENALTY, EGO_LENGTH, COLLISION_PENALTY)
    print(" done.", flush=True)
