from __future__ import annotations

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import traci
import traci.constants as tc
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*Could not connect.*")
_warnings.filterwarnings("ignore", message=".*TraCI server.*")
from sumolib import checkBinary

from scenario.build_highway import build_oneway_highway_3lane


_EGO_VARS = [
    tc.VAR_SPEED,
    tc.VAR_LANE_INDEX,
    tc.VAR_LANEPOSITION,
    tc.VAR_ROAD_ID,
    tc.VAR_LANEPOSITION_LAT
]

_CTX_VARS = [
    tc.VAR_LANEPOSITION,
    tc.VAR_POSITION,
    tc.VAR_SPEED,
    tc.VAR_LANE_INDEX,
    tc.VAR_LENGTH,
    tc.VAR_ACCEL,
    tc.VAR_DECEL,
    tc.VAR_MAXSPEED,
    tc.VAR_ACCELERATION,
    tc.VAR_LANEPOSITION_LAT


]


_SURR_VARS = [
    tc.VAR_LANEPOSITION,
]


class Highway(gym.Env):
    """
    Three-lane one-way highway in SUMO, seen from the ego truck.

    Action: one of 5 accelerations x 3 lane moves (left, keep, right), 15 in total.
    Observation (14 values): ego speed and lane, then gap and relative speed
    to the closest vehicle in front, front-left, front-right, back-left and
    back-right, and finally the lane-change progress and cooldown progress.
    Reward is minus the MPC stage cost, scaled by 1e-5, so the DQN and the
    MPC optimise the same thing.

    Only traffic in a window around the ego is simulated; vehicles are
    removed and re-inserted to keep the density roughly constant.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_params: dict,
        road_params: dict | None = None,
        use_gui: bool = False,
        env_label: str = "main",
    ):
        super().__init__()

        self.sim  = dict(sim_params)
        self.road = dict(road_params or {})
        self.use_gui   = use_gui
        self.env_label = env_label

        self.edge_id   = str(self.sim.get("spawn_edge_id",  "e0"))
        self.route_id  = str(self.sim.get("spawn_route_id", "r0"))
        self.n_lanes   = int(self.road.get("n_lanes",   3))
        self.length_m  = float(self.road.get("length_m", 30000.0))
        self.speed_mps = float(self.road.get("speed_mps", 20.0))

        self.sim_step   = float(self.sim.get("sim_step_length", 0.2))
        self.duration_s = float(self.sim.get("duration_s", 22000))
        self.init_steps = int(self.sim.get("init_steps", 10))

        self.sensor_range_m    = float(self.sim.get("sensor_range_m",    100.0))
        self.spawn_exclusion_m = float(self.sim.get("spawn_exclusion_m",   0.0))

        self.traffic_settle_s = float(self.sim.get("traffic_settle_s", 8.0))
        self.warmup_s         = float(self.sim.get("warmup_s", self.traffic_settle_s))
        self._t_episode0      = 0.0

        self.ego_id               = str(self.sim.get("ego_id", "ego"))
        self.ego_speed_mode       = int(self.sim.get("ego_speed_mode", 0))
        self._ego_start_pos_fixed = float(self.sim.get("ego_start_pos_m", 800.0))
        self._randomize_ego_start = bool(self.sim.get("randomize_ego_start", False))
        self._ego_start_margin_m  = float(self.sim.get("ego_start_margin_m", 1000.0))
        self.ego_start_pos_m      = self._ego_start_pos_fixed  # changes every reset() if randomised

        self.LANE_CHANGE_TIME       = float(self.sim.get("LANE_CHANGE_TIME",        2.0))
        self.LANE_CHANGE_COOLDOWN_S = float(self.sim.get("LANE_CHANGE_COOLDOWN_S",  5.0))

        self._lc_total = max(1, int(round(self.LANE_CHANGE_TIME        / self.sim_step)))  # 10
        self._cd_total = max(1, int(round(self.LANE_CHANGE_COOLDOWN_S  / self.sim_step)))  # 25
        self._lc_start = self._cd_total - self._lc_total                                   # 15

        self._c_ego      = 0
        self._step_count = 0

        self.window_behind_m    = float(self.sim.get("window_behind_m",    200.0))
        self.window_ahead_m     = float(self.sim.get("window_ahead_m",    1100.0))
        self.window_m           = self.window_behind_m + self.window_ahead_m
        self.window_buffer_m    = float(self.sim.get("window_buffer_m",     30.0))
        self.end_margin         = float(self.sim.get("edge_end_margin_m",  600.0))

        self.density_veh_per_km    = float(self.sim.get("traffic_density_veh_per_km", 80.0))
        self.nb_surr               = max(0, int(round((self.density_veh_per_km / 1000.0) * self.window_m)))
        self.max_replace_per_step  = int(self.sim.get("max_replacements_per_step", 10))
        self.reroute_every_n_steps = int(self.sim.get("reroute_every_n_steps", 5))

        self.place_clearance_m = float(self.sim.get("place_clearance_m", 10.0))

        self.car_type    = str(self.sim.get("spawn_car_type",  "car"))
        self.truck_type  = str(self.sim.get("spawn_truck_type", "truck"))
        self.truck_ratio = float(self.sim.get("truck_ratio", 0.1))

        self.v_ref         = float(self.sim.get("v_ref",         25.0))
        self.traffic_v_min = float(self.sim.get("traffic_v_min", 20.0))
        self.traffic_v_max = float(self.sim.get("traffic_v_max", 30.0))
        self.v_min         = float(self.sim.get("v_min",          0.0))
        self.v_max         = float(self.sim.get("v_max",         33.33))

        self.W_SPEED  = float(self.sim.get("W_SPEED",  1.0))
        self.W_SPEED_RL = float(self.sim.get("rl_speed_weight", self.W_SPEED))
        self.W_ACCEL  = float(self.sim.get("W_ACCEL",  1.0))
        self.W_FRONT  = float(self.sim.get("W_FRONT",  1000.0))
        self.W_BACK   = float(self.sim.get("W_BACK",   1000.0))

        self.LANE_CHANGE_PENALTY = float(self.sim.get("LANE_CHANGE_PENALTY", 10.0))
        self.COOLDOWN_PENALTY    = float(self.sim.get("COOLDOWN_PENALTY",    30.0))
        self.LEFT_LANE_PENALTY   = float(self.sim.get("LEFT_LANE_PENALTY",    2.0))

        self.D0     = float(self.sim.get("D0",      5.0))
        self.T_HEAD = float(self.sim.get("T_HEAD",  2.0))

        self.terminate_on_cost        = bool(self.sim.get("terminate_on_cost",         False))
        self.cost_terminate_threshold = float(self.sim.get("cost_terminate_threshold", 1e7))
        self.reward_scale             = float(self.sim.get("reward_scale",             1e-5))
        self.ego_missing_reward       = float(self.sim.get("ego_missing_reward",      -100.0))
        self.collision_penalty        = float(self.sim.get("collision_cost",           1e8))
        self.ego_length_m             = float(self.sim.get("ego_length_m",            12.0))
        self.collision_reward         = -(self.collision_penalty * self.reward_scale)
        self.terminate_on_collision   = bool(self.sim.get("terminate_on_collision",    True))

        self.lock_traffic_lane_changes = bool(self.sim.get("lock_traffic_lane_changes", True))

        self._rng = np.random.default_rng(int(self.sim.get("seed_spawn", 0)))

        accel_vals = self.sim.get("accel_vals", None)
        if accel_vals is None:
            n_acc      = int(self.sim.get("n_accel", 5))
            a_max      = float(self.sim.get("a_max",  2.0))
            accel_vals = np.linspace(-a_max, a_max, n_acc).tolist()
        self.accel_vals = [float(a) for a in accel_vals]
        if not self.accel_vals:
            raise ValueError("accel_vals is empty; cannot build action space.")

        self._a_min = float(min(self.accel_vals))
        self._a_max = float(max(self.accel_vals))

        self.lane_deltas  = [-1, 0, 1]
        self.action_space = spaces.Discrete(len(self.accel_vals) * len(self.lane_deltas))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        scenario_out = self.road.get(
            "scenario_dir", os.path.join(os.getcwd(), "scenario_out")
        )
        paths = build_oneway_highway_3lane(
            out_dir=scenario_out,
            length_m=self.length_m,
            speed_mps=self.speed_mps,
            sim_step=self.sim_step,
            duration_s=int(self.duration_s),
            car_vph=int(self.sim.get("car_vph",   0)),
            truck_vph=int(self.sim.get("truck_vph", 0)),
            ego_depart_s=float(self.sim.get("ego_depart_s", 0.0)),
        )
        self.sumocfg_path = paths["sumocfg"]

        collision_action = str(self.sim.get("collision_action",  "warn"))
        time_to_teleport = str(self.sim.get("time_to_teleport",  -1))
        self._sumo_common = [
            "--start", "true",
            "--step-length", str(self.sim_step),
            "--collision.action",    collision_action,
            "--time-to-teleport",    str(time_to_teleport),
        ]

        sumo_binary = checkBinary("sumo-gui" if self.use_gui else "sumo")
        traci.start(
            [sumo_binary, "-c", self.sumocfg_path, *self._sumo_common],
            label=self.env_label,
        )

        try:
            import socket as _sock
            traci.getConnection(self.env_label)._socket.settimeout(30.0)
        except Exception:
            pass

        self.surr_ids = [f"surr.{i}" for i in range(self.nb_surr)]

        self._ego_edge       = self.edge_id
        self._lane_v_cap     = {}
        self._ctx_sub_active = False
        self._last_reinsert  = {}
        self._reinsert_cooldown = int(self.sim.get("reinsert_cooldown_steps", 25))

    def _lane_id(self, edge: str, lane: int) -> str:
        return f"{edge}_{lane}"

    def _clamp_x(self, x: float) -> float:
        return float(np.clip(x, self.end_margin, self.length_m - self.end_margin))

    def _outside_exclusion(self, x: float, ego_x: float) -> bool:
        excl = float(self.spawn_exclusion_m)
        return True if excl <= 0.0 else abs(float(x) - float(ego_x)) >= excl

    def _get_ego_edge(self) -> str:
        return self._ego_edge

    def _refresh_ego_edge(self) -> None:
        for _ in range(5):
            try:
                road_id = traci.vehicle.getRoadID(self.ego_id)
                if road_id and not road_id.startswith(":"):
                    self._ego_edge = road_id
                    return
                lane_id_now = traci.vehicle.getLaneID(self.ego_id)
                if lane_id_now and not lane_id_now.startswith(":") and "_" in lane_id_now:
                    self._ego_edge = lane_id_now.rsplit("_", 1)[0]
                    return
            except traci.TraCIException:
                pass
            traci.simulationStep()

    def _cache_lane_speeds(self) -> None:
        edge = self._ego_edge
        self._lane_v_cap = {}
        for ln in range(self.n_lanes):
            try:
                v_lane = float(traci.lane.getMaxSpeed(self._lane_id(edge, ln)))
            except Exception:
                v_lane = float(self.speed_mps)
            self._lane_v_cap[ln] = min(self.v_max, v_lane)

    def _wait_for_ego(self, max_steps: int) -> None:
        for _ in range(max_steps):
            if self.ego_id in traci.vehicle.getIDList():
                return
            traci.simulationStep()
        raise RuntimeError(f"Ego '{self.ego_id}' did not spawn within {max_steps} steps.")

    def _window_bounds(self, ego_x: float):
        B = float(self.window_buffer_m)
        win_start    = max(self.end_margin, ego_x - self.window_behind_m)
        win_end      = min(self.length_m - self.end_margin, ego_x + self.window_ahead_m)
        back_target  = float(np.clip(win_start + B, self.end_margin, self.length_m - self.end_margin))
        front_target = float(np.clip(win_end   - B, self.end_margin, self.length_m - self.end_margin))
        return win_start, win_end, back_target, front_target

    def _sample_spawn_speed(self, lane: int) -> float:
        v_lo = max(0.1, float(self.traffic_v_min))
        v_hi = min(float(self.traffic_v_max), self._lane_v_cap.get(lane, self.speed_mps))
        return float(v_hi) if v_hi <= v_lo + 1e-9 else float(self._rng.uniform(v_lo, v_hi))

    def _sample_speed_ahead(self) -> float:
        """Speed for a vehicle spawned ahead of the ego, between v_min and v_ref."""
        v_lo = max(0.1, float(self.traffic_v_min))
        v_hi = float(self.v_ref)
        return float(self._rng.uniform(v_lo, v_hi))

    def _sample_speed_behind(self) -> float:
        """Speed for a vehicle spawned behind the ego, between v_ref and v_max."""
        v_lo = float(self.v_ref)
        v_hi = float(self.traffic_v_max)
        return float(v_lo if v_hi <= v_lo + 1e-9 else self._rng.uniform(v_lo, v_hi))

    def _choose_vtype(self) -> str:
        return self.truck_type if (self._rng.random() < self.truck_ratio) else self.car_type

    def _lock_surr_lane_change(self, vid: str) -> None:
        if not self.lock_traffic_lane_changes:
            return
        try:
            traci.vehicle.setLaneChangeMode(vid, 0)
        except traci.TraCIException:
            pass

    def _set_ego_fixed_start(self) -> None:
        if self.ego_id not in traci.vehicle.getIDList():
            return
        try:
            lane_idx = int(np.clip(int(traci.vehicle.getLaneIndex(self.ego_id)), 0, self.n_lanes - 1))
        except traci.TraCIException:
            lane_idx = 0
        x0 = self._clamp_x(float(self.ego_start_pos_m))
        try:
            traci.vehicle.moveTo(self.ego_id, self._lane_id(self._ego_edge, lane_idx), float(x0))
        except traci.TraCIException:
            pass

    def _setup_subscriptions(self) -> None:
        try:
            traci.vehicle.subscribe(self.ego_id, _EGO_VARS)
            traci.vehicle.subscribeContext(
                self.ego_id,
                tc.CMD_GET_VEHICLE_VARIABLE,
                self.sensor_range_m,
                _CTX_VARS,
            )
            self._ctx_sub_active = True
        except traci.TraCIException:
            self._ctx_sub_active = False

        active_ids = set(traci.vehicle.getIDList())
        for vid in self.surr_ids:
            if vid not in active_ids:
                continue
            try:
                traci.vehicle.subscribe(vid, _SURR_VARS)
            except traci.TraCIException:
                pass

    def _ego_state_from_sub(self):
        try:
            res = traci.vehicle.getSubscriptionResults(self.ego_id)
            if res:
                v_ego = float(res[tc.VAR_SPEED])
                lane  = int(res[tc.VAR_LANE_INDEX])
                ego_x = float(res[tc.VAR_LANEPOSITION])
                return v_ego, lane, ego_x
        except (traci.TraCIException, KeyError, TypeError):
            pass
        v_ego = float(traci.vehicle.getSpeed(self.ego_id))
        lane  = int(traci.vehicle.getLaneIndex(self.ego_id))
        ego_x = float(traci.vehicle.getLanePosition(self.ego_id))
        return v_ego, lane, ego_x

    def _build_lane_gaps(self, ego_x: float, ego_lane: int):
        sr = float(self.sensor_range_m)
        best_F = {ln: (sr, 0.0) for ln in range(self.n_lanes)}
        best_B = {ln: (sr, 0.0) for ln in range(self.n_lanes)}

        try:
            ctx = traci.vehicle.getContextSubscriptionResults(self.ego_id) or {}
        except traci.TraCIException:
            ctx = {}

        for vid, data in ctx.items():
            if vid == self.ego_id:
                continue
            try:
                x      = float(data[tc.VAR_LANEPOSITION])
                v      = float(data[tc.VAR_SPEED])
                ln     = int(data[tc.VAR_LANE_INDEX])
                length = float(data.get(tc.VAR_LENGTH, 4.8))
            except (KeyError, TypeError):
                continue

            if ln < 0 or ln >= self.n_lanes:
                continue

            dx  = x - ego_x
            adx = abs(dx)
            if adx > sr:
                continue

            if dx >= 0:
                real_gap = max(0.0, adx - length)
                if real_gap < best_F[ln][0]:
                    best_F[ln] = (real_gap, v)
            else:
                real_gap = max(0.0, adx - self.ego_length_m)
                if real_gap < best_B[ln][0]:
                    best_B[ln] = (real_gap, v)

        return best_F, best_B

    def _get_obs(self) -> np.ndarray | None:
        if self.ego_id not in traci.vehicle.getIDList():
            return None

        try:
            v_ego, lane, ego_x = self._ego_state_from_sub()
        except traci.TraCIException:
            return None

        sr = float(self.sensor_range_m)

        best_F, best_B = self._build_lane_gaps(ego_x, lane)

        gF,  vF  = best_F[lane]
        gFL, vFL = best_F.get(lane + 1, (sr, 0.0))
        gFR, vFR = best_F.get(lane - 1, (sr, 0.0))
        gBL, vBL = best_B.get(lane + 1, (sr, 0.0))
        gBR, vBR = best_B.get(lane - 1, (sr, 0.0))

        dvF  = (vF  - v_ego) if gF  < sr else 0.0
        dvFL = (vFL - v_ego) if gFL < sr else 0.0
        dvFR = (vFR - v_ego) if gFR < sr else 0.0
        dvBL = (vBL - v_ego) if gBL < sr else 0.0
        dvBR = (vBR - v_ego) if gBR < sr else 0.0

        c = self._c_ego
        lc_prog       = float(c - self._lc_start + 1) / float(self._lc_total) if c >= self._lc_start else 0.0
        cooldown_prog = float(c) / float(max(1, self._cd_total - 1))

        return np.array(
            [v_ego, lane,
             gF,  dvF,
             gFL, dvFL, gFR, dvFR,
             gBL, dvBL, gBR, dvBR,
             lc_prog, cooldown_prog],
            dtype=np.float32,
        )

    def _decode_action(self, a: int) -> tuple[float, int]:
        n_dl = len(self.lane_deltas)
        ai   = int(a) // n_dl
        di   = int(a) %  n_dl
        ai   = int(np.clip(ai, 0, len(self.accel_vals) - 1))
        di   = int(np.clip(di, 0, n_dl - 1))
        return float(self.accel_vals[ai]), int(self.lane_deltas[di])

    def setup_ego_control(self) -> None:
        try:
            traci.vehicle.setSpeedMode(self.ego_id, int(self.ego_speed_mode))
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        except traci.TraCIException:
            pass

    def apply_ego_action(self, action: int):

        if self.ego_id not in traci.vehicle.getIDList():
            return 0.0, 0, 0, 0.0, True, False

        accel, dl_req = self._decode_action(int(action))

        try:
            v_ego, lane, _ = self._ego_state_from_sub()
        except traci.TraCIException:
            return 0.0, 0, 0, 0.0, True, False

        dt = float(self.sim_step)
        try:
            v_cap  = self._lane_v_cap.get(lane, self.v_max)
            v_next = float(np.clip(v_ego + accel * dt, self.v_min, v_cap))
            traci.vehicle.setSpeed(self.ego_id, v_next)
        except traci.TraCIException:
            return 0.0, 0, 0, 0.0, True, False

        c                = self._c_ego
        lane_change_flag = 0.0
        dl_eff           = 0
        invalid_lc       = False
        blocked          = False

        if c >= self._lc_start:

            if dl_req != 0:
                invalid_lc = True
                blocked    = True
            self._c_ego = c - 1

        elif c == 0:

            if dl_req != 0:
                tgt_lane = lane + int(dl_req)
                if tgt_lane < 0 or tgt_lane >= self.n_lanes:
                    invalid_lc = True
                else:
                    try:
                        traci.vehicle.changeLane(self.ego_id, int(tgt_lane), float(self.LANE_CHANGE_TIME))
                        lane_change_flag = 1.0
                        dl_eff           = int(dl_req)
                        self._c_ego      = self._cd_total - 1
                    except traci.TraCIException:
                        invalid_lc = True

        else:
            if dl_req != 0:
                tgt_lane = lane + int(dl_req)
                if tgt_lane < 0 or tgt_lane >= self.n_lanes:
                    invalid_lc = True
                else:
                    try:
                        traci.vehicle.changeLane(self.ego_id, int(tgt_lane), float(self.LANE_CHANGE_TIME))
                        lane_change_flag = 1.0
                        dl_eff           = int(dl_req)
                        self._c_ego      = self._cd_total - 1
                    except traci.TraCIException:
                        invalid_lc = True
            else:
                self._c_ego = c - 1

        return float(accel), int(dl_req), int(dl_eff), float(lane_change_flag), bool(invalid_lc), bool(blocked)

    def _stage_cost(
        self,
        obs: np.ndarray,
        accel: float,
        dl_req: int,
        dl_eff: int,
        lane_change_flag: float,
        c_before: int,
    ) -> float:

        v_ego = float(obs[0])
        l_ego = int(obs[1])

        gF   = float(obs[2])
        gFL  = float(obs[4])
        gFR  = float(obs[6])
        gBL  = float(obs[8])
        dvBL = float(obs[9])
        gBR  = float(obs[10])
        dvBR = float(obs[11])

        cost  = self.W_SPEED_RL * (v_ego - self.v_ref) ** 2
        cost += self.W_ACCEL * (float(accel) ** 2)

        if float(lane_change_flag) != 0.0:
            cost += self.LANE_CHANGE_PENALTY

        c = c_before
        if int(dl_eff) != 0 and (0 < c < self._lc_start):
            cost += self.COOLDOWN_PENALTY

        if int(dl_eff) < 0:
            g_front = gFR
        elif int(dl_eff) > 0:
            g_front = gFL
        else:
            g_front = gF

        d_safe_front = self.D0 + self.T_HEAD * v_ego
        if g_front <= 0.0:
            cost += self.collision_penalty
        if g_front < d_safe_front:
            cost += self.W_FRONT * (d_safe_front - g_front) ** 2

        sr = float(self.sensor_range_m)
        if int(dl_eff) != 0:
            if int(dl_eff) < 0:
                g_back = gBR
                v_back = v_ego + dvBR if (gBR < sr) else None
            else:
                g_back = gBL
                v_back = v_ego + dvBL if (gBL < sr) else None

            if v_back is not None:
                d_safe_back = self.D0 + self.T_HEAD * float(v_back)
                if g_back <= 0.0:
                    cost += self.collision_penalty
                if g_back < d_safe_back:
                    cost += self.W_BACK * (d_safe_back - g_back) ** 2

        if l_ego > 0:
            cost += self.LEFT_LANE_PENALTY * l_ego

        return float(cost)

    def _stage_cost_decomposed(
        self, obs, accel, dl_req, dl_eff, lane_change_flag, c_before,
    ):
        v_ego = float(obs[0]); l_ego = int(obs[1])
        gF = float(obs[2]); gFL = float(obs[4]); gFR = float(obs[6])
        gBL = float(obs[8]); dvBL = float(obs[9])
        gBR = float(obs[10]); dvBR = float(obs[11])

        c_speed = self.W_SPEED * (v_ego - self.v_ref) ** 2
        c_accel = self.W_ACCEL * float(accel) ** 2
        c_lc    = self.LANE_CHANGE_PENALTY if float(lane_change_flag) != 0.0 else 0.0
        c_cd    = self.COOLDOWN_PENALTY if (int(dl_eff) != 0 and 0 < c_before < self._lc_start) else 0.0

        g_front = gFR if int(dl_eff) < 0 else (gFL if int(dl_eff) > 0 else gF)
        d_sf    = self.D0 + self.T_HEAD * v_ego
        c_col_f = self.collision_penalty if g_front <= 0.0 else 0.0
        c_front = self.W_FRONT * (d_sf - g_front) ** 2 if g_front < d_sf else 0.0

        c_col_b = c_back = 0.0
        sr = float(self.sensor_range_m)
        if int(dl_eff) != 0:
            g_back = gBR if int(dl_eff) < 0 else gBL
            dv     = dvBR if int(dl_eff) < 0 else dvBL
            g_sr   = gBR if int(dl_eff) < 0 else gBL
            v_back = v_ego + dv if g_sr < sr else None
            if v_back is not None:
                d_sb    = self.D0 + self.T_HEAD * float(v_back)
                c_col_b = self.collision_penalty if g_back <= 0.0 else 0.0
                c_back  = self.W_BACK * (d_sb - g_back) ** 2 if g_back < d_sb else 0.0

        c_left = self.LEFT_LANE_PENALTY * l_ego if l_ego > 0 else 0.0
        total  = c_speed + c_accel + c_lc + c_cd + c_col_f + c_front + c_col_b + c_back + c_left
        return dict(cost_total=total, cost_speed=c_speed, cost_accel=c_accel,
                    cost_front=c_front, cost_back=c_back,
                    cost_col_f=c_col_f, cost_col_b=c_col_b,
                    cost_lc=c_lc, cost_cd=c_cd, cost_left=c_left)

    def _reinsert_truck_style(
        self, vid: str, edge: str, lane: int, x_target: float, active_ids: set,
        v_override: float | None = None,
    ) -> None:
        if self.ego_id not in active_ids:
            return

        try:
            ego_x = float(traci.vehicle.getLanePosition(self.ego_id))
        except traci.TraCIException:
            return

        if not self._outside_exclusion(x_target, ego_x):
            return

        lane     = int(np.clip(int(lane), 0, self.n_lanes - 1))
        x_target = self._clamp_x(float(x_target))
        v_spawn  = float(v_override) if v_override is not None else self._sample_spawn_speed(lane)
        vtype    = self._choose_vtype()

        if vid in active_ids:
            try:
                traci.vehicle.remove(vid)
            except traci.TraCIException:
                pass

        try:
            traci.vehicle.add(
                vid, self.route_id,
                typeID=vtype,
                depart=None,
                departLane=int(lane),
                departPos="free",
                departSpeed="max",
                arrivalLane="current",
                arrivalPos="max",
                arrivalSpeed="current",
            )
            self._lock_surr_lane_change(vid)
            traci.vehicle.moveTo(vid, self._lane_id(edge, lane), float(x_target))
            traci.vehicle.setSpeedMode(vid, 31)
            traci.vehicle.setSpeed(vid, float(max(0.1, v_spawn)))
        except traci.TraCIException:
            pass

    def spawn_initial_traffic(self) -> None:
        if self.nb_surr <= 0 or self.ego_id not in traci.vehicle.getIDList():
            return

        edge = self._ego_edge
        try:
            ego_x = float(traci.vehicle.getLanePosition(self.ego_id))
        except traci.TraCIException:
            return

        win_start, win_end, _, _ = self._window_bounds(ego_x)

        active_ids = set(traci.vehicle.getIDList())
        for vid in self.surr_ids:
            if vid in active_ids:
                try:
                    traci.vehicle.remove(vid)
                except traci.TraCIException:
                    pass

        traci.simulationStep()
        active_ids = set(traci.vehicle.getIDList())

        excl     = float(self.spawn_exclusion_m)
        left_hi  = ego_x - excl
        right_lo = ego_x + excl

        bands = []
        if left_hi - win_start > 5.0:
            bands.append((win_start, left_hi))
        if win_end - right_lo > 5.0:
            bands.append((right_lo, win_end))
        if not bands:
            return

        spacing = max(15.0, float(self.place_clearance_m))

        rear_positions  = []
        front_positions = []
        for lo, hi in bands:
            is_front = (lo >= ego_x)
            if is_front:

                x = float(lo)
                while x <= hi:
                    front_positions.append(x)
                    x += spacing
            else:

                x = float(hi)
                while x >= lo:
                    rear_positions.append(x)
                    x -= spacing

        slots = []
        ri, fi = 0, 0
        while ri < len(rear_positions) or fi < len(front_positions):
            if ri < len(rear_positions):
                slots.append((rear_positions[ri], False))
                ri += 1
            if fi < len(front_positions):
                slots.append((front_positions[fi], True))
                fi += 1

        vid_idx = 0
        for x, is_front in slots:
            for lane in range(self.n_lanes):
                if vid_idx >= len(self.surr_ids):
                    break
                v = self._sample_speed_ahead() if is_front else self._sample_speed_behind()
                self._reinsert_truck_style(
                    self.surr_ids[vid_idx], edge, lane, x, active_ids,
                    v_override=v,
                )
                vid_idx += 1
            if vid_idx >= len(self.surr_ids):
                break

        traci.simulationStep()
        for vid in self.surr_ids:
            self._lock_surr_lane_change(vid)

    def maintain_traffic_pool(self) -> None:
        if self.nb_surr <= 0 or self.ego_id not in traci.vehicle.getIDList():
            return

        edge = self._ego_edge

        active_ids = set(traci.vehicle.getIDList())
        known_ids  = set(self.surr_ids) | {self.ego_id}
        unknown    = active_ids - known_ids
        if len(unknown) > 0:
            for vid in unknown:
                try:
                    traci.vehicle.remove(vid)
                except traci.TraCIException:
                    pass

        try:
            ego_res = traci.vehicle.getSubscriptionResults(self.ego_id)
            ego_x   = float(ego_res[tc.VAR_LANEPOSITION]) if ego_res else float(traci.vehicle.getLanePosition(self.ego_id))
        except (traci.TraCIException, KeyError, TypeError):
            try:
                ego_x = float(traci.vehicle.getLanePosition(self.ego_id))
            except traci.TraCIException:
                return

        win_start, win_end, back_target, front_target = self._window_bounds(ego_x)

        active_ids = set(traci.vehicle.getIDList())

        rear_out, front_out, missing = [], [], []
        for vid in self.surr_ids:
            if vid not in active_ids:
                missing.append(vid)
                continue
            try:
                res = traci.vehicle.getSubscriptionResults(vid)
                x   = float(res[tc.VAR_LANEPOSITION]) if res else None
            except (traci.TraCIException, KeyError, TypeError):
                x = None
            if x is None:
                try:
                    x = float(traci.vehicle.getLanePosition(vid))
                except traci.TraCIException:
                    missing.append(vid)
                    continue
            if x < win_start:
                rear_out.append(vid)
            elif x > win_end:
                front_out.append(vid)

        rear_out  = list(dict.fromkeys(rear_out + missing))
        front_out = list(dict.fromkeys(front_out))

        now = self._step_count
        rear_out  = [v for v in rear_out  if now - self._last_reinsert.get(v, -9999) > self._reinsert_cooldown]
        front_out = [v for v in front_out if now - self._last_reinsert.get(v, -9999) > self._reinsert_cooldown]

        k        = int(self.max_replace_per_step)
        to_front = rear_out[:  max(0, k // 2)]
        to_back  = front_out[: max(0, k - len(to_front))]

        spacing = max(15.0, float(self.place_clearance_m))

        for idx, vid in enumerate(to_front):
            lane = int(idx % self.n_lanes)
            x    = float(front_target - (idx // self.n_lanes) * spacing)
            self._reinsert_truck_style(vid, edge, lane, x, active_ids,
                                       v_override=self._sample_speed_ahead())
            self._last_reinsert[vid] = self._step_count
            try:
                traci.vehicle.subscribe(vid, _SURR_VARS)
            except traci.TraCIException:
                pass

        for idx, vid in enumerate(to_back):
            lane = int(idx % self.n_lanes)
            x    = float(back_target + (idx // self.n_lanes) * spacing)
            self._reinsert_truck_style(vid, edge, lane, x, active_ids,
                                       v_override=self._sample_speed_behind())
            self._last_reinsert[vid] = self._step_count
            try:
                traci.vehicle.subscribe(vid, _SURR_VARS)
            except traci.TraCIException:
                pass

        for vid in (to_front + to_back):
            self._lock_surr_lane_change(vid)

        if self.reroute_every_n_steps > 0 and (self._step_count % self.reroute_every_n_steps == 0):
            for vid in self.surr_ids:
                if vid not in active_ids:
                    continue
                try:
                    traci.vehicle.setRouteID(vid, self.route_id)
                except traci.TraCIException:
                    pass

    def _ego_collided(self) -> bool:
        try:
            return self.ego_id in traci.simulation.getCollidingVehiclesIDList()
        except traci.TraCIException:
            return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        traci.switch(self.env_label)

        self._step_count     = 0
        self._c_ego          = 0
        self._ctx_sub_active = False
        self._last_reinsert  = {}

        if self._randomize_ego_start:
            margin = float(self._ego_start_margin_m)
            lo     = margin
            hi     = float(self.length_m) - margin
            self.ego_start_pos_m = float(self._rng.uniform(lo, hi)) if hi > lo else float(self._ego_start_pos_fixed)
        else:
            self.ego_start_pos_m = float(self._ego_start_pos_fixed)

        def _restart_sumo():
            import threading, time as _time, subprocess as _sp
            # print to the real stderr, W&B console capture can hang otherwise
            import sys as _sys
            def _log(msg):
                pass

            _log(f"[reset] SUMO connection lost for {self.env_label}, restarting...")

            def _kill_sumo_proc(proc):
                """Kill a SUMO subprocess and its children."""
                if proc is None:
                    return
                try:
                    import signal
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            try:
                import signal as _signal
                conn = traci.getConnection(self.env_label)
                proc = getattr(conn, "_process", None)
                if proc is not None:
                    try:
                        proc.send_signal(_signal.SIGTERM)
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                traci.close()
            except Exception:
                pass
            _time.sleep(0.3)

            sumo_binary = checkBinary("sumo-gui" if self.use_gui else "sumo")

            for attempt in range(3):
                sumo_proc = [None]
                result    = [None]

                def _start():
                    try:

                        import socket as _sock
                        with _sock.socket() as _s:
                            _s.bind(("", 0))
                            port = _s.getsockname()[1]
                        cmd = [sumo_binary, "-c", self.sumocfg_path,
                               *self._sumo_common, "--remote-port", str(port)]
                        sumo_proc[0] = _sp.Popen(cmd,
                                                  stdout=_sp.DEVNULL,
                                                  stderr=_sp.DEVNULL)

                        traci.init(port=port, numRetries=5, label=self.env_label)

                        try:
                            traci.getConnection(self.env_label)._socket.settimeout(30.0)
                        except Exception:
                            pass
                        result[0] = "ok"
                    except Exception as e:
                        result[0] = str(e)

                t = threading.Thread(target=_start, daemon=True)
                t.start()
                t.join(timeout=8.0)

                if result[0] == "ok":
                    _log(f"[reset] SUMO restarted for {self.env_label}")
                    return

                if result[0] is None:
                    _log(f"[reset] SUMO restart timed out (attempt {attempt+1}), retrying...")
                else:
                    _log(f"[reset] SUMO restart failed: {result[0]} (attempt {attempt+1}), retrying...")

                _kill_sumo_proc(sumo_proc[0])
                try:
                    traci.close()
                except Exception:
                    pass
                _time.sleep(0.5)

            _log(f"[reset] SUMO restart failed after 3 attempts for {self.env_label}")
            raise RuntimeError(f"[reset] SUMO restart failed after 3 attempts for {self.env_label}")

        import sys as _sys

        def _safe_restart():
            """Restart SUMO; errors are written straight to stderr (see _restart_sumo)."""
            try:
                _restart_sumo()
            except RuntimeError as _re:
                _err = getattr(_sys, "__stderr__", _sys.stderr)
                try:
                    import os as _os
                    _os.write(2, f"[reset] FATAL: {_re}\n".encode("utf-8", errors="replace"))
                except Exception:
                    pass
                raise

        known_ids = [self.ego_id] + list(self.surr_ids)
        try:
            for vid in known_ids:
                try:
                    traci.vehicle.unsubscribe(vid)
                    traci.vehicle.remove(vid)
                except traci.TraCIException:
                    pass
            traci.simulationStep()
            traci.simulationStep()
        except Exception:
            _safe_restart()

        try:
            traci.vehicle.add(
                self.ego_id, self.route_id,
                typeID=str(self.sim.get("ego_type", "egoTruck")),
                depart=None,
                departLane="free",
                departPos="base",
                departSpeed=str(self.v_ref),
            )
            traci.simulationStep()
            traci.vehicle.setColor(self.ego_id, (255, 0, 0))
        except Exception:
            _restart_sumo()
            traci.vehicle.add(
                self.ego_id, self.route_id,
                typeID=str(self.sim.get("ego_type", "egoTruck")),
                depart=None,
                departLane="free",
                departPos="base",
                departSpeed=str(self.v_ref),
            )
            traci.simulationStep()

        try:
            traci.vehicle.moveTo(
                self.ego_id,
                self._lane_id(self.edge_id, 1),
                float(np.clip(self.ego_start_pos_m, self.end_margin, self.length_m - self.end_margin))
            )
            traci.vehicle.setSpeed(self.ego_id, float(self.v_ref))
        except traci.TraCIException:
            pass

        self._refresh_ego_edge()
        self._cache_lane_speeds()
        self.setup_ego_control()

        self.spawn_initial_traffic()
        self._setup_subscriptions()

        settle_steps = int(max(0.0, self.traffic_settle_s) / self.sim_step)
        for _ in range(settle_steps):
            self.maintain_traffic_pool()
            traci.simulationStep()

        warmup_steps = int(max(0.0, self.warmup_s) / self.sim_step)
        for _ in range(warmup_steps):
            try:
                traci.vehicle.setSpeed(self.ego_id, float(self.v_ref))
            except traci.TraCIException:
                pass
            self.maintain_traffic_pool()
            traci.simulationStep()

        self._t_episode0 = float(traci.simulation.getTime())

        obs = self._get_obs()
        if obs is None:
            obs = np.zeros((14,), dtype=np.float32)

        info = {
            "ego_id":                    self.ego_id,
            "nb_surr":                   self.nb_surr,
            "spawn_exclusion_m":         float(self.spawn_exclusion_m),
            "t_sumo":                    float(traci.simulation.getTime()),
            "t_episode":                 0.0,
            "ego_start_pos_m":           float(self.ego_start_pos_m),
            "lock_traffic_lane_changes": bool(self.lock_traffic_lane_changes),
            "c_ego":                     int(self._c_ego),
        }
        return obs, info

    def step(self, action):
        traci.switch(self.env_label)
        self._step_count += 1

        obs_before = self._get_obs()
        if obs_before is None:
            obs_before = np.zeros((14,), dtype=np.float32)

        c_before = int(self._c_ego)
        accel, dl_req, dl_eff, lane_change_flag, invalid_lc, blocked = (
            self.apply_ego_action(int(action))
        )
        cost = self._stage_cost(obs_before, accel, dl_req, dl_eff, lane_change_flag,
                                c_before=c_before)

        try:
            traci.simulationStep()
        except traci.TraCIException as e:
            print(f"[step] TraCI error: {e}", flush=True)
            obs  = np.zeros((14,), dtype=np.float32)
            info = {
                "ego_id": self.ego_id, "reason": "traci_error",
                "cost_raw": float(cost), "terminated_on_cost": False,
            }
            return obs, -(float(cost) * float(self.reward_scale)), True, False, info

        if traci.simulation.getMinExpectedNumber() == 0:
            obs = np.zeros((14,), dtype=np.float32)
            info = {
                "ego_id": self.ego_id, "reason": "simulation_empty",
                "cost_raw": float(cost), "terminated_on_cost": False,
                "c_ego": int(self._c_ego),
            }
            return obs, -(float(cost) * float(self.reward_scale)), True, False, info

        if self.ego_id not in traci.vehicle.getIDList():
            obs  = np.zeros((14,), dtype=np.float32)
            info = {
                "ego_id": self.ego_id, "reason": "ego_missing",
                "cost_raw": float("inf"), "terminated_on_cost": False,
                "c_ego": int(self._c_ego),
            }
            return obs, float(self.ego_missing_reward), True, False, info

        obs = self._get_obs()
        if obs is None:
            obs = np.zeros((14,), dtype=np.float32)

        if self.terminate_on_collision and self._ego_collided():
            info   = {
                "ego_id":             self.ego_id,
                "reason":             "ego_collision",
                "cost_raw":           float(self.sim.get("collision_cost", 1e7)),
                "t_sumo":             float(traci.simulation.getTime()),
                "t_episode":          float(traci.simulation.getTime() - self._t_episode0),
                "accel":              float(accel),
                "dl":                 int(dl_eff),
                "dl_req":             int(dl_req),
                "lane_change":        float(lane_change_flag),
                "invalid_lc":         bool(invalid_lc),
                "blocked":            bool(blocked),
                "c_ego":              int(self._c_ego),
                "terminated_on_cost": False,
            }
            return obs, float(self.collision_reward), True, False, info

        self.maintain_traffic_pool()

        obs = self._get_obs()
        if obs is None:
            obs = np.zeros((14,), dtype=np.float32)

        terminated         = False
        terminated_on_cost = False
        if self.terminate_on_cost and (cost > self.cost_terminate_threshold):
            terminated         = True
            terminated_on_cost = True

        reward    = -(float(cost) * float(self.reward_scale))
        t_sumo    = float(traci.simulation.getTime())
        t_episode = float(t_sumo - self._t_episode0)
        truncated = (t_episode >= float(self.duration_s))

        cost_decomp = self._stage_cost_decomposed(
            obs_before, accel, dl_req, dl_eff, lane_change_flag, c_before)

        info = {
            "ego_id":             self.ego_id,
            "t_sumo":             t_sumo,
            "t_episode":          t_episode,
            "accel":              float(accel),
            "dl":                 int(dl_eff),
            "dl_req":             int(dl_req),
            "lane_change":        float(lane_change_flag),
            "invalid_lc":         bool(invalid_lc),
            "blocked":            bool(blocked),
            "c_ego":              int(self._c_ego),
            "cost_raw":           float(cost),
            "terminated_on_cost": bool(terminated_on_cost),
            "cost_speed":         cost_decomp["cost_speed"],
            "cost_accel":         cost_decomp["cost_accel"],
            "cost_front":         cost_decomp["cost_front"],
            "cost_back":          cost_decomp["cost_back"],
            "cost_col_f":         cost_decomp["cost_col_f"],
            "cost_col_b":         cost_decomp["cost_col_b"],
            "cost_lc":            cost_decomp["cost_lc"],
            "cost_cd":            cost_decomp["cost_cd"],
            "cost_left":          cost_decomp["cost_left"],
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            traci.switch(self.env_label)
            traci.close()
        except Exception:
            pass
