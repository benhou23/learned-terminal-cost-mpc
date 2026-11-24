"""Simulation and road settings shared by training and evaluation."""

SIM_PARAMS = {

    "sensor_range_m":      150.0,
    "spawn_exclusion_m":   150.0,
    "sim_step_length":     0.2,
    "ego_speed_mode":      0,

    "v_ref":               25.0,
    "D0":                   5.0,
    "T_HEAD":               2.0,
    "ego_start_pos_m":     5000.0,
    "ego_type":            "egoTruck",
    "terminate_on_collision": True,
    "collision_cost":      1e8,
    "v_max":               33.33,
    "rl_speed_weight": 1.0,

    "W_SPEED":             1.0,
    "W_ACCEL":             1.0,
    "W_FRONT":          1000.0,
    "W_BACK":           1000.0,
    "traffic_v_mean":      25.0,
    "traffic_v_std":        5.0,
    "traffic_v_min":       20.0,
    "traffic_v_max":       30.0,
    "traffic_speed_delta":  2.0,

    "LANE_CHANGE_PENALTY": 10.0,
    "COOLDOWN_PENALTY":    30.0,
    "LEFT_LANE_PENALTY":    2.0,

    "LANE_CHANGE_TIME":     2.0,
    "LANE_CHANGE_COOLDOWN_S": 5.0,

    "terminate_on_cost":   False,
    "cost_terminate_threshold": 5e8,

    "reward_scale":        1e-5,
    "ego_missing_reward": -100.0,

    "init_steps":          10,
    "duration_s":          999999,
    "warmup_s":             5.0,
    "enable_lc_shield":    True,
    "enable_longitudinal_shield": True,
    "lock_traffic_lane_changes": False,

    "car_vph":              0,
    "truck_vph":            0,

    "window_behind_m":     200,
    "window_ahead_m":     500,
    "moving_window_m":    700,
    "traffic_density_veh_per_km": 30,

    "spawning_safe_dist_m": 10.0,
    "edge_end_margin_m":   600.0,
    "place_clearance_m":    10.0,

    "randomize_ego_start": True,   # training only; run_controller.py always evaluates from the fixed start
    "ego_start_margin_m":  1000.0,

    "collision_action":    "warn",
    "time_to_teleport":    -1,
    "traffic_settle_s":     2.0,
    "max_replacements_per_step": 10,
    "window_buffer_m":      30.0,
    "reinsert_cooldown_s":   1.0,
    "ego_spawn_wait_steps": 50,
}

ROAD_PARAMS = {
    "scenario_dir": "scenario_out",
    "length_m":     30000.0,
    "speed_mps":    35,
    "n_lanes":      3,
}

EVAL_HOLDOUT_SEEDS = list(range(10000, 10010))
EVAL_FINAL_SEEDS   = list(range(20000, 20500))
