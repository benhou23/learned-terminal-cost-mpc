"""
Measure how good the trajectory prediction in build_lane_trajs is.

At step t we store the predicted state at t+H of the five neighbours that
matter (front, front-left, front-right, back-left, back-right). When the
simulation reaches t+H we read their real state from SUMO and write one row
per vehicle with the predicted value, the actual value and the error.

The standard deviations in mpc_rl_utils.all_traj_s_stds/all_traj_v_stds
come from these logs (see analyze_pred_errors.py).

Usage inside an episode loop:

    logger = PredErrorLogger(H, dt, "pred_errors.csv")
    ...
    logger.flush_actuals(scenario_id, t, ego_id)        # before predicting
    trajs = build_lane_trajs(ego_id, H, dt, lanes)
    logger.record_prediction(scenario_id, t, s_ego, v_ego, l_ego,
                             len_ego, lanes, trajs)
    ...
    logger.flush_remaining(scenario_id)                 # end of episode
    logger.close()                                      # end of all episodes

eval_batch.py does this when called with --pred_errors.
"""

import csv
import os
import numpy as np
import traci
import traci.constants as tc


ROLES = ["front", "front_left", "front_right", "back_left", "back_right"]


def _find_role_vehicles(s_ego, l_ego, len_ego, lanes, final_trajs, h=0):
    """
    Closest vehicle for each of the five roles at trajectory step h.
    Returns {role: (vid, pred_s, pred_v, pred_lane)}, or None for an empty role.
    """
    sr = 1e9  # range is already limited in build_lane_trajs

    roles = {
        "front":       {"lane": l_ego,     "ahead": True},
        "front_left":  {"lane": l_ego + 1, "ahead": True},
        "front_right": {"lane": l_ego - 1, "ahead": True},
        "back_left":   {"lane": l_ego + 1, "ahead": False},
        "back_right":  {"lane": l_ego - 1, "ahead": False},
    }

    result = {role: None for role in ROLES}

    for role, spec in roles.items():
        target_lane = spec["lane"]
        if target_lane not in lanes:
            continue

        best_gap = np.inf
        best_vid = None
        best_s   = None
        best_v   = None
        best_l   = None

        for vid, traj in final_trajs.items():
            traj_s, traj_v, traj_l = traj[0], traj[1], traj[2]
            length_other = traj[3]

            if traj_l[h] != target_lane:
                continue

            s_other = traj_s[h]

            if spec["ahead"]:
                if s_other <= s_ego:
                    continue
                gap = s_other - length_other - s_ego
            else:
                if s_other >= s_ego:
                    continue
                gap = (s_ego - len_ego) - s_other

            if gap < best_gap:
                best_gap = gap
                best_vid = vid
                best_s   = float(s_other)
                best_v   = float(traj_v[h])
                best_l   = int(traj_l[h])

        if best_vid is not None:
            result[role] = (best_vid, best_s, best_v, best_l)

    return result


class PredErrorLogger:
    """Collects predicted vs. actual states and appends them to a CSV file."""

    COLUMNS = [
        "scenario", "t_predict", "t_actual", "H",
        "role",
        "vid",
        "pred_s",    "pred_v",    "pred_lane",
        "actual_s",  "actual_v",  "actual_lane",
        "error_s",   "error_v",   "error_lane",
        "abs_error_s", "abs_error_v",
        "vehicle_present",
        # ego state when the prediction was made
        "ego_s_at_predict", "ego_v_at_predict", "ego_lane_at_predict",
    ]

    def __init__(self, H: int, dt: float, csv_path: str):
        self.H        = H
        self.dt       = dt
        self.csv_path = csv_path

        # (scenario_id, t_actual) -> rows still waiting for the real values
        self._pending = {}

        file_is_new  = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self._file   = open(csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        if file_is_new:
            self._writer.writeheader()

    def record_prediction(
        self,
        scenario_id,
        t: int,
        s_ego: float,
        v_ego: float,
        l_ego: int,
        len_ego: float,
        lanes: list,
        final_trajs: dict,
    ):
        """
        Call right after build_lane_trajs() at step t. Stores the predicted
        state at t+H for the five neighbours.
        """
        t_actual = t + self.H

        role_vids = _find_role_vehicles(
            s_ego, l_ego, len_ego, lanes, final_trajs, h=0
        )

        pending_rows = []
        for role in ROLES:
            entry = role_vids[role]
            if entry is None:
                # nobody in this position
                pending_rows.append({
                    "scenario":           scenario_id,
                    "t_predict":          t,
                    "t_actual":           t_actual,
                    "H":                  self.H,
                    "role":               role,
                    "vid":                "",
                    "pred_s":             np.nan,
                    "pred_v":             np.nan,
                    "pred_lane":          np.nan,
                    "actual_s":           np.nan,
                    "actual_v":           np.nan,
                    "actual_lane":        np.nan,
                    "error_s":            np.nan,
                    "error_v":            np.nan,
                    "error_lane":         np.nan,
                    "abs_error_s":        np.nan,
                    "abs_error_v":        np.nan,
                    "vehicle_present":    0,
                    "ego_s_at_predict":   round(s_ego, 3),
                    "ego_v_at_predict":   round(v_ego, 3),
                    "ego_lane_at_predict": l_ego,
                })
                continue

            vid, pred_s, pred_v, pred_lane = entry

            # prediction for step H
            traj = final_trajs[vid]
            traj_s_H = float(traj[0][self.H])
            traj_v_H = float(traj[1][self.H])
            traj_l_H = int(  traj[2][self.H])

            pending_rows.append({
                "scenario":           scenario_id,
                "t_predict":          t,
                "t_actual":           t_actual,
                "H":                  self.H,
                "role":               role,
                "vid":                vid,
                "pred_s":             round(traj_s_H, 4),
                "pred_v":             round(traj_v_H, 4),
                "pred_lane":          traj_l_H,
                # actual values are filled in at t+H
                "actual_s":           np.nan,
                "actual_v":           np.nan,
                "actual_lane":        np.nan,
                "error_s":            np.nan,
                "error_v":            np.nan,
                "error_lane":         np.nan,
                "abs_error_s":        np.nan,
                "abs_error_v":        np.nan,
                "vehicle_present":    0,
                "ego_s_at_predict":   round(s_ego, 3),
                "ego_v_at_predict":   round(v_ego, 3),
                "ego_lane_at_predict": l_ego,
            })

        key = (scenario_id, t_actual)
        self._pending.setdefault(key, []).extend(pending_rows)

    def flush_actuals(
        self,
        scenario_id,
        t: int,
        ego_id: str,
    ):
        """
        Call at step t before predicting. Fills in the real SUMO state for
        every prediction that was made for this step.
        """
        key = (scenario_id, t)
        if key not in self._pending:
            return

        rows = self._pending.pop(key)
        active_ids = set(traci.vehicle.getIDList())

        for row in rows:
            vid = row["vid"]
            if not vid or vid not in active_ids:
                # vehicle has left the simulation
                row["vehicle_present"] = 0
                self._writer.writerow(row)
                continue

            try:
                actual_s    = float(traci.vehicle.getPosition(vid)[0])
                actual_v    = float(traci.vehicle.getSpeed(vid))
                actual_lane = int(traci.vehicle.getLaneIndex(vid))

                row["actual_s"]    = round(actual_s, 4)
                row["actual_v"]    = round(actual_v, 4)
                row["actual_lane"] = actual_lane
                row["vehicle_present"] = 1

                if not np.isnan(row["pred_s"]):
                    err_s = row["pred_s"] - actual_s
                    err_v = row["pred_v"] - actual_v
                    err_l = row["pred_lane"] - actual_lane
                    row["error_s"]     = round(err_s, 4)
                    row["error_v"]     = round(err_v, 4)
                    row["error_lane"]  = int(err_l)
                    row["abs_error_s"] = round(abs(err_s), 4)
                    row["abs_error_v"] = round(abs(err_v), 4)

            except traci.TraCIException:
                row["vehicle_present"] = 0

            self._writer.writerow(row)

        self._file.flush()

    def flush_remaining(self, scenario_id):
        """
        Write out rows that never got their real values, e.g. because the
        episode ended before t+H.
        """
        keys_to_flush = [k for k in self._pending if k[0] == scenario_id]
        for key in keys_to_flush:
            rows = self._pending.pop(key)
            for row in rows:
                row["vehicle_present"] = 0
                self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """Write everything that is left and close the file."""
        for key in list(self._pending.keys()):
            rows = self._pending.pop(key)
            for row in rows:
                row["vehicle_present"] = 0
                self._writer.writerow(row)
        self._file.flush()
        self._file.close()
