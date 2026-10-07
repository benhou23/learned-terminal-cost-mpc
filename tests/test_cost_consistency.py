"""
Check that the RL environment and the MPC use the same cost.

Compares Highway._stage_cost() with the cost terms in backward_dp_numba for a
few hand-picked states (safety, collisions, lane changes). SUMO is stubbed
out, so only numpy and numba are needed.

    python tests/test_cost_consistency.py
"""
import os
import sys
import types
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PASS = FAIL = 0

def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")

# fake traci/sumolib so highway_env can be imported without SUMO
def _install_stubs():
    traci = types.ModuleType("traci")
    tc    = types.ModuleType("traci.constants")
    tc.VAR_SPEED = 0x40; tc.VAR_LANE_INDEX = 0x52
    tc.VAR_LANEPOSITION = 0x56; tc.VAR_ROAD_ID = 0x50
    tc.VAR_POSITION = 0x42; tc.VAR_LENGTH = 0x44
    tc.VAR_LANEPOSITION_LAT = 0xb8; tc.VAR_ACCEL = 0x46; tc.VAR_DECEL = 0x47
    tc.VAR_MAXSPEED = 0x41; tc.VAR_ACCELERATION = 0x72; tc.CMD_GET_VEHICLE_VARIABLE = 0xa4
    traci.constants = tc
    class TE(Exception): pass
    traci.TraCIException = TE
    vs = types.SimpleNamespace(
        getIDList=lambda: [], getSubscriptionResults=lambda id: {},
        getContextSubscriptionResults=lambda id: {},
        subscribe=lambda *a,**kw: None, subscribeContext=lambda *a,**kw: None,
        unsubscribeContext=lambda *a,**kw: None,
        changeLane=lambda *a: None, setSpeed=lambda *a: None,
        setSpeedMode=lambda *a: None, setLaneChangeMode=lambda *a: None,
        setPreviousSpeed=lambda *a: None, setMaxSpeed=lambda *a: None,
        add=lambda *a,**kw: None, remove=lambda *a: None,
        moveTo=lambda *a: None, moveToXY=lambda *a,**kw: None,
        getLaneIndex=lambda *a: 1, getLanePosition=lambda *a: 1000.,
        getLaneID=lambda *a: "e0_1", getSpeed=lambda *a: 25.,
        getMaxSpeed=lambda *a: 35., getTypeID=lambda *a: "car",
        setAcceleration=lambda *a: None,
    )
    traci.vehicle = vs
    traci.lane = types.SimpleNamespace(getMaxSpeed=lambda *a: 33., getWidth=lambda *a: 3.2)
    traci.simulation = types.SimpleNamespace(
        getTime=lambda: 0., getCollidingVehiclesIDList=lambda: [],
        getMinExpectedNumber=lambda: 99)
    traci.start=lambda *a,**kw: None; traci.switch=lambda *a: None
    traci.close=lambda: None; traci.simulationStep=lambda: None
    sumolib = types.ModuleType("sumolib")
    sumolib.checkBinary = lambda n: n
    import pathlib, os
    sc = types.ModuleType("scenario")
    bm = types.ModuleType("scenario.build_highway")
    def fb(out_dir, **kw):
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        p = os.path.join(out_dir, "run.sumocfg")
        pathlib.Path(p).write_text("<configuration/>")
        return {"dir": out_dir, "nodes": "", "edges": "", "net": "", "routes": "", "sumocfg": p}
    bm.build_oneway_highway_3lane = fb; sc.build_highway = bm
    gym = types.ModuleType("gymnasium")
    class FE:
        def __init__(self): pass
        def reset(self, seed=None, options=None): pass
    gym.Env = FE
    sp = types.ModuleType("gymnasium.spaces")
    class D:
        def __init__(self, n): self.n = n
    class B:
        def __init__(self, **kw): self.shape = kw.get("shape")
    sp.Discrete = D; sp.Box = B; gym.spaces = sp
    for k, v in [("traci", traci), ("traci.constants", tc), ("sumolib", sumolib),
                 ("scenario", sc), ("scenario.build_highway", bm),
                 ("gymnasium", gym), ("gymnasium.spaces", sp)]:
        sys.modules[k] = v

_install_stubs()

from highway_env import Highway
from params import SIM_PARAMS as _P

# reference values come from params.py
V_REF = float(_P.get("v_ref",          25.0))
V_MAX = float(_P.get("v_max",          33.33))
CP    = float(_P.get("collision_cost",  1e8))

SIM = {
    "sim_step_length": 0.2, "LANE_CHANGE_TIME": 2.0, "LANE_CHANGE_COOLDOWN_S": 5.0,
    "sensor_range_m": 100.0, "v_ref": V_REF, "v_min": 0.0, "v_max": V_MAX,
    "D0": float(_P.get("D0", 5.0)), "T_HEAD": 2.0, "W_SPEED": 1.0, "W_ACCEL": 1.0,
    "W_FRONT": 1000.0, "W_BACK": 1000.0,
    "LANE_CHANGE_PENALTY": 10.0, "COOLDOWN_PENALTY": 30.0, "LEFT_LANE_PENALTY": 2.0,
    "reward_scale": 1e-5, "ego_missing_reward": -100.0,
    "collision_cost": CP,
    "cost_terminate_threshold": 5e8, "terminate_on_cost": False,
    "terminate_on_collision": True, "traffic_density_veh_per_km": 0,
    "moving_window_m": 1100.0, "edge_end_margin_m": 600.0, "window_buffer_m": 30.0,
    "warmup_s": 0.0, "traffic_settle_s": 0.0, "init_steps": 0, "duration_s": 22000,
    "lock_traffic_lane_changes": True, "n_accel": 5, "a_max": 2.0,
    "ego_speed_mode": 0, "ego_length_m": 12.0,
}
ROAD = {"n_lanes": 3, "length_m": 30000.0, "speed_mps": 33.0,
        "scenario_dir": "/tmp/test_cc"}

env = Highway(SIM, ROAD, use_gui=False, env_label="cc")
env._lane_v_cap = {0: 33., 1: 33., 2: 33.}

def o(v=V_REF, lane=0, gF=100., dvF=0., gFL=100., dvFL=0.,
      gFR=100., dvFR=0., gBL=100., dvBL=0., gBR=100., dvBR=0.,
      lcp=0., cdp=0.):
    return np.array([v, lane, gF, dvF, gFL, dvFL, gFR, dvFR,
                     gBL, dvBL, gBR, dvBR, lcp, cdp], dtype=np.float32)

print(f"\n{'='*60}")
print("  Cost consistency: RL _stage_cost vs MPC DP formulas")
print(f"{'='*60}")

env._c_ego = 0

# 1. Zero cost at v_ref
c = env._stage_cost(o(), 0., 0, 0, 0., 0)
check("v=v_ref, no gaps, lane 0 -> cost=0.0", abs(c) < 1e-6, f"got {c}")

# 2. Speed cost matches W_SPEED*(v-v_ref)^2
for dv in [-10, -5, 5, 10]:
    v = V_REF + dv
    c = env._stage_cost(o(v=v), 0., 0, 0, 0., 0)
    exp = 1.0 * dv**2
    check(f"W_SPEED*(v-v_ref)^2  dv={dv}: {exp}", abs(c - exp) < 1e-4, f"got {c}")

# 3. Front safety cost matches MPC formula
for gF in [5., 10., 20., 30.]:
    D0_val = float(_P.get("D0", 5.0))
    d_safe = D0_val + 2. * V_REF
    c = env._stage_cost(o(gF=gF), 0., 0, 0, 0., 0)
    if gF < d_safe:
        exp = 1000. * (d_safe - gF)**2
        check(f"W_FRONT*(d_safe-gF)^2  gF={gF}: {exp:.0f}", abs(c - exp) < 1., f"got {c:.2f}")
    else:
        check(f"gF={gF} >= d_safe={d_safe:.0f} -> 0", abs(c) < 1e-4, f"got {c}")

# 4. zero front gap: collision penalty and the quadratic term both apply
c_zero = env._stage_cost(o(gF=0.), 0., 0, 0, 0., 0)
d_safe_zero = float(_P.get("D0", 5.0)) + 2. * V_REF   # D0 + T_HEAD * v_ref
exp_stacked = CP + 1000. * d_safe_zero**2
check("gF=0 -> collision_penalty + quadratic",
      abs(c_zero - exp_stacked) < 1., f"got {c_zero:.2e}, expected {exp_stacked:.2e}")

c_tiny = env._stage_cost(o(gF=0.001), 0., 0, 0, 0., 0)
check("gF=0.001 -> quadratic only (no collision_penalty)",
      c_tiny < CP, f"got {c_tiny:.2e}")

# 5. rear collision while changing lane, same stacking
# plus the lane-change and left-lane penalties
c_rear = env._stage_cost(o(lane=1, gBL=0., dvBL=5.), 0., 1, 1, 1., 0)
v_back_test  = V_REF + 5.             # v_ego + dvBL = 25 + 5 = 30
D0_val       = float(_P.get("D0", 5.0))
d_safe_back  = D0_val + 2. * v_back_test
exp_rear_stacked = (CP
                    + 1000. * d_safe_back**2   # quadratic term
                    + 10.                       # LANE_CHANGE_PENALTY (lane_change_flag=1)
                    + 2.)                       # LEFT_LANE_PENALTY (lane=1)
check("gBL=0 during left LC -> CP + quadratic",
      abs(c_rear - exp_rear_stacked) < 1., f"got {c_rear:.6e}, expected {exp_rear_stacked:.6e}")

# 6. reward scale is the same for stage and collision cost
terminal = abs(env.collision_reward)  # 1e8 * 1e-5 = 1000
stage_scaled = CP * env.reward_scale  # 1e8 * 1e-5 = 1000
check(f"Bellman: |collision_reward| ({terminal}) == CP*reward_scale ({stage_scaled})",
      abs(terminal - stage_scaled) < 1e-6,
      f"|reward|={terminal}, CP*scale={stage_scaled}")

# 7. MPC DP uses same constants as env
print("\n  Checking mpc_rl_utils constants match env...")
try:
    from mpc_rl_utils import (v_ref as mpc_vref, v_max as mpc_vmax,
                               D0 as mpc_D0, T_HEAD as mpc_TH,
                               W_FRONT as mpc_WF, W_BACK as mpc_WB,
                               COLLISION_PENALTY as mpc_CP,
                               EGO_LENGTH as mpc_EL)

    check(f"v_ref: env={env.v_ref} == mpc={mpc_vref}",
          abs(env.v_ref - mpc_vref) < 0.01, f"env={env.v_ref}, mpc={mpc_vref}")
    check(f"v_max: env={env.v_max} == mpc={mpc_vmax}",
          abs(env.v_max - mpc_vmax) < 0.01)
    check(f"D0: env={env.D0} == mpc={mpc_D0}",
          abs(env.D0 - mpc_D0) < 0.01)
    check(f"T_HEAD: env={env.T_HEAD} == mpc={mpc_TH}",
          abs(env.T_HEAD - mpc_TH) < 0.01)
    check(f"W_FRONT: env={env.W_FRONT} == mpc={mpc_WF}",
          abs(env.W_FRONT - mpc_WF) < 0.01)
    check(f"W_BACK: env={env.W_BACK} == mpc={mpc_WB}",
          abs(env.W_BACK - mpc_WB) < 0.01)
    check(f"COLLISION_PENALTY: env={env.collision_penalty:.0e} == mpc={mpc_CP:.0e}",
          abs(env.collision_penalty - mpc_CP) < 1.)
    check(f"EGO_LENGTH: env={env.ego_length_m} == mpc={mpc_EL}",
          abs(env.ego_length_m - mpc_EL) < 0.01)

    # MPC safety term vs. env for one state
    v, gF = V_REF, 20.
    d_safe_mpc = mpc_D0 + mpc_TH * v
    mpc_safety = mpc_WF * (d_safe_mpc - gF)**2
    env_cost   = env._stage_cost(o(v=v, gF=gF), 0., 0, 0, 0., 0)
    check("MPC safety formula == env safety formula (v=25, gF=20)",
          abs(mpc_safety - env_cost) < 1., f"mpc={mpc_safety:.2f}, env={env_cost:.2f}")

except ImportError:
    print("  SKIP  mpc_rl_utils not importable (numba not installed)")

env.close()

total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  {PASS}/{total} checks passed")
if FAIL == 0:
    print("  RL and MPC cost structures are consistent")
else:
    print(f"  {FAIL} FAILURE(S): RL/MPC mismatch, fix before training")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
