"""
Unit tests for the plain Python parts of highway_env.py.

SUMO/TraCI are replaced by stubs, so this runs with numpy only:

    python tests/test_highway_env.py

Covers the action decoding, the lane-change/cooldown state machine (zones
A/B/C), the observation vector, every stage-cost term including collisions,
and that the constants agree with params.py and mpc_rl_utils.py.
"""

import os
import sys
import types
import numpy as np

def _install_stubs():
    traci = types.ModuleType("traci")
    tc    = types.ModuleType("traci.constants")
    tc.VAR_SPEED         = 0x40
    tc.VAR_LANE_INDEX    = 0x52
    tc.VAR_LANEPOSITION  = 0x56
    tc.VAR_ROAD_ID       = 0x50
    tc.VAR_POSITION      = 0x42
    tc.VAR_LENGTH        = 0x44
    tc.VAR_LANEPOSITION_LAT = 0xb8; tc.VAR_ACCEL = 0x46; tc.VAR_DECEL = 0x47
    tc.VAR_MAXSPEED = 0x41; tc.VAR_ACCELERATION = 0x72; tc.CMD_GET_VEHICLE_VARIABLE = 0xa4
    traci.constants      = tc

    class TraCIException(Exception): pass
    traci.TraCIException = TraCIException

    ego_state = {"ids": [], "lane": 1, "speed": 25.0, "x": 1000.0}

    v = types.SimpleNamespace()
    v.getIDList              = lambda:    ego_state["ids"]
    v.getSubscriptionResults = lambda id: {
        tc.VAR_SPEED:        ego_state["speed"],
        tc.VAR_LANE_INDEX:   ego_state["lane"],
        tc.VAR_LANEPOSITION: ego_state["x"],
    }
    v.getContextSubscriptionResults = lambda id: {}
    v.subscribe          = lambda id, vs: None
    v.subscribeContext   = lambda *a, **kw: None
    v.unsubscribeContext = lambda *a, **kw: None
    v.changeLane         = lambda id, l, t: None
    v.setSpeed           = lambda id, s: None
    v.setSpeedMode       = lambda id, m: None
    v.setLaneChangeMode  = lambda id, m: None
    v.setPreviousSpeed   = lambda id, s: None
    v.setMaxSpeed        = lambda id, s: None
    v.add                = lambda *a, **kw: None
    v.remove             = lambda id: None
    v.moveTo             = lambda id, lid, x: None
    v.moveToXY           = lambda *a, **kw: None
    v.setRouteID         = lambda id, r: None
    v.getLaneIndex       = lambda id: ego_state["lane"]
    v.getLanePosition    = lambda id: ego_state["x"]
    v.getLaneID          = lambda id: f"e0_{ego_state['lane']}"
    v.getSpeed           = lambda id: ego_state["speed"]
    v.getMaxSpeed        = lambda id: 35.0
    v.getTypeID          = lambda id: "car"
    v.setAcceleration    = lambda id, a, t: None
    traci.vehicle        = v

    lane = types.SimpleNamespace()
    lane.getMaxSpeed = lambda lid: 33.0
    lane.getWidth    = lambda lid: 3.2
    traci.lane       = lane

    sim = types.SimpleNamespace()
    sim.getTime                    = lambda: 0.0
    sim.getCollidingVehiclesIDList = lambda: []
    sim.getMinExpectedNumber       = lambda: 99
    traci.simulation = sim

    traci.start          = lambda cmd, label=None: None
    traci.switch         = lambda label: None
    traci.load           = lambda args: None
    traci.close          = lambda: None
    traci.simulationStep = lambda: None

    sumolib = types.ModuleType("sumolib")
    sumolib.checkBinary = lambda name: name

    import pathlib, os
    scenario  = types.ModuleType("scenario")
    build_mod = types.ModuleType("scenario.build_highway")
    def _fake_build(out_dir, **kw):
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        cfg = os.path.join(out_dir, "run.sumocfg")
        pathlib.Path(cfg).write_text("<configuration/>")
        return {"dir": out_dir, "nodes": "", "edges": "", "net": "",
                "routes": "", "sumocfg": cfg}
    build_mod.build_oneway_highway_3lane = _fake_build
    scenario.build_highway = build_mod

    sys.modules["traci"]                  = traci
    sys.modules["traci.constants"]        = tc
    sys.modules["sumolib"]                = sumolib
    sys.modules["scenario"]               = scenario
    sys.modules["scenario.build_highway"] = build_mod

    gym = types.ModuleType("gymnasium")
    class FakeEnv:
        def __init__(self): pass
        def reset(self, seed=None, options=None): pass
    gym.Env = FakeEnv
    spaces = types.ModuleType("gymnasium.spaces")
    class Discrete:
        def __init__(self, n): self.n = n
    class Box:
        def __init__(self, **kw): self.shape = kw.get("shape")
    spaces.Discrete = Discrete
    spaces.Box      = Box
    gym.spaces      = spaces
    sys.modules["gymnasium"]        = gym
    sys.modules["gymnasium.spaces"] = spaces

    return traci, ego_state


_traci, _ego_state = _install_stubs()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from highway_env import Highway

V_REF = 25.0
V_MAX = 35.0
CP    = 1e8

SIM = {
    "sim_step_length": 0.2, "LANE_CHANGE_TIME": 2.0, "LANE_CHANGE_COOLDOWN_S": 5.0,
    "sensor_range_m": 100.0, "v_ref": V_REF, "v_min": 0.0, "v_max": V_MAX,
    "D0": 10.0, "T_HEAD": 2.0, "W_SPEED": 1.0, "W_ACCEL": 1.0,
    "W_FRONT": 1000.0, "W_BACK": 1000.0,
    "LANE_CHANGE_PENALTY": 10.0, "COOLDOWN_PENALTY": 30.0, "LEFT_LANE_PENALTY": 2.0,
    "reward_scale": 1e-5, "ego_missing_reward": -100.0,
    "collision_cost": 1e8, "collision_penalty_mpc": 1e8,
    "cost_terminate_threshold": 5e8,
    "terminate_on_cost": False, "terminate_on_collision": True,
    "traffic_density_veh_per_km": 0,
    "moving_window_m": 1100.0, "edge_end_margin_m": 600.0, "window_buffer_m": 30.0,
    "warmup_s": 0.0, "traffic_settle_s": 0.0, "init_steps": 0, "duration_s": 22000,
    "lock_traffic_lane_changes": True, "n_accel": 5, "a_max": 2.0, "ego_speed_mode": 0,
    "ego_length_m": 12.0, "train_seed_max": 9999,
}
ROAD = {"n_lanes": 3, "length_m": 30000.0, "speed_mps": 33.0,
        "scenario_dir": "/tmp/test_highway_sc"}

_results = []

def check(name, got, expected, *, approx=False, tol=1e-4):
    ok = (abs(float(got) - float(expected)) < tol) if approx else (got == expected)
    _results.append(ok)
    tag = "" if ok else f"  <- got {got!r}, expected {expected!r}"
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{tag}")

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def make_env():
    e = Highway(SIM, ROAD, use_gui=False, env_label="test")
    e._lane_v_cap = {0: 33.0, 1: 33.0, 2: 33.0}
    return e

def set_ego(lane=1, speed=25.0, x=1000.0):
    _ego_state["ids"]   = ["ego"]
    _ego_state["lane"]  = lane
    _ego_state["speed"] = speed
    _ego_state["x"]     = x

def obs(v=V_REF, lane=0, gF=100., dvF=0., gFL=100., dvFL=0.,
        gFR=100., dvFR=0., gBL=100., dvBL=0., gBR=100., dvBR=0.,
        lcp=0., cdp=0.):
    return np.array([v, lane, gF, dvF, gFL, dvFL, gFR, dvFR,
                     gBL, dvBL, gBR, dvBR, lcp, cdp], dtype=np.float32)

STAY=7; RIGHT=8; LEFT=6  # zero-accel actions

# 1. Initialisation
section("1. Initialisation")
e = make_env()
check("_cd_total = 25",                   e._cd_total,  25)
check("_lc_total = 10",                   e._lc_total,  10)
check("_lc_start = 15",                   e._lc_start,  15)
check("_c_ego = 0",                       e._c_ego,      0)
check("action_space.n = 15",             e.action_space.n, 15)
check("obs_space.shape = (14,)",         e.observation_space.shape, (14,))
check("v_ref = 25.0",                    e.v_ref, 25.0, approx=True)
check("v_max = 35.0",                    e.v_max, 35.0, approx=True)
check("collision_penalty = 1e8",         e.collision_penalty, 1e8, approx=True)
check("collision_reward = -1000.0",      e.collision_reward, -1000.0, approx=True)
e.close()

# 2. Action decoder
section("2. Action decoder")
e = make_env()
check("action 0: (-2.0,-1)", tuple(round(x,2) for x in e._decode_action(0)), (-2.0,-1))
check("action 7: ( 0.0, 0)", tuple(round(x,2) for x in e._decode_action(7)), ( 0.0, 0))
check("action 14:(+2.0,+1)", tuple(round(x,2) for x in e._decode_action(14)), (2.0, 1))
all_valid = all(-2<=e._decode_action(i)[0]<=2 and e._decode_action(i)[1] in (-1,0,1) for i in range(15))
check("all 15 actions valid", all_valid, True)
e.close()

# 3. State machine
section("3. State machine – zone transitions")
e = make_env()
set_ego(lane=1)

e._c_ego = 0; e.apply_ego_action(STAY)
check("Zone A + stay -> c=0",  e._c_ego, 0)
e._c_ego = 0; e.apply_ego_action(RIGHT)
check("Zone A + change -> c=24", e._c_ego, 24)
e._c_ego = 0; _,_,dl,lf,_,bl = e.apply_ego_action(RIGHT)
check("Zone A change: not blocked", bl, False)
check("Zone A change: dl_eff=1", dl, 1)

e._c_ego = 20; _,_,dl,_,_,bl = e.apply_ego_action(RIGHT)
check("Zone C + change: blocked", bl, True)
check("Zone C + change: dl_eff=0", dl, 0)
check("Zone C + change: c decrements", e._c_ego, 19)

e._c_ego = 15; e.apply_ego_action(STAY)
check("Zone C c=15 + stay -> 14", e._c_ego, 14)

e._c_ego = 7; e.apply_ego_action(STAY)
check("Zone B + stay: 7->6", e._c_ego, 6)
e._c_ego = 7; e.apply_ego_action(RIGHT)
check("Zone B + change -> c=24", e._c_ego, 24)
e._c_ego = 1; e.apply_ego_action(STAY)
check("Zone B c=1 + stay -> 0", e._c_ego, 0)

e._c_ego = 0; e.apply_ego_action(RIGHT)
ok = (e._c_ego == 24)
for exp in range(23, -1, -1):
    e.apply_ego_action(STAY)
    if e._c_ego != exp: ok = False; break
check("full 24-step countdown", ok, True)

set_ego(lane=0); e._c_ego = 0
_,_,dl,_,inv,_ = e.apply_ego_action(LEFT)
check("dl=-1 from lane 0: invalid", (inv, dl), (True, 0))
e.close()

# 4. Observation encoding
section("4. Observation encoding")
e = make_env()
bad = [c for c in range(e._lc_start) if (float(c-e._lc_start)/e._lc_total if c>=e._lc_start else 0.) != 0.]
check("lc_prog=0 for Zone A+B (c=0..14)", bad, [])
bad2 = [c for c in range(e._lc_start, e._cd_total)
        if not (0 < float(c-e._lc_start+1)/e._lc_total <= 1.0)]
check("lc_prog in (0,1] for Zone C (c=15..24)", bad2, [])
check("lc_prog=1.0 at c=24", float(e._cd_total-1-e._lc_start+1)/e._lc_total, 1.0, approx=True)
fails = [(c, round(float(c)/float(e._cd_total-1)*(e._cd_total-1)))
         for c in range(e._cd_total)
         if round(float(c)/float(e._cd_total-1)*(e._cd_total-1)) != c]
check("c->obs[13]->c round-trip all 25 values", fails, [])
e.close()

# 5. Stage cost – normal terms
section("5. Stage cost – normal terms")
e = make_env(); e._c_ego = 0
check("v=v_ref, safe, lane 0 -> cost=0",
      e._stage_cost(obs(), 0., 0, 0, 0., 0), 0., approx=True)
check("v=v_ref-5 -> W_SPEED*25=25",
      e._stage_cost(obs(v=V_REF-5), 0., 0, 0, 0., 0), 25., approx=True)
check("accel=2 -> W_ACCEL*4=4",
      e._stage_cost(obs(), 2., 0, 0, 0., 0), 4., approx=True)
e._c_ego = 7
check("Zone B LC: 10+30+2=42",
      e._stage_cost(obs(lane=1), 0., 1, 1, 1., 7), 42., approx=True)
e._c_ego = 0
check("Zone A LC: 10+2=12",
      e._stage_cost(obs(lane=1), 0., 1, 1, 1., 0), 12., approx=True)
check("lane 0: no left-lane penalty",
      e._stage_cost(obs(v=V_REF, lane=0), 0., 0, 0, 0., 0), 0., approx=True)
check("lane 2: 2*2=4",
      e._stage_cost(obs(v=V_REF, lane=2), 0., 0, 0, 0., 0), 4., approx=True)
gF = 20.; d_safe = 10. + 2.*V_REF
check("gF=20 < d_safe -> W_FRONT*(d_safe-20)^2",
      e._stage_cost(obs(gF=gF), 0., 0, 0, 0., 0),
      1000.*(d_safe-gF)**2, approx=True)
check("gF > d_safe -> no safety cost",
      e._stage_cost(obs(gF=d_safe+1), 0., 0, 0, 0., 0), 0., approx=True)
neg = [(v_,l_) for v_ in [0,V_REF,V_MAX] for l_ in [0,1,2]
       if e._stage_cost(obs(v=v_,lane=l_), 0.,0,0,0.,0) < 0.]
check("cost >= 0 all v x lane", neg, [])
e.close()

# 6. Collision cost
section("6. Collision cost – zero-gap triggers collision_penalty (MPC parity)")
e = make_env(); e._c_ego = 0

# zero front gap -> collision penalty
cost = e._stage_cost(obs(gF=0.), 0., 0, 0, 0., 0)
check("gF=0.0 -> collision_penalty fires (>= 1e8)", cost >= CP, True)

# small positive gap -> only the quadratic term
gF_s = 1.; d_s = 10. + 2.*V_REF
cost_s = e._stage_cost(obs(gF=gF_s), 0., 0, 0, 0., 0)
check("gF=1.0 -> only quadratic, no collision_penalty",
      abs(cost_s - 1000.*(d_s-gF_s)**2) < 1., True)

# gFL=0 left lane change
cost = e._stage_cost(obs(lane=1, gFL=0.), 0., 1, 1, 1., 0)
check("gFL=0 during left LC -> collision_penalty", cost >= CP, True)

# gFR=0 right lane change
cost = e._stage_cost(obs(lane=1, gFR=0.), 0., -1, -1, 1., 0)
check("gFR=0 during right LC -> collision_penalty", cost >= CP, True)

# Rear gap=0 during lane change
cost = e._stage_cost(obs(lane=1, gBL=0., dvBL=5.), 0., 1, 1, 1., 0)
check("gBL=0 during left LC -> rear collision_penalty", cost >= CP, True)

cost = e._stage_cost(obs(lane=1, gBR=0., dvBR=5.), 0., -1, -1, 1., 0)
check("gBR=0 during right LC -> rear collision_penalty", cost >= CP, True)

# Rear gap=0 NO lane change -> no penalty
cost = e._stage_cost(obs(gBL=0., dvBL=5.), 0., 0, 0, 0., 0)
check("gBL=0 no LC -> no rear penalty", cost, 0., approx=True)

# Bellman consistency
check("collision_reward = -1000 (= -1e8 * 1e-5)",
      e.collision_reward, -1000., approx=True)
check("CP * reward_scale == |collision_reward| (Bellman consistent)",
      abs(CP * e.reward_scale - abs(e.collision_reward)) < 1e-6, True)

e.close()

# 7. MPC compatibility
section("7. MPC compatibility")
e = make_env()
dt = e.sim_step
cd = int(round(e.LANE_CHANGE_COOLDOWN_S / dt))
lc = int(round(e.LANE_CHANGE_TIME / dt))
check("_lc_start = cd - lc", e._lc_start, cd - lc)
check("W_SPEED=1",  e.W_SPEED,  1., approx=True)
check("W_FRONT=1000", e.W_FRONT, 1000., approx=True)
check("D0=10, T_HEAD=2", (e.D0, e.T_HEAD), (10., 2.))
check("LANE_CHANGE_PENALTY=10", e.LANE_CHANGE_PENALTY, 10., approx=True)
check("COOLDOWN_PENALTY=30",    e.COOLDOWN_PENALTY,    30., approx=True)
check("collision_penalty=1e8",  e.collision_penalty,   1e8, approx=True)
v, gap = V_REF, 20.
e._c_ego = 0
cost = e._stage_cost(obs(v=v, gF=gap), 0., 0, 0, 0., 0)
check("front-safety matches DP formula exactly",
      cost, e.W_FRONT*(e.D0+e.T_HEAD*v-gap)**2, approx=True)
bad = [(c, round(float(c)/float(e._cd_total-1)*(e._cd_total-1)))
       for c in range(e._cd_total)
       if round(float(c)/float(e._cd_total-1)*(e._cd_total-1)) != c]
check("c losslessly encoded in obs[13]", bad, [])
e.close()

# 8. params.py consistency
section("8. Parameter consistency with params.py and mpc_rl_utils.py")
try:
    from params import SIM_PARAMS, EVAL_HOLDOUT_SEEDS, EVAL_FINAL_SEEDS
    # Expected values read from mpc_rl_utils so params and MPC stay in sync.
    from mpc_rl_utils import v_ref as _mr, v_max as _mx, COLLISION_PENALTY as _mcp
    check("params v_ref matches mpc_rl_utils",
          SIM_PARAMS.get("v_ref"), _mr, approx=True)
    check("params v_max matches mpc_rl_utils",
          SIM_PARAMS.get("v_max"), _mx, approx=True)
    check("params collision_cost=1e8",
          SIM_PARAMS.get("collision_cost"), 1e8, approx=True)
    check("params collision_cost matches mpc_rl_utils.COLLISION_PENALTY",
          SIM_PARAMS.get("collision_cost"), _mcp, approx=True)
    check("params cost_terminate=5e8",
          SIM_PARAMS.get("cost_terminate_threshold"), 5e8, approx=True)
    check("EVAL_HOLDOUT_SEEDS has 10",    len(EVAL_HOLDOUT_SEEDS), 10)
    check("EVAL_FINAL_SEEDS has 500",     len(EVAL_FINAL_SEEDS),   500)
    tm = SIM_PARAMS.get("train_seed_max", 9999)
    ts, hs, fs = set(range(tm+1)), set(EVAL_HOLDOUT_SEEDS), set(EVAL_FINAL_SEEDS)
    check("train disjoint from holdout",  ts.isdisjoint(hs), True)
    check("train disjoint from final",    ts.isdisjoint(fs), True)
    check("holdout disjoint from final",  hs.isdisjoint(fs), True)
except ImportError:
    print("  SKIP  params.py not found: run from project root")


# Summary
total = len(_results); passed = sum(_results); failed = total - passed
print(f"\n{'='*60}")
print(f"  {passed}/{total} tests passed", end="")
if failed == 0:
    print("  -- safe to train")
else:
    print(f"  -- {failed} FAILURE(S), fix before training")
print("="*60)
sys.exit(0 if failed == 0 else 1)
