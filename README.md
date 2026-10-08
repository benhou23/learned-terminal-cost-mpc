# Learned terminal costs for short-horizon MPC

Code for the paper *Learned Terminal Costs for Short-Horizon Model Predictive
Control of Autonomous Trucks in Interactive Highway Traffic* (under review).

An autonomous truck drives on a three-lane highway in SUMO. It is controlled by
a model predictive controller (MPC) that only looks a few steps ahead, because
a long horizon is too expensive to solve online. With such a short horizon the
controller is myopic: it does not see that a lane change or slowing down now
pays off later. We close the horizon with a terminal cost that estimates the
cost-to-go, and compare three ways of getting it:

| `--terminal_cost` | Terminal cost J[H]                                        |
|-------------------|-----------------------------------------------------------|
| `zero`            | none, plain short-horizon MPC                             |
| `rollout`         | simulate a simple driving policy for 40 more steps        |
| `rl`              | Q-values of a DQN trained on the same cost                |
| `rl_only`         | (baseline) the DQN policy alone, without MPC              |

The DQN is trained in the same environment with reward = -(MPC stage cost), so
its Q-values are directly a cost-to-go in the units of the MPC. With a
one-step horizon the learned terminal cost took the collision rate in our
test scenarios from about 88% to zero.

## How it works

- **MPC** (`mpc_rl_utils.backward_dp_numba`): backward dynamic programming on
  a grid over position, lane, speed and lane-change cooldown, compiled with
  Numba. Only the cells the truck can reach within the horizon are evaluated.
- **Prediction** (`build_lane_trajs`): the other vehicles are predicted with
  a lane-keeping car-following model. Vehicles in the middle of a lane change
  occupy both lanes.
- **Learned terminal cost** (`fill_terminal_rl`): for every reachable end
  state we build the DQN observation and take `-max_a Q / reward_scale`.
- **Rollout terminal cost** (`initialize_terminal_cost`): a greedy ego policy
  and the traffic are simulated forward and the stage costs are summed.
- **Environment** (`highway_env.Highway`): Gymnasium env around SUMO/TraCI
  with 15 discrete actions (5 accelerations x left/keep/right) and a
  14-dimensional observation. It is used both for training and evaluation.

## Files

```
run_controller.py      run one episode with any controller
eval_batch.py          evaluate a controller over many seeds, write CSVs
train_dqn.py           train the DQN (settings via environment variables)
eval_checkpoint.py     evaluate a DQN checkpoint on the hold-out seeds
mpc_rl_utils.py        DP solver, trajectory prediction, terminal costs
highway_env.py         SUMO highway environment
params.py              shared simulation settings and seed ranges
pred_error_logger.py   log prediction errors of build_lane_trajs
analyze_pred_errors.py plot those errors
scenario/              generates the SUMO network and config
models/                the DQN checkpoint used in the paper
tests/                 sanity checks
```

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
export SUMO_HOME=$(python -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
```

If SUMO is installed some other way, point `SUMO_HOME` to that installation.

## Running

```bash
# one episode, MPC with the learned terminal cost, H = 3
python run_controller.py --terminal_cost rl --horizon 3 \
    --checkpoint models/model_t16600000.zip --seed 20000

# same with the SUMO GUI
python run_controller.py --terminal_cost rl --horizon 3 \
    --checkpoint models/model_t16600000.zip --gui

# evaluation on the 500 test seeds
python eval_batch.py --terminal_cost zero    --horizon 1 --seeds 20000-20499
python eval_batch.py --terminal_cost rl      --horizon 1 --seeds 20000-20499 \
    --checkpoint models/model_t16600000.zip
python eval_batch.py --terminal_cost rollout --horizon 1 --seeds 20000-20499

# training (we ran this on a cluster)
N_ENVS=8 TOTAL_STEPS=20000000 USE_WANDB=0 python train_dqn.py
```

Seeds 0-9999 are for training, 10000-10009 for model selection and
20000-20499 for the final evaluation (`tests/test_seeds.py` checks that they
do not overlap).

## Tests

```bash
python tests/test_highway_env.py      # no SUMO needed
python tests/test_cost_consistency.py # env cost == MPC cost, no SUMO needed
python tests/test_seeds.py
python tests/test_traffic_quality.py  # needs SUMO
python tests/test_learning.py         # short training run, needs SUMO
```

## Authors

Withheld for double-blind review.
