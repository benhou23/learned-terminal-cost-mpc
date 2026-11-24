# Learned terminal costs for short-horizon MPC

Code for the paper

> H. Hamze\*, B. Houshmand\*, D. Pathare, E. Börve, M. Haghir Chehreghani, N. Murgovski.
> *Learned Terminal Costs for Short-Horizon Model Predictive Control of Autonomous
> Trucks in Interactive Highway Traffic.* Under review, 2026.
> [arXiv:2610.40257](https://arxiv.org/abs/2610.40257) (\*equal contribution)

An autonomous truck drives on a three-lane highway in SUMO, among traffic that
reacts to it. It is controlled by a model predictive controller (MPC) that looks
only a few steps ahead, because a long horizon is too expensive to solve online.
A short horizon makes the controller myopic: it cannot see that a lane change or
slowing down now pays off later. We close the horizon with a terminal cost that
estimates the cost-to-go, and learn that terminal cost with a Deep Q-Network
trained on the MPC's own stage cost.

In the paper's main (stochastic) setting, adding the learned terminal cost to a
one-step MPC removes collisions entirely (86% of episodes collide without it),
and at a three-step horizon the hybrid reaches less than half the cost of
five-step MPC.

## Controllers

| `--terminal_cost` | Terminal cost J[H]                                            |
|-------------------|---------------------------------------------------------------|
| `zero`            | none: plain short-horizon MPC                                 |
| `rl`              | the DQN's Q-values (the paper's method)                       |
| `rollout`         | a simulated rollout of a simple driving policy                |
| `rl_only`         | baseline: the DQN policy alone, without MPC                   |

The DQN is trained with reward = -(MPC stage cost), so its Q-values are a
cost-to-go in the units of the MPC.

## Settings

| `--setting`               | What it means                                              | In the paper |
|---------------------------|------------------------------------------------------------|--------------|
| `stochastic` (default)    | Gaussian noise on the other vehicles' measured position, speed and acceleration (std 0.5 m, 0.5 m/s, 0.2 m/s²); every gap is tightened by 2.33 prediction-error standard deviations (chance constraint at 99%) | main results |
| `deterministic`           | exact measurements, no margin                              | appendix     |

Measurement noise is seeded with the episode seed, so results are
reproducible.

## How it works

- **MPC** (`mpc_rl_utils.backward_dp_numba`): backward dynamic programming on a
  grid over position, lane, speed and lane-change cooldown, compiled with Numba.
  Only the cells the truck can reach within the horizon are evaluated. In the
  stochastic setting the gaps are tightened by `Z_MARGIN` standard deviations.
- **Prediction** (`build_lane_trajs`): the other vehicles are predicted with a
  lane-keeping car-following model. Vehicles in the middle of a lane change
  occupy both lanes.
- **Learned terminal cost** (`fill_terminal_rl`): for every reachable end state
  we build the DQN observation and take `-max_a Q / reward_scale`.
- **Rollout terminal cost** (`initialize_terminal_cost`): a greedy ego policy and
  the traffic are simulated forward and the stage costs are summed.
- **Environment** (`highway_env.Highway`): Gymnasium environment around
  SUMO/TraCI with 15 discrete actions (5 accelerations × left/keep/right) and a
  14-dimensional observation, used for both training and evaluation.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
export SUMO_HOME=$(python -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
```

If SUMO is installed some other way, point `SUMO_HOME` to that installation.

## Quick start

```bash
# one episode: MPC with the learned terminal cost, H = 3, stochastic setting
python run_controller.py --terminal_cost rl --horizon 3 \
    --checkpoint models/model_t16100000.zip --seed 20000

# the same with the SUMO GUI
python run_controller.py --terminal_cost rl --horizon 3 \
    --checkpoint models/model_t16100000.zip --seed 20000 --gui
```

## Reproducing the paper

The paper evaluates every controller on the 100 seeds 20000–20099.

```bash
bash scripts/paper_table.sh                  # main table (stochastic)
bash scripts/paper_table.sh deterministic    # appendix table
python scripts/summarize.py results/stochastic
```

`paper_table.sh` runs pure MPC, MPC + learned terminal cost and MPC + rollout
terminal cost (rollout lengths 10 and 20) for horizons 1–5, plus the DQN alone.
It takes several hours on a laptop. A single configuration is one line, for
example the paper's best one:

```bash
python eval_batch.py --terminal_cost rl --horizon 3 \
    --checkpoint models/model_t16100000.zip --seeds 20000-20099
```

`eval_batch.py` writes one row per seed (`*_per_seed.csv`) and a summary row
(`*_summary.csv`). `scripts/summarize.py` reports collisions and the cost per
step as in the paper (total cost over total steps).

**A note on exact numbers.** The runs in the paper were made before the
measurement noise was seeded per episode, so a rerun gives slightly different
stochastic results. With seeding, MPC + learned terminal cost at H = 3 has
0 collisions and a cost per step of 342 (paper: 0 and 329), and pure MPC at
H = 5 has 712 (paper: 754); the conclusions are unchanged. Different
SUMO versions can also change the traffic slightly; `requirements.txt` lists the
versions this release was tested with.

## Further experiments

The `extra/` folder holds experiments run after submission:

| Script                      | What it runs                                                     |
|-----------------------------|------------------------------------------------------------------|
| `extra/five_models.sh`      | MPC + learned terminal cost (H = 3) for all five DQN training runs (`models/` and `models/other_runs/`) |
| `extra/longer_horizons.sh`  | pure MPC with H = 5, 7 and 10                                    |
| `extra/safety.sh`           | per-step logs and gap/braking statistics (`extra/safety_metrics.py`) |

The five training runs give different terminal costs: the learned terminal
cost helps with the paper's model, but the benefit varies across runs. Which
value functions work well inside the MPC, and why, is an open question.

## Training

```bash
N_ENVS=8 TOTAL_STEPS=20000000 USE_WANDB=0 python train_dqn.py
```

Settings are read from environment variables (see the top of `train_dqn.py`).
Training starts each episode at a random position; evaluation always uses the
fixed start. Seeds 0–9999 are for training, 10000–10009 for model selection and
20000–20499 are held out for evaluation (the paper uses the first 100);
`tests/test_seeds.py` checks that the ranges do not overlap.

## Files

```
run_controller.py      run one episode with any controller and setting
eval_batch.py          evaluate a controller over many seeds, write CSVs
train_dqn.py           train the DQN (settings via environment variables)
eval_checkpoint.py     evaluate a DQN checkpoint on the model-selection seeds
mpc_rl_utils.py        DP solver, prediction, terminal costs, noise settings
highway_env.py         SUMO highway environment
params.py              shared simulation settings and seed ranges
pred_error_logger.py   log prediction errors of build_lane_trajs
analyze_pred_errors.py plot those errors
scenario/              generates the SUMO network and config
models/                the DQN used in the paper (model_t16100000.zip)
models/other_runs/     the four other training runs (extra/five_models.sh)
scripts/               reproduce and summarise the paper's tables
extra/                 experiments run after submission
tests/                 sanity checks
```

Set `STEP_LOG=path.csv` to write the per-step log of an evaluation to its own
file (the default is `data.csv`), which is needed when runs go in parallel.

## Tests

```bash
python tests/test_highway_env.py      # no SUMO needed
python tests/test_cost_consistency.py # env cost == MPC cost, no SUMO needed
python tests/test_seeds.py
python tests/test_traffic_quality.py  # needs SUMO
python tests/test_learning.py         # short training run, needs SUMO
```

## Authors

Hooman Hamze and Benjamin Houshmand (joint first authors), with Deepthi Pathare,
Erik Börve, Morteza Haghir Chehreghani and Nikolce Murgovski. The work started
as a master's thesis at Volvo Group and Chalmers University of Technology (2026).

## Citation

```bibtex
@misc{hamze2026learned,
  title  = {Learned Terminal Costs for Short-Horizon Model Predictive Control of
            Autonomous Trucks in Interactive Highway Traffic},
  author = {Hamze, Hooman and Houshmand, Benjamin and Pathare, Deepthi and
            B{\"o}rve, Erik and Haghir Chehreghani, Morteza and Murgovski, Nikolce},
  year   = {2026},
  eprint = {2610.40257},
  archivePrefix = {arXiv}
}
```

## License

MIT, see `LICENSE`.
