"""
Train the DQN that is later used as the terminal cost of the MPC.

The agent is trained in the same Highway environment, with reward equal to
minus the MPC stage cost, so its Q-values estimate the cost-to-go. Settings
are read from environment variables (SEED, N_ENVS, TOTAL_STEPS, EVAL_EVERY,
LEARNING_RATE, USE_WANDB, ...), which made it easy to run on the cluster:

    SUMO_HOME=/path/to/sumo N_ENVS=8 TOTAL_STEPS=20000000 python train_dqn.py

Checkpoints and evaluation results go to checkpoints/<run_name>_<time>/.
"""

from __future__ import annotations

import os
import sys
import copy
import signal
import threading
import traceback
import subprocess
import time as _time
from dataclasses import dataclass, asdict

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from params import SIM_PARAMS, ROAD_PARAMS, EVAL_HOLDOUT_SEEDS
from highway_env import Highway


def _wandb_available():
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        return True
    except Exception:
        return False


@dataclass
class CFG:

    seed: int = 0

    n_envs: int = 4
    vec_type: str = "subproc"

    total_timesteps: int = 40_000_000
    eval_every_steps: int = 100_000
    eval_max_steps: int = 1500

    learning_rate: float = 1e-5
    buffer_size: int = 400_000
    learning_starts: int = 5_000
    batch_size: int = 64
    gamma: float = 0.98

    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000

    exploration_fraction: float = 0.15
    exploration_final_eps: float = 0.01

    net_arch: tuple[int, int] = (256, 256)

    tensorboard_log: str = "tb_dqn"
    out_base: str = ""
    device: str = "auto"

    use_wandb: bool = True
    wandb_project: str = "highway-rl-mpc"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None


class FreezeWatchdog:
    """
    Background thread that notices when training stops making progress.

    On the cluster SUMO/TraCI sometimes hung without any error. If no step
    is reported for `timeout_s` seconds this prints the stack of every
    thread and the running SUMO processes, and can kill the job
    (WATCHDOG_KILL=1) so it does not sit there until the time limit.
    """

    def __init__(self, timeout_s: float = 120.0, check_interval_s: float = 10.0):
        self.timeout_s        = timeout_s
        self.check_interval_s = check_interval_s
        self.heartbeat        = _time.time()
        self.last_timestep    = 0
        self._stop            = threading.Event()
        self._thread          = threading.Thread(
            target=self._run, daemon=True, name="FreezeWatchdog"
        )

    def start(self):
        self._thread.start()
        kill = os.getenv("WATCHDOG_KILL", "0") == "1"
        print(f"[watchdog] started  timeout={self.timeout_s:.0f}s  "
              f"kill_on_freeze={kill}", flush=True)

    def stop(self):
        self._stop.set()

    def update(self, timestep: int):
        self.heartbeat     = _time.time()
        self.last_timestep = timestep

    def _dump(self, silent_s: float):
        print(f"\n{'='*65}", flush=True)
        print(f"[watchdog] FREEZE DETECTED, no SB3 step for {silent_s:.0f}s",
              flush=True)
        print(f"[watchdog] Last timestep seen : {self.last_timestep}", flush=True)
        print(f"[watchdog] Wall time          : "
              f"{_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        # where is every thread stuck?
        print("\n[watchdog] --- Python thread stack traces ---", flush=True)
        for tid, frame in sys._current_frames().items():
            name = next(
                (t.name for t in threading.enumerate() if t.ident == tid), "?"
            )
            print(f"\n  Thread {tid} ({name}):", flush=True)
            for line in traceback.format_stack(frame):
                print("  " + line.rstrip(), flush=True)

        # and which SUMO processes are still around
        print("\n[watchdog] --- SUMO processes ---", flush=True)
        try:
            r = subprocess.run(["ps", "aux"], capture_output=True, text=True,
                               timeout=5)
            lines = [l for l in r.stdout.splitlines() if "sumo" in l.lower()]
            for l in lines:
                print(f"  {l}", flush=True)
            if not lines:
                print("  (no sumo processes found)", flush=True)
        except Exception as e:
            print(f"  (ps failed: {e})", flush=True)

        print("\n[watchdog] --- Open network fds (TraCI is TCP) ---", flush=True)
        try:
            r = subprocess.run(
                ["lsof", "-p", str(os.getpid()), "-n", "-P"],
                capture_output=True, text=True, timeout=5,
            )
            relevant = [l for l in r.stdout.splitlines()
                        if any(k in l for k in
                               ["TCP", "FIFO", "pipe", "IPv4", "IPv6", "UNIX"])]
            for l in relevant[:30]:
                print(f"  {l}", flush=True)
            if not relevant:
                print("  (no network fds found)", flush=True)
        except Exception as e:
            print(f"  (lsof failed: {e})", flush=True)

        print("\n[watchdog] --- Memory ---", flush=True)
        try:
            r = subprocess.run(["free", "-h"], capture_output=True, text=True,
                               timeout=5)
            print(f"  {r.stdout.strip()}", flush=True)
        except Exception as e:
            print(f"  (free failed: {e})", flush=True)

        print(f"{'='*65}\n", flush=True)

        if os.getenv("WATCHDOG_KILL", "0") == "1":
            print("[watchdog] WATCHDOG_KILL=1: sending SIGTERM", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)

    def _run(self):
        while not self._stop.is_set():
            self._stop.wait(self.check_interval_s)
            if self._stop.is_set():
                break
            silent_s = _time.time() - self.heartbeat
            if silent_s > self.timeout_s:
                self._dump(silent_s)
                self.heartbeat = _time.time()


class ProgressCallback(BaseCallback):

    def __init__(self, print_every: int = 10_000,
                 watchdog: "FreezeWatchdog | None" = None):
        super().__init__()
        self.print_every        = print_every
        self.watchdog           = watchdog
        self._next              = print_every
        self._t_last            = None
        self._steps_last        = 0
        self._cumulative_reward = 0.0
        self._ep_rewards        = []
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        now = _time.time()

        if self._t_last is None:
            self._t_last     = now
            self._steps_last = self.num_timesteps
        if self.watchdog is not None:
            self.watchdog.update(self.num_timesteps)

        rewards  = self.locals.get("rewards", None)
        dones    = self.locals.get("dones",   None)
        if rewards is not None:
            self._cumulative_reward     += float(np.sum(rewards))
            self._current_ep_reward     += float(np.sum(rewards))
        if dones is not None:
            for done in dones:
                if done:
                    self._ep_rewards.append(self._current_ep_reward)
                    self._current_ep_reward = 0.0

                    if len(self._ep_rewards) > 20:
                        self._ep_rewards.pop(0)

        if self.num_timesteps >= self._next:
            elapsed = max(now - self._t_last, 1e-6)
            steps   = self.num_timesteps - self._steps_last
            sps     = steps / elapsed

            mean_ep_rew = (float(np.mean(self._ep_rewards))
                           if self._ep_rewards else float("nan"))
            print(f"[train] timesteps={self.num_timesteps:>8}  "
                  f"steps/s={sps:>6.0f}  "
                  f"mean_ep_reward(last20)={mean_ep_rew:.2f}", flush=True)

            if self.logger:
                self.logger.record("time/steps_per_second", sps)
                self.logger.record("train/cumulative_reward",
                                   self._cumulative_reward)
                if self._ep_rewards:
                    self.logger.record("train/mean_ep_reward_last20",
                                       mean_ep_rew)
                self.logger.dump(self.num_timesteps)

            self._t_last     = now
            self._steps_last = self.num_timesteps
            self._next      += self.print_every

        return True


def make_env(label: str, seed: int, road_params: dict, sim_params: dict):
    def _thunk():
        np.random.seed(seed)
        sp = copy.deepcopy(sim_params)
        sp["seed_spawn"] = seed
        rp = copy.deepcopy(road_params)
        rp["scenario_dir"] = f"scenario_out_{label}"
        env = Highway(sp, rp, use_gui=False, env_label=label)
        return env
    return _thunk


def evaluate_holdout(model, seeds: list, max_steps: int,
                     sim_params: dict, road_params: dict) -> dict:
    import time
    ep_scaled, ep_raw, ep_len = [], [], []
    ego_collision = ego_missing = 0

    rp = copy.deepcopy(road_params)
    rp["scenario_dir"] = "scenario_out_eval"
    eval_env = Highway(sim_params, rp, use_gui=False, env_label="eval")

    t0 = time.time()
    try:
        for s in seeds:
            obs, _ = eval_env.reset(seed=s)
            done = False
            scaled = raw = steps = 0
            last_reason = ""

            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                done = terminated or truncated
                scaled += reward
                raw    += info.get("cost_raw", 0)
                last_reason = info.get("reason", "")
                steps += 1

            if last_reason == "ego_collision": ego_collision += 1
            if last_reason == "ego_missing":   ego_missing   += 1

            ep_scaled.append(scaled)
            ep_raw.append(raw)
            ep_len.append(steps)
            print(f"  [eval] seed={s}  steps={steps}  "
                  f"reward={scaled:.4f}  reason={last_reason or 'timeout'}", flush=True)

    finally:
        eval_env.close()

    elapsed = time.time() - t0
    ep_scaled = np.array(ep_scaled)
    ep_raw    = np.array(ep_raw)
    ep_len    = np.array(ep_len)

    print(f"  [eval] done in {elapsed:.1f}s  "
          f"mean_reward={ep_scaled.mean():.4f}  collisions={ego_collision}", flush=True)

    return dict(
        scaled_mean=float(ep_scaled.mean()),
        scaled_std=float(ep_scaled.std()),
        raw_mean=float(ep_raw.mean()),
        raw_std=float(ep_raw.std()),
        len_mean=float(ep_len.mean()),
        len_std=float(ep_len.std()),
        ego_collision=int(ego_collision),
        ego_missing=int(ego_missing),
        cumulative_reward=float(ep_scaled.sum()),
    )


def main():
    if not os.environ.get("SUMO_HOME"):
        print("[ERROR] SUMO_HOME is not set. Add it to your cluster job script:", flush=True)
        print("  export SUMO_HOME=/path/to/sumo", flush=True)
        sys.exit(1)

    watchdog_timeout = float(os.getenv("WATCHDOG_TIMEOUT_S", "120"))
    watchdog = FreezeWatchdog(timeout_s=watchdog_timeout, check_interval_s=10.0)
    watchdog.start()

    cfg = CFG(
        seed=int(os.getenv("SEED", "0")),
        n_envs=int(os.getenv("N_ENVS", "4")),
        total_timesteps=int(os.getenv("TOTAL_STEPS", "40000000")),
        eval_every_steps=int(os.getenv("EVAL_EVERY", "100000")),
        eval_max_steps=int(os.getenv("EVAL_MAX_STEPS", "1500")),
        device=os.getenv("DEVICE", "auto"),
        learning_rate=float(os.getenv("LEARNING_RATE", "0.0001")),
        use_wandb=os.getenv("USE_WANDB", "1") == "1",
        wandb_project=os.getenv("WANDB_PROJECT", "highway-rl-mpc"),
        wandb_run_name=os.getenv("WANDB_RUN_NAME", None),
    )

    sim_params = copy.deepcopy(SIM_PARAMS)
    if "WARMUP_S" in os.environ:
        sim_params["warmup_s"] = float(os.environ["WARMUP_S"])
    if "TRAFFIC_SETTLE_S" in os.environ:
        sim_params["traffic_settle_s"] = float(os.environ["TRAFFIC_SETTLE_S"])
    if "INIT_STEPS" in os.environ:
        sim_params["init_steps"] = int(os.environ["INIT_STEPS"])
    if "TRAFFIC_DENSITY" in os.environ:
        sim_params["traffic_density_veh_per_km"] = float(os.environ["TRAFFIC_DENSITY"])
    if "TERMINATE_ON_COST" in os.environ:
        sim_params["terminate_on_cost"] = os.environ["TERMINATE_ON_COST"] == "1"
    if "TERMINATE_ON_COLLISION" in os.environ:
        sim_params["terminate_on_collision"] = os.environ["TERMINATE_ON_COLLISION"] == "1"

    eval_sim_params = copy.deepcopy(sim_params)
    eval_sim_params["warmup_s"]         = float(os.getenv("EVAL_WARMUP_S",         "2.0"))
    eval_sim_params["traffic_settle_s"] = float(os.getenv("EVAL_TRAFFIC_SETTLE_S", "1.0"))
    eval_sim_params["init_steps"]       = int(os.getenv("EVAL_INIT_STEPS",         "5"))

    import datetime
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label  = cfg.wandb_run_name or "run"
    run_dir    = os.path.join("checkpoints", f"{run_label}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    cfg.out_base        = os.path.join(run_dir, "model")
    cfg.tensorboard_log = os.path.join(run_dir, "tb")
    print(f"[run_dir]     {os.path.abspath(run_dir)}", flush=True)
    print(f"[tensorboard] {os.path.abspath(cfg.tensorboard_log)}", flush=True)

    print("=== CONFIG ===")
    print(f"n_envs:               {cfg.n_envs}")
    print(f"total_timesteps:      {cfg.total_timesteps}")
    print(f"eval_every:           {cfg.eval_every_steps}")
    print(f"eval_max_steps:       {cfg.eval_max_steps}")
    print(f"device:               {cfg.device}")
    print(f"gradient_steps:       {cfg.gradient_steps}  (-1 = match collected)")
    print(f"warmup_s (train):     {sim_params['warmup_s']}")
    print(f"warmup_s (eval):      {eval_sim_params['warmup_s']}")
    print(f"traffic_settle(train):{sim_params['traffic_settle_s']}")
    print(f"traffic_settle(eval): {eval_sim_params['traffic_settle_s']}")
    print(f"traffic_density:      {sim_params['traffic_density_veh_per_km']}")
    print(f"learning_rate:        {cfg.learning_rate}")
    print("===============")

    wandb_run = None
    wandb_cb  = None
    if cfg.use_wandb and _wandb_available():
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            entity=cfg.wandb_entity,
            config=asdict(cfg),
            sync_tensorboard=True,
        )

        wandb_cb = WandbCallback(
            gradient_save_freq=0,
            verbose=0,
        )

    env_fns = [
        make_env(f"train{i}", cfg.seed + i, ROAD_PARAMS, sim_params)
        for i in range(cfg.n_envs)
    ]

    vec = DummyVecEnv(env_fns)
    vec = VecMonitor(vec)

    resume_path = os.getenv("RESUME_CHECKPOINT", "")
    if resume_path and os.path.exists(resume_path):
        print(f"\n=== RESUMING FROM {resume_path} ===")
        model = DQN.load(
            resume_path,
            env=vec,
            device=cfg.device,
            tensorboard_log=cfg.tensorboard_log if cfg.tensorboard_log else None,
        )

        model.learning_rate      = cfg.learning_rate
        model.exploration_fraction = cfg.exploration_fraction
        model.exploration_final_eps = cfg.exploration_final_eps
        try:
            from stable_baselines3.common.utils import constant_fn
            model.learning_rate = cfg.learning_rate
            model.lr_schedule = constant_fn(float(cfg.learning_rate))
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = cfg.learning_rate
            print(f"    Learning rate set to {cfg.learning_rate}", flush=True)
        except Exception as e:
            print(f"    Warning: could not set optimizer lr: {e}", flush=True)

        import re
        m = re.search(r"_t(\d+)", resume_path)
        steps_done_init = int(m.group(1)) if m else 0
        print(f"    Resuming from timestep {steps_done_init}")
    else:
        model = DQN(
            "MlpPolicy",
            vec,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            target_update_interval=cfg.target_update_interval,
            exploration_fraction=cfg.exploration_fraction,
            exploration_final_eps=cfg.exploration_final_eps,
            tensorboard_log=cfg.tensorboard_log if cfg.tensorboard_log else None,
            policy_kwargs=dict(net_arch=list(cfg.net_arch)),
            verbose=1,
            device=cfg.device,
        )
        steps_done_init = 0

    callbacks = [ProgressCallback(print_every=10_000, watchdog=watchdog)]
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    print("\n=== EVAL BEFORE TRAINING ===")
    stats = evaluate_holdout(model, EVAL_HOLDOUT_SEEDS, cfg.eval_max_steps,
                             eval_sim_params, ROAD_PARAMS)
    print(stats)
    if wandb_run:
        import wandb
        wandb.log({"timesteps": 0, **stats})

    steps_done = steps_done_init
    while steps_done < cfg.total_timesteps:

        chunk = min(cfg.eval_every_steps, cfg.total_timesteps - steps_done)
        print(f"\n=== TRAIN CHUNK {chunk} (done={steps_done}) ===")

        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            callback=callbacks,
        )
        steps_done += chunk

        path = f"{cfg.out_base}_t{steps_done}.zip"
        model.save(path)
        print(f"[save] {path}  (exploration_rate={model.exploration_rate:.4f})")

    model.save(cfg.out_base + "_final.zip")
    print("\nTraining finished.")
    vec.close()
    watchdog.stop()

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
