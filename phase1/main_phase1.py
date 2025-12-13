# phase1/main_phase1.py
import os
import random
import numpy as np
import torch
from datetime import datetime

from .config import Phase1Config
from .tasks import generate_taskset
from .env import HeteroEDFEnv
from .baseline import run_random_policy
from .ddqn import train_double_dqn, run_greedy_policy
from .plots import plot_qos_energy, plot_gantt, plot_training_reward
from .io_utils import ensure_dir, save_all_results, save_training_history_csv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = Phase1Config()
    set_seed(cfg.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", stamp)
    ensure_dir(outdir)

    # 1) Generate deterministic test task set
    np.random.seed(cfg.test.seed)
    test_tasks = generate_taskset(
        n_tasks=cfg.test.n_tasks,
        horizon=cfg.test.horizon,
        **cfg.test.taskgen.to_kwargs(),
    )

    # restore seed for training randomness
    np.random.seed(cfg.seed)

    # 2) Random baseline on test set
    env_random = HeteroEDFEnv(
        test_tasks,
        meet_bonus=cfg.train.meet_bonus,
        miss_penalty=cfg.train.miss_penalty,
        energy_weight=cfg.train.energy_weight,
    )
    res_random = run_random_policy(env_random)

    # 3) Train Double DQN (agent + history)
    agent, hist = train_double_dqn(
        episodes=cfg.train.episodes,
        n_tasks=cfg.train.n_tasks,
        horizon=cfg.train.horizon,
        epsilon_start=cfg.train.epsilon_start,
        epsilon_end=cfg.train.epsilon_end,
        epsilon_decay_steps=cfg.train.epsilon_decay_steps,
        warmup_steps=cfg.train.warmup_steps,
        max_steps_per_episode=cfg.train.max_steps_per_episode,
        seed=cfg.train.seed,
        lr=cfg.train.lr,
        gamma=cfg.train.gamma,
        batch_size=cfg.train.batch_size,
        buffer_capacity=cfg.train.buffer_capacity,
        target_update=cfg.train.target_update,
        meet_bonus=cfg.train.meet_bonus,
        miss_penalty=cfg.train.miss_penalty,
        energy_weight=cfg.train.energy_weight,
        taskgen_kwargs=cfg.train.taskgen.to_kwargs(),
    )

    # 4) Evaluate on test set
    env_rl = HeteroEDFEnv(
        test_tasks,
        meet_bonus=cfg.train.meet_bonus,
        miss_penalty=cfg.train.miss_penalty,
        energy_weight=cfg.train.energy_weight,
    )
    res_rl = run_greedy_policy(agent, env_rl)

    # 5) Plots
    fig_qos, fig_energy = plot_qos_energy(
        res_random["metrics"], res_rl["metrics"], title_prefix="Test set: "
    )
    fig_gantt_big = plot_gantt(res_rl["segments"]["big"], "big core", max_rows=80, label_every=10)
    fig_gantt_little = plot_gantt(res_rl["segments"]["little"], "little core", max_rows=80, label_every=10)
    fig_reward = plot_training_reward(hist.episode_return, window=20)

    figs = {
        "qos_per_core": fig_qos,
        "energy_per_core": fig_energy,
        "gantt_big_core": fig_gantt_big,
        "gantt_little_core": fig_gantt_little,
        "training_reward": fig_reward,
    }

    # 6) Save outputs
    save_all_results(
        outdir=outdir,
        config=cfg.to_dict(),
        test_tasks=test_tasks,
        agent=agent,
        res_random=res_random,
        res_rl=res_rl,
        figs=figs,
    )
    save_training_history_csv(hist, os.path.join(outdir, "training_history.csv"))

    print(f"\nSaved all outputs to: {outdir}")
    print("Key files:")
    print("  results.json, summary.txt")
    print("  test_tasks.csv")
    print("  assignments_random.csv, assignments_double_dqn.csv")
    print("  segments/*.csv")
    print("  plots/*.png (includes training_reward.png)")
    print("  training_history.csv")
    print("  double_dqn_weights.pt")


if __name__ == "__main__":
    main()
