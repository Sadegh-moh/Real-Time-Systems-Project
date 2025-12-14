# phase2/main_phase2.py
from __future__ import annotations
import os
import argparse
from datetime import datetime
import numpy as np
import torch

from phase1.config import Phase1Config
from phase1.env import HeteroEDFEnv
from phase1.tasks import generate_taskset
from phase1.ddqn import DoubleDQNAgent

from .io import ensure_dir, save_json, save_rows_csv, save_assignments_csv, load_tasks_csv
from .classifier import DistanceToMeanClassifier
from .label_dataset import collect_labeled_states
from .evaluate import run_method_ddqn, run_method_distance_to_mean, run_method_random
from .plots import plot_compare_qos_energy, plot_feature_distributions

# reuse gantt from phase1
from phase1.plots import plot_gantt


def load_ddqn_agent(weights_path: str) -> DoubleDQNAgent:
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])

    agent = DoubleDQNAgent(state_dim, action_dim)
    agent.q.load_state_dict(ckpt["q_state_dict"])
    agent.qt.load_state_dict(ckpt["target_state_dict"])
    agent.q.eval()
    agent.qt.eval()
    return agent


def safe_taskgen_kwargs(cfg: Phase1Config, which: str) -> dict:
    """
    Return taskgen kwargs if present in config, else {}.
    which: 'train' or 'test'
    """
    section = getattr(cfg, which)
    if hasattr(section, "taskgen") and hasattr(section.taskgen, "to_kwargs"):
        return section.taskgen.to_kwargs()
    return {}


def safe_power_kwargs(cfg: Phase1Config, which: str) -> dict:
    """
    Return power kwargs if present in config, else {}.
    which: 'train' or 'test'
    """
    section = getattr(cfg, which)
    if hasattr(section, "power") and hasattr(section.power, "to_kwargs"):
        return section.power.to_kwargs()
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to double_dqn_weights.pt from Phase 1")
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory")
    parser.add_argument("--label_episodes", type=int, default=60, help="How many episodes to generate for labeling dataset")
    parser.add_argument("--label_n_tasks", type=int, default=120, help="Tasks per labeling episode")
    args = parser.parse_args()

    cfg = Phase1Config()  # reuse whatever config exists (old/new)
    agent = load_ddqn_agent(args.weights)

    # infer phase1 results dir from weights path
    phase1_dir = os.path.dirname(os.path.abspath(args.weights))
    tasks_csv = os.path.join(phase1_dir, "test_tasks.csv")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("results_phase2", stamp)
    ensure_dir(outdir)

    # Gather optional kwargs (backward compatible)
    train_taskgen = safe_taskgen_kwargs(cfg, "train")
    test_taskgen = safe_taskgen_kwargs(cfg, "test")
    train_power = safe_power_kwargs(cfg, "train")
    test_power = safe_power_kwargs(cfg, "test")

    # Reward weights exist in old config too (but guard anyway)
    meet_bonus = getattr(cfg.train, "meet_bonus", 1.0)
    miss_penalty = getattr(cfg.train, "miss_penalty", 5.0)
    energy_weight = getattr(cfg.train, "energy_weight", 0.02)

    # 1) Load test task set (preferred) or generate if missing
    if os.path.exists(tasks_csv):
        test_tasks = load_tasks_csv(tasks_csv)
        test_source = {"type": "loaded_csv", "path": tasks_csv}
    else:
        # fall back generation
        test_seed = getattr(cfg.test, "seed", getattr(cfg, "seed", 0))
        np.random.seed(test_seed)
        test_tasks = generate_taskset(
            n_tasks=getattr(cfg.test, "n_tasks", 160),
            horizon=getattr(cfg.test, "horizon", 600),
            **test_taskgen,
        )
        test_source = {"type": "generated", "seed": int(test_seed)}

    # 2) Build labeled dataset using DDQN (Phase 2 output #1)
    X, y = collect_labeled_states(
        agent=agent,
        episodes=args.label_episodes,
        n_tasks=args.label_n_tasks,
        horizon=getattr(cfg.train, "horizon", 600),
        taskgen_kwargs=train_taskgen,
        power_kwargs=train_power,
        meet_bonus=meet_bonus,
        miss_penalty=miss_penalty,
        energy_weight=energy_weight,
    )

    # 3) Train distance-to-mean classifier (Phase 2 output #2)
    clf = DistanceToMeanClassifier(standardize=True).fit(X, y)
    mean0, mean1 = clf.means()

    # 4) Evaluate three methods on the same test set (Phase 2 output #3)
    def make_env():
        return HeteroEDFEnv(
            test_tasks,
            **test_power,  # may be {}
            meet_bonus=meet_bonus,
            miss_penalty=miss_penalty,
            energy_weight=energy_weight,
        )

    res_ddqn = run_method_ddqn(agent, make_env())
    res_dist = run_method_distance_to_mean(clf, make_env())
    res_rand = run_method_random(make_env(), seed=getattr(cfg, "seed", 0))

    results = {"ddqn": res_ddqn, "dist_mean": res_dist, "random": res_rand}

    # 5) Charts
    figs = {}

    # gantt per method per core
    for m, res in results.items():
        figs[f"{m}_gantt_big"] = plot_gantt(res["segments"]["big"], f"{m} - big core", max_rows=80, label_every=10)
        figs[f"{m}_gantt_little"] = plot_gantt(res["segments"]["little"], f"{m} - little core", max_rows=80, label_every=10)

    # feature distributions per method
    for m, res in results.items():
        figs.update(plot_feature_distributions(res["feature_rows"], method_name=m))

    # comparison charts
    figs.update(plot_compare_qos_energy(results))

    # 6) Save outputs
    ensure_dir(os.path.join(outdir, "plots"))
    for name, fig in figs.items():
        fig.savefig(os.path.join(outdir, "plots", f"{name}.png"), dpi=200, bbox_inches="tight")

    summary = {
        "phase1_dir_inferred": phase1_dir,
        "test_source": test_source,
        "label_dataset": {
            "X_shape": list(X.shape),
            "y_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        },
        "classifier": {
            "standardize": True,
            "mean0_norm": mean0.tolist(),
            "mean1_norm": mean1.tolist(),
        },
        "methods": {
            m: {
                "total_reward": results[m]["total_reward"],
                "steps": results[m]["steps"],
                "metrics": results[m]["metrics"],
            }
            for m in results
        },
        "config_used": cfg.to_dict(),
        "kwargs_used": {
            "train_taskgen": train_taskgen,
            "test_taskgen": test_taskgen,
            "train_power": train_power,
            "test_power": test_power,
            "reward": {"meet_bonus": meet_bonus, "miss_penalty": miss_penalty, "energy_weight": energy_weight},
        },
    }
    save_json(summary, os.path.join(outdir, "phase2_results.json"))

    for m, res in results.items():
        save_assignments_csv(res["assignments"], os.path.join(outdir, f"assignments_{m}.csv"))
        save_rows_csv(res["feature_rows"], os.path.join(outdir, f"task_features_{m}.csv"))

    print(f"\nPhase 2 saved to: {outdir}")
    print("Saved:")
    print("  phase2_results.json")
    print("  assignments_*.csv, task_features_*.csv")
    print("  plots/*.png (gantt + feature distributions + comparisons)")


if __name__ == "__main__":
    main()
