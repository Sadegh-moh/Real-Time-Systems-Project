# phase1/io_utils.py
from __future__ import annotations
from typing import Dict, List
import os
import json
import csv
import torch
from .edf import Segment
from .ddqn import DoubleDQNAgent
from .tasks import TaskSpec, save_tasks_csv

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_assignments_csv(assignments: Dict[int, int], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "assigned_core"])  # 0 big, 1 little
        for tid in sorted(assignments.keys()):
            w.writerow([tid, assignments[tid]])

def save_segments_csv(segments: Dict[str, List[Segment]], outdir: str, prefix: str) -> None:
    ensure_dir(outdir)
    for core_name, segs in segments.items():
        p = os.path.join(outdir, f"{prefix}_segments_{core_name}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tid", "start", "end"])
            for s in segs:
                w.writerow([s.tid, s.start, s.end])

def write_summary_text(path: str, res_random: Dict, res_rl: Dict) -> None:
    def fmt(res: Dict) -> str:
        m = res["metrics"]
        lines = []
        lines.append(f"total_reward: {res['total_reward']:.3f}")
        lines.append(f"steps(decisions): {res['steps']}")
        for core in ["big", "little"]:
            lines.append(
                f"{core:6s}: completed={m[core]['completed']:4d} met={m[core]['met']:4d} "
                f"qos={m[core]['qos']:.3f} energy={m[core]['energy']:.2f}"
            )
        return "\n".join(lines)

    with open(path, "w") as f:
        f.write("=== Random ===\n")
        f.write(fmt(res_random) + "\n\n")
        f.write("=== Double DQN ===\n")
        f.write(fmt(res_rl) + "\n")

def save_model(agent: DoubleDQNAgent, path: str) -> None:
    torch.save(
        {
            "q_state_dict": agent.q.state_dict(),
            "target_state_dict": agent.qt.state_dict(),
            "device": agent.device,
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
        },
        path,
    )

def save_all_results(
    outdir: str,
    config: Dict,
    test_tasks: List[TaskSpec],
    agent: DoubleDQNAgent,
    res_random: Dict,
    res_rl: Dict,
    figs: Dict[str, "object"],  # matplotlib Figure
) -> None:
    ensure_dir(outdir)

    # results.json
    payload = {
        "config": config,
        "random": {
            "total_reward": res_random["total_reward"],
            "steps": res_random["steps"],
            "metrics": res_random["metrics"],
        },
        "double_dqn": {
            "total_reward": res_rl["total_reward"],
            "steps": res_rl["steps"],
            "metrics": res_rl["metrics"],
        },
    }
    save_json(payload, os.path.join(outdir, "results.json"))

    # tasks
    save_tasks_csv(test_tasks, os.path.join(outdir, "test_tasks.csv"))

    # assignments
    save_assignments_csv(res_random["assignments"], os.path.join(outdir, "assignments_random.csv"))
    save_assignments_csv(res_rl["assignments"], os.path.join(outdir, "assignments_double_dqn.csv"))

    # segments
    seg_dir = os.path.join(outdir, "segments")
    save_segments_csv(res_random["segments"], seg_dir, prefix="random")
    save_segments_csv(res_rl["segments"], seg_dir, prefix="double_dqn")

    # model
    save_model(agent, os.path.join(outdir, "double_dqn_weights.pt"))

    # summary
    write_summary_text(os.path.join(outdir, "summary.txt"), res_random, res_rl)

    # plots
    plot_dir = os.path.join(outdir, "plots")
    ensure_dir(plot_dir)
    for name, fig in figs.items():
        fig.savefig(os.path.join(plot_dir, f"{name}.png"), dpi=200, bbox_inches="tight")

def save_training_history_csv(history, path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return", "steps", "epsilon", "mean_loss"])
        for i in range(len(history.episode_return)):
            w.writerow([i, history.episode_return[i], history.episode_steps[i],
                        history.episode_epsilon[i], history.episode_mean_loss[i]])

