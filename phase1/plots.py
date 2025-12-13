# phase1/plots.py
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .edf import Segment
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from .edf import Segment

def plot_training_reward(episode_returns, window: int = 20):
    fig, ax = plt.subplots(figsize=(9, 4))

    r = np.array(episode_returns, dtype=float)
    ax.plot(r, label="episode return")

    if len(r) >= window:
        ma = np.convolve(r, np.ones(window)/window, mode="valid")
        ax.plot(np.arange(window-1, len(r)), ma, label=f"{window}-ep moving avg")

    ax.set_title("Training reward (episode return)")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_qos_energy(metrics_random: Dict, metrics_rl: Dict, title_prefix: str = "") -> Tuple[plt.Figure, plt.Figure]:
    cores = ["big", "little"]
    x = np.arange(len(cores))
    w = 0.35

    qos_rand = [metrics_random[c]["qos"] for c in cores]
    qos_rl = [metrics_rl[c]["qos"] for c in cores]
    en_rand = [metrics_random[c]["energy"] for c in cores]
    en_rl = [metrics_rl[c]["energy"] for c in cores]

    fig1 = plt.figure()
    plt.bar(x - w/2, qos_rand, width=w, label="Random")
    plt.bar(x + w/2, qos_rl, width=w, label="Double DQN")
    plt.xticks(x, cores)
    plt.ylim(0.0, 1.0)
    plt.ylabel("QoS (deadline-met ratio)")
    plt.title(f"{title_prefix}QoS per core")
    plt.legend()

    fig2 = plt.figure()
    plt.bar(x - w/2, en_rand, width=w, label="Random")
    plt.bar(x + w/2, en_rl, width=w, label="Double DQN")
    plt.xticks(x, cores)
    plt.ylabel("Energy (arbitrary units)")
    plt.title(f"{title_prefix}Energy per core")
    plt.legend()

    return fig1, fig2

def plot_gantt(
    segments: List[Segment],
    core_name: str,
    max_rows: int = 80,
    label_every: int = 10,     # show only every N labels (set to 0 to hide all)
    label_fontsize: int = 7,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    show = segments[:max_rows]
    y = np.arange(len(show))
    left = [s.start for s in show]
    width = [s.end - s.start for s in show]
    labels = [f"T{s.tid}" for s in show]

    ax.barh(y, width, left=left)

    # --- Improve vertical axis readability ---
    if label_every <= 0:
        ax.set_yticks([])  # hide labels completely
        ax.set_ylabel("segment index")
    else:
        idx = np.arange(0, len(show), label_every)
        ax.set_yticks(idx)
        ax.set_yticklabels([labels[i] for i in idx], fontsize=label_fontsize)
        ax.set_ylabel(f"segments (showing every {label_every})")

    # make the vertical axis line/ticks cleaner
    ax.spines["left"].set_linewidth(1.5)
    ax.tick_params(axis="y", length=3, width=1)

    ax.set_xlabel("time")
    ax.set_title(f"Task scheduling (EDF) on {core_name} core (first {len(show)} segments)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig

