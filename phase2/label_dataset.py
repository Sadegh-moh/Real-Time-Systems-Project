# phase2/label_dataset.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from phase1.env import HeteroEDFEnv
from phase1.tasks import generate_taskset
from phase1.ddqn import DoubleDQNAgent


def collect_labeled_states(
    agent: DoubleDQNAgent,
    episodes: int,
    n_tasks: int,
    horizon: int,
    taskgen_kwargs: Optional[dict] = None,
    power_kwargs: Optional[dict] = None,
    meet_bonus: float = 1.0,
    miss_penalty: float = 3.0,
    energy_weight: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the DDQN policy to label states at arrival times:
      X = state vectors
      y = action labels (0 big, 1 little)
    """
    taskgen_kwargs = taskgen_kwargs or {}
    power_kwargs = power_kwargs or {}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for _ in range(episodes):
        tasks = generate_taskset(n_tasks=n_tasks, horizon=horizon, **taskgen_kwargs)
        env = HeteroEDFEnv(
            tasks,
            **power_kwargs,
            meet_bonus=meet_bonus,
            miss_penalty=miss_penalty,
            energy_weight=energy_weight,
        )
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state, epsilon=0.0)
            X_list.append(state.copy())
            y_list.append(int(action))
            state, r, done, info = env.step(action)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y
