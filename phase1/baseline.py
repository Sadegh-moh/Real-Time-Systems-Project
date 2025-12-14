# phase1/baseline.py
import numpy as np
from .env import HeteroEDFEnv

def run_random_policy(env: HeteroEDFEnv) -> dict:
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = np.random.randint(0, env.action_dim())
        state, r, done, info = env.step(action)
        total_reward += r
        steps += 1

    return {
        "total_reward": float(total_reward),
        "steps": int(steps),
        "metrics": env.metrics_by_core(),
        "segments": env.gantt_segments(),
        "assignments": dict(env.assigned_core),
    }
