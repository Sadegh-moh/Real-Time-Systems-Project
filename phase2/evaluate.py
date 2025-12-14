# phase2/evaluate.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from phase1.env import HeteroEDFEnv
from phase1.tasks import TaskSpec
from phase1.ddqn import DoubleDQNAgent
from .classifier import DistanceToMeanClassifier


def _current_task_feature_row(env: HeteroEDFEnv, method: str, action: int) -> Dict[str, Any]:
    """
    Record raw task features (not normalized) for distribution plots.
    """
    t = env.current_arriving
    assert t is not None
    slack = int(max(0, t.deadline - env.time))
    return {
        "method": method,
        "tid": int(t.tid),
        "time": int(env.time),
        "assigned_core": int(action),  # 0 big, 1 little
        "exec_big": int(t.exec_big),
        "exec_little": int(t.exec_little),
        "slack": slack,
        "deadline": int(t.deadline),
        "arrival": int(t.arrival),
    }


def run_method_ddqn(agent: DoubleDQNAgent, env: HeteroEDFEnv) -> Dict[str, Any]:
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    rows: List[Dict[str, Any]] = []

    while not done:
        action = agent.act(state, epsilon=0.0)
        rows.append(_current_task_feature_row(env, "ddqn", action))
        state, r, done, info = env.step(action)
        total_reward += float(r)
        steps += 1

    return {
        "method": "ddqn",
        "total_reward": float(total_reward),
        "steps": int(steps),
        "metrics": env.metrics_by_core(),
        "segments": env.gantt_segments(),
        "assignments": dict(env.assigned_core),
        "feature_rows": rows,
    }


def run_method_random(env: HeteroEDFEnv, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    rows: List[Dict[str, Any]] = []

    while not done:
        action = int(rng.randint(0, 2))
        rows.append(_current_task_feature_row(env, "random", action))
        state, r, done, info = env.step(action)
        total_reward += float(r)
        steps += 1

    return {
        "method": "random",
        "total_reward": float(total_reward),
        "steps": int(steps),
        "metrics": env.metrics_by_core(),
        "segments": env.gantt_segments(),
        "assignments": dict(env.assigned_core),
        "feature_rows": rows,
    }


def run_method_distance_to_mean(clf: DistanceToMeanClassifier, env: HeteroEDFEnv) -> Dict[str, Any]:
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    rows: List[Dict[str, Any]] = []

    while not done:
        action = int(clf.predict_one(state))
        rows.append(_current_task_feature_row(env, "dist_mean", action))
        state, r, done, info = env.step(action)
        total_reward += float(r)
        steps += 1

    return {
        "method": "dist_mean",
        "total_reward": float(total_reward),
        "steps": int(steps),
        "metrics": env.metrics_by_core(),
        "segments": env.gantt_segments(),
        "assignments": dict(env.assigned_core),
        "feature_rows": rows,
    }
