# phase1/tasks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
import csv

@dataclass(frozen=True)
class TaskSpec:
    tid: int
    arrival: int
    deadline: int
    exec_big: int
    exec_little: int

def generate_taskset(
    n_tasks: int,
    horizon: int,
    interarrival_mean: float = 5.0,
    base_exec_range: Tuple[int, int] = (2, 20),
    big_speedup: float = 2,
    little_slowdown: float = 1.0,
    deadline_slack_range: Tuple[float, float] = (1.5, 4.0),
) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    t = 0.0
    for tid in range(n_tasks):
        dt = np.random.exponential(interarrival_mean)
        t = min(horizon, t + dt)
        arrival = int(round(t))

        base = np.random.randint(base_exec_range[0], base_exec_range[1] + 1)
        exec_l = int(math.ceil(base / little_slowdown))
        exec_b = max(1, int(math.ceil(base / big_speedup)))

        slack = np.random.uniform(deadline_slack_range[0], deadline_slack_range[1])
        rel_deadline = max(1, int(math.ceil(slack * exec_l)))
        deadline = arrival + rel_deadline

        tasks.append(TaskSpec(tid=tid, arrival=arrival, deadline=deadline,
                              exec_big=exec_b, exec_little=exec_l))

    tasks.sort(key=lambda x: x.arrival)
    return tasks

def save_tasks_csv(tasks: List[TaskSpec], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "arrival", "deadline", "exec_big", "exec_little"])
        for t in tasks:
            w.writerow([t.tid, t.arrival, t.deadline, t.exec_big, t.exec_little])
