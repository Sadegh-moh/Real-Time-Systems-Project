# phase2/io.py
from __future__ import annotations
from typing import List, Dict, Any
import os
import json
import csv

from phase1.tasks import TaskSpec


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_rows_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_assignments_csv(assignments: Dict[int, int], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "assigned_core"])  # 0 big, 1 little
        for tid in sorted(assignments.keys()):
            w.writerow([tid, assignments[tid]])


def load_tasks_csv(path: str) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tasks.append(
                TaskSpec(
                    tid=int(row["tid"]),
                    arrival=int(row["arrival"]),
                    deadline=int(row["deadline"]),
                    exec_big=int(row["exec_big"]),
                    exec_little=int(row["exec_little"]),
                )
            )
    tasks.sort(key=lambda t: t.arrival)
    return tasks
