# phase1/edf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .tasks import TaskSpec

@dataclass
class CoreTask:
    spec: TaskSpec
    remaining: int

@dataclass
class Segment:
    tid: int
    start: int
    end: int  # [start, end)

class EDFCore:
    """
    Preemptive EDF, 1-tick time base.
    """
    def __init__(self, name: str, active_power: float, idle_power: float):
        self.name = name
        self.active_power = active_power
        self.idle_power = idle_power

        self.ready: List[CoreTask] = []
        self.segments: List[Segment] = []
        self.energy: float = 0.0
        self.completed: List[Tuple[int, int, bool]] = []  # (tid, completion_time, met_deadline)

        self._last_running_tid: Optional[int] = None
        self._last_seg_start: Optional[int] = None

    def __len__(self) -> int:
        return len(self.ready)

    def add_task(self, spec: TaskSpec, is_big: bool) -> None:
        exec_time = spec.exec_big if is_big else spec.exec_little
        self.ready.append(CoreTask(spec=spec, remaining=exec_time))

    def _pick_edf_index(self) -> Optional[int]:
        if not self.ready:
            return None
        best_i = 0
        best_deadline = self.ready[0].spec.deadline
        for i in range(1, len(self.ready)):
            d = self.ready[i].spec.deadline
            if d < best_deadline:
                best_deadline = d
                best_i = i
        return best_i

    def _close_segment(self, now: int) -> None:
        if self._last_running_tid is not None and self._last_seg_start is not None:
            if self._last_seg_start < now:
                self.segments.append(Segment(self._last_running_tid, self._last_seg_start, now))
        self._last_running_tid = None
        self._last_seg_start = None

    def step_one_tick(self, now: int) -> None:
        idx = self._pick_edf_index()
        if idx is None:
            self.energy += self.idle_power
            if self._last_running_tid is not None:
                self._close_segment(now)
            return

        task = self.ready[idx]
        tid = task.spec.tid

        # segment start/continue
        if self._last_running_tid != tid:
            if self._last_running_tid is not None:
                self._close_segment(now)
            self._last_running_tid = tid
            self._last_seg_start = now

        self.energy += self.active_power
        task.remaining -= 1

        if task.remaining <= 0:
            completion_time = now + 1
            met = completion_time <= task.spec.deadline
            self.completed.append((tid, completion_time, met))
            self.ready.pop(idx)

            # close segment exactly at completion boundary
            if self._last_running_tid == tid and self._last_seg_start is not None:
                if self._last_seg_start < completion_time:
                    self.segments.append(Segment(tid, self._last_seg_start, completion_time))
                self._last_running_tid = None
                self._last_seg_start = None

    def finalize(self, end_time: int) -> None:
        if self._last_running_tid is not None:
            self._close_segment(end_time)
