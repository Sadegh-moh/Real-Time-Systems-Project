# phase1/env.py
from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np
from .tasks import TaskSpec
from .edf import EDFCore, Segment

class HeteroEDFEnv:
    """
    MDP:
      - State: (time, arriving task features, queue summaries for each core)
      - Action: choose core for arriving task (0 big, 1 little)
      - Transition: assign task then EDF-simulate until next arrival (or finish)
      - Reward: met bonuses - miss penalties - energy cost over interval
    """
    def __init__(
        self,
        tasks: List[TaskSpec],
        power_big_active: float = 3.5,
        power_big_idle: float = 0.6,
        power_little_active: float = 1.2,
        power_little_idle: float = 0.25,
        meet_bonus: float = 1.0,
        miss_penalty: float = 5.0,
        energy_weight: float = 0.02,
        max_time_cap: int = 20000,
    ):
        self.tasks = tasks
        self.max_time_cap = max_time_cap

        self.big = EDFCore("big", power_big_active, power_big_idle)
        self.little = EDFCore("little", power_little_active, power_little_idle)

        self.meet_bonus = meet_bonus
        self.miss_penalty = miss_penalty
        self.energy_weight = energy_weight

        self.time: int = 0
        self.idx: int = 0
        self.current_arriving: Optional[TaskSpec] = None

        self.assigned_core: Dict[int, int] = {}
        self._energy_prev_total = 0.0
        self._completed_prev = 0
        self._met_prev = 0
        self._miss_prev = 0

        self.horizon = max([t.deadline for t in tasks]) if tasks else 1
        self.max_exec = max([max(t.exec_big, t.exec_little) for t in tasks]) if tasks else 1
        self.max_slack = max([(t.deadline - t.arrival) for t in tasks]) if tasks else 1
        self.max_queue = max(10, int(len(tasks) * 0.5))
        self.max_work = max(20, int(sum([t.exec_little for t in tasks]) * 0.2))

    def state_dim(self) -> int:
        return 1 + 3 + 3 * 2

    def action_dim(self) -> int:
        return 2

    def reset(self) -> np.ndarray:
        # rebuild cores (keeping power values)
        self.big = EDFCore("big", self.big.active_power, self.big.idle_power)
        self.little = EDFCore("little", self.little.active_power, self.little.idle_power)

        self.time = 0
        self.idx = 0
        self.current_arriving = None
        self.assigned_core = {}

        self._energy_prev_total = 0.0
        self._completed_prev = 0
        self._met_prev = 0
        self._miss_prev = 0

        if not self.tasks:
            return np.zeros(self.state_dim(), dtype=np.float32)

        first_arrival = self.tasks[0].arrival
        self._simulate_until(first_arrival)

        self.current_arriving = self.tasks[self.idx]
        return self._get_state()

    def done(self) -> bool:
        no_more_arrivals = self.idx >= len(self.tasks)
        empty = (len(self.big) == 0 and len(self.little) == 0)
        return no_more_arrivals and empty

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.current_arriving is None:
            raise RuntimeError("step() called with no arriving task.")
        if action not in (0, 1):
            raise ValueError("Action must be 0 (big) or 1 (little).")

        spec = self.current_arriving
        self.assigned_core[spec.tid] = action
        if action == 0:
            self.big.add_task(spec, is_big=True)
        else:
            self.little.add_task(spec, is_big=False)

        self.idx += 1

        if self.idx < len(self.tasks):
            next_time = self.tasks[self.idx].arrival
            self._simulate_until(next_time)
            self.current_arriving = self.tasks[self.idx]
            next_state = self._get_state()
            terminal = False
        else:
            self._simulate_until_empty()
            self.current_arriving = None
            next_state = np.zeros(self.state_dim(), dtype=np.float32)
            terminal = True

        reward, info = self._compute_interval_reward()
        if self.time > self.max_time_cap:
            terminal = True
        return next_state, reward, terminal, info

    # ---------- simulation ----------
    def _simulate_until(self, target_time: int) -> None:
        while self.time < target_time:
            self.big.step_one_tick(self.time)
            self.little.step_one_tick(self.time)
            self.time += 1

    def _simulate_until_empty(self) -> None:
        guard = 0
        while not self.done():
            self.big.step_one_tick(self.time)
            self.little.step_one_tick(self.time)
            self.time += 1
            guard += 1
            if guard > self.max_time_cap:
                break
        self.big.finalize(self.time)
        self.little.finalize(self.time)

    # ---------- state ----------
    def _get_state(self) -> np.ndarray:
        tnorm = float(self.time) / float(max(1, self.horizon))

        task = self.current_arriving
        assert task is not None

        slack = max(0, task.deadline - self.time)
        task_feat = np.array([
            task.exec_big / self.max_exec,
            task.exec_little / self.max_exec,
            slack / max(1, self.max_slack),
        ], dtype=np.float32)

        big_feat = self._core_features(self.big)
        little_feat = self._core_features(self.little)

        state = np.concatenate([np.array([tnorm], dtype=np.float32), task_feat, big_feat, little_feat], axis=0)
        return state.astype(np.float32)

    def _core_features(self, core: EDFCore) -> np.ndarray:
        q_len = len(core.ready)
        total_work = sum(ct.remaining for ct in core.ready) if core.ready else 0
        if core.ready:
            min_slack = min(max(0, ct.spec.deadline - self.time) for ct in core.ready)
        else:
            min_slack = self.max_slack

        return np.array([
            min(1.0, q_len / self.max_queue),
            min(1.0, total_work / self.max_work),
            min(1.0, min_slack / max(1, self.max_slack)),
        ], dtype=np.float32)

    # ---------- reward ----------
    def _compute_interval_reward(self) -> Tuple[float, Dict]:
        energy_total = self.big.energy + self.little.energy

        completed_total = len(self.big.completed) + len(self.little.completed)
        met_total = sum(1 for _, _, met in self.big.completed if met) + sum(1 for _, _, met in self.little.completed if met)
        miss_total = completed_total - met_total

        energy_delta = energy_total - self._energy_prev_total
        met_delta = met_total - self._met_prev
        miss_delta = miss_total - self._miss_prev

        self._energy_prev_total = energy_total
        self._met_prev = met_total
        self._miss_prev = miss_total

        reward = (self.meet_bonus * met_delta) - (self.miss_penalty * miss_delta) - (self.energy_weight * energy_delta)

        info = {
            "energy_delta": float(energy_delta),
            "met_delta": int(met_delta),
            "miss_delta": int(miss_delta),
            "time": int(self.time),
        }
        return float(reward), info

    # ---------- metrics ----------
    def metrics_by_core(self) -> Dict:
        def summarize(core: EDFCore) -> Dict:
            total = len(core.completed)
            met = sum(1 for _, _, m in core.completed if m)
            qos = (met / total) if total > 0 else 0.0
            return {"completed": total, "met": met, "qos": qos, "energy": float(core.energy)}
        return {"big": summarize(self.big), "little": summarize(self.little)}

    def gantt_segments(self) -> Dict[str, List[Segment]]:
        return {"big": self.big.segments, "little": self.little.segments}
