# phase1/config.py
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple


@dataclass
class TaskGenConfig:
    # arrival process
    interarrival_mean: float = 5.0

    # base workload (converted to exec_big/exec_little)
    base_exec_min: int = 2
    base_exec_max: int = 20

    # heterogeneity (big faster)
    big_speedup: float = 1.7
    little_slowdown: float = 1.0

    # deadline slack factor range (used to set relative deadlines)
    deadline_slack_min: float = 1.5
    deadline_slack_max: float = 4.0

    def to_kwargs(self) -> Dict:
        """Convert to arguments expected by generate_taskset()."""
        return {
            "interarrival_mean": self.interarrival_mean,
            "base_exec_range": (self.base_exec_min, self.base_exec_max),
            "big_speedup": self.big_speedup,
            "little_slowdown": self.little_slowdown,
            "deadline_slack_range": (self.deadline_slack_min, self.deadline_slack_max),
        }


@dataclass
class TrainConfig:
    # training workload size
    episodes: int = 360
    n_tasks: int = 120
    horizon: int = 600
    seed: int = 0

    # task generation hyperparameters (ALL HERE)
    taskgen: TaskGenConfig = field(default_factory=TaskGenConfig)

    # DQN hyperparams
    lr: float = 1e-3
    gamma: float = 0.9
    batch_size: int = 128
    buffer_capacity: int = 200_000
    target_update: int = 500

    # exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000

    # training mechanics
    warmup_steps: int = 5_000
    max_steps_per_episode: int = 2_000

    # reward weights
    meet_bonus: float = 1.0
    miss_penalty: float = 2.0
    energy_weight: float = 0.04


@dataclass
class TestConfig:
    n_tasks: int = 160
    horizon: int = 600
    seed: int = 0  # deterministic test set

    # task generation hyperparameters (can be same or different)
    taskgen: TaskGenConfig = field(default_factory=TaskGenConfig)


@dataclass
class Phase1Config:
    seed: int = 0
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)

    def to_dict(self):
        return asdict(self)
