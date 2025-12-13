# phase1/ddqn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import HeteroEDFEnv
from .tasks import generate_taskset


# ----------------------------
# Training history (for reward plotting/logging)
# ----------------------------
@dataclass
class TrainHistory:
    episode_return: List[float]
    episode_steps: List[int]
    episode_epsilon: List[float]
    episode_mean_loss: List[float]  # NaN if no updates happened


# ----------------------------
# Replay buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.s = None
        self.ns = None
        self.a = None
        self.r = None
        self.d = None

    def _init(self, state_dim: int) -> None:
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.ns = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.d = np.zeros((self.capacity,), dtype=np.float32)

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        if self.s is None:
            self._init(s.shape[0])

        self.s[self.ptr] = s
        self.ns[self.ptr] = ns
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.d[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "s": torch.from_numpy(self.s[idx]),
            "ns": torch.from_numpy(self.ns[idx]),
            "a": torch.from_numpy(self.a[idx]),
            "r": torch.from_numpy(self.r[idx]),
            "d": torch.from_numpy(self.d[idx]),
        }


# ----------------------------
# Q network
# ----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Double DQN agent
# ----------------------------
class DoubleDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 2e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 200_000,
        batch_size: int = 256,
        target_update: int = 1000,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_steps = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNetwork(state_dim, action_dim).to(self.device)
        self.qt = QNetwork(state_dim, action_dim).to(self.device)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.rb = ReplayBuffer(capacity=buffer_capacity)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.action_dim))
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)
            qvals = self.q(s)
            return int(torch.argmax(qvals, dim=1).item())

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.rb.add(s, a, r, ns, done)

    def learn(self) -> Optional[float]:
        if self.rb.size < self.batch_size:
            return None

        b = self.rb.sample(self.batch_size)
        s = b["s"].to(self.device)
        ns = b["ns"].to(self.device)
        a = b["a"].to(self.device)
        r = b["r"].to(self.device)
        d = b["d"].to(self.device)

        # Q(s,a)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # a* = argmax_a Q_online(s',a)
        # y = r + gamma*(1-done)*Q_target(s', a*)
        with torch.no_grad():
            next_actions = torch.argmax(self.q(ns), dim=1)  # online selects
            q_next = self.qt(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # target evaluates
            target = r + self.gamma * (1.0 - d) * q_next

        # Huber loss (Smooth L1)
        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.qt.load_state_dict(self.q.state_dict())

        return float(loss.item())


# ----------------------------
# Train / Evaluate
# ----------------------------
def train_double_dqn(
    episodes: int,
    n_tasks: int,
    horizon: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    warmup_steps: int,
    max_steps_per_episode: int,
    seed: int,
    lr: float = 1e-4,
    gamma: float = 0.95,
    batch_size: int = 128,
    buffer_capacity: int = 200_000,
    target_update: int = 500,
    # Optional: pass reward weights into env
    meet_bonus: float = 1.0,
    miss_penalty: float = 5.0,
    energy_weight: float = 0.02,
    # NEW: all task-generation hyperparams are passed in here from config.py
    taskgen_kwargs: Optional[dict] = None,
) -> Tuple[DoubleDQNAgent, TrainHistory]:
    """
    Returns:
      agent: trained DoubleDQNAgent
      history: per-episode reward/loss logs (for plotting/saving)
    """
    taskgen_kwargs = taskgen_kwargs or {}

    # seeding
    np.random.seed(seed)
    torch.manual_seed(seed)

    # build demo env for dims
    demo_tasks = generate_taskset(50, horizon, **taskgen_kwargs)
    demo_env = HeteroEDFEnv(
        demo_tasks,
        meet_bonus=meet_bonus,
        miss_penalty=miss_penalty,
        energy_weight=energy_weight,
    )

    agent = DoubleDQNAgent(
        demo_env.state_dim(),
        demo_env.action_dim(),
        lr=lr,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update=target_update,
    )

    history = TrainHistory(
        episode_return=[],
        episode_steps=[],
        episode_epsilon=[],
        episode_mean_loss=[],
    )

    global_step = 0
    losses_all: List[float] = []

    for ep in range(episodes):
        env = HeteroEDFEnv(
            generate_taskset(n_tasks, horizon, **taskgen_kwargs),
            meet_bonus=meet_bonus,
            miss_penalty=miss_penalty,
            energy_weight=energy_weight,
        )

        state = env.reset()
        done = False

        ep_steps = 0
        ep_return = 0.0
        ep_losses: List[float] = []

        epsilon = epsilon_start

        while not done and ep_steps < max_steps_per_episode:
            # epsilon schedule
            if global_step < epsilon_decay_steps:
                frac = global_step / max(1, epsilon_decay_steps)
                epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            else:
                epsilon = epsilon_end

            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)

            agent.add(state, action, float(reward), next_state, done)
            state = next_state

            ep_return += float(reward)

            if global_step > warmup_steps:
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)
                    losses_all.append(loss)

            global_step += 1
            ep_steps += 1

        # log per-episode stats
        history.episode_return.append(float(ep_return))
        history.episode_steps.append(int(ep_steps))
        history.episode_epsilon.append(float(epsilon))
        history.episode_mean_loss.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))

        # progress print
        if (ep + 1) % max(1, episodes // 10) == 0:
            avg_loss_200 = float(np.mean(losses_all[-200:])) if losses_all else float("nan")
            print(
                f"[ep {ep+1:4d}/{episodes}] steps={ep_steps:4d} "
                f"eps={epsilon:.3f} avg_loss(200)={avg_loss_200:.4f} return={ep_return:.2f}"
            )

    return agent, history


def run_greedy_policy(agent: DoubleDQNAgent, env: HeteroEDFEnv) -> dict:
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = agent.act(state, epsilon=0.0)
        state, r, done, info = env.step(action)
        total_reward += float(r)
        steps += 1

    return {
        "total_reward": float(total_reward),
        "steps": int(steps),
        "metrics": env.metrics_by_core(),
        "segments": env.gantt_segments(),
        "assignments": dict(env.assigned_core),
    }
