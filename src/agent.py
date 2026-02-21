# src/agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


@dataclass
class BanditPGConfig:
    eta: float = 0.002     # learning rate
    beta: float = 0.01     # baseline EMA rate
    seed: int = 42


class BanditPGPopulation:
    """
    Population of N agents with Bernoulli policy p_i = sigmoid(theta_i).
    Update rule (baseline-corrected):
      theta_i <- theta_i + eta * (r_i - b) * (a_i - p_i)
    Baseline b updated as EMA of average reward.
    """

    def __init__(self, n_agents: int, cfg: BanditPGConfig):
        if n_agents < 2:
            raise ValueError("n_agents must be >= 2")
        self.n = n_agents
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # parameters and baseline
        self.theta = np.zeros(self.n, dtype=float)
        self.baseline = 0.0

    def set_theta_normal(self, mean: float, std: float):
        self.theta = self.rng.normal(loc=mean, scale=std, size=self.n)

    def warm_start(self, x_star: float, delta: float = 0.15, std: float = 0.4):
        """
        Initialize near replicator equilibrium (optionally with offset).
        target_p = clip(x_star + delta, 0, 0.99) to create visible convergence.
        """
        target_p = min(max(x_star + delta, 0.0), 0.99)
        mu = logit(target_p)
        self.set_theta_normal(mean=mu, std=std)

    def random_start_low_coop(self, p_low: float = 0.02, std: float = 0.5):
        """
        Initialize near strong defection prior p(C) ~ p_low.
        """
        mu = logit(p_low)
        self.set_theta_normal(mean=mu, std=std)

    def probs(self) -> np.ndarray:
        return sigmoid(self.theta)

    def sample_actions(self) -> np.ndarray:
        """
        Sample a_i ~ Bernoulli(p_i) for all agents.
        """
        p = self.probs()
        a = self.rng.binomial(1, p)
        return a.astype(int)

    def update_baseline(self, rewards: np.ndarray):
        avg_r = float(np.mean(rewards))
        self.baseline = (1.0 - self.cfg.beta) * self.baseline + self.cfg.beta * avg_r

    def policy_gradient_update(self, rewards: np.ndarray, actions: np.ndarray):
        """
        theta_i <- theta_i + eta * (r_i - b) * (a_i - p_i)
        """
        p = self.probs()
        centered = rewards - self.baseline
        self.theta = self.theta + self.cfg.eta * centered * (actions - p)

    def step_update(self, rewards: np.ndarray, actions: np.ndarray):
        """
        One learning step: baseline update then parameter update.
        """
        self.update_baseline(rewards)
        self.policy_gradient_update(rewards, actions)

    def avg_coop_prob(self) -> float:
        return float(np.mean(self.probs()))
