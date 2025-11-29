"""
Firm cooperation model:
Replicator equilibrium vs bandit-style RL learning (warm-start vs random-start).

- Firms play a symmetric 2x2 game:
    C = invest in joint R&D / shared infrastructure
    D = act alone / free-ride

    C,C -> (3,3)
    C,D -> (1,4)
    D,C -> (4,1)
    D,D -> (0,0)

  This is a Hawk–Dove style game with a stable interior equilibrium x* = 0.5
  under the replicator dynamics.

- Multi-agent bandit RL:
    Each agent i has parameter theta_i, p_i = sigmoid(theta_i) = P(C).
    At each round:
        - Each agent samples action a_i ∈ {0,1} (C=1, D=0) from Bernoulli(p_i).
        - Agents are randomly paired and receive payoffs.
        - Policy gradient update:
            theta_i ← theta_i + η (r_i - b) (a_i - p_i),

- Warm-start:
    theta initialized so that p_i ≈ 0.65 (slightly above x*),
    reflecting a replicator-informed prior that cooperation is relatively attractive.
- Random-start:
    theta initialized so that p_i ≈ 0.02 (almost full defection),
    so it takes many rounds to climb toward x*.

- Both eventually converge toward the same equilibrium (x*),
  but warm-start converges in far fewer rounds.

Dependencies:
    pip install numpy matplotlib
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# 1. Payoff & Config
# ==============================

@dataclass
class PayoffParams:
    """
    Firm-level 2x2 game:

        C = invest in joint R&D / shared infrastructure
        D = act alone / free-ride

        C,C -> (3,3)
        C,D -> (1,4)
        D,C -> (4,1)
        D,D -> (0,0)
    """
    R: float
    T: float
    S: float
    P: float


def default_payoff() -> PayoffParams:
    # Hawk–Dove style firm game with stable interior equilibrium x* = 0.5
    return PayoffParams(
        R=3.0,
        T=4.0,
        S=1.0,
        P=0.0,
    )


@dataclass
class TrainingConfig:
    num_agents: int = 200         # population size
    num_rounds: int = 10000        # learning steps
    eta: float = 0.025            # learning rate (작게 해서 수렴 속도 차이 보이게)
    rng_seed: int = 42
    log_every: int = 500          # print every X rounds
    plot_path: str = "cooperation.png"


# ==============================
# 2. Replicator Equilibrium
# ==============================

def expected_payoffs(x: float, params: PayoffParams) -> Tuple[float, float]:
    """
    Expected payoff of C and D when fraction of cooperators is x.
    (Stateless population-level approximation)
    """
    R, T, S, P = params.R, params.T, params.S, params.P
    pi_C = x * R + (1.0 - x) * S
    pi_D = x * T + (1.0 - x) * P
    return pi_C, pi_D


def replicator_rhs(x: float, params: PayoffParams) -> float:
    pi_C, pi_D = expected_payoffs(x, params)
    pi_bar = x * pi_C + (1.0 - x) * pi_D
    return x * (pi_C - pi_bar)


def replicator_equilibrium(params: PayoffParams) -> Optional[float]:
    """
    Analytic interior equilibrium x* such that pi_C(x*) = pi_D(x*).

    xR + (1-x)S = xT + (1-x)P
    x(R - S) + S = x(T - P) + P
    x[(R - S) - (T - P)] = P - S
    x* = (P - S) / ((R - S) - (T - P))
    """
    R, T, S, P = params.R, params.T, params.S, params.P
    denom = (R - S) - (T - P)
    if abs(denom) < 1e-8:
        return None
    x_star = (P - S) / denom
    if 0.0 < x_star < 1.0:
        return x_star
    return None


# ==============================
# 3. Bandit-style RL agents
# ==============================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_bandit_learning(
    cfg: TrainingConfig,
    payoff: PayoffParams,
    warm_start: bool,
    x_star: float,
) -> np.ndarray:
    """
    Multi-agent bandit-style RL:
      Each agent i has parameter theta_i; p_i = sigmoid(theta_i) = P(C).
      At each round:
        - Each agent samples action a_i ∈ {0,1} from Bernoulli(p_i).
        - Randomly paired → payoffs.
        - Policy gradient update:
            theta_i ← theta_i + η (r_i - b) (a_i - p_i).

    Returns:
      coop_history[t] = average cooperation probability at round t.
    """
    rng = np.random.default_rng(cfg.rng_seed)

    # --- initialize theta ---
    if warm_start:
        # warm-start는 x*보다 살짝 높은 협력률에서 시작하도록 설정
        # 예: target_p ≈ 0.65
        target_p = min(max(x_star + 0.15, 0.0), 0.99)  # x*+0.15, capped below 1
        base = np.log(target_p / (1.0 - target_p))     # logit(target_p)
        theta = rng.normal(loc=base, scale=0.4, size=cfg.num_agents)
    else:
        # random-start: 거의 전부 defection에서 시작 (p ≈ 0.02)
        base = -4.0  # sigmoid(-4) ≈ 0.018
        theta = rng.normal(loc=base, scale=0.5, size=cfg.num_agents)

    # baseline for variance reduction
    baseline = 0.0
    beta = 0.01  # baseline update rate

    coop_history = []
    R, T, S, P = payoff.R, payoff.T, payoff.S, payoff.P

    for t in range(cfg.num_rounds):
        # current cooperation probabilities
        p = sigmoid(theta)
        # actions: 1 (C), 0 (D)
        actions = rng.binomial(1, p)

        # random pairing
        indices = np.arange(cfg.num_agents)
        rng.shuffle(indices)
        rewards = np.zeros(cfg.num_agents, dtype=float)

        num_pairs = len(indices) // 2
        for i in range(num_pairs):
            idx1 = indices[2 * i]
            idx2 = indices[2 * i + 1]
            a1 = actions[idx1]
            a2 = actions[idx2]

            # payoff matrix
            if a1 == 1 and a2 == 1:
                r1, r2 = R, R
            elif a1 == 1 and a2 == 0:
                r1, r2 = S, T
            elif a1 == 0 and a2 == 1:
                r1, r2 = T, S
            else:
                r1, r2 = P, P

            rewards[idx1] += r1
            rewards[idx2] += r2

        # update baseline (running average)
        avg_reward = rewards.mean()
        baseline = (1.0 - beta) * baseline + beta * avg_reward

        # policy gradient update for each agent
        theta += cfg.eta * (rewards - baseline) * (actions - p)

        coop_history.append(p.mean())

        # logging
        if (t + 1) % cfg.log_every == 0:
            label = "WARM" if warm_start else "RAND"
            print(f"[{label}] round {t+1:5d}/{cfg.num_rounds} | avg p(C) = {p.mean():.3f}")

    return np.array(coop_history, dtype=float)


def plot_results(
    warm_hist: np.ndarray,
    rand_hist: np.ndarray,
    x_star: float,
    cfg: TrainingConfig,
):
    t = np.arange(len(warm_hist))

    plt.figure(figsize=(8, 5))
    plt.plot(t, warm_hist, label="Warm-start", linewidth=2)
    plt.plot(t, rand_hist, label="Random-start", linewidth=2, linestyle="--")
    plt.axhline(x_star, color="gray", linestyle=":", linewidth=1.5, label="Replicator equilibrium x*")

    plt.xlabel("Round")
    plt.ylabel("Average cooperation probability")
    plt.title("Convergence to the same equilibrium: warm-start vs random-start")
    plt.ylim(0.0, 1.0)  # y-axis fixed between 0 and 1
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=200)
    print(f"[INFO] Saved plot to {cfg.plot_path}")
    plt.close()


# ==============================
# 4. Main
# ==============================

def main():
    cfg = TrainingConfig()
    payoff = default_payoff()

    x_star = replicator_equilibrium(payoff)
    if x_star is None:
        raise RuntimeError("No interior equilibrium for this payoff.")

    print(f"[INFO] Replicator equilibrium x* ≈ {x_star:.4f}")

    print("=== Running warm-start experiment (bandit RL) ===")
    warm_hist = run_bandit_learning(cfg, payoff, warm_start=True, x_star=x_star)

    print("\n=== Running random-start experiment (bandit RL) ===")
    rand_hist = run_bandit_learning(cfg, payoff, warm_start=False, x_star=x_star)

    plot_results(warm_hist, rand_hist, x_star, cfg)


if __name__ == "__main__":
    main()
