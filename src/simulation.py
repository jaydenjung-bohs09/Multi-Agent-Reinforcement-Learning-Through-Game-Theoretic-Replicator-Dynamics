# src/simulation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import os
import numpy as np
import matplotlib.pyplot as plt

from src.game import PayoffMatrix, default_payoff_hawk_dove, load_payoff_time_series, payoff_from_series
from src.agent import BanditPGPopulation, BanditPGConfig


@dataclass
class SimConfig:
    n_agents: int = 200
    rounds: int = 8000
    log_every: int = 500

    # convergence metric params
    eps_list: Tuple[float, float] = (0.02, 0.01)
    hold_H: int = 200

    # warm-start params
    warm_delta: float = 0.15
    warm_std: float = 0.4
    rand_p_low: float = 0.02
    rand_std: float = 0.5

    # plotting
    plot_path: str = "results_coop_curve.png"
    ylim_0_1: bool = True

    # optional dynamic payoff
    use_dynamic_payoff: bool = False
    payoff_series_path: str = "data/cordis_processed/payoff_time_series.csv"
    start_year: int = 2014  # if dynamic, year = start_year + t


def random_pairing_rewards(
    rng: np.random.Generator,
    payoff: PayoffMatrix,
    actions: np.ndarray,
) -> np.ndarray:
    """
    Pair agents uniformly at random without replacement, assign rewards via payoff matrix.
    actions: array of shape (N,), a=1 -> C, a=0 -> D
    Returns rewards array shape (N,).
    """
    n = actions.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    rewards = np.zeros(n, dtype=float)
    pairs = n // 2
    for k in range(pairs):
        i = int(idx[2 * k])
        j = int(idx[2 * k + 1])
        r_i, r_j = payoff.payoff(int(actions[i]), int(actions[j]))
        rewards[i] += r_i
        rewards[j] += r_j
    return rewards


def compute_tau_eps(history: np.ndarray, target: float, eps: float, H: int) -> Optional[int]:
    """
    tau_eps = first t such that for all t' in [t, t+H], |history[t'] - target| <= eps.
    Returns 1-indexed round number, or None if not reached.
    """
    T = len(history)
    for t in range(T):
        end = min(T, t + H)
        window = history[t:end]
        if np.all(np.abs(window - target) <= eps) and (end - t) >= H:
            return t + 1  # 1-indexed
    return None


def tracking_error(history: np.ndarray, target: float) -> float:
    return float(np.mean(np.abs(history - target)))


def stability_var(history: np.ndarray, tail: int = 2000) -> float:
    tail = min(tail, len(history))
    return float(np.var(history[-tail:]))


def avg_payoff_per_round(payoff_history: np.ndarray) -> float:
    return float(np.mean(payoff_history))


def run_one(
    sim: SimConfig,
    mode: str,
    seed: int,
) -> Dict[str, object]:
    """
    mode: 'warm' or 'random'
    Returns dict with histories and metrics.
    """
    if sim.n_agents % 2 != 0:
        raise ValueError("n_agents must be even for pairing without leftover.")

    # payoff
    static_payoff = default_payoff_hawk_dove()

    payoff_series = None
    if sim.use_dynamic_payoff:
        payoff_series = load_payoff_time_series(sim.payoff_series_path)

    # population
    agent_cfg = BanditPGConfig(eta=0.002, beta=0.01, seed=seed)
    pop = BanditPGPopulation(sim.n_agents, agent_cfg)

    # target equilibrium x*
    # If dynamic, use x*_t of the current payoff (we log instantaneous target too)
    xstar_static = static_payoff.replicator_equilibrium()
    if xstar_static is None:
        raise RuntimeError("No interior equilibrium for the default payoff.")

    if mode == "warm":
        pop.warm_start(x_star=xstar_static, delta=sim.warm_delta, std=sim.warm_std)
    elif mode == "random":
        pop.random_start_low_coop(p_low=sim.rand_p_low, std=sim.rand_std)
    else:
        raise ValueError("mode must be 'warm' or 'random'")

    rng = np.random.default_rng(seed)

    coop_hist: List[float] = []
    payoff_hist: List[float] = []
    xstar_hist: List[float] = []

    for t in range(sim.rounds):
        # payoff possibly changes with time
        if sim.use_dynamic_payoff and payoff_series is not None:
            year = sim.start_year + t
            payoff_t = payoff_from_series(payoff_series, year)
        else:
            payoff_t = static_payoff

        xstar_t = payoff_t.replicator_equilibrium()
        # If no interior equilibrium at some t, fall back to last known value
        if xstar_t is None:
            xstar_t = xstar_hist[-1] if xstar_hist else xstar_static

        actions = pop.sample_actions()
        rewards = random_pairing_rewards(rng, payoff_t, actions)

        # learning update
        pop.step_update(rewards=rewards, actions=actions)

        # logs
        coop_hist.append(pop.avg_coop_prob())
        payoff_hist.append(float(np.mean(rewards)))
        xstar_hist.append(float(xstar_t))

        if (t + 1) % sim.log_every == 0:
            print(f"[{mode.upper()}] round {t+1:5d}/{sim.rounds} | avg p(C)={coop_hist[-1]:.3f} | avg r={payoff_hist[-1]:.3f}")

    coop_arr = np.array(coop_hist, dtype=float)
    payoff_arr = np.array(payoff_hist, dtype=float)
    xstar_arr = np.array(xstar_hist, dtype=float)

    # metrics (static target uses xstar_static; dynamic uses xstar_arr)
    metrics = {}
    if not sim.use_dynamic_payoff:
        target = float(xstar_static)
        for eps in sim.eps_list:
            metrics[f"tau_{eps}"] = compute_tau_eps(coop_arr, target, eps, sim.hold_H)
        metrics["TrackErr"] = tracking_error(coop_arr, target)
    else:
        # dynamic tracking to x*_t
        metrics["TrackErr"] = float(np.mean(np.abs(coop_arr - xstar_arr)))
        # define tau_eps against moving target: check band against x*_t in window
        for eps in sim.eps_list:
            tau = None
            T = len(coop_arr)
            H = sim.hold_H
            for t in range(T):
                end = min(T, t + H)
                if (end - t) < H:
                    continue
                if np.all(np.abs(coop_arr[t:end] - xstar_arr[t:end]) <= eps):
                    tau = t + 1
                    break
            metrics[f"tau_{eps}"] = tau

    metrics["VarTail"] = stability_var(coop_arr, tail=2000)
    metrics["AvgPayoff"] = avg_payoff_per_round(payoff_arr)

    return {
        "mode": mode,
        "seed": seed,
        "xstar_static": float(xstar_static),
        "coop": coop_arr,
        "xstar": xstar_arr,
        "payoff": payoff_arr,
        "metrics": metrics,
    }


def plot_curves(sim: SimConfig, warm: np.ndarray, rand: np.ndarray, xstar: float, out_path: str):
    t = np.arange(len(warm))
    plt.figure(figsize=(8, 5))
    plt.plot(t, warm, label="Warm-start", linewidth=2)
    plt.plot(t, rand, label="Random-start", linewidth=2, linestyle="--")
    plt.axhline(xstar, color="gray", linestyle=":", linewidth=1.5, label="Replicator equilibrium x*")
    plt.xlabel("Round")
    plt.ylabel("Average cooperation probability")
    plt.title("Convergence speed: warm-start vs random-start")
    if sim.ylim_0_1:
        plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved plot to {out_path}")


def summarize_metrics(results: List[Dict[str, object]]):
    # group by mode
    by_mode: Dict[str, List[Dict[str, object]]] = {"warm": [], "random": []}
    for r in results:
        by_mode[str(r["mode"])].append(r)

    def mean_std(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    for mode, lst in by_mode.items():
        print(f"\n=== Summary: {mode.upper()} ===")
        if not lst:
            continue
        metrics_keys = lst[0]["metrics"].keys()
        for k in metrics_keys:
            # tau may be None -> ignore None in stats
            vals = [m["metrics"][k] for m in lst]
            vals_num = [float(v) for v in vals if v is not None]
            if len(vals_num) == 0:
                print(f"{k}: None")
                continue
            m, s = mean_std(vals_num)
            if k.startswith("tau_"):
                print(f"{k}: {m:.1f} ± {s:.1f}")
            else:
                print(f"{k}: {m:.4f} ± {s:.4f}")


def main():
    sim = SimConfig()
    # multiple seeds for robustness
    seeds = [42, 43, 44, 45, 46]

    all_results: List[Dict[str, object]] = []
    for seed in seeds:
        print(f"\n--- SEED {seed} | WARM ---")
        all_results.append(run_one(sim, mode="warm", seed=seed))
        print(f"\n--- SEED {seed} | RANDOM ---")
        all_results.append(run_one(sim, mode="random", seed=seed))

    summarize_metrics(all_results)

    # plot using the first seed's curves for visualization
    warm0 = next(r["coop"] for r in all_results if r["mode"] == "warm" and r["seed"] == seeds[0])
    rand0 = next(r["coop"] for r in all_results if r["mode"] == "random" and r["seed"] == seeds[0])
    xstar = float(next(r["xstar_static"] for r in all_results if r["seed"] == seeds[0]))

    plot_curves(sim, warm0, rand0, xstar=xstar, out_path=sim.plot_path)


if __name__ == "__main__":
    main()
