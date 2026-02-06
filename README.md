# Accelerating Global Innovation via Co-opetition Learning: Replicator-Guided Warm-Start Multi-Agent Reinforcement Learning


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange)]()

> **Accelerating convergence in cooperative and competitive strategies via Game Theoretic Replicator Dynamics.**

This repository contains the implementation and simulation code for the research paper: **"Accelerating Global Innovation via Co-opetition Learning: Replicator-Guided Warm-Start Multi-Agent Reinforcement Learning
"**.

## 📌 Abstract
Cooperation among self-interested firms is a central driver of innovation ecosystems, yet learning-based strategic adaptation often exhibits long transient phases before reaching stable behavior. This project addresses the convergence-speed gap between equilibrium reasoning and learning-based adaptation.

We propose a **Replicator-Guided Warm-Start framework**. By deriving an interior equilibrium cooperation rate using replicator dynamics from evolutionary game theory, we use it as a principled prior to initialize a population of learning agents. The results demonstrate that this method significantly reduces convergence time and lowers equilibrium tracking error without altering the long-run strategic outcome.

## 🚀 Key Features
* **Game Theoretic Initialization:** Uses Replicator Dynamics to calculate the theoretical equilibrium ($x^*$) and initializes agents near this manifold.
* **Bandit-Style Learning:** Agents update cooperation probabilities via a baseline-corrected policy-gradient rule.
* **Static & Dynamic Regimes:** Validated under both fixed and time-varying payoff structures.
* **Performance:** Achieves approx. **4.6x to 6.5x faster convergence** compared to random initialization.

## 🛠️ Methodology

### 1. Problem Definition
We model a population of $N$ firms interacting in a repeated symmetric $2 \times 2$ game (e.g., Prisoner's Dilemma context).
* **Actions:** Cooperation ($C$) vs. Defection ($D$).
* **Payoff Matrix:** Defined by $(R, T, S, P)$.

### 2. Replicator Dynamics & Equilibrium
Instead of starting from scratch, we calculate the interior equilibrium $x^*$ where the expected payoff of cooperation equals defection:

$$
x^* = \frac{P-S}{(R-S)-(T-P)}
$$

This $x^*$ serves as the target for our **Warm-Start Initialization**.

### 3. Algorithm: Replicator-Guided Warm-Start
The agents are initialized using a normal distribution centered around the logit of the equilibrium target $x_0$. This places the population near the equilibrium manifold at the start, significantly bypassing the initial exploration phase:

$$
\theta_i(0) \sim \mathcal{N}\left(\log\frac{x_0}{1-x_0}, \sigma_0^2\right)
$$

### 4. Learning Update Rule
Agents update their policies using a baseline-corrected policy-gradient rule:

$$
\theta_i(t+1) = \theta_i(t) + \eta(r_i(t) - b(t))(a_i(t) - p_i(t))
$$

Where $b(t)$ is an exponential moving average baseline to reduce variance.

## 📊 Experimental Results

Simulations were conducted to compare **Random-Start** vs. **Warm-Start** strategies.

| Metric | Random-Start | Warm-Start (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| **Convergence Time ($\tau_{0.02}$)** | 6,800 rounds | **1,050 rounds** | ~6.5x Faster |
| **Convergence Time ($\tau_{0.01}$)** | 8,900 rounds | **1,920 rounds** | ~4.6x Faster |
| **Tracking Error** | 0.088 | **0.020** | Significantly Lower |
| **Stability ** | $1.7 \times 10^{-3}$ | **$4.1 \times 10^{-4}$** | More Stable |

> *Data sourced from Table I of the paper.*

**Ablation Study Findings:**
1.  **Speed:** Equilibrium-guided initialization is the dominant contributor to speedup.
2.  **Robustness:** The benefit remains meaningful even when swapping learning rules (e.g., NeuRD, Softmax PG).
3.  **Welfare:** Reduces time spent in inefficient transient regimes, improving overall welfare.

## 💻 Tech Stack & Simulation
* **Language:** Python
* **Environment:** Google Colab Pro
* **Libraries:**
    * `NumPy`: For numerical routines and vectorized operations.
    * `Matplotlib`: For generating figures and plots.

## 📂 Project Structure
```bash
├── src
│   ├── agent.py          # Implementation of Learning Agents (Bandit/PG)
│   ├── game.py           # Payoff Matrix and Replicator Dynamics Logic
│   └── simulation.py     # Main loop for multi-agent interaction
├── notebooks
│   └── analysis.ipynb    # Data analysis and visualization
├── data
│   └── cordis_processed  # (Optional) Processed CORDIS dataset
└── README.md
