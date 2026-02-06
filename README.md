# Multi-Agent Reinforcement Learning with Replicator-Guided Warm-Start

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange?style=flat-square)]()

> **Accelerating convergence to cooperative equilibrium using Evolutionary Game Theory priors.**

## 📌 Project Overview

This repository contains the simulation code for the research paper **"Multi-Agent Reinforcement Learning for Cooperative and Competitive Strategies via Game Theoretic Replicator Dynamics"**.

In multi-agent environments, learning-based agents often suffer from long transient phases before reaching stable strategies. This project introduces a **Replicator-Guided Warm-Start** framework that:
1.  Derives a theoretical equilibrium target ($x^*$) using **Replicator Dynamics**.
2.  Initializes agent policies near this target to bypass early exploration inefficiencies.
3.  Achieves significantly faster convergence without altering the final strategic outcome.

---

## 🔑 Key Features

* **Game-Theoretic Initialization:** Bridges the gap between static equilibrium concepts and dynamic learning.
* **Drastic Speedup:** Reduces convergence time by approximately **4.6x to 6.5x** compared to random initialization.
* **Stability:** Lowers equilibrium tracking error and reduces variance in population behavior.
* **Algorithm Agnostic:** The initialization method is complementary to various learning rules (e.g., Policy Gradient, NeuRD).

---

## ⚙️ Methodology

### 1. The Stage Game
We simulate a population of $N$ firms playing a symmetric 2x2 game (Cooperation vs. Defection).
The payoff matrix is defined as:

| | Cooperate (C) | Defect (D) |
| :---: | :---: | :---: |
| **Cooperate (C)** | $R$ (Reward) | $S$ (Sucker) |
| **Defect (D)** | $T$ (Temptation) | $P$ (Punishment) |

### 2. Deriving the Target ($x^*$)
Using **Replicator Dynamics** ($\dot{x}=x(\pi_{C}(x)-\overline{\pi}(x))$), we calculate the interior equilibrium cooperation rate:

$$
x^{*} = \frac{P-S}{(R-S)-(T-P)}
$$

### 3. Warm-Start Initialization
Instead of initializing agents with random policies (e.g., $p \approx 0.5$ or $p \approx 0$), we sample initial policy parameters $\theta_i(0)$ such that the population average starts near $x^*$:

$$
\theta_{i}(0) \sim \mathcal{N}\left(\log\frac{x^{*}}{1-x^{*}}, \sigma_{0}^{2}\right)
$$

---

## 🚀 Installation

### Prerequisites
* Python 3.8 or higher
* `numpy`
* `matplotlib`

```bash
# Clone the repository
git clone [https://github.com/yourusername/marl-replicator-warmstart.git](https://github.com/yourusername/marl-replicator-warmstart.git)
cd marl-replicator-warmstart

# Install dependencies
pip install numpy matplotlib

## 💻 Usage

You can run the simulation to reproduce the paper's results comparing **Random-Start** vs. **Warm-Start**.

### Basic Run
Run the simulation with default settings using the warm-start method:
```bash
python main.py --rounds 10000 --agents 100 --init_mode warm_start
