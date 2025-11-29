# Replicator-Guided Warm-Start for Multi-Agent Firm Cooperation

This repository implements a simple but insightful model of **firm cooperation and competition** using tools from **evolutionary game theory** and **reinforcement learning**.

We consider a repeated interaction among many firms playing a **binary cooperation game** (join a joint R&D initiative vs. act alone/free-ride), and compare:

- **Replicator dynamics** (evolutionary game theory) — which predicts a stable interior equilibrium cooperation rate  
- **Multi-agent bandit-style RL** — firms adapt their cooperation probabilities via a policy gradient–type update

The key idea is to use the **replicator equilibrium** as a **warm-start prior** for RL agents and study how this affects the **speed of convergence** to cooperative behavior, compared to a purely **random initialization**.

---

## 📦 Model Overview

We model a symmetric 2×2 game between two firms:

- `C` = Cooperate (invest in joint R&D / shared infrastructure)  
- `D` = Defect (act alone or free-ride on others)
  
---

## 🧮 Evolutionary Equilibrium (Replicator Dynamics)

Let \( x \in [0,1] \) denote the fraction of firms that choose `C`.

- Expected payoff of `C`:
  \[
  \pi_C(x) = x R + (1-x) S
  \]
- Expected payoff of `D`:
  \[
  \pi_D(x) = x T + (1-x) P
  \]

The **replicator dynamics** is given by:
\[
\dot{x} = x(\pi_C(x) - \bar{\pi}(x)),
\]
where \( \bar{\pi}(x) = x \pi_C(x) + (1-x) \pi_D(x) \) is the population average payoff.

The equilibrium condition \( \pi_C(x^*) = \pi_D(x^*) \) yields an analytic **interior equilibrium** \( x^* \in (0,1) \).  
For the chosen payoffs, this gives \( x^* = 0.5 \), which is **stable** under the replicator dynamics.

---

## 🤖 Multi-Agent Bandit-Style RL

Instead of assuming evolutionary dynamics directly, we model each firm as a learning agent:

- Each agent \(i\) has a parameter \( \theta_i \in \mathbb{R} \)
- The probability of cooperating is
  \[
  p_i = \sigma(\theta_i) = \frac{1}{1 + e^{-\theta_i}}.
  \]
- At each round:
  1. Each agent samples `C` (1) or `D` (0) from Bernoulli(\(p_i\)).
  2. Agents are randomly paired; payoffs are computed using the 2×2 matrix above.
  3. Each agent’s parameter is updated using a **policy gradient–like rule**:
     \[
     \theta_i \leftarrow \theta_i + \eta \, (r_i - b)\,(a_i - p_i),
     \]
     where:
     - \( \eta \) is the learning rate
     - \( r_i \) is the realized payoff
     - \( b \) is a running **baseline** (approximate average reward)
     - \( a_i \in \{0,1\} \) is the sampled action

In expectation, stationary points of this learning rule satisfy the same **payoff indifference condition** \( \pi_C(x) = \pi_D(x) \), so **the RL dynamics and replicator dynamics share the same equilibrium \(x^*\)**.  

This allows us to compare **how different initializations affect the *speed*** with which RL converges to the evolutionary equilibrium.

---

## 🔥 Warm-Start vs Random-Start

We compare two initialization strategies:

1. **Warm-Start (Replicator-Guided)**  
   - We first compute the replicator equilibrium \( x^* \).  
   - We then set initial \( \theta_i \) so that each agent’s cooperation probability is slightly above the equilibrium (e.g., around 0.65):
     \[
     p_i(0) \approx 0.65.
     \]
   - This represents a prior belief that “cooperation is relatively attractive,” guided by game-theoretic analysis.

2. **Random-Start (Low-Cooperation Baseline)**  
   - We initialize \( \theta_i \) such that almost all agents have a very low cooperation probability (e.g., \( p_i(0) \approx 0.02 \)).  
   - This corresponds to a market environment where firms are initially almost purely self-interested and rarely cooperate.

Both learning processes use the **same game, same update rule, and same learning rate**.  
In both cases, with enough rounds, the **average cooperation probability** converges toward the same equilibrium \(x^* \approx 0.5\).  
However, the **number of rounds required to get close to \(x^*\)** is dramatically smaller for the warm-started population.

---

## 📈 What the Script Produces

Running the script:

```bash
python firm_replicator_bandit.py
