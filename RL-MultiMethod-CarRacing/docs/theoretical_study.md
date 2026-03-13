# Theoretical Study: Reinforcement Learning Methods for Autonomous Driving

## 1. Introduction

This document provides a theoretical overview of the four reinforcement learning algorithms
implemented in this project and their application to the CarRacing-v2 environment.

---

## 2. Problem Formulation

We model the autonomous driving task as a **Markov Decision Process (MDP)**:

- **State** $s_t$: RGB image of the racing track at time $t$ (96×96×3)
- **Action** $a_t$: Continuous control vector $[\text{steering}, \text{gas}, \text{brake}] \in [-1,1]^3$
- **Reward** $r_t$: $-0.1$ per step + $+1000/N$ per track tile visited
- **Discount factor** $\gamma$: controls the weight of future rewards

---

## 3. Shared Architecture: CNN Backbone

All agents use a shared **Convolutional Neural Network (CNN)** to extract visual features
from raw pixel observations:

```
Input: (3, 96, 96)
Conv2d(3 → 32, kernel=8, stride=4) → ReLU
Conv2d(32 → 64, kernel=4, stride=2) → ReLU
Conv2d(64 → 64, kernel=3, stride=1) → ReLU
Flatten → Linear(64×8×8, 256) → ReLU
Output: feature vector ∈ ℝ^256
```

---

## 4. Algorithm Descriptions

### 4.1 PPO – Proximal Policy Optimization

**Type**: On-policy, continuous actions, stochastic policy

PPO optimises a clipped surrogate objective to prevent overly large policy updates:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ is the probability ratio and $\hat{A}_t$ is the GAE advantage estimate.

**Key hyperparameters**: clip $\varepsilon=0.2$, GAE $\lambda=0.95$, rollout length 2048.

---

### 4.2 DQN – Deep Q-Network

**Type**: Off-policy, discrete actions, deterministic policy

DQN learns the optimal action-value function by minimising the Bellman error:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

where $\theta^-$ denotes the frozen target network updated every $C$ steps.

**Key hyperparameters**: replay buffer 50K, target update 1000 steps, $\varepsilon$-greedy exploration.

---

### 4.3 TD3 – Twin Delayed DDPG

**Type**: Off-policy, continuous actions, deterministic policy

TD3 addresses overestimation bias in DDPG via three improvements:
1. **Twin critics** – take the minimum of two Q-value estimates
2. **Delayed policy updates** – actor updated every 2 critic steps
3. **Target policy smoothing** – add clipped noise to target actions

$$y = r + \gamma \min_{i=1,2} Q_{\theta_i^-}\left(s', \pi_{\phi^-}(s') + \epsilon\right), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

**Key hyperparameters**: $\tau=0.005$, policy noise $\sigma=0.2$, noise clip $c=0.5$.

---

### 4.4 SAC – Soft Actor-Critic

**Type**: Off-policy, continuous actions, stochastic policy

SAC maximises a trade-off between expected return and policy entropy:

$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

The temperature $\alpha$ is automatically tuned to maintain a target entropy
$\mathcal{H}^* = -\dim(\mathcal{A})$.

**Key hyperparameters**: replay buffer 100K, batch size 256, soft update $\tau=0.005$.

---

## 5. Comparative Analysis

| Property           | PPO        | DQN      | TD3          | SAC          |
|--------------------|------------|----------|--------------|--------------|
| Policy type        | Stochastic | Greedy   | Deterministic| Stochastic   |
| Action space       | Continuous | Discrete | Continuous   | Continuous   |
| Sample efficiency  | Low        | Medium   | High         | High         |
| Stability          | High       | Medium   | Medium-High  | High         |
| Exploration        | Entropy    | ε-greedy | Gaussian     | Entropy reg. |
| On/Off policy      | On         | Off      | Off          | Off          |

---

## 6. Environment Wrappers

- **TransposeObservation**: converts HWC → CHW for PyTorch compatibility
- **NormalizeReward**: clips rewards to $[-1, 1]$ for training stability

---

## 7. References

1. Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
2. Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
3. Fujimoto et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML.
4. Haarnoja et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL*. ICML.
