# Technical Report: Advanced Off-Policy Methods in Deep Reinforcement Learning
**Author:** Eduardo Martín Postigo
**Subject:** Comparative Analysis of DQN, TD3, and SAC

## 1. Introduction
Modern Deep Reinforcement Learning (DRL) has evolved from simple tabular methods to complex architectures capable of handling high-dimensional state spaces and continuous action domains. This report examines the transition from value-based discrete methods to state-of-the-art actor-critic stochastic frameworks, focusing on the mechanical improvements that provide stability and efficiency.

---
## 2. Core Taxonomy
To classify any Reinforcement Learning algorithm, we evaluate it across these four fundamental dimensions.

### A. Action Spaces 
* **Discrete:** The agent chooses from a finite, fixed set of actions.
    * *LaTeX:* $a \in \{0, 1, ..., n-1\}$.
    * *Analogy:* Using a D-pad or a keyboard (binary choices).
* **Continuous:** The agent outputs real-valued vectors, allowing for infinite precision within bounds.
    * *LaTeX:* $a \in \mathbb{R}^d$ typically within $[-1, 1]$.
    * *Analogy:* Using a steering wheel or a throttle pedal (gradient choices).



### B. Policy Types (The Decision Logic)
* **Deterministic:** Maps a state directly to a single, specific action.
    * *Formula:* $a = \mu(s)$.
    * *Behavior:* Exploitative and precise; requires external noise (like Gaussian) to explore.
* **Stochastic:** Maps a state to a probability distribution (usually a Gaussian mean $\mu$ and variance $\sigma$).
    * *Formula:* $a \sim \pi(a|s)$.
    * *Behavior:* Naturally exploratory; better for complex environments and handling uncertainty.



### C. Paradigms (The Data Efficiency)
* **On-Policy:** The agent learns only from data collected by its *current* version. Data is discarded after the update.
    * *Example:* **PPO**.
    * *Trait:* Stable but "expensive" (requires constant new environment interactions).
* **Off-Policy:** The agent learns from a **Replay Buffer** containing experiences from past versions of itself.
    * *Example:* **DQN, SAC, TD3**.
    * *Trait:* Highly sample-efficient; reuses data many times to squeeze out more learning.



### D. Architectures (The Brain Structure)
* **Value-Based:** Focuses exclusively on estimating the "quality" ($Q$) of every possible action. The policy is implicitly to "pick the highest $Q$."
    * *Constraint:* Hard to use in continuous spaces because you cannot calculate $\max$ over infinite values.
    * *Example:* **DQN**.
* **Actor-Critic:** Splits the model into two specialized networks.
    * **The Actor:** Learns *how* to behave (the policy).
    * **The Critic:** Learns to estimate the value of the Actor's actions.
    * *Example:* **DDPG, TD3, SAC, PPO**.
---
## 2. Comparative Methodology Overview

|  | **DQN** | **DDPG** | **TD3** | **SAC** | **PPO** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Paradigm** | Off-Policy | Off-Policy | Off-Policy | Off-Policy | On-Policy |
| **Action Space** | Discrete | Continuous | Continuous | Continuous | Continuous |
| **Policy Type** | Deterministic | Deterministic | Deterministic | **Stochastic** | Stochastic |
| **Architecture** | Value-Based | Actor-Critic | Actor-Critic | Actor-Critic | Actor-Critic |
| **Exploration** | $\epsilon$-greedy | Added Noise | Added Noise | **Max Entropy** | Probability Dist. |
| **Stability** | Medium | Low | High | **Very High** | High |
| **Sample Efficiency**| High | High | High | **Very High** | Low |

---

## 2. Deep Q-Networks (DQN): The Value-Based Foundation
DQN represents the first major successful integration of Deep Learning with Reinforcement Learning. It approximates the optimal action-value function $Q^*(s, a)$ using a neural network.

### 2.1 Theoretical Framework
DQN relies on the **Bellman Equation** to iteratively improve its estimation of the total discounted reward:
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

### 2.2 Stability Mechanisms
To solve the instability caused by using a non-linear function approximator (Neural Networks), DQN introduces:
* **Experience Replay:** A buffer that stores transitions $(s, a, r, s')$. Sampling random batches breaks the temporal correlation of data, ensuring Independent and Identically Distributed (IID) samples.
* **Target Networks:** A secondary network $\theta^{-}$ used to compute the target $y$. This prevents the "moving target" problem by freezing the target values for a set number of iterations.



[Image of Deep Q-Network architecture]


---

## 3. Twin Delayed DDPG (TD3): Mastering Continuity
While DQN excels in discrete spaces, it fails in continuous domains. TD3 was developed to address the **Overestimation Bias** inherent in the Deep Deterministic Policy Gradient (DDPG) algorithm.

### 3.1 The Overestimation Problem
Value-based methods often overestimate Q-values because they use the maximum of noisy estimates. In continuous control, this leads to the accumulation of errors and suboptimal policies.

### 3.2 The TD3 "Tricks" for Stability
1.  **Clipped Double-Q Learning:** TD3 maintains two independent critics ($Q_1, Q_2$) and uses the minimum of their estimates to calculate the target:
    $$y = r + \gamma \min_{i=1,2} Q_{\theta_{target, i}}(s', \mu_{\phi_{target}}(s') + \epsilon)$$
2.  **Target Policy Smoothing:** Small random noise is added to the target actions to smooth out the value function surface, making it less susceptible to exploitation by policy errors.
3.  **Delayed Policy Updates:** The Actor network is updated at a lower frequency than the Critic, ensuring that the value function is sufficiently accurate before adjusting the policy.



---

## 4. Soft Actor-Critic (SAC): Maximum Entropy Framework
SAC is an off-policy actor-critic algorithm that utilizes a stochastic policy. It is distinguished by its focus on **Maximum Entropy Reinforcement Learning**.

### 4.1 Entropy Regularization
Instead of just maximizing the expected reward, SAC aims to maximize the reward plus the entropy ($\mathcal{H}$) of the policy. Entropy represents the "randomness" or "uncertainty" of the agent's choices.
$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

### 4.2 Key Advantages
* **Automatic Exploration:** The agent is incentivized to explore all possible actions that lead to high rewards, preventing premature convergence to local optima.
* **Robustness:** Because the policy is stochastic, it is naturally more robust to noise and changes in environment dynamics compared to deterministic methods like TD3.
* **Reparameterization Trick:** To allow backpropagation through stochastic sampling, SAC expresses the action as a deterministic function of the state and independent noise: $a = f_{\theta}(\epsilon, s)$.



---

## 5. Comparative Analysis

| Feature | DQN | TD3 | SAC |
| :--- | :--- | :--- | :--- |
| **Action Space** | Discrete | Continuous | Continuous |
| **Policy Nature** | Deterministic | Deterministic | Stochastic |
| **Stability Strategy** | Target Network | Clipped Double-Q | Entropy Regularization |
| **Exploration** | $\epsilon$-greedy | Additive Noise | Entropy Maximization |

---

## 6. Conclusion
The progression from DQN to SAC shows a shift from stabilizing value targets toward stabilizing the exploration process itself. DQN introduced the necessary infrastructure (Buffers and Targets), TD3 refined the value estimation through pessimism (Clipped Q), and SAC integrated exploration directly into the mathematical objective through Entropy. Understanding these differences allows researchers to select the most appropriate tool based on the complexity and continuity of the task at hand.



## 4. The Evolutionary Journey: Why we moved forward

### 1. DQN to DDPG: Crossing the Continuous Gap
DQN was limited to buttons. **DDPG** introduced the Actor-Critic framework to off-policy learning, allowing for "pedal" control. However, DDPG was prone to "divergence"—it would often learn a bad policy and never recover because its Critic was too optimistic.

### 2. DDPG to TD3: Implementing "Pessimism"
**TD3** fixed DDPG by adding three stability mechanisms:
* **Clipped Double-Q:** Using two critics and taking the minimum prevents the agent from overestimating how good a state is.
* **Target Policy Smoothing:** Adding noise to targets prevents the actor from exploiting sharp "peaks" in the value function.
* **Update Rule:** $$y = r + \gamma \min_{i=1,2} Q_{\theta_{target, i}}(s', \mu_{\phi_{target}}(s') + \epsilon)$$

### 3. TD3 to SAC: Embracing Randomness (Entropy)
While TD3 is precise, it can be rigid. **SAC** introduced **Entropy Regularization**. Instead of just trying to get a high score, SAC tries to get a high score while being as "random" as possible.
* **Entropy ($\mathcal{H}$):** This ensures the agent never stops exploring until it is absolutely certain of the optimal path.
* **Objective:** $$J(\pi) = \sum_{t=0}^{T} \mathbb{E} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

---

## 5. Summary for Students
* If your task is **discrete** (like Chess): Use **DQN**.
* If you need **simple, reliable** continuous control and have time to train: Use **PPO**.
* If you need **maximum efficiency** and robust exploration in continuous space: Use **SAC**.
* If you need **deterministic precision** (robotics): Use **TD3**.