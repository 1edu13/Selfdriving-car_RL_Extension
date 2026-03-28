## 7. Soft Actor-Critic (SAC)

After working with the rigidity of DDPG and the "safety-first" approach of TD3, we arrive at **SAC**. In the world of continuous control, SAC is often considered the "gold standard." If I had to describe it simply, SAC is the most "open-minded" algorithm because it doesn't just try to find a single perfect path; it tries to learn every possible way to succeed.

While DDPG and TD3 are **deterministic** (the Actor tries to output one specific "best" number), SAC is **stochastic**. This means the Actor outputs a probability distribution (usually a mean and a standard deviation), which completely changes how the agent explores its environment.

### 7.1. The Heart of SAC: Maximum Entropy Reinforcement Learning
The "Soft" in SAC comes from **Entropy**. In information theory, entropy is a measure of randomness. In standard Reinforcement Learning, the goal is simply to maximize the sum of rewards: $\sum r_t$. 

The problem with this "reward-only" focus is that agents often become obsessed with the first decent strategy they find (exploitation) and stop looking for better ones (exploration). SAC fixes this by adding an entropy term ($H$) to the objective function:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E} [r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]$$

We achive this improvements:
1. **Deeper Exploration**: The agent is literally "paid" to be random. If there are multiple ways to reach a goal, SAC will try them all instead of getting stuck on one.
2. **Robustness**: Because the agent has practiced a wide variety of actions during training (due to that randomness), it becomes much more resilient. if the real-world environment has noise or slight changes, the agent doesn't "break" because it has seen similar variations before.

### 7.2. The Temperature Parameter ($\alpha$)
The symbol $\alpha$ is known as the **temperature**. It controls the balance between Reward and Entropy:
* **High $\alpha$**: The agent prioritizes being random and exploring over getting points.
* **Low $\alpha$**: The agent gets "serious" and focuses almost entirely on maximizing reward.
In modern SAC implementations, we don't even have to tune this manually; the agent learns the optimal $\alpha$ value as it trains.

### 7.3. Architecture: Twin Critics and the Reparameterization Trick
To keep training stable, SAC adopts the **Twin Critic** ($Q_1$ and $Q_2$) setup from TD3 to prevent overestimation bias. 

However, there is a mathematical challenge: since the Actor is a random distribution, we normally couldn't pass gradients through it (you can't derive "luck"). SAC solves this using the **Reparameterization Trick**. It separates the deterministic part of the action from the noise, allowing the network to learn through backpropagation while still behaving stochastically.

---

### 7.4. SAC Training Flow

This is the step-by-step logic implemented in the training loop:

#### 1. Initialization
* **Stochastic Actor $\pi_\phi$**: Outputs Gaussian parameters ($\mu, \sigma$).
* **Twin Critics $Q_{\theta_1}, Q_{\theta_2}$**: Two networks to estimate action-values.
* **Target Critics**: Stable copies of the critics.
* **Temperature $\alpha$**: Entropy coefficient.

#### 2. Interaction Phase
* Observe state $s$.
* **Sample action** $a$ from the distribution: $a \sim \pi_\phi(\cdot|s)$.
* Execute action, receive $r$, and observe $s'$.
* Store transition $(s, a, r, s', d)$ in the **Replay Buffer**.

#### 3. Learning Phase (Minibatch Update)
For each update step:
1.  **Future Value**: Sample next action $a' \sim \pi_\phi(\cdot|s')$ and calculate its log-probability $\log \pi_\phi(a'|s')$.
2.  **Compute Target ($y$)**:
    $$y = r + \gamma \left( \min_{i=1,2} Q_{\text{targ}, i}(s', a') - \alpha \log \pi_\phi(a'|s') \right)$$
    *Note: The entropy term is subtracted here to reward high-entropy future states.*
3.  **Update Critics**: Minimize MSE loss between $Q_{\theta_i}(s, a)$ and the target $y$.
4.  **Update Actor**: Adjust weights to maximize the expected Q-value while maintaining high entropy:
    $$\max_\phi \mathbb{E}_{a \sim \pi_\phi} [Q_{\text{min}}(s, a) - \alpha \log \pi_\phi(a|s)]$$
5.  **Update Temperature**: Adjust $\alpha$ automatically to meet a target entropy.
6.  **Soft Update**: Gradually move Target weights toward Current weights: 
    $\theta_{\text{targ}} \leftarrow \tau \theta + (1 - \tau) \theta_{\text{targ}}$.

![Soft Actor-Critic (SAC) Architectural Framework.](image-7.png)
*Figure 6: Soft Actor-Critic (SAC) Architectural Framework.*