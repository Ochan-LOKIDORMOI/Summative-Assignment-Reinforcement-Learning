# **Summative Assignment - Reinforcement Learning**

# Wildlife Intrusion Prevention using Reinforcement Learning

## Project Overview
This project implements reinforcement learning techniques to prevent wildlife intrusion in agricultural areas. The simulation models a farm environment where an agent must protect crops by capturing approaching wildlife before they cause damage. Two algorithms, **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)**, were implemented and compared for effectiveness in this discrete grid-world environment.

## Environment Description
### Agent(s)
- A single agent representing a wildlife prevention unit.
- Moves on a 5x5 grid with four cardinal directions (up, down, left, right).
- Objective: Intercept wildlife before they reach the central farm area.

### Action Space
- Discrete with 8 possible actions:
  - 0: Move up
  - 1: Move right
  - 2: Move down
  - 3: Move left
  - 4-7: Diagonal movements (partially implemented).

### State Space
Includes:
- Agent position (2D coordinates).
- Wildlife positions (up to 5 animals, each with 2D coordinates).
- Current state enum (`NO_WILDLIFE`, `WILDLIFE_DISTANT`, `WILDLIFE_APPROACHING`, `WILDLIFE_IN_CROP`).

### Reward Structure
- **+5.0** for capturing wildlife.
- **-2.0** for wildlife reaching the farm.
- **-0.1** per step to encourage efficiency.
- Additional penalties for wildlife near the farm.

Mathematically:  
`R = 5 × captures - 2 × intrusions - 0.1 × steps`

## Implemented Methods
### Deep Q-Network (DQN)
- Neural network with two hidden layers (256 and 128 nodes).
- Experience replay buffer (size 50,000).
- Target network with periodic updates (every 500 steps).
- ε-greedy exploration (initial ε=1.0, final ε=0.01).
- Double DQN update rule.

### Proximal Policy Optimization (PPO)
- Separate policy and value networks (both 256-128 architecture).
- Generalized Advantage Estimation (λ=0.92).
- Clipped objective (clip_range=0.15).
- 10 epochs per update with batch size 64.
- Entropy coefficient of 0.02 for exploration.

## Hyperparameter Optimization
### DQN Hyperparameters
| Hyperparameter         | Optimal Value | Summary                                                                 |
|------------------------|---------------|-------------------------------------------------------------------------|
| Learning Rate          | 1e-3          | Faster convergence but required careful tuning for stability.           |
| Gamma (Discount Factor)| 0.95          | Emphasized immediate rewards for wildlife capture.                      |
| Replay Buffer Size     | 50,000        | Balanced diversity and computational efficiency.                        |
| Batch Size             | 64            | Stable learning with reasonable computational cost.                     |
| Exploration Strategy   | ε-greedy      | Longer exploration period (30% of training) improved strategy discovery.|

### PPO Hyperparameters
| Hyperparameter         | Optimal Value | Summary                                                                 |
|------------------------|---------------|-------------------------------------------------------------------------|
| Learning Rate          | 1e-4          | Lower rate ensured stable policy updates.                               |
| Gamma (Discount Factor)| 0.97          | Slightly higher than DQN to account for policy method.                 |
| clip_range             | 0.15          | Tighter than standard for conservative updates.                         |
| Entropy Coefficient    | 0.02          | Balanced exploration and exploitation.                                  |

## Results
### Quantitative Findings
| Metric                  | DQN           | PPO           |
|-------------------------|---------------|---------------|
| Steps to >0 reward      | ~15,000       | ~25,000       |
| Max reward achieved     | ~8.5          | ~6.2          |
| Reward stability (σ)    | 0.9           | 1.4           |

### Key Insights
- **DQN Advantages**:
  - Faster initial learning and higher final performance (~27% better).
  - More consistent episode rewards.
- **PPO Characteristics**:
  - Slower but steadier improvement.
  - Greater variance, suggesting more exploratory behavior.

## Conclusion
**DQN outperformed PPO** in this task, demonstrating:
- Higher success rates (92% vs 80%).
- Faster adaptation (2.1 vs 3.7 median steps to capture).
- Better generalization (<13% performance drop on unseen configurations vs PPO's 25%).

### Recommendations for Improvement
- **For DQN**:
  - Implement prioritized experience replay.
  - Use dynamic ε schedules for faster exploration reduction.
- **For PPO**:
  - Apply reward shaping or curriculum learning.
  - Consider hybrid approaches combining DQN's value estimation with PPO's policy gradients.

## Repository Structure
- `dqn/`: Code and models for DQN implementation.
- `ppo/`: Code and models for PPO implementation.
- `environment/`: Simulation environment and visualization tools.
- `results/`: Training logs and performance metrics.

## Links
- **Video Recording**: [Link to your Video]
- **GitHub Repository**: [Link to your repository]
