# **Summative Assignment - Reinforcement Learning**

## Wildlife Intrusion Prevention using Reinforcement Learning

## Project Overview
This project implements reinforcement learning techniques to prevent wildlife intrusion in agricultural areas. 

The simulation models a farm environment where an agent must protect crops by capturing approaching wildlife before they cause damage.

Two algorithms, **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)**, were implemented and compared for effectiveness in this discrete grid-world environment.

## Repository Structure
```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Visualization components using PyOpenGL
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO/other PG using SB3
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for running experiments
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Environment Description

![Image](https://github.com/user-attachments/assets/a6d929fc-6c4b-491f-88e8-42828294db91)

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
- 
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

![Image](https://github.com/user-attachments/assets/ef501f1f-6dd3-44fd-bd78-2244bd4afb2f)
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
 
## Training Stability
![Image](https://github.com/user-attachments/assets/7d7df591-b215-46f7-ab67-ed8a47660639)   ![Image](https://github.com/user-attachments/assets/bc492a1a-59e1-415f-a0db-9a83e880a933)

- DQN's loss convergence  demonstrates stable learning, with TD error plateauing at 0.1 after 40k steps.
   
- PPO maintained controlled exploration through entropy decay , reaching optimal stochasticity (-1.8) by 60k steps.

## Generalization

![Image](https://github.com/user-attachments/assets/0059a073-e30a-418e-8825-101511f8255b)  ![Image](https://github.com/user-attachments/assets/1f409cc3-19cd-4dba-bbff-d5f1c43fac43)


Both models demonstrated reasonable generalization, but DQN outperformed PPO by 17-22% in success rates across unseen spawn configurations.


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
 
## Installation Instructions

Clone this repository:

`git clone https://github.com/Ochan-LOKIDORMOI/Summative-Assignment-Reinforcement-Learning.git`

To install the required dependencies, run:

`pip install -r requirements.txt`

## Usage
To train a DQN agent, run:

`python training/dqn_training.py`

To train a policy gradient agent (e.g., PPO), run:

`python training/pg_training.py`

To see the model performace, run:

`python main.py`

## License

This project is open-source and available under the MIT License.

## Contact

For any questions or contributions, feel free to submit an issue or a pull request.
