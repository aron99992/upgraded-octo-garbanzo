# BipedalWalker-v3 with PPO

A reinforcement learning project implementing Proximal Policy Optimization (PPO) with simulated annealing to train a bipedal walker robot in the Gymnasium environment.

Overview

This project explores  RL techniques to train a bipedal walking agent, achieving 288 mean reward through:
- custom reward shaping with minimal penalties
- action space restriction as well as a reward restriction
- simulated annealing for exploration-to-exploitation transition
- state-dependent exploration (gSDE)
- various hyperparameter changes


Aaron Lee  
lalalaaaron@gmail.com  
Performance: 288 ± 95 reward over 600,000 training steps


Quick Start

### Installation
```
# Clone the repository
git clone https://github.com/aron99992/upgraded-octo-garbanzo.git
cd upgraded-octo-garbanzo

# Install dependencies
pip install -r requirements.txt
```

### Training
```
# Run training
python train.py
```



## Key Techniques

### Reward Shaping
- Angle penalty: `0.04 × |hull_angle|`
- Angular velocity penalty: `0.015 × |angular_velocity|`
- Action smoothness: `0.005 × action_diff²`
- Velocity smoothness: `0.025 × |velocity_change|`
- Airborne penalty: `0.1` when feet off ground

### Action Clipping
- Restricts action space from `[-1, 1]` to `[-0.7, 0.7]`
- Prevents extreme torque and encourages smooth movements

### Simulated Annealing
- Learning rate: `5e-4 → 1e-6` (exponential decay over 500k steps)
- Entropy coefficient: `0.05 → 0.0001`
- Clip range: `0.3 → 0.02`

### Hyperparameters
- `n_steps=4096` (longer rollouts)
- `batch_size=512` (reduced variance)
- `n_epochs=20` 
- `gamma=0.995` (long-term reward focus)
- `use_sde=True` (gSDE)




## Training Progress

### Key Milestones
- 0-200k steps: Rapid learning (50 → 220 reward)
- 200k-400k steps: Plateau phase (~280-290 reward)
- 400k-600k steps: Annealing refinement (stabilized at 288)




## Project Structure
```
├── train.py                  # Main training script
├── requirements.txt          # Python dependencies
├── wrappers/                 # Custom Gym wrappers
├── docs/                     # Detailed documentation
└── experiments/              # Ablation studies
```


## Acknowledgments

**Papers:**
- Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- Heess et al. (2017) - "Emergence of Locomotion Behaviours in Rich Environments"
- Kohl & Stone (2004) - "Policy Gradient Methods for Robotics"
- Kim Jiwon(2023) - "Deep Reinforcement Learning for Asset Allocation: Reward Clipping"
- A. F. Atiya, A. G. Parlos and L. Ingber (2003) - "A reinforcement learning method based on adaptive simulated annealing,"

**Libraries:**
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://gymnasium.farama.org/)



## Contact

Aaron Lee - lalalaaaron@gmail.com

Project Link: https://github.com/aron99992/upgraded-octo-garbanzo
