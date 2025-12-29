# BipedalWalker-v3 Project Report

**Name:** Aaron Lee  
**Email:** lalalaaaron@gmail.com  
**Date:** December 2024



## Prior Experience

**Reinforcement Learning:**
- No prior experience 
**Deep Learning:**
- No prior experience

**Coding:**
- Baseline experience


## Time Spent

**Total:** ~7-8 hours

**Breakdown:**
- **12/19 (1-2 hours):** Learning basic RL terminology, setting up PPO solution with AI assistance
- **12/21 (2 hours):** Initial experimentation with reward shaping
- **12/23 (2 hours):** Debugging normalization issues, hyperparameter tuning
- **12/24 (2-3 hours):** Implementing simulated annealing, final optimization



## Compute Resources

**Hardware:**
- **CPU:** Apple Silicon M4 Pro - 12 cores, 12 logical threads
- **GPU:** Apple Silicon - 16 cores (MPS acceleration)



## Walker Performance

**Video:** https://screenapp.io/app/v/dNt03cshq1 

**Final Results:**
- **Total timesteps trained:** 600,000
- **Mean reward:** 269.85 +/- 25.30
- **Success rate:** Consistently walks forward without falling



## Techniques Used

### Reward Shaping

**Implementation:**
```python
# Penalties applied to base environment reward
angle_penalty = 0.04 × |hull_angle|
angular_velocity_penalty = 0.015 × |hull_angular_velocity|
action_smoothness_penalty = 0.005 × mean(action_diff²)
velocity_smoothness_penalty = 0.025 × |velocity_change|
airborne_penalty = 0.1 (when mean_lidar > 0.7)
fall_softening = -100 → -50
```

**Justification:**
- Prevents catastrophic behaviors 
- Encourages smooth, natural walking gait

### Action Clipping

**Implementation:**
```python
# Restrict action space from [-1, 1] to [-0.7, 0.7]
ActionClippingWrapper(env, clip_range=0.7)
```

**Justification:**
- Restricts torque magnitudes to prevent extreme movements
- Encourages smoother, more energy-efficient locomotion

### Reward Clipping

**Implementation:**
```python
# Clip all rewards to [-10, 10] range
TransformReward(env, lambda r: np.clip(r, -10, 10))
```

**Justification:**
- Normalizes reward scale for more stable learning
- Prevents extreme reward values from dominating gradient updates


### Hyperparameter Tuning

**Implementation:**
```python
learning_rate = 2e-4
n_steps = 4096        # Doubled from default 2048
batch_size = 512      # 8× larger than default 64
n_epochs = 20         # Doubled from default 10
gamma = 0.995         # Increased from 0.99
gae_lambda = 0.95
clip_range = 0.1      # Tightened from 0.2
vf_coef = 0.5
max_grad_norm = 0.5
```


### Simulated Annealing

**Implementation:**
```python
# Exponential decay schedules over 500k steps
learning_rate: 5e-4 → 1e-6
entropy_coefficient: 0.05 → 0.0001
clip_range: 0.3 → 0.02
temperature: 1.0 → 0.0
```

**Justification:**
- High initial values encourage exploration to find diverse strategies early
- Gradual reduction allows policy refinement and exploitation of best strategies
- Prevents catastrophic forgetting late in training through smaller updates


## Ablation Studies

Final reward: 270

### Impact of Reward Shaping Penalties
Baseline (no penalties): 228 median reward
Conclusion: While the agent can learn to walk without penalties, it lacks discipline to stay upright consistently. Adding the penalties resulted in a 18% performance boost 
### Impact of Reward/Action Clipping
Without clipping: 232 median reward
Conclusion: Extreme torque or massive reward spikes can cause good behaviors to overwrite bad ones. The clipping restricted the agent's ability to make catastrophic updates.
### Simulated Annealing
Without annealing (constant parameters): 232 median reward
 Without the annealing schedule, the agent never settled into its best wait, it kept trying to change its movements even after finding a winning strategy. 




## Issues Encountered & Solutions

### Overly Strict Penalties

**Problem:** Robot learned to crouch/jitter in place rather than walk. Performance dropped from 200 to -100 reward when multiple large penalties were added simultaneously.

**Root Cause:** Penalties (angle=1.0, energy=0.05, smoothness=0.1) were overwhelming the base reward signal. Agent learned "do nothing to avoid penalties" instead of "walk forward."

**Solution:** Reduced all penalty magnitudes by 5-10×. Final values of 0.04-0.05 provided guidance without dominating reward.



### Plateau at 280-300 Reward

**Problem:** Performance consistently stagnated around 290 reward after 300k-400k steps, sometimes degrading afterward.

**Attempted Solutions:**
- Increased training duration to 1M+ steps
- Removed reward clipping
- Reduced n_epochs from 20 to 10
- Tried multiple random seeds
- Added simulated annealing

**Outcome:** Simulated annealing provided modest improvement (+18 reward).


### Parameter Tuning Difficulty

**Problem:** Small hyperparameter changes caused large, unpredictable swings in performance. Couldn't isolate which changes helped vs hurt.

**Pivot:** Adopted one-change-at-a-time methodology. Established baseline performance before each modification. Documented all results systematically.



## Conclusion

### Final Performance

- **Best mean reward:** 269.85 +/- 25.30
- **Training timesteps:** 600,000
- **Key success factor:** Network capacity and action space restriction were primary drivers. Reward shaping had minimal impact due to small penalty magnitudes.

### What Could Be Done With More Time
**Extended Training**
   - Train for 2M-5M timesteps
   - Multiple random seeds (5-10 runs) to find lucky initializations
   - May naturally break through 300 reward threshold
**Action Space Annealing**
   - Start with restricted range [-0.5, 0.5]
   - Gradually expand to [-0.9, 0.9]
   - Curriculum approach to action space


## Citations

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347.*

2. Heess, N., TB, D., Sriram, S., Lemmon, J., Merel, J., Wayne, G., ... & Silver, D. (2017). "Emergence of Locomotion Behaviours in Rich Environments." *arXiv preprint arXiv:1707.02286.*

3. Kohl, N., & Stone, P. (2004). "Policy Gradient Reinforcement Learning for Fast Quadrupedal Locomotion." *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, Vol. 3, pp. 2619-2624.

4. Andrychowicz, M., Raichuk, A., Stańczyk, P., Orsini, M., Girgin, S., Marinier, R., ... & Riedmiller, M. (2021). "What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study." *arXiv preprint arXiv:2006.05990.*

5. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *arXiv preprint arXiv:1801.01290.*



**End**
```

