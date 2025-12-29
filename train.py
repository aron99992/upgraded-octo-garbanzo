import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os

video_folder = "videos/"
log_dir = "./ppo_bipedal_tensorboard/"
stats_path = "vec_normalize.pkl"
os.makedirs(video_folder, exist_ok=True)


class ActionClipping(gym.ActionWrapper):
    def __init__(self, env, clip_range=0.7):
        super().__init__(env)
        self.clip_range = clip_range
    
    def action(self, action):
        return np.clip(action, -self.clip_range, self.clip_range)
    

class AnnealingCallback(BaseCallback):
    def __init__(self, annealing_steps=500000, verbose=0):
        super().__init__(verbose)
        self.annealing_steps = annealing_steps
    
        self.initial_lr = 5e-4
        self.initial_clip = 0.3
        self.initial_ent = 0.05
        
        self.final_lr = 1e-6
        self.final_clip = 0.02
        self.final_ent = 0.0001
    
    def _on_step(self):
        if self.num_timesteps < self.annealing_steps:
            progress = self.num_timesteps / self.annealing_steps
            temp = (1 - progress)  # 1.0 → 0.0
            
            # anneal lr
            new_lr = self.final_lr + (self.initial_lr - self.final_lr) * temp
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # anneal clip range
            new_clip = self.final_clip + (self.initial_clip - self.final_clip) * temp
            self.model.clip_range = lambda _: new_clip
            
            # anneal entropy
            new_ent = self.final_ent + (self.initial_ent - self.final_ent) * temp
            self.model.ent_coef = new_ent
            
            self.logger.record("train/learning_rate", new_lr)
            self.logger.record("train/clip_range", new_clip)
            self.logger.record("train/entropy_coef", new_ent)
            self.logger.record("train/temperature", temp)
        
        return True

anneal_callback = AnnealingCallback(annealing_steps=500000)



class Reward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(4)
        self.prev_vel = 0.0
        
    def reset(self, **kwargs):
        self.prev_action = np.zeros(4)
        self.prev_vel = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        hull_angle = obs[2]
        hull_ang_vel = obs[3]
        vel_x = obs[4] 
        
        # angle penalty
        angle_penalty = 0.04 * abs(hull_angle)
        
        # angular velocity penalty
        ang_vel_penalty = 0.015 * abs(hull_ang_vel)  
        
        # action smoothness
        action_diff = np.mean(np.square(action - self.prev_action))
        smoothness_penalty = 0.005 * action_diff
        self.prev_action = action.copy()
        
        # velocity smoothness
        vel_change = abs(vel_x - self.prev_vel)
        vel_smoothness_penalty = 0.025 * vel_change
        self.prev_vel = vel_x
        
        # airborne penalty
        lidar = obs[14:24]
        airborne_penalty = 0.0
        if np.mean(lidar) > 0.7:  # mostly in air
            airborne_penalty = 0.1
        
        if reward <= -100:
            reward = -50
        
        shaped_reward = (reward 
                        - angle_penalty 
                        - ang_vel_penalty
                        - smoothness_penalty 
                        - vel_smoothness_penalty
                        - airborne_penalty)
        
        return obs, shaped_reward, terminated, truncated, info


def make_env():
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
    env = Monitor(env)
    env = ActionClipping(env, clip_range=0.7)
    env = Reward(env)

    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))


    return env


env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x % 100000 == 0, video_length=2500, name_prefix="ppo-walker")


eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
eval_env = VecVideoRecorder(eval_env, video_folder, record_video_trigger=lambda x: x % 100000 == 0, video_length=2500, name_prefix="ppo-walker")
eval_env.training = False 
eval_env.norm_reward = False  

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./eval_logs/',
    eval_freq=5000,
    n_eval_episodes=15,
    deterministic=True
)

total_timesteps = 600000

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=2e-4,        
    n_steps=4096,             
    batch_size=512,     
    n_epochs=20,              
    gamma=0.995,         
    gae_lambda=0.95,          
    clip_range=0.1,            
    ent_coef=0.003,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=True,
    sde_sample_freq=4,
    policy_kwargs=dict(
        log_std_init=-2.5,     
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    ),
    tensorboard_log=log_dir,
    device="mps",            
    verbose=1
)

# tensorboard --logdir=./ppo_bipedal_tensorboard/
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, anneal_callback]
)

model.save("ppo_bipedal_simple")
env.save(stats_path)


eval_env.obs_rms = env.obs_rms

mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=50, deterministic=True
)
print(f"\nFinal Results:")
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")



eval_env.close()
env.close()





