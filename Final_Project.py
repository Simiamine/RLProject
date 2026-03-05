# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Final Project: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv/tree/master)
#
# Ressources:
# - **Highway-env** [👨‍💻Repo](https://github.com/Farama-Foundation/HighwayEnv/tree/master) | [📜Documentation](http://highway-env.farama.org/quickstart/)
# - **OpenAI Gym**
# - **Stable-Baselines3**: [👨‍💻Repo](https://github.com/DLR-RM/stable-baselines3) | [📜Documentation](https://stable-baselines.readthedocs.io/en/master/)
#
# ### Your task: Solve the Highway
# ![](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway.gif?raw=true)
# - By Group of two, three
# - Use *at least* two different RL Algorithms
#   - try to implement at least one 'by hand'
#
# ### Evaluation
# *Based on the report (showing that you understood what you did), the performances and the code (you built something that works).*
#
# - **Produce a notebook**
#   -  The notebook must run one one go, I will not loose time trying to fix your env...
#   - Possible to send a git repo with the weight so that I ca nrun them locally.
# - **Produce a 2-5 pages report**
#   - Describe Your choices and explain the algorithms used.
#   - Benchmark and compare them depending on their hyperparameters.
#
# *Analysis could include exploration of hyperparameters, figures of training, explainations of how your algorithm works*

# %% [markdown]
# ## Setup
# ⚠️ *Do not Modify anything here !*
#
# but always read everything to be sure of what is available
#

# %% [markdown]
# ### Install (if in Colab)

# %%
import sys
IN_COLAB = True if 'google.colab' in sys.modules else False

if IN_COLAB:
    print("Running in Google Colab! Installing packages")
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # !pip -q install highway_env
    # !pip -q install gymnasium>=1.0.0a2
    # !pip -q install farama-notifications>=0.0.1
    # !pip -q install numpy>=1.21.0
    # !pip -q install pygame>=2.0.2
    # !pip -q install stable-baselines3[extra]
    # !pip -q install moviepy==2.2.1
    # %load_ext tensorboard
else:
    print("Running locally or on a different server.")


# %% [markdown]
# ### Constants
# We begin by setting the constants for evaluation, look carefully, you may want to create your own training configuration later (but your model will still be evaluated on the eval config)

# %%

ENV_ID = "highway-v0"
# Custom evaluation config to make the environment more challenging
# see https://highway-env.farama.org/environments/highway/#default-configuration for more details on the available configuration parameters

EVAL_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 40,
    "initial_spacing": .1,
    # this config makes the _Eval_ environment more challenging by increasing the speed of other vehicles and making them more aggressive
    "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle", 
    "duration": 40,  # [s]
    }


# %% [markdown]
# ### Utilities
# _We define here utilities that allows to evaluate models and record videos for you to visualize the behavior or your agents (Do do modify Anything)_

# %%
### VIDEO RECORDER
# Set up fake display; otherwise rendering will fail
import os
import numpy as np
import base64
from pathlib import Path
from IPython import display as ipythondisplay
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env  # noqa: F401
# from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from moviepy import VideoFileClip, concatenate_videoclips
import shutil

os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


def record_video(model, env_id:str = ENV_ID, video_length=500, prefix="final_render", video_folder="videos/", fps = 10):
    """Record a video of the agent's performance in the environment.
    :param model: (BaseAlgorithm) The trained RL model to evaluate.
    :param env_id: (str) The ID of the Gym environment to use for recording.
    :param video_length: (int) The number of steps to record in the video.
    :param prefix: (str) A prefix for the video file name.
    :param video_folder: (str) The folder where the video will be saved.
    :param fps: (int) Frames per second for the output video.
    """
    
    # 1. Setup paths
    base_path = Path(video_folder)
    temp_folder = base_path / "temp"
    temp_folder.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Env
    env = gym.make(env_id, render_mode="rgb_array", config=EVAL_CONFIG )
    
    env = RecordVideo(env,
                      video_folder=str(temp_folder), 
                      episode_trigger=lambda x: True, 
                      name_prefix=prefix,
                      fps=fps)
    
    # Enable HighwayEnv smooth rendering
    env.unwrapped.set_record_video_wrapper(env)
    ep_lens = []
    rewards = []
    ep_rewards = []


    # 3. Run Simulation
    obs, info = env.reset()
    for _ in tqdm(range(video_length)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            obs, info = env.reset()
            ep_rewards.append(np.sum(rewards))
            ep_lens.append(len(rewards))
            rewards = []
    
    print(f"✅ Finished recording {len(ep_rewards)} episodes. Average reward: {np.mean(ep_rewards):.2f}, Average episode length: {np.mean(ep_lens):.2f}")
    
    env.close() # Important: This flushes the final video buffer to disk

    # 4. CONCATENATION LOGIC
    video_files = sorted(list(temp_folder.glob(f"{prefix}*.mp4")))
    
    if video_files:
        print(f"Concatenating {len(video_files)} episodes...")
        clips = [VideoFileClip(str(v)) for v in video_files]
        final_clip = concatenate_videoclips(clips)
        
        final_output_path = base_path / f"{prefix}_{video_length}_steps.mp4"
        final_clip.write_videofile(str(final_output_path), logger=None)
        
        # Close clips to release file locks
        for clip in clips:
            clip.close()
            
        print(f"✅ Saved merged video of {len(video_files)} episodes to: {final_output_path}")
    
    # 5. Cleanup
    shutil.rmtree(temp_folder)

def show_videos(video_path="", prefix=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<h3>{}</h3>
                <video alt="{}" autoplay
                    loop controls style="height: 200px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


# %%
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

def evaluate(model, num_episodes=30):
    """
    Evaluates a reinforcement learning agent.

    Args:
        model: The trained RL model.
        env: The environment to evaluate the model on.
        num_episodes: The number of episodes to run for evaluation.

    Returns:
        A tuple containing the mean reward and the mean elapsed time per episode.
    """
    env = make_vec_env(ENV_ID, env_kwargs={"config":EVAL_CONFIG})

    episode_rewards = []
    episode_times = []
    print(f"evaluating Model on {num_episodes} episodes ...")
    for _ in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        total_reward = 0
        start_time = 0
        current_time = 0

        while not done:
          action, _states = model.predict(obs, deterministic=True)
          obs, reward, done, info = env.step(action)
          total_reward += reward
          current_time += 1

        episode_rewards.append(total_reward)
        episode_times.append(current_time - start_time)

    mean_reward = np.mean(episode_rewards)
    mean_time = np.mean(episode_times)
    std_reward = np.std(episode_rewards)
    std_time = np.std(episode_times)
    print(f"\n{'-'*50}\nResults :\n\t- Mean Reward: {mean_reward:.3f} ± {std_reward:.2f} \n\t- Mean elapsed Time per episode: {mean_time:.3f} ± {std_time:.2f}\n{'-'*50}")
    return mean_reward, mean_time


def evaluate_vectorized(model, num_episodes=30, n_envs=4):
    """
    Evaluates a model using parallel vectorized environments with a tqdm progress bar.
    """
    env = make_vec_env(ENV_ID, n_envs=n_envs, env_kwargs={"config": EVAL_CONFIG})
    
    episode_rewards = []
    episode_lengths = []
    
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs)
    
    obs = env.reset()
    
    pbar = tqdm(total=num_episodes, desc="Evaluating")

    while len(episode_rewards) < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        current_rewards += rewards
        current_lengths += 1 

        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])
                
                current_rewards[i] = 0
                current_lengths[i] = 0
                
                pbar.update(1)
                
                if len(episode_rewards) >= num_episodes:
                    break
    
    pbar.close()
    
    mean_reward = np.mean(episode_rewards)
    mean_time = np.mean(episode_lengths)
    std_reward = np.std(episode_rewards)
    std_time = np.std(episode_lengths)

    print(f"\n{'-'*50}\nResults :\n\t- Mean Reward: {mean_reward:.3f} ± {std_reward:.2f} \n\t- Mean elapsed Time per episode: {mean_time:.3f} ± {std_time:.2f}\n{'-'*50}")
    
    return mean_reward, mean_time


# %% [markdown]
# # The Highway Environment

# %% [markdown]
# ### Load and explore Environment
# Lets first load an untrained model and see how it behaves in the environment.

# %%
## IMPORTS
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(ENV_ID, 
                   env_kwargs={"config":EVAL_CONFIG}
                   )

#instanciate model
model = PPO("MlpPolicy", env, verbose=1)

#generate video of random model
record_video(model, video_length=90, prefix="random-agent")
show_videos("videos", prefix="random-agent")

# %%
_ = evaluate(model)

# %% [markdown]
# Let's now explore the environments settings:
# ### Action Space
# Look at the action space, what actions can the model take ?

# %%
env_explore = gym.make(ENV_ID, config=EVAL_CONFIG)
env_explore.reset()

print("Action space:", env_explore.action_space)
print("Number of actions:", env_explore.action_space.n)
print()

print("Action mapping (DiscreteMetaAction):")
for idx, action_name in env_explore.unwrapped.action_type.actions_indexes.items():
    print(f"  {action_name} -> index {idx}")

print()
print("Currently available actions:", env_explore.unwrapped.action_type.get_available_actions())
env_explore.close()

# %% [markdown]
# ### Observation Space
# Look at the [documentation](http://highway-env.farama.org/observations/) for possibles observations of the agents on the Highway
#
# Look at the observation space in our case

# %%
env_explore = gym.make(ENV_ID, config=EVAL_CONFIG)
obs, info = env_explore.reset()

print("Observation space:", env_explore.observation_space)
print("Observation shape:", obs.shape)
print()
print("Default observation type: Kinematics")
print("Each row = 1 vehicle (first row = ego vehicle)")
print("Columns: [presence, x, y, vx, vy]")
print()
print("Example observation:")
print(obs)
print()
print(f"Ego vehicle state: presence={obs[0,0]:.1f}, x={obs[0,1]:.3f}, y={obs[0,2]:.3f}, vx={obs[0,3]:.3f}, vy={obs[0,4]:.3f}")

print("\nEnvironment config:")
import pprint
pprint.pprint(env_explore.unwrapped.config)
env_explore.close()

# %% [markdown]
# # Training an Agent on the Environment
# **Now it is your turn**, train your agents
# Recall:
# - you must try and compare different RL Algorithms
# - part of your grade will be the evaluation of your best Agent.
#
# Tips
# - Use tensorboard to monitor your trainings
# - install it locally to get faster and longer trainings (not mandatory, colab should be ok)
# - try to train the agent on a [faster variant](https://highway-env.farama.org/environments/highway/#faster-variant) before evaluating on the main.

# %%
if IN_COLAB:
    pass
    # %tensorboard --logdir "highway"

# %% [markdown]
# ## Training Configuration
# We define a training config (can differ from EVAL_CONFIG) and a flag to skip
# retraining when weights already exist.

# %%
import time
import torch
from pathlib import Path

TRAIN_ENV_ID = "highway-fast-v0"

TRAIN_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 30,
    "duration": 40,
    "policy_frequency": 1,
}

RETRAIN = True

# Device detection: MPS (Apple Silicon) > CUDA > CPU
# SB3 with MlpPolicy is CPU-bound (env simulation bottleneck), so we keep
# SB3 on CPU but use MPS for the manual DQN which benefits from GPU tensor ops.
import os

if torch.backends.mps.is_available():
    DEVICE_MANUAL = "mps"
elif torch.cuda.is_available():
    DEVICE_MANUAL = "cuda"
else:
    DEVICE_MANUAL = "cpu"
DEVICE_SB3 = "auto"

N_ENVS = min(8, max(4, os.cpu_count() // 2))

print(f"SB3 device: {DEVICE_SB3} | Manual DQN device: {DEVICE_MANUAL} | "
      f"Parallel envs: {N_ENVS} | CPU cores: {os.cpu_count()}")

Path("models").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

training_histories = {}

# %% [markdown]
# ## Algorithm 1 : DQN (Stable-Baselines3)
#
# Deep Q-Network -- value-based, off-policy method. Natural fit for discrete
# action spaces. Uses a replay buffer and a target network for stability.

# %%
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

DQN_TIMESTEPS = 100_000

if RETRAIN or not Path("models/highway_dqn.zip").exists():
    print("=== Training DQN (SB3) ===")
    
    train_env_dqn = make_vec_env(
        TRAIN_ENV_ID, n_envs=N_ENVS,
        env_kwargs={"config": TRAIN_CONFIG}
    )
    
    eval_env_dqn = make_vec_env(
        TRAIN_ENV_ID, n_envs=2,
        env_kwargs={"config": EVAL_CONFIG}
    )
    
    eval_callback_dqn = EvalCallback(
        eval_env_dqn,
        best_model_save_path="models/dqn_best/",
        log_path="logs/dqn_eval/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    model_dqn = DQN(
        "MlpPolicy",
        train_env_dqn,
        device=DEVICE_SB3,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="logs/highway_dqn/",
    )
    
    t0 = time.time()
    model_dqn.learn(total_timesteps=DQN_TIMESTEPS, callback=eval_callback_dqn)
    dqn_train_time = time.time() - t0
    
    model_dqn.save("models/highway_dqn")
    print(f"DQN training done in {dqn_train_time:.1f}s")
    
    train_env_dqn.close()
    eval_env_dqn.close()
else:
    print("Loading pre-trained DQN model...")
    model_dqn = DQN.load("models/highway_dqn")
    dqn_train_time = 0

# %%
print("=== DQN Evaluation ===")
dqn_reward, dqn_time = evaluate_vectorized(model_dqn, num_episodes=30)
training_histories["DQN (SB3)"] = {
    "reward": dqn_reward, "time": dqn_time, "train_time": dqn_train_time
}

# %% [markdown]
# ## Algorithm 2 : PPO (Stable-Baselines3)
#
# Proximal Policy Optimization -- actor-critic, on-policy method. Clips the
# policy ratio to prevent destructively large updates.

# %%
from stable_baselines3 import PPO

PPO_TIMESTEPS = 100_000

if RETRAIN or not Path("models/highway_ppo.zip").exists():
    print("=== Training PPO (SB3) ===")
    
    train_env_ppo = make_vec_env(
        TRAIN_ENV_ID, n_envs=N_ENVS,
        env_kwargs={"config": TRAIN_CONFIG}
    )
    
    eval_env_ppo = make_vec_env(
        TRAIN_ENV_ID, n_envs=2,
        env_kwargs={"config": EVAL_CONFIG}
    )
    
    eval_callback_ppo = EvalCallback(
        eval_env_ppo,
        best_model_save_path="models/ppo_best/",
        log_path="logs/ppo_eval/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    model_ppo = PPO(
        "MlpPolicy",
        train_env_ppo,
        device=DEVICE_SB3,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.8,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="logs/highway_ppo/",
    )
    
    t0 = time.time()
    model_ppo.learn(total_timesteps=PPO_TIMESTEPS, callback=eval_callback_ppo)
    ppo_train_time = time.time() - t0
    
    model_ppo.save("models/highway_ppo")
    print(f"PPO training done in {ppo_train_time:.1f}s")
    
    train_env_ppo.close()
    eval_env_ppo.close()
else:
    print("Loading pre-trained PPO model...")
    model_ppo = PPO.load("models/highway_ppo")
    ppo_train_time = 0

# %%
print("=== PPO Evaluation ===")
ppo_reward, ppo_time = evaluate_vectorized(model_ppo, num_episodes=30)
training_histories["PPO (SB3)"] = {
    "reward": ppo_reward, "time": ppo_time, "train_time": ppo_train_time
}

# %% [markdown]
# ## Algorithm 3 : DQN from scratch (PyTorch)
#
# Manual implementation of DQN to demonstrate understanding of the algorithm.
# Components: Q-Network (MLP), Replay Buffer, Target Network, epsilon-greedy
# policy, and the full training loop.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    """Multi-layer perceptron Q-value approximator."""
    def __init__(self, obs_size, n_actions, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular replay buffer storing (s, a, r, s', done) tuples."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Hand-coded DQN agent with an SB3-compatible predict() interface."""
    
    def __init__(self, obs_size, n_actions, hidden_sizes=(256, 256),
                 lr=5e-4, gamma=0.8, buffer_size=15000, batch_size=32,
                 target_update_freq=50, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=10000, device="cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        self.q_net = QNetwork(obs_size, n_actions, hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_size, n_actions, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.step_count = 0
        self.training_rewards = []
    
    def select_action(self, state_flat):
        """Epsilon-greedy action selection on a flat 1-D state."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
            return self.q_net(t).argmax(dim=1).item()
    
    def predict(self, obs, deterministic=False):
        """SB3-compatible interface for evaluate() and record_video()."""
        obs_np = np.array(obs, dtype=np.float32)
        if obs_np.ndim == 3:
            obs_flat = obs_np.reshape(obs_np.shape[0], -1)
        elif obs_np.ndim == 2:
            obs_flat = obs_np.reshape(1, -1)
        else:
            obs_flat = obs_np.reshape(1, -1)
        
        with torch.no_grad():
            t = torch.FloatTensor(obs_flat).to(self.device)
            actions = self.q_net(t).argmax(dim=1).cpu().numpy()
        
        if obs_np.ndim == 2 and obs_np.shape[0] != 1:
            return actions[0], None
        return actions, None
    
    def update(self):
        """Single gradient step on a batch from the replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1)[0]
            target = r + self.gamma * next_q * (1 - d)
        
        loss = nn.functional.mse_loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "training_rewards": self.training_rewards,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.step_count = checkpoint.get("step_count", 0)
        self.training_rewards = checkpoint.get("training_rewards", [])


# %% [markdown]
# ### Training loop for DQN from scratch

# %%
MANUAL_DQN_EPISODES = 600

env_manual = gym.make(TRAIN_ENV_ID, config=TRAIN_CONFIG)
obs_shape = env_manual.observation_space.shape
obs_size = int(np.prod(obs_shape))
n_actions = env_manual.action_space.n

print(f"Obs shape: {obs_shape} -> flat size: {obs_size}, Actions: {n_actions}")

agent = DQNAgent(
    obs_size=obs_size,
    n_actions=n_actions,
    hidden_sizes=(256, 256),
    lr=5e-4,
    gamma=0.8,
    buffer_size=15000,
    batch_size=32,
    target_update_freq=50,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=15000,
    device=DEVICE_MANUAL,
)

manual_dqn_train_time = 0

if not RETRAIN and Path("models/highway_dqn_manual.pt").exists():
    print("Loading pre-trained manual DQN agent...")
    agent.load("models/highway_dqn_manual.pt")
else:
    print(f"=== Training DQN from scratch ({MANUAL_DQN_EPISODES} episodes) ===")
    t0 = time.time()
    
    for ep in range(MANUAL_DQN_EPISODES):
        obs, info = env_manual.reset()
        state = obs.flatten()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env_manual.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()
            
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            
            state = next_state
            episode_reward += reward
        
        agent.training_rewards.append(episode_reward)
        
        if (ep + 1) % 50 == 0:
            avg = np.mean(agent.training_rewards[-50:])
            print(f"  Episode {ep+1}/{MANUAL_DQN_EPISODES} | "
                  f"Avg reward (last 50): {avg:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    manual_dqn_train_time = time.time() - t0
    agent.save("models/highway_dqn_manual.pt")
    print(f"Manual DQN training done in {manual_dqn_train_time:.1f}s")

env_manual.close()

# %%
print("=== Manual DQN Evaluation ===")
manual_reward, manual_time = evaluate_vectorized(agent, num_episodes=30)
training_histories["DQN (manual)"] = {
    "reward": manual_reward, "time": manual_time,
    "train_time": manual_dqn_train_time if RETRAIN else 0,
}

# %% [markdown]
# ## Benchmark : comparing the 3 algorithms

# %%
import matplotlib.pyplot as plt

print("=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"{'Algorithm':<20} {'Mean Reward':>12} {'Mean Length':>12} {'Train Time':>12}")
print("-" * 60)
for name, data in training_histories.items():
    print(f"{name:<20} {data['reward']:>12.3f} {data['time']:>12.1f} {data['train_time']:>10.1f}s")
print("=" * 60)

# %%
if hasattr(agent, "training_rewards") and len(agent.training_rewards) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    rewards = agent.training_rewards
    window = 20
    smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
    
    ax.plot(rewards, alpha=0.3, label="Episode reward")
    ax.plot(smoothed, linewidth=2, label=f"Moving avg (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("DQN from scratch - Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/manual_dqn_training.png", dpi=150)
    plt.show()
    print("Saved to figures/manual_dqn_training.png")

# %%
fig, ax = plt.subplots(figsize=(8, 5))

names = list(training_histories.keys())
rewards = [training_histories[n]["reward"] for n in names]
colors = ["#2196F3", "#4CAF50", "#FF9800"]

bars = ax.bar(names, rewards, color=colors[:len(names)], edgecolor="black", linewidth=0.5)
ax.set_ylabel("Mean Reward (30 episodes, EVAL_CONFIG)")
ax.set_title("Benchmark : DQN (SB3) vs PPO (SB3) vs DQN (manual)")
ax.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, rewards):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("figures/benchmark_comparison.png", dpi=150)
plt.show()
print("Saved to figures/benchmark_comparison.png")

# %% [markdown]
# ## Select best model

# %%
best_name = max(training_histories, key=lambda k: training_histories[k]["reward"])
print(f"Best model: {best_name} (reward={training_histories[best_name]['reward']:.3f})")

if best_name == "DQN (SB3)":
    model_final = model_dqn
elif best_name == "PPO (SB3)":
    model_final = model_ppo
else:
    model_final = agent

print(f"model_final set to: {best_name}")

# %% [markdown]
# # Evalutation
# ⚠️ *Do not Modify anything here !*
#
# Now that your Agents are trained, we evaluate them

# %%
evaluate(model_final)

# %%
env_id = "highway-v0"
# Generate video of trained model
# NOTE: fixed argument order (original had env_id and model swapped)
record_video(model_final, env_id, video_length=70, prefix="trained-agent", fps=5)
show_videos("videos", prefix="trained-agent")

# %% [markdown]
# # Bonus
# If it was too easy for your, you can also try to train an agent on an even more difficult environment, for instance the `racetrack` *(see the highway env repo for other possible environments)*
#
# ---
# ![](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/racetrack-env.gif?raw=true)
#
