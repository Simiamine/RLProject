"""
Bonus: Racetrack training script.
Run: uv run python bonus/train_racetrack.py

Two algorithms:
  1. PPO (SB3) with continuous actions (native racetrack)
  2. DQN manual (PyTorch) with discretized steering

Applies all lessons from the highway project:
  - lr schedules, gradient clipping, eval-based best model,
    no early stopping for manual DQN, best checkpoint copy for PPO.
"""
import os
import sys
import time
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gymnasium as gym
import highway_env  # noqa: F401

from pathlib import Path
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# ── Config ────────────────────────────────────────────────────────────────
ENV_ID = "racetrack-v0"

RACETRACK_CONFIG_PPO = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ["presence", "on_road"],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True,
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True,
    },
    "duration": 300,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "other_vehicles": 1,
}

RACETRACK_CONFIG_DQN = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True,
    },
    "duration": 300,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "other_vehicles": 1,
}

STEERING_BINS = np.array([-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0])
N_BINS = len(STEERING_BINS)

N_ENVS = min(8, max(4, os.cpu_count() // 2))

if torch.backends.mps.is_available():
    DEVICE_MANUAL = "mps"
elif torch.cuda.is_available():
    DEVICE_MANUAL = "cuda"
else:
    DEVICE_MANUAL = "cpu"

Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
Path("videos").mkdir(exist_ok=True)

print(f"{'='*60}")
print(f"BONUS: RACETRACK TRAINING")
print(f"{'='*60}")
print(f"  Parallel envs    : {N_ENVS}")
print(f"  Manual DQN device: {DEVICE_MANUAL}")
print(f"  Steering bins    : {N_BINS}")
print(f"{'='*60}\n")

results = {}


def evaluate_model(model, config, num_episodes=20):
    """Evaluate on racetrack with vectorized envs."""
    env = make_vec_env(ENV_ID, n_envs=4, env_kwargs={"config": config})
    ep_rewards, ep_lengths = [], []
    current_rewards = np.zeros(4)
    current_lengths = np.zeros(4)
    obs = env.reset()
    while len(ep_rewards) < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        current_rewards += rewards
        current_lengths += 1
        for i, done in enumerate(dones):
            if done:
                ep_rewards.append(current_rewards[i])
                ep_lengths.append(current_lengths[i])
                current_rewards[i] = 0
                current_lengths[i] = 0
                if len(ep_rewards) >= num_episodes:
                    break
    env.close()
    return np.mean(ep_rewards), np.std(ep_rewards), np.mean(ep_lengths)


# ═══════════════════════════════════════════════════════════════════════════
# ALGO 1: PPO (SB3) — Continuous actions, OccupancyGrid obs
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ALGO 1: PPO (SB3) — Continuous steering")
print(f"{'='*60}")

PPO_TIMESTEPS = 300_000


def ppo_lr_schedule(progress_remaining):
    return 3e-4 * max(0.1, progress_remaining)


try:
    train_env = make_vec_env(ENV_ID, n_envs=N_ENVS, env_kwargs={"config": RACETRACK_CONFIG_PPO})
    eval_env = make_vec_env(ENV_ID, n_envs=2, env_kwargs={"config": RACETRACK_CONFIG_PPO})

    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=8, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/ppo_best/",
        log_path="logs/ppo_eval/",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    model_ppo = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=ppo_lr_schedule,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log="logs/racetrack_ppo/",
    )

    t0 = time.time()
    model_ppo.learn(total_timesteps=PPO_TIMESTEPS, callback=eval_cb, progress_bar=True)
    ppo_time = time.time() - t0
    model_ppo.save("models/racetrack_ppo")
    train_env.close()
    eval_env.close()

    best_path = Path("models/ppo_best/best_model.zip")
    final_path = Path("models/racetrack_ppo.zip")
    if best_path.exists():
        shutil.copy2(best_path, final_path)
        print(f"  Copied best checkpoint -> {final_path}")

    model_ppo_best = PPO.load(str(final_path))
    mean_r, std_r, mean_l = evaluate_model(model_ppo_best, RACETRACK_CONFIG_PPO)
    results["PPO (continuous)"] = {
        "reward": float(mean_r), "std": float(std_r),
        "length": float(mean_l), "train_time": ppo_time,
    }
    print(f"\n  PPO DONE in {ppo_time:.0f}s | Eval: {mean_r:.2f} +/- {std_r:.2f} | Length: {mean_l:.0f}")

except Exception as e:
    print(f"\n  PPO FAILED: {e}")
    import traceback; traceback.print_exc()
    results["PPO (continuous)"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# ALGO 2: DQN manual — Discretized steering, Kinematics obs
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ALGO 2: DQN manual — Discretized steering (7 bins)")
print(f"{'='*60}")


class QNetwork(nn.Module):
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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

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


class RacetrackDQNAgent:
    """DQN agent with discretized continuous actions for racetrack."""

    def __init__(self, obs_size, steering_bins, hidden_sizes=(256, 256),
                 lr=5e-4, gamma=0.95, buffer_size=30000, batch_size=64,
                 target_update_freq=100, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=10000, max_grad_norm=10.0, device="cpu"):
        self.device = torch.device(device)
        self.steering_bins = steering_bins
        self.n_actions = len(steering_bins)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.q_net = QNetwork(obs_size, self.n_actions, hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_size, self.n_actions, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.step_count = 0
        self.training_rewards = []

    def _get_action_idx(self, state_flat):
        with torch.no_grad():
            t = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
            return self.q_net(t).argmax(dim=1).item()

    def select_action(self, state_flat):
        if random.random() < self.epsilon:
            idx = random.randrange(self.n_actions)
        else:
            idx = self._get_action_idx(state_flat)
        return idx, np.array([self.steering_bins[idx]])

    def predict(self, obs, deterministic=False):
        obs_np = np.array(obs, dtype=np.float32)
        if obs_np.ndim == 3:
            obs_flat = obs_np.reshape(obs_np.shape[0], -1)
        elif obs_np.ndim == 2:
            obs_flat = obs_np.reshape(1, -1)
        else:
            obs_flat = obs_np.reshape(1, -1)
        with torch.no_grad():
            t = torch.FloatTensor(obs_flat).to(self.device)
            indices = self.q_net(t).argmax(dim=1).cpu().numpy()
        actions = np.array([[self.steering_bins[i]] for i in indices])
        if obs_np.ndim == 2 and obs_np.shape[0] != 1:
            return actions[0], None
        return actions, None

    def update(self):
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
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
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
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.step_count = ckpt.get("step_count", 0)
        self.training_rewards = ckpt.get("training_rewards", [])


def deterministic_eval_dqn(agent, config, num_episodes=10):
    env = gym.make(ENV_ID, config=config)
    rewards, lengths = [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        ep_reward, done = 0, False
        ep_len = 0
        while not done:
            idx = agent._get_action_idx(state)
            action = np.array([agent.steering_bins[idx]])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = obs.flatten()
            ep_reward += reward
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
    env.close()
    return np.mean(rewards), np.std(rewards), np.mean(lengths)


MANUAL_EPISODES = 2000
EVAL_EVERY = 100
EVAL_EPS = 10

try:
    env_manual = gym.make(ENV_ID, config=RACETRACK_CONFIG_DQN)
    obs_shape = env_manual.observation_space.shape
    obs_size = int(np.prod(obs_shape))
    print(f"  Obs: {obs_shape} -> flat {obs_size} | Steering bins: {N_BINS}")
    print(f"  Episodes: {MANUAL_EPISODES} | Eval every {EVAL_EVERY} eps")

    agent = RacetrackDQNAgent(
        obs_size=obs_size, steering_bins=STEERING_BINS,
        hidden_sizes=(256, 256), lr=5e-4, gamma=0.95,
        buffer_size=30000, batch_size=64, target_update_freq=100,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000,
        max_grad_norm=10.0, device=DEVICE_MANUAL,
    )

    t0 = time.time()
    best_eval_reward = -float("inf")

    for ep in range(MANUAL_EPISODES):
        obs, _ = env_manual.reset()
        state = obs.flatten()
        ep_reward, done = 0, False

        while not done:
            idx, action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env_manual.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()
            agent.buffer.push(state, idx, reward, next_state, float(done))
            agent.update()
            state = next_state
            ep_reward += reward

        agent.training_rewards.append(ep_reward)

        if (ep + 1) % 50 == 0:
            recent = agent.training_rewards[-50:]
            avg = np.mean(recent)
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            eta = (MANUAL_EPISODES - ep - 1) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"  Ep {ep+1:5d}/{MANUAL_EPISODES} | "
                  f"Train avg(50): {avg:7.2f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.buffer):6d} | "
                  f"{eps_per_sec:.1f} ep/s | ETA: {eta:.0f}s")

        if (ep + 1) % EVAL_EVERY == 0:
            ev_r, ev_std, ev_l = deterministic_eval_dqn(agent, RACETRACK_CONFIG_DQN, EVAL_EPS)
            improved = ""
            if ev_r > best_eval_reward:
                best_eval_reward = ev_r
                agent.save("models/racetrack_dqn_manual_best.pt")
                improved = " *BEST*"
            print(f"  >>> EVAL ep {ep+1}: reward={ev_r:.2f} +/- {ev_std:.2f}, "
                  f"length={ev_l:.0f}{improved}")

    manual_time = time.time() - t0
    agent.save("models/racetrack_dqn_manual_final.pt")
    env_manual.close()

    best_path = Path("models/racetrack_dqn_manual_best.pt")
    final_path = Path("models/racetrack_dqn_manual.pt")
    if best_path.exists():
        shutil.copy2(best_path, final_path)
        print(f"\n  Copied best eval model -> {final_path}")

    agent.load(str(final_path))
    mean_r, std_r, mean_l = deterministic_eval_dqn(agent, RACETRACK_CONFIG_DQN, 20)
    results["DQN (discretized)"] = {
        "reward": float(mean_r), "std": float(std_r),
        "length": float(mean_l), "train_time": manual_time,
        "best_eval_reward": float(best_eval_reward),
    }
    print(f"  DQN DONE in {manual_time:.0f}s | Eval: {mean_r:.2f} +/- {std_r:.2f} | Length: {mean_l:.0f}")

except Exception as e:
    print(f"\n  DQN FAILED: {e}")
    import traceback; traceback.print_exc()
    results["DQN (discretized)"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print("RACETRACK TRAINING COMPLETE")
print(f"{'='*60}")
print(f"{'Algorithm':<25} {'Reward':>10} {'Std':>8} {'Length':>8} {'Time':>10}")
print(f"{'-'*65}")
for name, data in results.items():
    if "error" in data:
        print(f"{name:<25} {'FAILED':>10}")
    else:
        print(f"{name:<25} {data['reward']:>10.2f} {data['std']:>8.2f} "
              f"{data['length']:>8.0f} {data['train_time']:>8.0f}s")
print(f"{'='*60}")

with open("models/racetrack_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to models/racetrack_results.json")
