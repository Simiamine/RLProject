"""
Training script v2 — Phase 3 corrections applied.
Run: uv run python trainv2.py

Changes vs train.py (Phase 2):
  DQN SB3:    gamma 0.8->0.9, lr fixed 5e-4 -> schedule 1e-3->1e-4,
              target_update_interval 50->100
  PPO SB3:    gamma 0.8->0.9, lr fixed 5e-4 -> schedule 3e-4->3e-5,
              n_steps 256->512, added ent_coef=0.01
  DQN Manual: epsilon_decay_steps 15000->5000

Safeguards:
- EvalCallback with early stopping (patience=10 evals without improvement)
- CheckpointCallback every 10k steps
- Per-algo try/except so a failure doesn't kill the whole run
- Progress printed every eval cycle
- All models saved to models/
"""
import os
import sys
import time
import json
import numpy as np
import torch
import gymnasium as gym
import highway_env  # noqa: F401

from pathlib import Path
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
    CallbackList,
)

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# ── Config ────────────────────────────────────────────────────────────────
TRAIN_ENV_ID = "highway-fast-v0"
ENV_ID = "highway-v0"

TRAIN_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 30,
    "duration": 40,
    "policy_frequency": 1,
}
EVAL_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 40,
    "initial_spacing": 0.1,
    "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
    "duration": 40,
}

N_ENVS = min(8, max(4, os.cpu_count() // 2))

if torch.backends.mps.is_available():
    DEVICE_MANUAL = "mps"
elif torch.cuda.is_available():
    DEVICE_MANUAL = "cuda"
else:
    DEVICE_MANUAL = "cpu"

Path("models").mkdir(exist_ok=True)
Path("models/checkpoints").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

print(f"{'='*60}")
print(f"TRAINING v2 (Phase 3 corrections)")
print(f"{'='*60}")
print(f"  Parallel envs   : {N_ENVS}")
print(f"  Manual DQN device: {DEVICE_MANUAL}")
print(f"  CPU cores        : {os.cpu_count()}")
print(f"{'='*60}\n")

results = {}


def quick_evaluate(model, num_episodes=10):
    """Fast evaluation on EVAL_CONFIG."""
    env = make_vec_env(ENV_ID, n_envs=4, env_kwargs={"config": EVAL_CONFIG})
    ep_rewards = []
    ep_lengths = []
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
# ALGO 1: DQN (SB3) — Phase 3: gamma=0.9, lr schedule 1e-3->1e-4, target=100
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ALGO 1: DQN (Stable-Baselines3)")
print(f"{'='*60}")

DQN_TIMESTEPS = 100_000


def dqn_lr_schedule(progress_remaining):
    """Linear decay from 1e-3 to 1e-4 to avoid Q-value overestimation."""
    return 1e-3 * max(0.1, progress_remaining)


try:
    train_env = make_vec_env(TRAIN_ENV_ID, n_envs=N_ENVS, env_kwargs={"config": TRAIN_CONFIG})
    eval_env = make_vec_env(TRAIN_ENV_ID, n_envs=2, env_kwargs={"config": EVAL_CONFIG})

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/dqn_best/",
        log_path="logs/dqn_eval/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_callback,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=10000,
        save_path="models/checkpoints/",
        name_prefix="dqn",
        verbose=0,
    )

    model_dqn = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=dqn_lr_schedule,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.9,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=100,
        verbose=0,
        tensorboard_log="logs/highway_dqn/",
    )

    t0 = time.time()
    model_dqn.learn(
        total_timesteps=DQN_TIMESTEPS,
        callback=CallbackList([eval_cb, ckpt_cb]),
        progress_bar=True,
    )
    dqn_time = time.time() - t0
    model_dqn.save("models/highway_dqn")
    train_env.close()
    eval_env.close()

    mean_r, std_r, mean_l = quick_evaluate(model_dqn)
    results["DQN (SB3)"] = {"reward": float(mean_r), "std": float(std_r),
                            "length": float(mean_l), "train_time": dqn_time}
    print(f"\n  DQN DONE in {dqn_time:.0f}s | Eval reward: {mean_r:.2f} +/- {std_r:.2f} | Ep length: {mean_l:.0f}")

except Exception as e:
    print(f"\n  DQN FAILED: {e}")
    import traceback; traceback.print_exc()
    results["DQN (SB3)"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# ALGO 2: PPO (SB3) — Phase 3: gamma=0.9, lr schedule 3e-4->3e-5,
#                                n_steps=512, ent_coef=0.01
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ALGO 2: PPO (Stable-Baselines3)")
print(f"{'='*60}")

PPO_TIMESTEPS = 100_000


def ppo_lr_schedule(progress_remaining):
    """Linear decay from 3e-4 to 3e-5 to stabilize late training."""
    return 3e-4 * max(0.1, progress_remaining)


try:
    train_env = make_vec_env(TRAIN_ENV_ID, n_envs=N_ENVS, env_kwargs={"config": TRAIN_CONFIG})
    eval_env = make_vec_env(TRAIN_ENV_ID, n_envs=2, env_kwargs={"config": EVAL_CONFIG})

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/ppo_best/",
        log_path="logs/ppo_eval/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_callback,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=10000,
        save_path="models/checkpoints/",
        name_prefix="ppo",
        verbose=0,
    )

    model_ppo = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=ppo_lr_schedule,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.9,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log="logs/highway_ppo/",
    )

    t0 = time.time()
    model_ppo.learn(
        total_timesteps=PPO_TIMESTEPS,
        callback=CallbackList([eval_cb, ckpt_cb]),
        progress_bar=True,
    )
    ppo_time = time.time() - t0
    model_ppo.save("models/highway_ppo")
    train_env.close()
    eval_env.close()

    mean_r, std_r, mean_l = quick_evaluate(model_ppo)
    results["PPO (SB3)"] = {"reward": float(mean_r), "std": float(std_r),
                            "length": float(mean_l), "train_time": ppo_time}
    print(f"\n  PPO DONE in {ppo_time:.0f}s | Eval reward: {mean_r:.2f} +/- {std_r:.2f} | Ep length: {mean_l:.0f}")

except Exception as e:
    print(f"\n  PPO FAILED: {e}")
    import traceback; traceback.print_exc()
    results["PPO (SB3)"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# ALGO 3: DQN from scratch (PyTorch) — Phase 3: epsilon_decay_steps=5000
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ALGO 3: DQN from scratch (PyTorch)")
print(f"{'='*60}")

import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


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
    def __init__(self, obs_size, n_actions, hidden_sizes=(256, 256),
                 lr=5e-4, gamma=0.8, buffer_size=15000, batch_size=32,
                 target_update_freq=50, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=5000, device="cpu"):
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
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
            return self.q_net(t).argmax(dim=1).item()

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
            actions = self.q_net(t).argmax(dim=1).cpu().numpy()
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.step_count = checkpoint.get("step_count", 0)
        self.training_rewards = checkpoint.get("training_rewards", [])


MANUAL_EPISODES = 600
PATIENCE = 80

try:
    env_manual = gym.make(TRAIN_ENV_ID, config=TRAIN_CONFIG)
    obs_shape = env_manual.observation_space.shape
    obs_size = int(np.prod(obs_shape))
    n_actions = env_manual.action_space.n

    agent = DQNAgent(
        obs_size=obs_size, n_actions=n_actions,
        hidden_sizes=(256, 256), lr=5e-4, gamma=0.8,
        buffer_size=15000, batch_size=32, target_update_freq=50,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=5000,
        device=DEVICE_MANUAL,
    )
    print(f"  Device: {agent.device} | Obs: {obs_shape} -> {obs_size} | Actions: {n_actions}")
    print(f"  Episodes: {MANUAL_EPISODES} | Early stop patience: {PATIENCE}")
    print(f"  epsilon_decay_steps: 5000 (Phase 3 fix)")

    t0 = time.time()
    best_avg = -float("inf")
    no_improve_count = 0

    for ep in range(MANUAL_EPISODES):
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

        if (ep + 1) % 25 == 0:
            recent = agent.training_rewards[-25:]
            avg = np.mean(recent)
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            eta = (MANUAL_EPISODES - ep - 1) / eps_per_sec if eps_per_sec > 0 else 0

            improved = ""
            if avg > best_avg:
                best_avg = avg
                no_improve_count = 0
                agent.save("models/highway_dqn_manual_best.pt")
                improved = " *BEST*"
            else:
                no_improve_count += 25

            print(f"  Ep {ep+1:4d}/{MANUAL_EPISODES} | "
                  f"Avg(25): {avg:6.2f}{improved} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.buffer):5d} | "
                  f"{eps_per_sec:.1f} ep/s | "
                  f"ETA: {eta:.0f}s")

            if no_improve_count >= PATIENCE:
                print(f"  EARLY STOP: no improvement for {PATIENCE} episodes")
                break

        if (ep + 1) % 100 == 0:
            agent.save(f"models/checkpoints/dqn_manual_ep{ep+1}.pt")

    manual_time = time.time() - t0
    agent.save("models/highway_dqn_manual.pt")
    env_manual.close()

    mean_r, std_r, mean_l = quick_evaluate(agent)
    results["DQN (manual)"] = {"reward": float(mean_r), "std": float(std_r),
                               "length": float(mean_l), "train_time": manual_time}
    print(f"\n  Manual DQN DONE in {manual_time:.0f}s | Eval reward: {mean_r:.2f} +/- {std_r:.2f} | Ep length: {mean_l:.0f}")

except Exception as e:
    print(f"\n  Manual DQN FAILED: {e}")
    import traceback; traceback.print_exc()
    results["DQN (manual)"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print("TRAINING v2 COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"{'Algorithm':<20} {'Reward':>10} {'Std':>8} {'Length':>8} {'Time':>10}")
print(f"{'-'*60}")
for name, data in results.items():
    if "error" in data:
        print(f"{name:<20} {'FAILED':>10}")
    else:
        print(f"{name:<20} {data['reward']:>10.2f} {data['std']:>8.2f} {data['length']:>8.0f} {data['train_time']:>8.0f}s")
print(f"{'='*60}")

with open("models/training_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to models/training_results.json")
print("Models saved to models/")

print("\nSaved models:")
for p in sorted(Path("models").glob("*.zip")) + sorted(Path("models").glob("*.pt")):
    print(f"  {p} ({p.stat().st_size / 1024:.0f} KB)")
