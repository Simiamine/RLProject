"""
Hyperparameter grid search for all 3 RL algorithms.
Run: PYTHONUNBUFFERED=1 uv run python -u hyperparam_search.py

Phase 1: Quick exploration (30k steps SB3, 200 episodes manual)
Phase 2: Re-train best config per algo at full power (100k steps / 600 episodes)

Results saved to models/hyperparam_results.json
"""
import os
import sys
import time
import json
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList

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

EXPLORE_STEPS_SB3 = 30_000
EXPLORE_EPISODES_MANUAL = 200
FULL_STEPS_SB3 = 100_000
FULL_EPISODES_MANUAL = 600
EVAL_EPISODES = 15

Path("models").mkdir(exist_ok=True)
Path("models/hyperparam").mkdir(exist_ok=True)

print(f"{'='*70}")
print(f"HYPERPARAMETER GRID SEARCH")
print(f"{'='*70}")
print(f"  N_ENVS={N_ENVS} | Manual device={DEVICE_MANUAL}")
print(f"  Explore: {EXPLORE_STEPS_SB3} steps (SB3), {EXPLORE_EPISODES_MANUAL} eps (manual)")
print(f"  Full retrain: {FULL_STEPS_SB3} steps (SB3), {FULL_EPISODES_MANUAL} eps (manual)")
print(f"{'='*70}\n")


# ── DQN Manual components ────────────────────────────────────────────────

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
        return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_size, n_actions, hidden_sizes=(256, 256),
                 lr=5e-4, gamma=0.8, buffer_size=15000, batch_size=32,
                 target_update_freq=50, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=15000, device="cpu"):
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
        torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict(),
                     "epsilon": self.epsilon, "step_count": self.step_count,
                     "training_rewards": self.training_rewards}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)  # weights_only=False needed for numpy arrays in checkpoint
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.step_count = checkpoint.get("step_count", 0)
        self.training_rewards = checkpoint.get("training_rewards", [])


# ── Evaluation helper ─────────────────────────────────────────────────────

def quick_evaluate(model, num_episodes=EVAL_EPISODES):
    env = make_vec_env(ENV_ID, n_envs=4, env_kwargs={"config": EVAL_CONFIG})
    ep_rewards, ep_lengths = [], []
    cr, cl = np.zeros(4), np.zeros(4)
    obs = env.reset()
    while len(ep_rewards) < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        cr += rewards; cl += 1
        for i, done in enumerate(dones):
            if done:
                ep_rewards.append(cr[i]); ep_lengths.append(cl[i])
                cr[i] = 0; cl[i] = 0
                if len(ep_rewards) >= num_episodes:
                    break
    env.close()
    return float(np.mean(ep_rewards)), float(np.std(ep_rewards)), float(np.mean(ep_lengths))


# ── Grids ─────────────────────────────────────────────────────────────────

GAMMAS = [0.8, 0.9, 0.99]
LRS_DQN = [5e-4, 1e-3]
LRS_PPO = [3e-4, 5e-4]
LRS_MANUAL = [5e-4, 1e-3]

dqn_grid = [{"gamma": g, "lr": lr} for g, lr in itertools.product(GAMMAS, LRS_DQN)][:5]
ppo_grid = [{"gamma": g, "lr": lr} for g, lr in itertools.product(GAMMAS, LRS_PPO)][:5]
manual_grid = [{"gamma": g, "lr": lr} for g, lr in itertools.product(GAMMAS, LRS_MANUAL)][:5]

all_results = {"dqn_sb3": [], "ppo_sb3": [], "dqn_manual": []}


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PHASE 1: QUICK EXPLORATION")
print(f"{'='*70}\n")

# ── DQN SB3 grid ──
print(f"--- DQN SB3 ({len(dqn_grid)} configs, {EXPLORE_STEPS_SB3} steps each) ---")
for i, cfg in enumerate(dqn_grid):
    t0 = time.time()
    try:
        env = make_vec_env(TRAIN_ENV_ID, n_envs=N_ENVS, env_kwargs={"config": TRAIN_CONFIG})
        model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]),
                     learning_rate=cfg["lr"], gamma=cfg["gamma"],
                     buffer_size=15000, learning_starts=200, batch_size=32,
                     train_freq=1, gradient_steps=1, target_update_interval=50,
                     verbose=0)
        model.learn(total_timesteps=EXPLORE_STEPS_SB3)
        env.close()
        mean_r, std_r, mean_l = quick_evaluate(model)
        dt = time.time() - t0
        result = {"gamma": cfg["gamma"], "lr": cfg["lr"], "reward": mean_r,
                  "std": std_r, "length": mean_l, "time": dt}
        all_results["dqn_sb3"].append(result)
        best_mark = " <-- BEST" if mean_r == max(r["reward"] for r in all_results["dqn_sb3"]) else ""
        print(f"  [{i+1}/{len(dqn_grid)}] gamma={cfg['gamma']}, lr={cfg['lr']:.0e} "
              f"-> reward={mean_r:.2f} +/- {std_r:.1f} ({dt:.0f}s){best_mark}")
    except Exception as e:
        print(f"  [{i+1}/{len(dqn_grid)}] gamma={cfg['gamma']}, lr={cfg['lr']:.0e} -> FAILED: {e}")
        all_results["dqn_sb3"].append({"gamma": cfg["gamma"], "lr": cfg["lr"], "error": str(e)})

# ── PPO SB3 grid ──
print(f"\n--- PPO SB3 ({len(ppo_grid)} configs, {EXPLORE_STEPS_SB3} steps each) ---")
for i, cfg in enumerate(ppo_grid):
    t0 = time.time()
    try:
        env = make_vec_env(TRAIN_ENV_ID, n_envs=N_ENVS, env_kwargs={"config": TRAIN_CONFIG})
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                     learning_rate=cfg["lr"], gamma=cfg["gamma"],
                     n_steps=256, batch_size=64, n_epochs=10, clip_range=0.2,
                     verbose=0)
        model.learn(total_timesteps=EXPLORE_STEPS_SB3)
        env.close()
        mean_r, std_r, mean_l = quick_evaluate(model)
        dt = time.time() - t0
        result = {"gamma": cfg["gamma"], "lr": cfg["lr"], "reward": mean_r,
                  "std": std_r, "length": mean_l, "time": dt}
        all_results["ppo_sb3"].append(result)
        best_mark = " <-- BEST" if mean_r == max(r["reward"] for r in all_results["ppo_sb3"]) else ""
        print(f"  [{i+1}/{len(ppo_grid)}] gamma={cfg['gamma']}, lr={cfg['lr']:.0e} "
              f"-> reward={mean_r:.2f} +/- {std_r:.1f} ({dt:.0f}s){best_mark}")
    except Exception as e:
        print(f"  [{i+1}/{len(ppo_grid)}] gamma={cfg['gamma']}, lr={cfg['lr']:.0e} -> FAILED: {e}")
        all_results["ppo_sb3"].append({"gamma": cfg["gamma"], "lr": cfg["lr"], "error": str(e)})

# ── DQN manual grid ──
print(f"\n--- DQN Manual ({len(manual_grid)} configs, {EXPLORE_EPISODES_MANUAL} episodes each) ---")
for i, cfg in enumerate(manual_grid):
    t0 = time.time()
    try:
        env = gym.make(TRAIN_ENV_ID, config=TRAIN_CONFIG)
        obs_size = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n
        agent = DQNAgent(obs_size=obs_size, n_actions=n_actions,
                         lr=cfg["lr"], gamma=cfg["gamma"], device=DEVICE_MANUAL)
        for ep in range(EXPLORE_EPISODES_MANUAL):
            obs, _ = env.reset()
            state = obs.flatten()
            done, ep_reward = False, 0
            while not done:
                action = agent.select_action(state)
                nobs, rew, term, trunc, _ = env.step(action)
                done = term or trunc
                nstate = nobs.flatten()
                agent.buffer.push(state, action, rew, nstate, float(done))
                agent.update()
                state = nstate
                ep_reward += rew
            agent.training_rewards.append(ep_reward)
        env.close()
        mean_r, std_r, mean_l = quick_evaluate(agent)
        dt = time.time() - t0
        result = {"gamma": cfg["gamma"], "lr": cfg["lr"], "reward": mean_r,
                  "std": std_r, "length": mean_l, "time": dt}
        all_results["dqn_manual"].append(result)
        best_mark = " <-- BEST" if mean_r == max(r["reward"] for r in all_results["dqn_manual"]) else ""
        print(f"  [{i+1}/{len(manual_grid)}] gamma={cfg['gamma']}, lr={cfg['lr']:.0e} "
              f"-> reward={mean_r:.2f} +/- {std_r:.1f} ({dt:.0f}s){best_mark}")
    except Exception as e:
        print(f"  [{i+1}/{len(manual_grid)}] gamma={cfg['gamma']}, lr={cfg['lr']:.0e} -> FAILED: {e}")
        all_results["dqn_manual"].append({"gamma": cfg["gamma"], "lr": cfg["lr"], "error": str(e)})

# Save exploration results
with open("models/hyperparam_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nExploration results saved to models/hyperparam_results.json")


# ── Find best configs ──
def best_config(results):
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None
    return max(valid, key=lambda r: r["reward"])

best_dqn = best_config(all_results["dqn_sb3"])
best_ppo = best_config(all_results["ppo_sb3"])
best_manual = best_config(all_results["dqn_manual"])

print(f"\n{'='*70}")
print("BEST CONFIGS")
print(f"{'='*70}")
if best_dqn:
    print(f"  DQN SB3:   gamma={best_dqn['gamma']}, lr={best_dqn['lr']:.0e} -> reward={best_dqn['reward']:.2f}")
if best_ppo:
    print(f"  PPO SB3:   gamma={best_ppo['gamma']}, lr={best_ppo['lr']:.0e} -> reward={best_ppo['reward']:.2f}")
if best_manual:
    print(f"  DQN Manual: gamma={best_manual['gamma']}, lr={best_manual['lr']:.0e} -> reward={best_manual['reward']:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: RETRAIN BEST CONFIGS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*70}")
print("PHASE 2: RETRAIN BEST CONFIGS AT FULL POWER")
print(f"{'='*70}\n")

final_results = {}

# ── DQN SB3 best ──
if best_dqn:
    print(f"--- Retraining DQN SB3 (gamma={best_dqn['gamma']}, lr={best_dqn['lr']:.0e}, {FULL_STEPS_SB3} steps) ---")
    try:
        env = make_vec_env(TRAIN_ENV_ID, n_envs=N_ENVS, env_kwargs={"config": TRAIN_CONFIG})
        eval_env = make_vec_env(TRAIN_ENV_ID, n_envs=2, env_kwargs={"config": EVAL_CONFIG})
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, verbose=1)
        eval_cb = EvalCallback(eval_env, best_model_save_path="models/dqn_best/",
                               log_path="logs/dqn_eval/", eval_freq=5000,
                               n_eval_episodes=10, deterministic=True,
                               callback_after_eval=stop_cb, verbose=1)
        model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]),
                     learning_rate=best_dqn["lr"], gamma=best_dqn["gamma"],
                     buffer_size=15000, learning_starts=200, batch_size=32,
                     train_freq=1, gradient_steps=1, target_update_interval=50,
                     verbose=0, tensorboard_log="logs/highway_dqn/")
        t0 = time.time()
        model.learn(total_timesteps=FULL_STEPS_SB3, callback=eval_cb, progress_bar=True)
        dt = time.time() - t0
        model.save("models/highway_dqn")
        env.close(); eval_env.close()
        mean_r, std_r, mean_l = quick_evaluate(model, num_episodes=30)
        final_results["DQN (SB3)"] = {"gamma": best_dqn["gamma"], "lr": best_dqn["lr"],
                                       "reward": mean_r, "std": std_r, "length": mean_l, "time": dt}
        print(f"  DONE in {dt:.0f}s | reward={mean_r:.2f} +/- {std_r:.1f} | length={mean_l:.0f}")
    except Exception as e:
        print(f"  FAILED: {e}")

# ── PPO SB3 best ──
if best_ppo:
    print(f"\n--- Retraining PPO SB3 (gamma={best_ppo['gamma']}, lr={best_ppo['lr']:.0e}, {FULL_STEPS_SB3} steps) ---")
    try:
        env = make_vec_env(TRAIN_ENV_ID, n_envs=N_ENVS, env_kwargs={"config": TRAIN_CONFIG})
        eval_env = make_vec_env(TRAIN_ENV_ID, n_envs=2, env_kwargs={"config": EVAL_CONFIG})
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, verbose=1)
        eval_cb = EvalCallback(eval_env, best_model_save_path="models/ppo_best/",
                               log_path="logs/ppo_eval/", eval_freq=5000,
                               n_eval_episodes=10, deterministic=True,
                               callback_after_eval=stop_cb, verbose=1)
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                     learning_rate=best_ppo["lr"], gamma=best_ppo["gamma"],
                     n_steps=256, batch_size=64, n_epochs=10, clip_range=0.2,
                     verbose=0, tensorboard_log="logs/highway_ppo/")
        t0 = time.time()
        model.learn(total_timesteps=FULL_STEPS_SB3, callback=eval_cb, progress_bar=True)
        dt = time.time() - t0
        model.save("models/highway_ppo")
        env.close(); eval_env.close()
        mean_r, std_r, mean_l = quick_evaluate(model, num_episodes=30)
        final_results["PPO (SB3)"] = {"gamma": best_ppo["gamma"], "lr": best_ppo["lr"],
                                       "reward": mean_r, "std": std_r, "length": mean_l, "time": dt}
        print(f"  DONE in {dt:.0f}s | reward={mean_r:.2f} +/- {std_r:.1f} | length={mean_l:.0f}")
    except Exception as e:
        print(f"  FAILED: {e}")

# ── DQN manual best ──
if best_manual:
    print(f"\n--- Retraining DQN Manual (gamma={best_manual['gamma']}, lr={best_manual['lr']:.0e}, {FULL_EPISODES_MANUAL} eps) ---")
    try:
        env = gym.make(TRAIN_ENV_ID, config=TRAIN_CONFIG)
        obs_size = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n
        agent = DQNAgent(obs_size=obs_size, n_actions=n_actions,
                         lr=best_manual["lr"], gamma=best_manual["gamma"],
                         device=DEVICE_MANUAL)
        t0 = time.time()
        best_avg = -float("inf")
        no_improve = 0
        for ep in range(FULL_EPISODES_MANUAL):
            obs, _ = env.reset()
            state = obs.flatten()
            done, ep_reward = False, 0
            while not done:
                action = agent.select_action(state)
                nobs, rew, term, trunc, _ = env.step(action)
                done = term or trunc
                nstate = nobs.flatten()
                agent.buffer.push(state, action, rew, nstate, float(done))
                agent.update()
                state = nstate
                ep_reward += rew
            agent.training_rewards.append(ep_reward)
            if (ep + 1) % 50 == 0:
                avg = np.mean(agent.training_rewards[-50:])
                improved = ""
                if avg > best_avg:
                    best_avg = avg
                    no_improve = 0
                    agent.save("models/highway_dqn_manual_best.pt")
                    improved = " *BEST*"
                else:
                    no_improve += 50
                print(f"    Ep {ep+1}/{FULL_EPISODES_MANUAL} | Avg(50): {avg:.2f}{improved} | Eps: {agent.epsilon:.3f}")
                if no_improve >= 100:
                    print(f"    EARLY STOP at ep {ep+1}")
                    break
        dt = time.time() - t0
        agent.save("models/highway_dqn_manual.pt")
        env.close()
        mean_r, std_r, mean_l = quick_evaluate(agent, num_episodes=30)
        final_results["DQN (manual)"] = {"gamma": best_manual["gamma"], "lr": best_manual["lr"],
                                          "reward": mean_r, "std": std_r, "length": mean_l, "time": dt}
        print(f"  DONE in {dt:.0f}s | reward={mean_r:.2f} +/- {std_r:.1f} | length={mean_l:.0f}")
    except Exception as e:
        print(f"  FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*70}")
print("FINAL RESULTS (best hyperparameters, full training)")
print(f"{'='*70}")
print(f"{'Algorithm':<20} {'gamma':>6} {'lr':>8} {'Reward':>10} {'Std':>8} {'Length':>8} {'Time':>8}")
print(f"{'-'*70}")
for name, data in final_results.items():
    print(f"{name:<20} {data['gamma']:>6} {data['lr']:>8.0e} {data['reward']:>10.2f} "
          f"{data['std']:>8.2f} {data['length']:>8.0f} {data['time']:>6.0f}s")
print(f"{'='*70}")

# Save everything
all_results["final"] = final_results
with open("models/hyperparam_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nAll results saved to models/hyperparam_results.json")
print("\nSaved models:")
for p in sorted(Path("models").glob("*.zip")) + sorted(Path("models").glob("*.pt")):
    print(f"  {p} ({p.stat().st_size / 1024:.0f} KB)")
