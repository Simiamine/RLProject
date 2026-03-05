"""
Automated tests for the RL Highway-env project.
Run: uv run python test_project.py
"""
import sys
import os
import tempfile
import shutil
import traceback
import time
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

PASS = 0
FAIL = 0

def test(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

def ok(msg=""):
    global PASS
    PASS += 1
    print(f"  [PASS] {msg}")

def fail(msg=""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


# =========================================================================
# TEST 1: Dependencies
# =========================================================================
test("1. Dependencies import")
deps = [
    "gymnasium", "highway_env", "stable_baselines3", "torch",
    "numpy", "matplotlib", "moviepy", "tensorboard", "tqdm",
    "pandas", "scipy", "seaborn",
]
for d in deps:
    try:
        __import__(d)
        ok(d)
    except ImportError:
        fail(f"{d} -- not installed")


# =========================================================================
# TEST 2: Environment creation and shapes
# =========================================================================
test("2. Environment creation and shapes")

import gymnasium as gym
import highway_env  # noqa: F401

EVAL_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 40,
    "initial_spacing": .1,
    "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
    "duration": 40,
}
TRAIN_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 30,
    "duration": 40,
    "policy_frequency": 1,
}

try:
    env = gym.make("highway-v0", config=EVAL_CONFIG)
    obs, info = env.reset()
    assert obs.shape == (5, 5), f"Expected (5,5), got {obs.shape}"
    assert env.action_space.n == 5, f"Expected 5 actions, got {env.action_space.n}"
    ok(f"highway-v0 EVAL_CONFIG: obs={obs.shape}, actions={env.action_space.n}")
    
    obs2, rew, term, trunc, info = env.step(1)  # IDLE
    assert obs2.shape == (5, 5)
    assert isinstance(rew, float)
    ok(f"step() returns correct types: obs={obs2.shape}, reward={rew:.3f}")
    env.close()
except Exception as e:
    fail(f"highway-v0 EVAL_CONFIG: {e}")

try:
    env_fast = gym.make("highway-fast-v0", config=TRAIN_CONFIG)
    obs, _ = env_fast.reset()
    assert obs.shape == (5, 5)
    ok(f"highway-fast-v0 TRAIN_CONFIG: obs={obs.shape}")
    env_fast.close()
except Exception as e:
    fail(f"highway-fast-v0 TRAIN_CONFIG: {e}")


# =========================================================================
# TEST 3: DQN from scratch components
# =========================================================================
test("3. DQN from scratch components")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

exec_globals = {}
exec("""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, obs_size, n_actions, hidden_sizes=(256, 256),
                 lr=5e-4, gamma=0.8, buffer_size=15000, batch_size=32,
                 target_update_freq=50, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=10000):
        self.device = torch.device("cpu")
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
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.step_count = checkpoint.get("step_count", 0)
        self.training_rewards = checkpoint.get("training_rewards", [])
""", exec_globals)

QNetwork = exec_globals["QNetwork"]
ReplayBuffer = exec_globals["ReplayBuffer"]
DQNAgent = exec_globals["DQNAgent"]

# 3a. QNetwork
try:
    qn = QNetwork(25, 5)
    x = torch.randn(4, 25)
    out = qn(x)
    assert out.shape == (4, 5), f"Expected (4,5), got {out.shape}"
    ok(f"QNetwork output shape: {out.shape}")
except Exception as e:
    fail(f"QNetwork: {e}")

# 3b. ReplayBuffer
try:
    buf = ReplayBuffer(100)
    for i in range(50):
        buf.push(np.random.randn(25), i % 5, float(i), np.random.randn(25), False)
    assert len(buf) == 50
    s, a, r, ns, d = buf.sample(16)
    assert s.shape == (16, 25), f"States shape {s.shape}"
    assert a.shape == (16,)
    ok(f"ReplayBuffer: len={len(buf)}, sample shapes OK")
except Exception as e:
    fail(f"ReplayBuffer: {e}")

# 3c. DQNAgent.predict() shapes
try:
    agent = DQNAgent(25, 5)
    agent.epsilon = 0.0  # deterministic

    # record_video sends obs shape (5, 5)
    obs_2d = np.random.randn(5, 5).astype(np.float32)
    act, none_val = agent.predict(obs_2d, deterministic=True)
    assert none_val is None
    assert isinstance(act, (int, np.integer)), f"Expected int, got {type(act)}"
    assert 0 <= act < 5
    ok(f"predict((5,5)) -> action={act}, type={type(act).__name__}")

    # evaluate_vectorized sends obs shape (1, 5, 5)
    obs_3d = np.random.randn(1, 5, 5).astype(np.float32)
    act2, _ = agent.predict(obs_3d, deterministic=True)
    assert isinstance(act2, np.ndarray), f"Expected ndarray, got {type(act2)}"
    ok(f"predict((1,5,5)) -> action={act2}, type={type(act2).__name__}")

    # evaluate_vectorized with 4 envs sends (4, 5, 5)
    obs_batch = np.random.randn(4, 5, 5).astype(np.float32)
    act3, _ = agent.predict(obs_batch, deterministic=True)
    assert isinstance(act3, np.ndarray)
    assert act3.shape == (4,), f"Expected (4,), got {act3.shape}"
    ok(f"predict((4,5,5)) -> actions={act3}, shape={act3.shape}")

except Exception as e:
    fail(f"DQNAgent.predict: {e}")
    traceback.print_exc()


# =========================================================================
# TEST 4: Mini training of all 3 algorithms
# =========================================================================
test("4. Mini training (500 steps SB3, 5 episodes manual)")

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

# 4a. DQN SB3
try:
    env_dqn = make_vec_env("highway-fast-v0", n_envs=2, env_kwargs={"config": TRAIN_CONFIG})
    m_dqn = DQN("MlpPolicy", env_dqn, learning_rate=5e-4, buffer_size=1000,
                learning_starts=50, batch_size=32, gamma=0.8, verbose=0)
    m_dqn.learn(total_timesteps=500)
    env_dqn.close()
    ok("DQN SB3 trained 500 steps")
except Exception as e:
    fail(f"DQN SB3 training: {e}")
    traceback.print_exc()

# 4b. PPO SB3
try:
    env_ppo = make_vec_env("highway-fast-v0", n_envs=2, env_kwargs={"config": TRAIN_CONFIG})
    m_ppo = PPO("MlpPolicy", env_ppo, n_steps=64, batch_size=32, gamma=0.8, verbose=0)
    m_ppo.learn(total_timesteps=500)
    env_ppo.close()
    ok("PPO SB3 trained 500 steps")
except Exception as e:
    fail(f"PPO SB3 training: {e}")
    traceback.print_exc()

# 4c. DQN manual
try:
    env_m = gym.make("highway-fast-v0", config=TRAIN_CONFIG)
    ag = DQNAgent(25, 5, hidden_sizes=(64, 64), buffer_size=500, batch_size=16,
                  epsilon_decay_steps=200)
    for ep in range(5):
        obs, _ = env_m.reset()
        state = obs.flatten()
        done = False
        while not done:
            action = ag.select_action(state)
            nobs, rew, term, trunc, info = env_m.step(action)
            done = term or trunc
            nstate = nobs.flatten()
            ag.buffer.push(state, action, rew, nstate, float(done))
            ag.update()
            state = nstate
    env_m.close()
    ok(f"Manual DQN trained 5 episodes, buffer={len(ag.buffer)}, epsilon={ag.epsilon:.3f}")
except Exception as e:
    fail(f"Manual DQN training: {e}")
    traceback.print_exc()


# =========================================================================
# TEST 5: Save / Load
# =========================================================================
test("5. Save and Load")

tmpdir = tempfile.mkdtemp()

# 5a. DQN SB3
try:
    path_dqn = os.path.join(tmpdir, "test_dqn")
    m_dqn.save(path_dqn)
    m_dqn_loaded = DQN.load(path_dqn)
    test_obs = np.random.randn(1, 5, 5).astype(np.float32)
    a1, _ = m_dqn.predict(test_obs, deterministic=True)
    a2, _ = m_dqn_loaded.predict(test_obs, deterministic=True)
    assert np.array_equal(a1, a2), f"Predictions differ: {a1} vs {a2}"
    ok("DQN SB3 save/load: predictions match")
except Exception as e:
    fail(f"DQN SB3 save/load: {e}")

# 5b. PPO SB3
try:
    path_ppo = os.path.join(tmpdir, "test_ppo")
    m_ppo.save(path_ppo)
    m_ppo_loaded = PPO.load(path_ppo)
    a1, _ = m_ppo.predict(test_obs, deterministic=True)
    a2, _ = m_ppo_loaded.predict(test_obs, deterministic=True)
    assert np.array_equal(a1, a2), f"Predictions differ: {a1} vs {a2}"
    ok("PPO SB3 save/load: predictions match")
except Exception as e:
    fail(f"PPO SB3 save/load: {e}")

# 5c. DQN manual
try:
    path_manual = os.path.join(tmpdir, "test_manual.pt")
    ag.save(path_manual)
    ag2 = DQNAgent(25, 5, hidden_sizes=(64, 64))
    ag2.load(path_manual)
    a1, _ = ag.predict(test_obs, deterministic=True)
    a2, _ = ag2.predict(test_obs, deterministic=True)
    assert np.array_equal(a1, a2), f"Predictions differ: {a1} vs {a2}"
    ok("Manual DQN save/load: predictions match")
except Exception as e:
    fail(f"Manual DQN save/load: {e}")

shutil.rmtree(tmpdir)


# =========================================================================
# TEST 6: Compatibility with professor's evaluate()
# =========================================================================
test("6. Professor's evaluate() compatibility")

ENV_ID = "highway-v0"

def evaluate_quick(model, num_episodes=3):
    """Stripped-down version of prof's evaluate() with fewer episodes for speed."""
    from stable_baselines3.common.env_util import make_vec_env
    from tqdm import tqdm
    env = make_vec_env(ENV_ID, env_kwargs={"config": EVAL_CONFIG})
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if steps > 200:
                break
        episode_rewards.append(total_reward)
    env.close()
    return np.mean(episode_rewards)

# 6a. DQN SB3
try:
    r = evaluate_quick(m_dqn, num_episodes=2)
    ok(f"evaluate() with DQN SB3: mean_reward={float(np.squeeze(r)):.3f}")
except Exception as e:
    fail(f"evaluate() with DQN SB3: {e}")
    traceback.print_exc()

# 6b. PPO SB3
try:
    r = evaluate_quick(m_ppo, num_episodes=2)
    ok(f"evaluate() with PPO SB3: mean_reward={float(np.squeeze(r)):.3f}")
except Exception as e:
    fail(f"evaluate() with PPO SB3: {e}")
    traceback.print_exc()

# 6c. Manual DQN
try:
    r = evaluate_quick(ag, num_episodes=2)
    ok(f"evaluate() with Manual DQN: mean_reward={float(np.squeeze(r)):.3f}")
except Exception as e:
    fail(f"evaluate() with Manual DQN: {e}")
    traceback.print_exc()


# =========================================================================
# SUMMARY
# =========================================================================
print(f"\n{'='*60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
