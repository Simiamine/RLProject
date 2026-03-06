"""
Bonus v2: Racetrack training — SAC only (PPO + DQN already trained in v1).
Run: uv run python bonus/train_racetrack_v2.py

SAC (Soft Actor-Critic) is the state-of-the-art off-policy algorithm for
continuous control. It maximizes both expected return and entropy, leading
to more robust exploration and better sample efficiency than PPO.
"""
import os
import time
import json
import shutil
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401

from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

ENV_ID = "racetrack-v0"

RACETRACK_CONFIG_SAC = {
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

N_ENVS = min(8, max(4, os.cpu_count() // 2))

Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

print(f"{'='*60}")
print(f"BONUS v2: SAC RACETRACK TRAINING")
print(f"{'='*60}")
print(f"  Parallel envs: {N_ENVS}")
print(f"{'='*60}\n")


def evaluate_model(model, config, num_episodes=20):
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
# SAC (SB3) — Off-policy, continuous, entropy-regularized
# ═══════════════════════════════════════════════════════════════════════════
print("Training SAC (Soft Actor-Critic)...")
print("  Off-policy + continuous + entropy bonus = ideal for racetrack\n")

SAC_TIMESTEPS = 300_000


def sac_lr_schedule(progress_remaining):
    return 3e-4 * max(0.1, progress_remaining)


try:
    train_env = make_vec_env(ENV_ID, n_envs=1, env_kwargs={"config": RACETRACK_CONFIG_SAC})
    eval_env = make_vec_env(ENV_ID, n_envs=2, env_kwargs={"config": RACETRACK_CONFIG_SAC})

    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=8, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/sac_best/",
        log_path="logs/sac_eval/",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    model_sac = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=sac_lr_schedule,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        tensorboard_log="logs/racetrack_sac/",
    )

    t0 = time.time()
    model_sac.learn(total_timesteps=SAC_TIMESTEPS, callback=eval_cb, progress_bar=True)
    sac_time = time.time() - t0
    model_sac.save("models/racetrack_sac")
    train_env.close()
    eval_env.close()

    best_path = Path("models/sac_best/best_model.zip")
    final_path = Path("models/racetrack_sac.zip")
    if best_path.exists():
        shutil.copy2(best_path, final_path)
        print(f"  Copied best checkpoint -> {final_path}")

    model_sac_best = SAC.load(str(final_path))
    mean_r, std_r, mean_l = evaluate_model(model_sac_best, RACETRACK_CONFIG_SAC)

    results = {
        "SAC (continuous)": {
            "reward": float(mean_r), "std": float(std_r),
            "length": float(mean_l), "train_time": sac_time,
        }
    }
    print(f"\n  SAC DONE in {sac_time:.0f}s | Eval: {mean_r:.2f} +/- {std_r:.2f} | Length: {mean_l:.0f}")

except Exception as e:
    print(f"\n  SAC FAILED: {e}")
    import traceback; traceback.print_exc()
    results = {"SAC (continuous)": {"error": str(e)}}

# Merge with existing results
existing_path = Path("models/racetrack_results.json")
if existing_path.exists():
    with open(existing_path) as f:
        existing = json.load(f)
    existing.update(results)
    results = existing

print(f"\n{'='*60}")
print("ALL RACETRACK RESULTS")
print(f"{'='*60}")
print(f"{'Algorithm':<25} {'Reward':>10} {'Std':>8} {'Length':>8} {'Time':>10}")
print(f"{'-'*65}")
for name, data in results.items():
    if "error" in data:
        print(f"{name:<25} {'FAILED':>10}")
    else:
        t = f"{data['train_time']:.0f}s" if data.get("train_time", 0) > 0 else "N/A"
        print(f"{name:<25} {data['reward']:>10.2f} {data['std']:>8.2f} "
              f"{data['length']:>8.0f} {t:>10}")
print(f"{'='*60}")

with open("models/racetrack_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to models/racetrack_results.json")
