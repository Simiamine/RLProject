# Livrables du projet RL Highway-env

**Equipe** : Amine M'ZALI, Mehdi SAADI, Samy Bouaissa
**Formation** : BDML 2 -- Efrei Paris

---

## 1. Notebook principal (obligatoire)

| Fichier | Description |
|---------|-------------|
| `Final_Project.ipynb` | Notebook complet : 3 algos (DQN SB3, PPO SB3, DQN manual PyTorch), benchmark, hyperparameters, analyse critique, videos. Tourne d'un seul trait avec `RETRAIN=False`. |

## 2. Rapport (obligatoire)

| Fichier | Description |
|---------|-------------|
| `rapport.md` | Rapport ~5 pages : introduction, algorithmes, exploration hyperparametres, resultats iteratifs (4 phases), bonus racetrack, conclusion + references. |

## 3. Modeles pre-entraines (obligatoire)

| Fichier | Algo | Reward (eval) |
|---------|------|---------------|
| `models/highway_dqn.zip` | DQN SB3 (Phase 3) | 27.14 |
| `models/highway_ppo.zip` | PPO SB3 best checkpoint (Phase 4) | 27.88 |
| `models/highway_dqn_manual.pt` | DQN manual best eval (Phase 4) | 27.54 |
| `models/hyperparam_results.json` | Resultats du grid search (15 configs) | -- |
| `models/training_results.json` | Resultats finaux des 3 algos | -- |

## 4. Figures

| Fichier | Description |
|---------|-------------|
| `figures/benchmark_comparison.png` | Bar chart comparatif des 3 algos |
| `figures/manual_dqn_training.png` | Courbe d'entrainement DQN manual (1000 episodes) |
| `figures/hyperparam_exploration.png` | Impact de gamma et lr par algo |

## 5. Videos

| Fichier | Description |
|---------|-------------|
| `videos/random-agent_90_steps.mp4` | Agent aleatoire (baseline) |
| `videos/trained-agent_70_steps.mp4` | Meilleur agent entraine sur highway |

## 6. Scripts d'entrainement

| Fichier | Description |
|---------|-------------|
| `train.py` | Script Phase 2 (hyperparametres initiaux) |
| `trainv2.py` | Script Phase 3 (lr schedules, epsilon fix) |
| `trainv3.py` | Script Phase 4 (PPO 80k + best copy, DQN manual 1000 eps + eval deterministe + grad clip) |
| `hyperparam_search.py` | Grid search : 15 configs (5 par algo) |

## 7. Documentation

| Fichier | Description |
|---------|-------------|
| `RESULTATS.md` | Resultats experimentaux detailles (4 phases, tableaux, analyse) |
| `ANALYSE.md` | Analyse du projet (env, algos, plan, bilan) |
| `README.md` | Presentation generale du repo |

## 8. Bonus : Racetrack (dossier `bonus/`)

| Fichier | Description |
|---------|-------------|
| `bonus/racetrack_notebook.ipynb` | Notebook bonus : 3 algos sur racetrack-v0, tourne avec `RETRAIN=False` |
| `bonus/train_racetrack.py` | Script training PPO + DQN discretise |
| `bonus/train_racetrack_v2.py` | Script training SAC |
| `bonus/models/racetrack_ppo.zip` | PPO continu (reward=637) |
| `bonus/models/racetrack_sac.zip` | SAC continu (reward=1117) |
| `bonus/models/racetrack_dqn_manual.pt` | DQN discretise 7 bins (reward=39) |
| `bonus/models/racetrack_results.json` | Resultats des 3 algos racetrack |
| `bonus/figures/racetrack_benchmark.png` | Bar chart PPO vs SAC vs DQN |
| `bonus/figures/racetrack_dqn_training.png` | Courbe d'entrainement DQN racetrack |
| `bonus/videos/racetrack-random_300_steps.mp4` | Agent aleatoire sur racetrack |
| `bonus/videos/racetrack-ppo_500_steps.mp4` | PPO sur racetrack |
| `bonus/videos/racetrack-sac_500_steps.mp4` | SAC sur racetrack |
| `bonus/videos/racetrack-dqn_300_steps.mp4` | DQN discretise sur racetrack |
| `bonus/videos/racetrack-best_600_steps.mp4` | Meilleur agent (SAC) sur racetrack |

---

## Fichiers NON necessaires dans le livrable

Ces fichiers sont dans le repo mais ne sont pas essentiels :

- `.venv/` -- environnement virtuel (le prof installe ses propres deps)
- `logs/` -- logs TensorBoard (utiles pour le dev, pas pour le rendu)
- `models/checkpoints/` -- checkpoints intermediaires d'entrainement
- `models/*_best.pt`, `models/*_final.pt` -- modeles intermediaires
- `bonus/logs/` -- logs TensorBoard du bonus
- `bonus/models/*_best/`, `bonus/models/*_final.pt` -- checkpoints bonus
- `SKILL.md`, `WORKFLOW.md` -- fichiers internes de travail
- `test_project.py` -- tests automatises (utiles pour le dev)
