# SKILL : Projet RL Highway-env

## Identite du projet

- **Projet** : Projet final du cours de Reinforcement Learning
- **Ecole** : Efrei Paris
- **Formation** : BDML 2 (Big Data & Machine Learning)
- **Professeur** : Victor Morand
- **Equipe** :
  - Amine M'ZALI
  - Mehdi SAADI
  - Samy Bouaissa
- **Deadline** : 25 mars 2026

---

## Contexte et objectif

L'objectif est d'entrainer un agent de Reinforcement Learning a conduire sur une autoroute simulee dans l'environnement `highway-v0` de la librairie [highway-env](https://github.com/Farama-Foundation/HighwayEnv). L'agent doit maximiser sa vitesse tout en evitant les collisions et en privilegiant la voie de droite.

Le groupe etant compose de 3 personnes (et non 2), **3 algorithmes de RL** doivent etre implementes et compares, dont **au moins 1 implemente "a la main"** (sans SB3).

---

## Contraintes techniques strictes

### Notebook

- Le fichier principal est `Final_Project.ipynb`.
- Les sections marquees "Do not Modify" (Setup, Constants, Utilities, Evaluation) ne doivent **jamais** etre modifiees.
- Le notebook doit tourner **d'un seul trait** sans intervention manuelle.
- Les poids des modeles peuvent etre sauvegardes dans le repo et charges dans le notebook pour eviter de re-entrainer a chaque execution.

### Configuration d'evaluation

L'evaluation se fait avec cette configuration imposee (ne pas modifier) :

```python
ENV_ID = "highway-v0"
EVAL_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 40,
    "initial_spacing": 0.1,
    "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
    "duration": 40,
}
```

### Entrainement

- Utiliser `highway-fast-v0` pour l'entrainement (x15 plus rapide).
- Utiliser TensorBoard pour monitorer les entrainements (`tensorboard --logdir highway`).
- Sauvegarder les modeles entraines pour pouvoir les charger sans re-entrainer.

### Livrables

1. **Notebook** (`Final_Project.ipynb`) : code complet qui tourne d'une traite
2. **Rapport** (5-10 pages PDF) : algorithmes, choix, benchmarks, figures
3. **Poids des modeles** : `.zip` (SB3) et `.pt` (PyTorch)
4. **Repo Git** : tout le code versionne

---

## Les 3 algorithmes a implementer

### Algorithme 1 : DQN (Stable-Baselines3)

**Type** : Value-based (off-policy)

Utiliser l'implementation DQN de SB3. C'est l'approche la plus naturelle pour un espace d'actions discret.

**Configuration de reference** :

```python
from stable_baselines3 import DQN

env = make_vec_env("highway-fast-v0", n_envs=4, env_kwargs={"config": TRAIN_CONFIG})

model_dqn = DQN(
    "MlpPolicy",
    env,
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
    tensorboard_log="highway_dqn/",
)
model_dqn.learn(total_timesteps=int(2e4))
model_dqn.save("models/highway_dqn")
```

**Hyperparametres a explorer** :
- `learning_rate` : [1e-4, 5e-4, 1e-3]
- `gamma` : [0.7, 0.8, 0.9, 0.99]
- `buffer_size` : [10000, 15000, 50000]
- `net_arch` : [[256, 256], [128, 128], [256, 128]]
- `target_update_interval` : [25, 50, 100]

---

### Algorithme 2 : PPO (Stable-Baselines3)

**Type** : Policy gradient / Actor-Critic (on-policy)

Utiliser l'implementation PPO de SB3 pour comparer une approche policy gradient.

**Configuration de reference** :

```python
from stable_baselines3 import PPO

env = make_vec_env("highway-fast-v0", n_envs=4, env_kwargs={"config": TRAIN_CONFIG})

model_ppo = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.8,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="highway_ppo/",
)
model_ppo.learn(total_timesteps=int(2e4))
model_ppo.save("models/highway_ppo")
```

**Hyperparametres a explorer** :
- `learning_rate` : [1e-4, 3e-4, 5e-4]
- `gamma` : [0.8, 0.9, 0.99]
- `n_steps` : [128, 256, 512]
- `clip_range` : [0.1, 0.2, 0.3]
- `n_epochs` : [5, 10, 20]

---

### Algorithme 3 : DQN from scratch (PyTorch)

**Type** : Value-based (off-policy) - Implementation manuelle

Implementer un DQN complet avec PyTorch pour satisfaire l'exigence "au moins un algorithme a la main".

**Composants a coder** :

1. **Q-Network** : MLP avec couches lineaires + ReLU
2. **Replay Buffer** : structure circulaire de taille fixe stockant `(state, action, reward, next_state, done)`
3. **Target Network** : copie du Q-Network mise a jour tous les N steps
4. **Politique epsilon-greedy** : epsilon decroit lineairement de 1.0 a 0.05
5. **Boucle d'entrainement** : sampling du buffer, calcul de la TD loss, backpropagation

**Structure du code** :

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_size = obs_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h), nn.ReLU()])
            prev_size = h
        layers.append(nn.Linear(prev_size, n_actions))
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
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)
```

**Interface de compatibilite** : Le DQN manuel doit exposer une methode `predict(obs, deterministic=True)` compatible avec les fonctions `evaluate()` et `record_video()` du notebook fournies par le professeur. Wrapper recommande :

```python
class DQNAgent:
    def __init__(self, ...):
        ...

    def predict(self, obs, deterministic=False):
        """Compatible avec l'interface SB3 pour evaluation."""
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            obs = obs.reshape(1, -1)
        elif isinstance(obs, np.ndarray) and obs.ndim == 2:
            obs = obs.reshape(1, -1)
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(obs).to(self.device))
            action = q_values.argmax(dim=1).cpu().numpy()
        if action.shape[0] == 1:
            return action[0], None
        return action, None

    def save(self, path):
        torch.save(self.q_network.state_dict(), path + ".pt")

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path + ".pt"))
```

**Extensions possibles** :
- Double DQN : utiliser le Q-network pour selectionner l'action, le target network pour l'evaluer
- Dueling DQN : separer V(s) et A(s,a) dans l'architecture

---

## Observations et types d'inputs

L'observation par defaut est de type **Kinematics** : une matrice `(V, F)` ou `V=5` vehicules et `F=5` features (`presence`, `x`, `y`, `vx`, `vy`). Pour un MLP, cette matrice est aplatie en vecteur 1D de taille `V*F = 25`.

Experimentation possible avec d'autres types :
- **GrayscaleImage** : necessite un CNN (plus lent mais potentiellement plus riche)
- **OccupancyGrid** : representation spatiale discretisee
- Augmenter `vehicles_count` dans l'observation (ex: 10 ou 15 vehicules observes)

---

## Structure de fichiers attendue

```
ProjetRL/
├── Final_Project.ipynb    # Notebook principal (les sections vides sont a remplir)
├── ANALYSE.md             # Document d'analyse du projet
├── SKILL.md               # Ce fichier
├── README.md              # README du repo original
├── requirements.txt       # Dependances Python
├── pyproject.toml         # Config du projet
├── models/                # Poids des modeles sauvegardes
│   ├── highway_dqn.zip    # DQN SB3
│   ├── highway_ppo.zip    # PPO SB3
│   └── highway_dqn_manual.pt  # DQN from scratch
├── videos/                # Videos des agents
├── figures/               # Graphiques d'entrainement pour le rapport
└── report/                # Rapport PDF
```

---

## Workflow d'entrainement recommande

1. **Explorer l'environnement** : comprendre actions, observations, rewards
2. **Definir une config d'entrainement** separee de EVAL_CONFIG (potentiellement plus simple pour accelerer l'apprentissage initial)
3. **Entrainer sur `highway-fast-v0`** pour le speedup x15
4. **Monitorer avec TensorBoard** : verifier que le reward augmente
5. **Evaluer sur `highway-v0` avec `EVAL_CONFIG`** pour la performance finale
6. **Sauvegarder les poids** pour ne pas re-entrainer
7. **Comparer** : courbes d'entrainement, reward moyenne, duree des episodes
8. **Iterer** sur les hyperparametres pour ameliorer les performances

---

## Benchmark et comparaison

Pour chaque algorithme, mesurer et reporter :

| Metrique                    | Description                                              |
|----------------------------|----------------------------------------------------------|
| Mean Reward (eval)         | Reward moyenne sur 30 episodes avec EVAL_CONFIG          |
| Std Reward                 | Ecart-type de la reward                                  |
| Mean Episode Length        | Nombre moyen de steps par episode (plus long = mieux)    |
| Training Time              | Temps d'entrainement total                               |
| Training Curve             | Reward en fonction du nombre de steps                    |
| Collision Rate             | Pourcentage d'episodes terminant par collision           |

Presenter les resultats sous forme de :
- Tableau comparatif des metriques finales
- Courbes d'apprentissage superposees (reward vs steps pour les 3 algos)
- Graphiques d'exploration des hyperparametres (impact de gamma, learning_rate, etc.)

---

## Conventions de code

- Tout le code d'entrainement va dans les sections "YOUR CODE HERE" du notebook
- Utiliser des noms de variables clairs : `model_dqn`, `model_ppo`, `model_dqn_manual`
- Logger les entrainements dans des dossiers separes pour TensorBoard
- Le modele final a evaluer doit etre dans la variable `model_final`
- Les imports supplementaires se font au debut de chaque section de code

---

## Points d'attention

1. **Le notebook doit tourner d'une traite** : prevoir un mode "load weights" qui charge les modeles pre-entraines au lieu de re-entrainer
2. **Ne pas modifier les sections du prof** : Setup, Constants, Utilities, Evaluation
3. **L'observation Kinematics est une matrice** : il faut la flatten pour un MLP (`obs.reshape(1, -1)`)
4. **gamma = 0.8 est un bon point de depart** : l'horizon temporel en conduite est relativement court
5. **`highway-fast-v0` pour l'entrainement** : ne pas entrainer sur `highway-v0` (trop lent)
6. **Sauvegarder regulierement** : les entrainements peuvent etre longs
7. **Le DQN manuel doit etre compatible** avec `evaluate()` et `record_video()` via la methode `predict()`
