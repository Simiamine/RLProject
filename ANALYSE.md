# Analyse du Projet RL : Highway-env

**Equipe** : Amine M'ZALI, Mehdi SAADI, Samy Bouaissa
**Formation** : BDML 2 - Efrei Paris
**Cours** : Reinforcement Learning (Victor Morand)
**Deadline** : 25 mars 2026

---

## 1. Objectif du projet

Entrainer un agent de Reinforcement Learning a conduire sur une autoroute simulee (environnement `highway-v0` de la librairie [highway-env](https://github.com/Farama-Foundation/HighwayEnv)). L'agent doit apprendre a :

- Rouler le plus vite possible
- Eviter les collisions avec les autres vehicules
- Privilegier la voie de droite

Etant un groupe de 3 (au lieu de 2), nous devons implementer **3 algorithmes de RL** au lieu de 2, dont **au moins 1 implemente "a la main"** (sans librairie haut niveau type Stable-Baselines3).

---

## 2. L'environnement Highway-env

### 2.1 Description generale

L'environnement simule une autoroute a plusieurs voies avec du trafic. Le vehicule ego (controle par l'agent) doit naviguer parmi les autres vehicules qui suivent des modeles de comportement pre-programmes (IDM - Intelligent Driver Model).

### 2.2 Espace d'actions (DiscreteMetaAction)

L'espace d'actions par defaut est **discret** avec 5 meta-actions :

| Index | Action       | Description                           |
|-------|-------------|---------------------------------------|
| 0     | LANE_LEFT   | Changer de voie vers la gauche        |
| 1     | IDLE        | Maintenir la trajectoire actuelle     |
| 2     | LANE_RIGHT  | Changer de voie vers la droite        |
| 3     | FASTER      | Accelerer                             |
| 4     | SLOWER      | Ralentir                              |

Ces meta-actions sont gerees par des controleurs bas niveau (PID) pour le suivi de voie et la regulation de vitesse. Certaines actions peuvent etre indisponibles selon le contexte (ex: pas de LANE_LEFT si on est deja sur la voie la plus a gauche).

### 2.3 Espace d'observations (Kinematics)

L'observation par defaut est une matrice **V x F** (V vehicules proches, F features par vehicule) :

| Feature    | Description                                    |
|-----------|------------------------------------------------|
| presence  | 1 si vehicule reel, 0 si placeholder           |
| x         | Position longitudinale                          |
| y         | Position laterale                               |
| vx        | Vitesse longitudinale                           |
| vy        | Vitesse laterale                                |

Par defaut, `V = 5` vehicules observes et les coordonnees sont relatives a l'ego-vehicule (sauf pour l'ego lui-meme qui reste en coordonnees absolues). Les valeurs sont normalisees dans `[-1, 1]`.

Autres types d'observations possibles :
- **GrayscaleImage** : image en niveaux de gris avec empilement temporel (pour CNN)
- **OccupancyGrid** : grille d'occupation discretisee autour du vehicule
- **TimeToCollision** : temps avant collision predit
- **Lidar** : simulation de capteur laser

### 2.4 Fonction de recompense

La recompense est composee de :
- **Vitesse** : recompense lineaire dans `[0, HIGH_SPEED_REWARD]` mappee depuis `reward_speed_range = [20, 30]` m/s
- **Collision** : penalite de `-1` en cas de collision (fin d'episode)
- **Voie de droite** : petit bonus `right_lane_reward = 0.1`
- **Changement de voie** : `lane_change_reward = 0` (pas de penalite par defaut)

La recompense totale est normalisee dans `[0, 1]` avec `normalize_reward = True`.

### 2.5 Configuration d'evaluation imposee

Le notebook fourni definit une configuration d'evaluation plus difficile que celle par defaut :

```python
EVAL_CONFIG = {
    "lanes_count": 3,           # 3 voies (au lieu de 4)
    "vehicles_count": 40,       # 40 vehicules
    "initial_spacing": 0.1,     # espacement initial tres faible
    "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",  # conducteurs agressifs
    "duration": 40,             # 40 secondes
}
```

Les vehicules sont de type **AggressiveVehicle**, ce qui rend l'environnement significativement plus difficile.

### 2.6 Variante rapide pour l'entrainement

L'environnement `highway-fast-v0` offre un speedup x15 par rapport a `highway-v0`, ce qui est recommande pour l'entrainement. L'evaluation finale se fait sur `highway-v0` avec `EVAL_CONFIG`.

---

## 3. Structure du notebook fourni

Le notebook `Final_Project.ipynb` est pre-structure :

| Section                  | Contenu                                                        | Modifiable |
|--------------------------|----------------------------------------------------------------|------------|
| Setup / Install          | Detection Colab, installation des packages                     | Non        |
| Constants                | `ENV_ID`, `EVAL_CONFIG`                                        | Non        |
| Utilities                | `record_video()`, `show_videos()`, `evaluate()`, `evaluate_vectorized()` | Non |
| Exploration              | Chargement d'un agent aleatoire, visualisation                | Non        |
| Action Space             | Section vide a remplir (exploration de l'espace d'actions)    | Oui        |
| Observation Space        | Section vide a remplir (exploration des observations)         | Oui        |
| Training                 | Sections vides pour l'entrainement des agents                 | Oui        |
| Evaluation               | Evaluation du modele final avec `evaluate()` et `record_video()` | Non     |
| Bonus                    | Environnements supplementaires (racetrack, etc.)              | Optionnel  |

### Fonctions utilitaires fournies

- **`record_video(model, ...)`** : enregistre une video de l'agent en action sur `EVAL_CONFIG`
- **`show_videos(path, prefix)`** : affiche les videos dans le notebook
- **`evaluate(model, num_episodes=30)`** : evaluation sequentielle (mean reward, mean time)
- **`evaluate_vectorized(model, num_episodes=30, n_envs=4)`** : evaluation parallelisee

---

## 4. Les 3 algorithmes choisis

### 4.1 DQN via Stable-Baselines3 (Value-based)

**Deep Q-Network** est l'approche la plus naturelle pour cet environnement a actions discretes.

- Utilise un reseau de neurones pour approximer la Q-function
- Replay buffer pour decoreler les experiences
- Target network pour stabiliser l'entrainement
- Exploration epsilon-greedy

Hyperparametres cles a explorer :
- `learning_rate` (ex: 5e-4)
- `buffer_size` (ex: 15000)
- `batch_size` (ex: 32)
- `gamma` (ex: 0.8 - horizon court pour la conduite)
- `target_update_interval` (ex: 50)
- `net_arch` (ex: [256, 256])

### 4.2 PPO via Stable-Baselines3 (Policy gradient)

**Proximal Policy Optimization** est un algorithme de type actor-critic qui offre un bon equilibre stabilite/performance.

- Optimise directement la politique (policy gradient)
- Clipping du ratio de probabilite pour limiter les mises a jour trop agressives
- Avantage : pas besoin de replay buffer, fonctionne en on-policy

Hyperparametres cles a explorer :
- `learning_rate` (ex: 3e-4)
- `n_steps` (ex: 256)
- `batch_size` (ex: 64)
- `gamma` (ex: 0.8)
- `clip_range` (ex: 0.2)
- `n_epochs` (ex: 10)
- `net_arch` pour policy et value (ex: [256, 256])

### 4.3 DQN implemente "a la main" avec PyTorch

Implementation from scratch d'un DQN pour demontrer la comprehension de l'algorithme :

- **Q-Network** : MLP en PyTorch (couches lineaires + ReLU)
- **Replay Buffer** : structure de donnees circulaire pour stocker les transitions `(s, a, r, s', done)`
- **Target Network** : copie du Q-Network, mise a jour periodiquement
- **Politique epsilon-greedy** : decroissance lineaire de epsilon
- **Boucle d'entrainement** : sampling du buffer, calcul de la loss MSE sur les Q-values cibles

Extensions possibles pour de meilleures performances :
- Double DQN (utiliser le Q-network pour selectionner l'action, le target pour evaluer)
- Dueling DQN (separer l'estimation de V(s) et A(s,a))
- Prioritized Experience Replay

---

## 5. Plan de travail et repartition

### Repartition des taches

| Membre         | Algorithme principal       | Taches complementaires                         |
|----------------|---------------------------|-------------------------------------------------|
| Amine M'ZALI   | DQN (SB3)                | Setup initial, exploration de l'env             |
| Mehdi SAADI    | PPO (SB3)                | Benchmark et comparaison des hyperparametres    |
| Samy Bouaissa  | DQN from scratch (PyTorch)| Implementation manuelle, rapport technique      |

### Taches communes

- Comparaison des 3 algorithmes (courbes d'entrainement, reward moyenne, duree des episodes)
- Redaction du rapport (5-10 pages)
- Integration finale dans le notebook unique
- Tests et verification que le notebook tourne d'un seul trait

### Livrables

1. **Notebook** (`Final_Project.ipynb`) : code complet, entrainement, evaluation, videos
2. **Rapport** (5-10 pages PDF) : choix, explications, benchmarks, figures
3. **Poids des modeles** : fichiers `.zip` pour les modeles SB3, fichier `.pt` pour le DQN manuel
4. **Repo Git** : tout le code, les poids, et les videos

---

## 6. Criteres d'evaluation

D'apres le professeur, la note est basee sur :

1. **Le rapport** : montrer que vous avez compris ce que vous avez fait
2. **Les performances** : qualite des agents entraines
3. **Le code** : quelque chose qui fonctionne, notebook qui tourne d'une traite

L'analyse attendue inclut :
- Exploration des hyperparametres
- Figures de l'entrainement (courbes de reward, loss)
- Explications de comment les algorithmes fonctionnent
- Benchmark et comparaison entre les differents algorithmes

---

## 7. Ressources techniques

- **Highway-env** : [Repo](https://github.com/Farama-Foundation/HighwayEnv) | [Documentation](http://highway-env.farama.org/quickstart/)
- **Stable-Baselines3** : [Repo](https://github.com/DLR-RM/stable-baselines3) | [Documentation](https://stable-baselines.readthedocs.io/en/master/)
- **Gymnasium** : [Documentation](https://gymnasium.farama.org/)
- **PyTorch** : [Documentation](https://pytorch.org/docs/stable/)
- **TensorBoard** : pour le monitoring des entrainements (`tensorboard --logdir highway`)
