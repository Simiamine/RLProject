# Rapport : Reinforcement Learning sur Highway-Env

**Equipe** : Amine M'ZALI, Mehdi SAADI, Samy Bouaissa
**Formation** : BDML 2 -- Efrei Paris
**Cours** : Reinforcement Learning (Victor Morand)
**Date** : 6 mars 2026

---

## 1. Introduction

L'objectif de ce projet est d'entrainer un agent de Reinforcement Learning a conduire sur une autoroute simulee via l'environnement `highway-v0` de la librairie highway-env (Leurent, 2018). L'agent doit maximiser sa vitesse tout en evitant les collisions avec 40 vehicules agressifs (`AggressiveVehicle`) sur 3 voies, pendant 40 secondes.

L'espace d'actions est discret (5 meta-actions : changer de voie a gauche/droite, accelerer, ralentir, maintenir). L'observation est une matrice 5x5 (cinematique de 5 vehicules proches : presence, position x/y, vitesse vx/vy). La recompense est normalisee dans [0,1] : bonus de vitesse, penalite de -1 pour collision.

Nous avons implemente **3 algorithmes** : DQN et PPO via Stable-Baselines3, et un DQN implemente entierement a la main avec PyTorch. L'entrainement utilise la variante rapide `highway-fast-v0` (x15), l'evaluation se fait sur `highway-v0` avec la configuration imposee (EVAL_CONFIG).

---

## 2. Algorithmes utilises

### 2.1 DQN via Stable-Baselines3

**Deep Q-Network** (Mnih et al., 2015) approxime la Q-function optimale par un reseau de neurones (MLP 256x256). Deux mecanismes stabilisent l'entrainement :

- **Replay Buffer** (15 000 transitions) : decorele les echantillons en tirant aleatoirement depuis un buffer circulaire.
- **Target Network** (mise a jour toutes les 100 steps) : fournit des cibles stables $y = r + \gamma \max_{a'} Q_{target}(s', a')$.

La perte est l'erreur TD quadratique : $\mathcal{L} = \mathbb{E}[(Q(s,a) - y)^2]$.

**Hyperparametres finaux** : gamma=0.9, lr schedule 1e-3 -> 1e-4 (lineaire), target_update_interval=100, 100k timesteps, 7 environnements paralleles.

### 2.2 PPO via Stable-Baselines3

**Proximal Policy Optimization** (Schulman et al., 2017) est une methode actor-critic on-policy. L'acteur optimise directement la politique $\pi_\theta(a|s)$, le critique estime $V(s)$ pour calculer l'avantage $A_t = R_t - V(s_t)$.

Le clipping du ratio d'importance empeche les mises a jour destructrices :
$$\mathcal{L}^{CLIP} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1\pm\varepsilon) A_t)]$$

**Hyperparametres finaux** : gamma=0.9, lr schedule 3e-4 -> 3e-5, n_steps=512, clip_range=0.2, ent_coef=0.01, 80k timesteps.

### 2.3 DQN from scratch (PyTorch)

Pour demontrer notre comprehension de l'algorithme, nous avons implemente un DQN complet sans librairie :

- **QNetwork** : MLP 25 -> 256 -> 256 -> 5 (ReLU)
- **ReplayBuffer** : deque de capacite 15 000
- **Target Network** : copie synchronisee toutes les 50 gradient steps
- **Epsilon-greedy** : decay lineaire de 1.0 a 0.05 sur 5 000 gradient steps
- **Gradient clipping** : `clip_grad_norm_` avec max_norm=10

**Difference avec SB3 DQN** : pas de vectorized envs (1 seul env), pas de callbacks automatiques, ~20k gradient steps vs 100k pour SB3. Malgre cela, les performances sont comparables une fois correctement configure (voir section 4).

**Hyperparametres finaux** : gamma=0.8, lr=5e-4, 1000 episodes, eval deterministe periodique (15 eps / 50 eps).

---

## 3. Exploration des hyperparametres

### Grid search (Phase 1)

Nous avons teste 15 configurations (5 par algo) en faisant varier gamma (0.8, 0.9, 0.99) et le learning rate. Chaque configuration a ete entrainee a budget reduit (30k steps SB3 / 200 episodes DQN manual) puis evaluee sur EVAL_CONFIG (15 episodes).

**Impact de gamma** (meilleur lr par algo) :

| gamma | DQN SB3 | PPO SB3 | DQN Manual |
|-------|--------:|--------:|-----------:|
| 0.8   | 13.79   | 23.24   | **28.46**  |
| 0.9   | **28.81** | **26.08** | 26.95  |
| 0.99  | 26.13   | 21.19   | 21.14      |

gamma=0.9 est optimal pour SB3 (horizon effectif ~10 steps). Le DQN manual prefere gamma=0.8 (horizon ~5 steps) car l'entrainement sur un seul environnement genere des transitions correlees temporellement, et un horizon plus court stabilise les cibles Q.

**Impact du learning rate** : DQN SB3 necessite un lr eleve (1e-3) pour converger en 30k steps. PPO fonctionne mieux avec un lr conservatif (3e-4) grace au clipping qui protege deja contre les updates trop agressifs. Le DQN manual est optimal avec lr=5e-4.

---

### 2.4 Justification des choix

Nous avons retenu le **DQN via Stable-Baselines3** comme premier algorithme parce que l'environnement HighwayEnv expose un espace d'actions discret et de petite taille (cinq meta-actions), ce pour quoi DQN est naturellement concu. En apprenant une fonction de valeur $Q(s,a)$ sur un ensemble d'actions fini, DQN constitue un algorithme de reference bien compris et facile a comparer aux autres methodes. L'implementation fournie par Stable-Baselines3 (replay buffer, target network, normalisation) nous permet de nous concentrer sur le probleme de conduite et le réglage des hyperparametres plutot que sur les details bas niveau, tout en obtenant deja un bon compromis entre performance et simplicite ; il sert ainsi de baseline solide pour evaluer les gains eventuels des autres algorithmes.

Le **PPO** a ete choisi en complement du DQN pour representer la famille des methodes de policy gradient. Il optimise directement la politique $\pi_\theta(a|s)$ en limitant les mises a jour grace a un objectif tronque (clipped), ce qui donne un entrainement generalement plus stable que les policy gradient classiques. PPO est repute robuste aux choix d'hyperparametres, ce qui est appreciable sur un environnement complexe et bruite comme HighwayEnv (trafic dense, vehicules agressifs). Sa nature on-policy et l'utilisation d'une estimation d'avantage (GAE) permettent de bien gerer le compromis exploration/exploitation, ce qui favorise des comportements de conduite plus fluides et plus surs. Enfin, comparer PPO a DQN sur le meme environnement permet d'evaluer si une methode policy-based surpasse une methode value-based dans ce contexte et d'observer les differences de stabilite d'entrainement et de qualite finale.

L'implementation d'un **DQN from scratch en PyTorch** repond a l'objectif pedagogique du projet : demontrer une comprehension fine de l'algorithme plutot que d'utiliser une boite noire. En ecrivant nous-memes le reseau Q, le replay buffer, l'epsilon-greedy, la target network et la boucle d'entrainement, nous controlons chaque detail (taille du reseau, strategie d'exploration, gestion du buffer) et pouvons tester des variantes et en mesurer l'impact sur les courbes d'apprentissage. Cela permet aussi une comparaison equitable avec DQN (SB3) et PPO en termes de performances, stabilite et sensibilite aux hyperparametres ; des resultats proches valident notre implementation, des ecarts mettent en evidence l'importance de certains choix. Enfin, presenter une implementation complete d'un algorithme de Deep RL renforce la valeur du rapport et de la soutenance en montrant la maitrise des fondements mathematiques et algorithmiques derriere les resultats obtenus sur HighwayEnv.

---

## 4. Resultats et analyse iterative

Notre approche experimentale a suivi 4 phases iteratives de type **diagnostic -> correction -> validation**.

### Phase 2 : Problemes identifies

L'entrainement a pleine puissance (100k steps / 600 episodes) a revele 3 problemes majeurs :

**DQN SB3 -- Q-value overestimation** : avec lr=1e-3 fixe, la reward monte a 23.98 (70k steps) puis s'effondre a 12.92 (100k), soit -46%. Le mecanisme $\max_{a'} Q_{target}(s',a')$ amplifie les surestimations a chaque update (Van Hasselt et al., 2016). Avec un lr eleve, ces erreurs se propagent rapidement.

**PPO -- Overtraining on-policy** : la reward chute de 28.63 (35k) a 19.51 (70k) avant de remonter partiellement a 23.53 (100k). Quand la politique s'ameliore, la distribution des etats rencontres change brutalement (distribution shift), forcant une readaptation couteuse.

**DQN Manual -- Epsilon decay trop lent** : configure en gradient steps (15 000), l'agent avait encore 62% d'actions aleatoires a l'episode 400. L'early stopping coupait avant que l'agent ne devienne exploitant, masquant la vraie qualite de la politique apprise.

### Phase 3 : Corrections et impact

| Correction | Algo | Avant | Apres | Impact |
|-----------|------|------:|------:|--------|
| Lr schedule 1e-3 -> 1e-4 | DQN SB3 | 12.92 | **27.14** | **+110%** |
| Lr schedule 3e-4 -> 3e-5 + n_steps=512 + ent_coef=0.01 | PPO | 23.53 | 21.92 (final) / 28.74 (best) | Best ckpt ameliore |
| epsilon_decay 15000 -> 5000 | DQN Manual | 19.73 | 8.35 | Training ameliore, eval decevante |

Le **lr scheduler** a ete la correction la plus impactante : en decroissant le learning rate lineairement, les updates deviennent plus prudents en fin d'entrainement, ce qui stabilise les Q-values et empeche l'effondrement post-pic.

Le DQN Manual montrait une bonne progression en training (reward 24.12 a ep 275, vs 14.26 en Phase 2) mais l'eval finale restait basse (8.35) a cause de l'early stop trop agressif et de la selection du best model basee sur la reward d'entrainement (bruitee par epsilon).

### Phase 4 : Corrections finales

| Correction | Algo | Phase 3 | **Phase 4** |
|-----------|------|--------:|------------:|
| 80k steps + copie best checkpoint | PPO | 21.92 | **27.88** |
| 1000 eps, eval det., grad clip, pas d'early stop | DQN Manual | 8.35 | **27.54** |
| (inchange) | DQN SB3 | 27.14 | **27.14** |

Pour le PPO, reduire a 80k steps et copier automatiquement le best checkpoint evite l'overtraining post-pic. Pour le DQN Manual, les 5 corrections cumulees (1000 episodes, eval deterministe periodique, gradient clipping, best model sur eval, pas d'early stop) ont transforme les resultats : le best_eval pendant l'entrainement atteint **29.35**, le meilleur score absolu du projet.

### Tableau comparatif toutes phases

| Algo        | Phase 1 | Phase 2 | Phase 3 | **Phase 4** |
|-------------|--------:|--------:|--------:|------------:|
| DQN SB3     | 28.81   | 12.92   | 27.14   | **27.14**   |
| PPO SB3     | 26.08   | 23.53   | 21.92   | **27.88**   |
| DQN Manual  | 28.46   | 19.73   | 8.35    | **27.54**   |

Les 3 algorithmes convergent vers **~27-28 de reward** en Phase 4, un resultat remarquable montrant que des architectures fondamentalement differentes (off-policy value-based, on-policy actor-critic, implementation manuelle) atteignent des performances similaires une fois correctement configurees.

---

## 5. Conclusion

### Verdict

**PPO SB3** est le meilleur modele global : reward=27.88, std=4.80 (meilleure consistance), survie 38.2/40 steps. Sa stabilite inherente (clipping) le rend plus fiable que le DQN malgre des performances brutes similaires.

**DQN Manual** est la plus grande reussite technique : passer de 8.35 a 27.54 (+230%) en identifiant et corrigeant 5 problemes distincts demontre une comprehension profonde de l'algorithme. Le best_eval de 29.35 montre que notre implementation from scratch peut rivaliser avec les librairies optimisees.

**DQN SB3** illustre l'importance du lr scheduling : une seule correction (lr decay) a suffi pour passer de 12.92 a 27.14 (+110%).

### Limites

- **Un seul run par configuration** : pas de moyenne multi-seeds (3-5 seeds seraient necessaires pour des resultats statistiquement robustes).
- **Pas de Double DQN** : reduirait l'overestimation residuelle.
- **DQN Manual sur 1 seul env** : les transitions correlees temporellement augmentent la variance par rapport aux 7 envs paralleles de SB3.
- **Evaluation sur 30 episodes** : echantillon relativement petit pour un environnement stochastique.

### Pistes d'amelioration

- Double DQN pour limiter le maximization bias
- Prioritized Experience Replay pour un apprentissage plus efficace
- Vectorized envs pour le DQN manual (decorrelation des transitions)
- Curriculum learning : entrainer progressivement sur des configs de plus en plus difficiles
- Multi-seed averaging pour des resultats plus robustes

---

## 6. Bonus : Racetrack

En complement du projet principal, nous avons aborde l'environnement `racetrack-v0` de highway-env, un probleme fondamentalement different : l'agent doit **suivre un circuit courbe** avec des **actions continues** (angle de braquage dans [-1, 1]) sur des episodes de 300 steps.

Nous avons teste 3 approches :

| Algorithme | Type | Reward | Ep. Length |
|-----------|------|-------:|----------:|
| **SAC (SB3)** | Off-policy, continu natif | **1117** | **1216** |
| PPO (SB3) | On-policy, continu natif | 637 | 760 |
| DQN manual (discretise) | Off-policy, 7 bins | 39 | 58 |

**SAC (Soft Actor-Critic)** (Haarnoja et al., 2018) domine largement. Cet algorithme combine replay buffer (off-policy, sample-efficient) et actions continues natives (politique gaussienne), avec un bonus d'entropie qui encourage l'exploration. Son meilleur eval pendant le training atteignait 1382 avec une survie parfaite (1501/1501 steps, std=8).

**PPO** obtient des resultats intermediaires. Etant on-policy, il est moins sample-efficient que SAC -- il jette les donnees apres chaque update, ce qui est couteux pour des episodes de 300 steps.

**Le DQN discretise** (notre implementation manuelle avec 7 bins de braquage) montre les limites de la discretisation pour le controle continu : les transitions brusques entre bins (ex: 0.0 -> 0.33) ne permettent pas de negocier finement les virages.

**Enseignement principal** : pour le controle continu de vehicules, SAC est l'algorithme de reference. La combinaison off-policy + actions continues + entropie automatique le rend nettement superieur aux alternatives. Tout le code et les modeles sont dans le dossier `bonus/`.

---

**References** :
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- Van Hasselt, H. et al. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Haarnoja, T. et al. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning. *ICML*.
- Leurent, E. (2018). An environment for autonomous driving decision-making. *GitHub: highway-env*.
