# Resultats experimentaux : Grid Search et Analyse

**Equipe** : Amine M'ZALI, Mehdi SAADI, Samy Bouaissa
**Date** : 5 mars 2026

Ce document rassemble tous les resultats chiffres de nos experiences, les
problemes observes, et leur analyse. Il servira de base pour le rapport et
les commentaires du notebook.

---

## 1. Grid Search - Phase 1 (exploration rapide)

15 configurations testees (5 par algo). Chaque config entraine a budget
reduit (30k steps SB3, 200 episodes DQN manual) puis evaluee sur EVAL_CONFIG
(3 voies, 40 vehicules agressifs, 40s). Evaluation sur 15 episodes.

### 1.1 DQN (Stable-Baselines3)

| gamma | lr   | Reward | Std  | Temps | Notes       |
|-------|------|--------|------|-------|-------------|
| 0.8   | 5e-4 | 13.79  | 8.1  | 443s  |             |
| 0.8   | 1e-3 | 13.01  | 8.2  | 441s  |             |
| 0.9   | 5e-4 | 10.27  | 3.9  | 433s  |             |
| **0.9**   | **1e-3** | **28.81**  | **0.8**  | 467s  | **BEST**        |
| 0.99  | 5e-4 | 26.13  | 5.7  | 465s  |             |

**Observations** :
- gamma=0.9 avec lr=1e-3 domine largement (+15 points vs gamma=0.8)
- gamma=0.99 reste competitif (26.13), horizon plus long aide aussi
- gamma=0.8 sous-performe systematiquement pour DQN SB3
- lr=1e-3 > lr=5e-4 pour gamma eleve (convergence plus rapide au budget 30k)

### 1.2 PPO (Stable-Baselines3)

| gamma | lr   | Reward | Std   | Temps | Notes       |
|-------|------|--------|-------|-------|-------------|
| 0.8   | 3e-4 | 23.24  | 10.5  | 466s  |             |
| 0.8   | 5e-4 | 20.61  | 12.1  | 461s  |             |
| **0.9**   | **3e-4** | **26.08**  | **6.9**   | 630s  | **BEST**        |
| 0.9   | 5e-4 | 23.61  | 9.9   | 700s  |             |
| 0.99  | 3e-4 | 21.19  | 10.2  | 694s  |             |

**Observations** :
- gamma=0.9 est aussi le meilleur pour PPO
- lr=3e-4 > lr=5e-4 systematiquement (lr plus petit est meilleur pour PPO)
- gamma=0.99 degrade (PPO on-policy supporte mal un horizon trop long)
- Temps d'entrainement augmente avec gamma eleve (episodes plus longues)
- Variance (std) plus faible avec gamma=0.9 qu'avec gamma=0.8

### 1.3 DQN from scratch (PyTorch)

| gamma | lr   | Reward | Std  | Temps | Notes       |
|-------|------|--------|------|-------|-------------|
| **0.8**   | **5e-4** | **28.46**  | **1.2**  | 144s  | **BEST**        |
| 0.8   | 1e-3 | 25.42  | 9.4  | 136s  |             |
| 0.9   | 5e-4 | 23.34  | 9.7  | 138s  |             |
| 0.9   | 1e-3 | 26.95  | 5.7  | 149s  |             |
| 0.99  | 5e-4 | 21.14  | 11.4 | 126s  |             |

**Observations** :
- Contrairement a SB3, gamma=0.8 est meilleur pour notre DQN manual
- lr=5e-4 donne la meilleure stabilite (std=1.2, remarquablement faible)
- gamma=0.99 est le pire choix (21.14, haute variance)
- L'entrainement est 3x plus rapide que SB3 (pas de vectorized envs overhead)

### 1.4 Analyse transversale - Impact de gamma

| gamma | DQN SB3 (best lr) | PPO SB3 (best lr) | DQN Manual (best lr) |
|-------|-------------------|--------------------|----------------------|
| 0.8   | 13.79             | 23.24              | **28.46**            |
| 0.9   | **28.81**         | **26.08**          | 26.95                |
| 0.99  | 26.13             | 21.19              | 21.14                |

**Tendance** : gamma=0.9 est le sweet spot pour les algos SB3. Le DQN manual
prefere gamma=0.8 -- probablement parce que notre implementation plus simple
(pas de vectorized envs, gradient steps par transition) beneficie d'un horizon
plus court qui stabilise les cibles Q.

### 1.5 Analyse transversale - Impact du learning rate

| lr    | DQN SB3 (gamma=0.9) | PPO SB3 (gamma=0.9) | DQN Manual (gamma=0.8) |
|-------|---------------------|---------------------|------------------------|
| 3e-4  | -                   | **26.08**           | -                      |
| 5e-4  | 10.27               | 23.61               | **28.46**              |
| 1e-3  | **28.81**           | -                   | 25.42                  |

**Tendance** : DQN SB3 a besoin d'un lr eleve pour converger vite a 30k steps.
PPO prefere un lr plus conservatif (le clipping protege contre les updates
trop agressifs, un lr eleve n'est pas necessaire). Le DQN manual fonctionne
mieux avec un lr modere (5e-4).

---

## 2. Phase 2 - Retrain des best configs a pleine puissance

Chaque best config de la Phase 1 est re-entrainee a 100k steps (SB3) ou
600 episodes (DQN manual), avec EvalCallback et early stopping.
Evaluation finale sur 30 episodes.

### 2.1 DQN SB3 (gamma=0.9, lr=1e-3, 100k steps)

| Timestep | Reward (eval) | Std   | Ep. Length | Interpretation                |
|----------|---------------|-------|------------|-------------------------------|
| 35,000   | 9.46          | 8.22  | 12.3       | Encore en exploration          |
| 70,000   | **23.98**     | 11.04 | 30.8       | **Pic de performance**         |
| 100,000 (final) | 12.92  | 9.47  | 16.0       | Effondrement post-pic          |

**Temps total** : 1975s (~33 min)

**Best model sauvegarde** : celui a 70k steps (reward=23.98) via EvalCallback.

**Probleme identifie** : instabilite du DQN avec lr=1e-3 sur le long terme
(voir section 3 pour l'analyse detaillee).

### 2.2 PPO SB3 (gamma=0.9, lr=3e-4, 100k steps)

| Timestep | Reward (eval) | Std   | Ep. Length | Interpretation                |
|----------|---------------|-------|------------|-------------------------------|
| 35,000   | **28.63**     | 3.02  | 38.8       | Pic de performance, tres stable |
| 70,000   | 19.51         | 11.89 | 26.2       | Baisse moderee                |
| 100,000 (final) | 23.53  | 10.33 | 31.0       | Rebond partiel                |

**Temps total** : 1461s (~24 min)

**Best model sauvegarde** : celui a 35k steps (reward=28.63) via EvalCallback.

**Observation** : PPO est plus stable que DQN mais montre aussi une legere
degradation post-pic. Le mecanisme de clipping limite la chute (de 28.63 a
19.51, soit -32%) par rapport au DQN (de 23.98 a 12.92, soit -46%). Le
rebond a 23.53 montre que PPO recupere partiellement, contrairement au DQN
qui continue de se degrader.

### 2.3 DQN Manual (gamma=0.8, lr=5e-4, 600 episodes)

| Episode | Avg Reward (50 eps) | Epsilon | Interpretation                |
|---------|---------------------|---------|-------------------------------|
| 50      | 8.24                | 0.967   | Debut, forte exploration       |
| 100     | 9.41                | 0.927   | Progression lente              |
| 150     | 9.48                | 0.886   | Plateau                        |
| 200     | 10.09               | 0.842   | Legere amelioration            |
| 250     | 11.84               | 0.791   | Apprentissage accelere         |
| 300     | **14.26**           | 0.731   | **Pic de performance**         |
| 350     | 12.33               | 0.678   | Debut de descente              |
| 400     | 13.91               | 0.619   | Early stop (pas d'amelioration) |

**Evaluation finale (30 episodes)** : reward=19.73 +/- 11.0, length=27

**Temps total** : 103s (~2 min)

**Early stop** a l'episode 400/600 (patience=100 episodes sans amelioration).

**Observation** : L'agent apprend progressivement mais plafonne vers 14.26 de
reward moyenne d'entrainement. L'evaluation finale de 19.73 est superieure
a la moyenne d'entrainement car l'evaluation est en mode deterministe
(pas d'exploration epsilon).

**Comparaison avec la Phase 1** : en Phase 1 (200 episodes), le meme algo avec
les memes hyperparametres donnait 28.46. La chute a 19.73 illustre la variance
inter-runs du RL (voir section 3.4).

---

## 3. Problemes identifies et analyse

### 3.1 Instabilite du DQN SB3 a long terme

**Symptome** : le DQN SB3 (gamma=0.9, lr=1e-3) montre une courbe en cloche :
- Monte jusqu'a 23.98 a 70k steps
- Retombe a 12.92 a 100k steps
- Perd 11 points de reward (-46%) dans les derniers 30k steps

**Explication : Q-value overestimation**

Avec un learning rate eleve (1e-3), les mises a jour des Q-values sont
agressives. Au debut, cela accelere l'apprentissage. Mais a mesure que
l'entrainement progresse, les Q-values surestiment la vraie valeur des
actions. L'agent commence a prendre des decisions basees sur des estimations
fausses, ce qui degrade la politique.

Le mecanisme en detail :
1. Le Q-network predit Q(s,a) trop haut pour certaines actions
2. Le target y = r + gamma * max Q_target(s',a') utilise le max, qui
   amplifie les surestimations (maximization bias)
3. Ces surestimations se propagent dans le replay buffer
4. La politique choisit des actions "surestimees" qui sont en realite
   sous-optimales

C'est le probleme historique du DQN, documente par Van Hasselt et al. (2016)
dans leur papier sur Double DQN.

### 3.2 Le PPO aussi decline, mais moins

Le PPO passe de 28.63 (35k) a 19.51 (70k) puis remonte a 23.53 (100k).
La baisse est moins severe car PPO n'estime pas de Q-values. Il optimise
directement la politique par policy gradient, avec un mecanisme de clipping
qui limite la taille des mises a jour. Meme avec un lr "trop grand", le
clipping empeche les catastrophes. La capacite de rebond (19.51 -> 23.53)
montre la resilience de PPO.

### 3.3 Le probleme general de l'overtraining en RL

Les 3 algos montrent des resultats **Phase 2 inferieurs a la Phase 1** :

| Algo        | Phase 1 (30k/200ep) | Phase 2 (100k/600ep) | Delta  |
|-------------|---------------------|----------------------|--------|
| DQN SB3     | 28.81               | 12.92 (final) / 23.98 (best) | -17% a -55% |
| PPO SB3     | 26.08               | 23.53 (final) / 28.63 (best) | -10% a +10% |
| DQN Manual  | 28.46               | 19.73                | -31%   |

Ce phenomene n'est pas un bug : c'est une propriete fondamentale du RL.
Entrainer plus longtemps ne garantit pas un meilleur resultat. Contrairement
au supervised learning ou plus de donnees = meilleur modele, en RL le modele
genere ses propres donnees via sa politique. Une politique qui se degrade
genere des donnees de mauvaise qualite, creant un cercle vicieux.

**Conclusion pratique** : l'EvalCallback qui sauvegarde le best model est
indispensable. Le modele "final" (dernier step) n'est souvent pas le meilleur.

### 3.4 La variance inter-runs du RL

La comparaison Phase 1 vs Phase 2 revele aussi une forte variance inter-runs :
le DQN SB3 passe de 28.81 (Phase 1) a 9.46 a 35k steps (Phase 2), alors que
c'est le meme algo avec les memes hyperparametres et le meme budget de steps.

Cela s'explique par 4 facteurs :

1. **Seeds aleatoires** : les poids initiaux du reseau, l'ordre des
   transitions dans le buffer, les episodes rencontrees sont tous
   differents d'un run a l'autre. En RL, le seed determine la trajectoire
   d'entrainement de maniere chaotique.

2. **Environnement stochastique** : les vehicules apparaissent a des
   positions aleatoires avec des comportements legerement differents.
   Un run peut tomber sur des scenarios "faciles" au debut (bonnes
   transitions dans le buffer qui accelerent l'apprentissage), un autre
   sur des scenarios difficiles (collisions repetees qui remplissent le
   buffer de transitions negatives).

3. **Le replay buffer est different** : a 30k steps, le buffer contient
   des transitions differentes entre deux runs. Le DQN apprend directement
   de ce buffer -- sa composition determine la qualite de l'apprentissage.
   Un buffer avec des transitions variees (differentes voies, vitesses,
   situations) produit un meilleur modele qu'un buffer domine par des
   collisions repetitives.

4. **Echantillonnage de l'evaluation** : 15 episodes (Phase 1) vs
   10 episodes (EvalCallback Phase 2) vs 30 episodes (eval finale) sont
   des petits echantillons, sujets a forte variance. Le resultat de 28.81
   en Phase 1 sur 15 episodes pourrait etre un echantillon chanceux, tandis
   que 12.92 sur 30 episodes est plus representatif.

C'est pour cette raison que les publications en RL reportent systematiquement
des resultats **moyennes sur 3 a 5 seeds**, avec ecart-type. Un seul run
n'est pas representatif. Dans le cadre de ce projet, le temps limite ne nous
a pas permis de faire des runs multi-seeds, mais nous reconnaissons cette
limite.

### 3.5 Diagnostic detaille par algorithme et solutions

#### A. DQN SB3 : Q-value overestimation + lr trop agressif

**Probleme principal** : lr=1e-3 permet une convergence rapide mais
destabilise les Q-values sur le long terme. Le mecanisme du max dans
la cible DQN (y = r + gamma * **max** Q_target) amplifie les surestimations
a chaque update. Avec un lr eleve, ces erreurs se propagent vite.

**Solutions** :

1. **lr scheduler** (priorite haute) : decroitre le lr lineairement de
   1e-3 a 1e-4 au cours de l'entrainement :

```python
def lr_schedule(progress_remaining):
    return 1e-3 * max(0.1, progress_remaining)
```

   Hypothese : la courbe ne s'effondrera plus apres 70k steps car les
   updates deviennent plus prudents en fin d'entrainement.

2. **Augmenter `target_update_interval`** de 50 a 100 : synchroniser le
   target network moins souvent ralentit la propagation des erreurs de
   surestimation.

3. **Gradient clipping** (`max_grad_norm=10`) : limiter la norme des
   gradients pour empecher les updates explosifs.

4. **Reduire a 70k steps** : puisque le best est a 70k, ne pas aller
   au-dela pourrait suffire (pragmatique mais pas elegant).

#### B. PPO SB3 : distribution shift on-policy

**Probleme principal** : PPO est on-policy -- la politique est mise a
jour puis les anciennes donnees sont jetees. Quand la politique s'ameliore
(l'agent va plus vite, atteint des positions nouvelles), la distribution
des etats rencontres change brutalement. La politique doit se readapter
a cette nouvelle distribution, ce qui cause le creux entre 35k et 70k.

Le clipping empeche les catastrophes (la baisse est de -32% vs -46% pour
DQN) et permet un rebond (19.51 -> 23.53), mais ne resout pas
completement le probleme.

**Solutions** :

1. **lr scheduler** (priorite haute) : decroitre lr de 3e-4 a 3e-5.
   Les updates deviennent plus conservatifs en fin de training, ce qui
   limite la derive de la politique.

2. **Augmenter `n_steps`** de 256 a 512 : collecter plus de donnees
   avant chaque update. Des batches plus grands = gradients plus stables
   = moins de variance dans les mises a jour de la politique.

3. **Ajouter un entropy bonus** (`ent_coef=0.01`) : par defaut SB3 met
   ent_coef=0.0 pour PPO. Un petit bonus d'entropie empeche la politique
   de se specialiser trop vite dans un sous-ensemble de situations, ce
   qui la rend plus robuste au distribution shift.

4. **Reduire a 50k steps** : le best est a 35k. Entrainer 50k au lieu
   de 100k evite l'overtraining tout en laissant une marge de progression
   au-dela du pic.

#### C. DQN Manual : epsilon decay BEAUCOUP trop lent (BUG)

**Probleme principal** : c'est le plus grave et le plus simple a corriger.
L'epsilon decay est configure en **gradient steps** et non en episodes.
Avec `epsilon_decay_steps=15000`, voici l'evolution reelle :

| Episode | Gradient steps (~) | Epsilon | Actions aleatoires |
|---------|-------------------|---------|-------------------|
| 50      | ~750              | 0.967   | 97%               |
| 100     | ~1500             | 0.927   | 93%               |
| 200     | ~3000             | 0.842   | 84%               |
| 300     | ~4500             | 0.731   | 73%               |
| 400     | ~6000             | 0.619   | **62%**           |
| 600     | ~9000             | 0.430   | 43%               |
| 1000    | ~15000            | 0.050   | 5% (enfin!)       |

**L'agent ne serait devenu exploitant (epsilon<0.1) qu'apres ~900 episodes.**
Or l'early stopping coupe a 400 episodes car la reward d'entrainement
(incluant 62% d'actions aleatoires) ne progresse plus. Mais la politique
sous-jacente est potentiellement bonne -- elle est simplement noyee sous
le bruit de l'exploration.

C'est confirme par le fait que l'evaluation (en mode deterministe, epsilon=0)
donne 19.73, bien mieux que la moyenne d'entrainement de 14.26. La
politique apprise est meilleure que ce que le training reward suggere.

En Phase 1 (200 episodes), le meme algo avec les memes hyperparametres
donnait 28.46 en eval. Ce n'est pas contradictoire : a 200 episodes
(epsilon~0.84), la politique est deja raisonnable, et l'evaluation en mode
deterministe revele sa vraie qualite sans le bruit de l'epsilon.

**Solutions** :

1. **Reduire `epsilon_decay_steps`** de 15000 a **5000** (priorite
   critique). Avec 5000 steps, epsilon atteint 0.05 vers l'episode
   300-350, ce qui laisse 250-300 episodes d'exploitation pure.

   Impact attendu : la reward d'entrainement montera significativement
   apres episode 300 (au lieu de plafonner a ~14), et le modele final
   sera bien meilleur.

2. **Baser l'early stopping sur l'eval deterministe** plutot que la
   reward d'entrainement. La reward d'entrainement est biaisee par
   epsilon et ne reflete pas la vraie qualite de la politique. Evaluer
   periodiquement en mode deterministe (epsilon=0) sur quelques episodes
   donnerait un signal bien plus fiable.

3. **Augmenter la patience d'early stopping** de 100 a 150 episodes :
   avec un epsilon qui decroit plus vite, la reward d'entrainement
   devrait continuer de monter plus longtemps.

4. **Gradient clipping** (`torch.nn.utils.clip_grad_norm_`) : limiter
   les gradients pour stabiliser l'entrainement en fin de course.

#### Recapitulatif des fixes par priorite

| Algo | Fix prioritaire | Impact attendu | Complexite |
|------|----------------|----------------|------------|
| **DQN Manual** | `epsilon_decay_steps`: 15000 -> 5000 | +++ (fix critique) | Trivial (1 parametre) |
| **DQN SB3** | lr scheduler (1e-3 -> 1e-4) | ++ (evite l'effondrement) | Simple (callable) |
| **PPO SB3** | lr scheduler (3e-4 -> 3e-5) + `n_steps=512` | + (stabilise post-pic) | Simple |
| DQN Manual | Early stop sur eval deterministe | + (arret intelligent) | Modere |
| PPO SB3 | `ent_coef=0.01` | + (exploration douce) | Trivial (1 parametre) |
| DQN SB3 | `target_update_interval`: 50 -> 100 | + (ralentit overestimation) | Trivial |

---

## 4. Comparaison DQN SB3 vs PPO vs DQN Manual

### 4.1 Resultats Phase 2 (modeles finaux vs best checkpoints)

| Algo        | Modele final | Best checkpoint | Amelioration |
|-------------|-------------|-----------------|-------------|
| DQN SB3     | 12.92       | **23.98** (70k) | +85%        |
| PPO SB3     | 23.53       | **28.63** (35k) | +22%        |
| DQN Manual  | **19.73**   | N/A             | -           |

En utilisant les best checkpoints, le classement devient :
1. **PPO SB3** : 28.63 (best checkpoint)
2. **DQN SB3** : 23.98 (best checkpoint)
3. **DQN Manual** : 19.73

### 4.2 Stabilite de l'entrainement

| Critere               | DQN SB3         | PPO SB3         | DQN Manual       |
|-----------------------|-----------------|-----------------|------------------|
| Stabilite training    | Fragile (chute -46%) | Moderee (chute -32%, rebond) | Stable (early stop) |
| Variance inter-runs   | Tres haute      | Haute           | Haute            |
| Sensibilite au lr     | Tres haute      | Moderee         | Moderee          |
| Risque d'overtraining | Eleve           | Modere          | Modere           |
| Besoin d'EvalCallback | Critique        | Important       | Utile            |

### 4.3 Vitesse de convergence

| Algo       | Temps 100k steps | Temps total grid (5 configs) | Env paralleles |
|------------|-----------------|------------------------------|----------------|
| DQN SB3    | ~33 min         | ~37 min                      | 7              |
| PPO SB3    | ~24 min         | ~48 min                      | 7              |
| DQN Manual | ~2 min (600 ep) | ~12 min                      | 1 (sequentiel) |

Le DQN manual est le plus rapide malgre l'absence de parallelisation :
l'overhead de SB3 (callbacks, logging, vectorized env management) est
significatif pour des reseaux aussi petits (MLP 256x256).

### 4.4 Off-policy vs On-policy

- **DQN (off-policy)** : apprend du replay buffer, peut reutiliser les
  donnees anciennes. Plus sample-efficient en theorie mais moins stable
  car les donnees du buffer deviennent "stale" (generees par une ancienne
  politique).
- **PPO (on-policy)** : collecte des donnees avec la politique actuelle,
  apprend dessus, les jette. Moins sample-efficient mais plus stable grace
  au clipping et a la fraicheur des donnees.

Pour highway-env avec des vehicules agressifs, la stabilite de PPO est un
avantage significatif : l'environnement est hautement stochastique et les
situations de conduite changent rapidement, ce qui favorise un algorithme
qui s'adapte en temps reel (on-policy) plutot qu'un qui apprend de vieilles
experiences (off-policy).

### 4.5 Pourquoi le DQN manual prefere gamma=0.8

Notre DQN manual prefere gamma=0.8 alors que les versions SB3 preferent 0.9.
Hypotheses :

1. **Pas de vectorized envs** : le DQN manual s'entraine sur un seul
   environnement, donc les transitions sont correlees temporellement.
   Un gamma plus court (0.8) limite la propagation de cette correlation
   dans les cibles Q.

2. **Gradient steps par transition** : notre implementation fait un update
   a chaque step, alors que SB3 peut batched de maniere plus efficace.
   Des updates plus frequents avec gamma=0.9 peuvent destabiliser les
   Q-values plus vite.

3. **Horizon de planification** : avec gamma=0.8, l'horizon effectif est
   de ~5 steps (1/(1-0.8)). Sur une autoroute, c'est suffisant pour
   anticiper les vehicules proches. gamma=0.9 donne un horizon de ~10
   steps, qui peut etre trop ambitieux pour un reseau simple.

---

## 5. Enseignements pour le rapport

### Points cles a couvrir

- [ ] Presenter les tableaux du grid search avec analyse de gamma et lr
- [ ] Expliquer le probleme d'instabilite DQN (Q-value overestimation)
- [ ] Documenter l'overtraining (Phase 2 < Phase 1 pour les 3 algos)
- [ ] Contraster avec la stabilite relative de PPO (clipping)
- [ ] Mentionner la variance inter-runs comme limite fondamentale du RL
- [ ] Insister sur l'importance de l'EvalCallback (best model != final model)
- [ ] Documenter le bug epsilon decay du DQN manual (62% aleatoire a ep 400)
- [ ] Expliquer la difference entre reward d'entrainement et reward d'eval
- [ ] Documenter les lr schedulers comme solution a l'overtraining
- [ ] Montrer les courbes d'entrainement (training reward vs episodes/steps)
- [ ] Comparer off-policy (DQN) vs on-policy (PPO) en termes de stabilite
- [ ] Expliquer le distribution shift de PPO et pourquoi il rebondit
- [ ] Justifier le choix de gamma=0.9 (SB3) vs gamma=0.8 (manual)
- [ ] Discuter les limites : un seul run par config, pas de multi-seed
- [ ] Presenter les resultats Phase 3 (avec corrections) vs Phase 2

### Structure suggeree du rapport

1. **Introduction** : l'environnement highway-env, la tache, les metriques
2. **Algorithmes** : DQN, PPO, DQN from scratch -- theorie et implementation
3. **Exploration des hyperparametres** : grid search, impact gamma et lr
4. **Resultats** : benchmark, courbes, tableaux comparatifs
5. **Analyse critique** : instabilite DQN, overtraining, variance inter-runs
6. **Conclusion** : le meilleur algo pour highway-env, limites, pistes

---

## 6. Resultats finaux (Phase 2, evaluation 30 episodes)

### Modeles finaux (dernier step)

| Algo              | Best gamma | Best lr | Reward | Std   | Ep. Length | Temps   |
|-------------------|-----------|---------|--------|-------|------------|---------|
| DQN SB3           | 0.9       | 1e-3    | 12.92  | 9.47  | 16         | 33 min  |
| PPO SB3           | 0.9       | 3e-4    | 23.53  | 10.33 | 31         | 24 min  |
| DQN Manual        | 0.8       | 5e-4    | 19.73  | 11.04 | 27         | 2 min   |

### Best checkpoints (EvalCallback)

| Algo              | Checkpoint  | Reward | Std   | Ep. Length |
|-------------------|-------------|--------|-------|------------|
| DQN SB3           | 70k steps   | 23.98  | 11.04 | 30.8       |
| **PPO SB3**       | **35k steps** | **28.63** | **3.02** | **38.8** |
| DQN Manual        | Ep 300      | N/A    | N/A   | N/A        |

### Modele selectionne pour l'evaluation finale

**PPO SB3 (best checkpoint, 35k steps)** : reward=28.63, std=3.02

C'est le meilleur modele en termes de reward et de consistance (std la plus
faible). L'agent survit en moyenne 38.8 steps sur les 40 possibles, ce qui
signifie qu'il evite presque toutes les collisions meme avec les vehicules
agressifs.

### Phase 3 prevue : retrain avec corrections (TBD)

Objectif : relancer les 3 algos avec les fixes identifies en section 3.5.

| Algo                 | Fix applique                          | Reward attendu | Status |
|----------------------|---------------------------------------|---------------|--------|
| DQN SB3 (lr sched)   | lr: 1e-3->1e-4, target_update=100    | >25           | TBD    |
| PPO SB3 (stabilise)  | lr: 3e-4->3e-5, n_steps=512, ent=0.01| >28           | TBD    |
| DQN Manual (epsilon fix) | epsilon_decay: 15000->5000, patience=150 | >25    | TBD    |
