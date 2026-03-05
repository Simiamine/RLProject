# Workflow : fichier .py + Jupyter

## Pourquoi travailler en .py ?

Le fichier `Final_Project.ipynb` est au format JSON (structure complexe, metadonnees, outputs encodes en base64). C'est penible a editer et a versionner avec Git.

On travaille dans **`Final_Project.py`** au format "percent" : chaque cellule est delimitee par `# %%` (code) ou `# %% [markdown]` (texte). C'est du Python pur, facile a lire, editer et versionner.

## Regle d'or

> **Ne jamais editer `Final_Project.ipynb` a la main.**
> Toujours travailler dans `Final_Project.py` et reconvertir.

## Setup

```bash
pip install jupytext
```

## Commandes

### Convertir .ipynb -> .py (une seule fois au debut)

```bash
jupytext --to py:percent Final_Project.ipynb
```

### Convertir .py -> .ipynb (pour soumission / tests)

```bash
jupytext --to ipynb Final_Project.py
```

### Synchroniser les deux fichiers (apres modification du .py)

```bash
jupytext --sync Final_Project.py
```

Pour activer la synchronisation automatique, ajouter dans le header du `.py` :

```python
# jupytext:
#   formats: ipynb,py:percent
```

## Executer les cellules dans Cursor / VS Code

1. Ouvrir `Final_Project.py` dans Cursor
2. Les cellules `# %%` apparaissent avec des boutons "Run Cell" / "Run Below"
3. Cliquer sur "Run Cell" pour executer une cellule
4. Les resultats s'affichent dans le panneau interactif Jupyter

Prerequis : extension **Jupyter** installee dans Cursor/VS Code.

## Procedure de soumission

1. S'assurer que `Final_Project.py` est a jour
2. Convertir : `jupytext --to ipynb Final_Project.py`
3. Ouvrir `Final_Project.ipynb` et executer "Run All" pour verifier
4. Commiter les deux fichiers + le dossier `models/` avec les poids

## Structure des cellules

```python
# %% [markdown]
# # Titre en Markdown
# Texte explicatif...

# %%
# Cellule de code Python
import numpy as np
x = np.array([1, 2, 3])
print(x)
```
