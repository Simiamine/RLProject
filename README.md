# Final Project: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv/tree/master)

Welcome in the repository for the final project of the Reinforcement learning course !

## 👉 Your task: Solve the Highway

Your objective will be to teach an agent how to drive on a highway, among many cars.

<div align="center">

![](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway.gif?raw=true)
</div>

- By Group of two
- Use *at least* two different RL Algorithms
  - try to implement at least one 'by hand'


### Evaluation
*Based on the report (showing that you understood what you did), the performances and the code (you built something that works).*

- **Produce a notebook**
  -  The notebook must run one one go, I will not loose time trying to fix your env...
  - Possible to send a git repo with the weight so that I can run them locally.
- **Produce a 2-5 pages report**
  - Describe Your choices and explain the algorithms used.
  - Benchmark and compare them depending on their hyperparameters.

*Analysis could include exploration of hyperparameters, figures of training, explainations of how your algorithm works*

### Roadmap
- 📆 **23 Feb**: your groups are all complete.
- 📆 **25 March** : Deadline for the report (5-10 pages) and code (notebook / script)



### Ressources

Here are some useful links for documentation and examples.
- **Highway-env** [👨‍💻Repo](https://github.com/Farama-Foundation/HighwayEnv/tree/master) | [📜Documentation](http://highway-env.farama.org/quickstart/)
- **OpenAI Gym**
- **Stable-Baselines3**: [👨‍💻Repo](https://github.com/DLR-RM/stable-baselines3) | [📜Documentation](https://stable-baselines.readthedocs.io/en/master/)


# Usage

### Install Locally 

The best way to to perform well is to install this repo on your machine, so that you can train for longer, test more intensive stuff without relying on the limited colab features.

This repository hosts all requirements necessary, 

```bash
git clone https://github.com/VictorMorand/Highway-env-project.git 
cd Highway-env-project
```
I recommend using [`uv`](https://docs.astral.sh/uv/) to automatically create a working venv in a few seconds. You can optionally fork this repository to share between the group.

```bash
uv sync 
```
you can also install with pip (works using python 3.10)

```bash
pip install -r requirements.txt
```

### Colab 

If you don't have a powerful enough machine, you can still use the main notebook in colab : 
[![Try in Colab !](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VictorMorand/Highway-env-project/blob/main/Final_Project.ipynb)

# Problems 

If you ahave any issues with the code, do not hesitae to reach out !



