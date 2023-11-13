# DUQN: User Response Modeling in Reinforcement Learning for Ads Allocation

## Project Overview

DUQN is the codebase for the public dataset experiment segment of the "User Response Modeling in Reinforcement Learning for Ads Allocation" study. This project aims to optimize ad allocation through reinforcement learning techniques.

## Dataset

A fragment of the RL4RS dataset is provided in the `dataset` folder for running and debugging the code. For the complete dataset, please visit [RL4RS GitHub Repository](https://github.com/fuxiAIlab/RL4RS).

## Comparative Baseline

This project uses the open-source Hyper-Actor-Critic (HAC) method as a comparative baseline, adopting the same environmental settings. For more details on HAC, visit the [HAC GitHub Repository](https://github.com/CharlieMat/Hyper-Actor-Critic-for-Recommendation).

## Installation Guide

To set up the DUQN environment, follow these steps:

```bash
# Create a new conda environment
conda create -n duqn python=3.9

# Activate the environment
conda activate duqn

# Install PyTorch and torchvision
conda install pytorch torchvision -c pytorch

# Install other dependencies
conda install pandas matplotlib scikit-learn
pip install tqdm
conda install -c anaconda ipykernel

# Set up the IPython kernel
python -m ipykernel install --user --name duqn --display-name "DUQN"
