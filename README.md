# DUQN: User Response Modeling in Reinforcement Learning for Ads Allocation

## Project Overview

DUQN is the codebase for the public dataset experiment segment of the "User Response Modeling in Reinforcement Learning for Ads Allocation" study. This project aims to optimize ad allocation through reinforcement learning techniques.

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
