from baselines.IPPO import ippo_ff_overcooked as ppo
from baselines.QLearning import vdn_animate as vdn

import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union

import chex
from flax.struct import PyTreeNode
from typing import Any

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, SMAXLogWrapper, CTRolloutManager
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
import matplotlib.pyplot as plt
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict

def plot_rewards(metrics, filename, num_seeds):
    test_metrics = metrics["test_metrics"]
    test_returns = test_metrics["test_returns"].mean(-1).reshape((num_seeds, -1))

    reward_mean = test_returns.mean(0)  # mean
    reward_std = test_returns.std(0) / np.sqrt(num_seeds)  # standard error
    
    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'{filename}.png')
    
    
@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    # tune(config)
    config = OmegaConf.to_container(config)

    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]
    env = make(config["ENV_NAME"], **config['ENV_KWARGS'])
    env = LogWrapper(env)
if __name__ == "__main__":
    main()