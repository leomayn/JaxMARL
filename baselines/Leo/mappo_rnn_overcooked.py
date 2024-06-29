import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import chex
import os
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State


import wandb
import functools
import matplotlib.pyplot as plt


class OvercookedWorldStateWrapper(JaxMARLWrapper):
    # Implement environment wrapper methods
    pass

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi

class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    # Initialize environment, networks, and training loop
    pass

@hydra.main(version_base=None, config_path="config", config_name="mappo_homogenous_rnn_overcooked")
def main(config):
    # Set up training and start the process
    pass

if __name__ == "__main__":
    main()
