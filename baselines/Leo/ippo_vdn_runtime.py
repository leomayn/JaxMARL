import sys
import os

# gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"

# Add the path to the baselines module
sys.path.append("/homes/gws/leomayn/leo/JaxMARL")

import time
import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager
import hydra

# Importing make_train functions
from baselines.IPPO import ippo_ff_overcooked as ppo
from baselines.QLearning import vdn_animate as vdn

def measure_ippo_update_time(config):
    # Initialize environment for IPPO
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Measure time for a single update step for IPPO
    start_time = time.time()
    rng = jax.random.PRNGKey(0)
    train_fn = ppo.make_train(config)
    train_fn(rng)
    end_time = time.time()

    update_time = end_time - start_time
    return update_time

def measure_vdn_update_time(config):
    # Initialize environment for VDN
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Measure time for a single update step for VDN
    start_time = time.time()
    rng = jax.random.PRNGKey(0)
    train_fn = vdn.make_train(config, env)
    train_fn(rng)
    end_time = time.time()

    update_time = end_time - start_time
    return update_time

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):
    # Define IPPO config
    ippo_config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 5e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "overcooked",
        "ENV_KWARGS": {"layout": "cramped_room"},
        "ANNEAL_LR": True
    }

    # Define VDN config
    vdn_config = {
        "NUM_ENVS": 8,
        "NUM_STEPS": 256,
        "BUFFER_SIZE": 1024,
        "BUFFER_BATCH_SIZE": 32,
        "TOTAL_TIMESTEPS": 10e7,
        "AGENT_HIDDEN_DIM": 64,
        "AGENT_INIT_SCALE": 2.0,
        "PARAMETERS_SHARING": True,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.1,
        "EPSILON_ANNEAL_TIME": 1e6,
        "MIXER_EMBEDDING_DIM": 32,
        "MIXER_HYPERNET_HIDDEN_DIM": 64,
        "MIXER_INIT_SCALE": 0.0001,
        "MAX_GRAD_NORM": 10,
        "TARGET_UPDATE_INTERVAL": 200,
        "LR": 0.0005,
        "LR_LINEAR_DECAY": False,
        "EPS_ADAM": 0.0001,
        "WEIGHT_DECAY_ADAM": 0.00001,
        "TD_LAMBDA_LOSS": False,
        "TD_LAMBDA": 0.6,
        "GAMMA": 0.99,
        "VERBOSE": False,
        "WANDB_ONLINE_REPORT": True,
        "NUM_TEST_EPISODES": 32,
        "ENV_NAME": "overcooked",
        "ENV_KWARGS": {"layout": "cramped_room"},
        "TEST_INTERVAL": 50000,
        "ENTITY": "",
        "PROJECT": "",
        "WANDB_MODE": "disabled",
        "NUM_SEEDS": 10,
        "SEED": 30
    }

    # Measure IPPO update time
    ippo_update_time = measure_ippo_update_time(ippo_config)
    print(f"IPPO Update Time: {ippo_update_time} seconds")

    # Measure VDN update time
    vdn_update_time = measure_vdn_update_time(vdn_config)
    print(f"VDN Update Time: {vdn_update_time} seconds")

if __name__ == "__main__":
    main()
