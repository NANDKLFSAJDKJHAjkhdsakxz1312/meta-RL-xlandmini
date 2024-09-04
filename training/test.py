import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from functools import partial
from typing import Literal, Optional

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax
import pyrallis
import wandb
from flax import core, struct
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from new_level_sampler import (
    LevelSampler,
    compute_max_returns,
    compute_score,
    make_level_generator,
)
from nn import ActorCriticRNN
from utils import Transition, calculate_gae, ppo_update_networks, rollout

import xminigrid
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import GymAutoResetWrapper

benchmark = xminigrid.load_benchmark("trivial-1m")
sample_random_level = make_level_generator(benchmark.num_rulesets())
pholder_level = sample_random_level(jax.random.PRNGKey(0))
print(pholder_level)
