from new_level_sampler import LevelSampler,make_level_generator,compute_max_returns,compute_score
import os
import shutil
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional, Literal
import chex
import jax
import logging
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax
import pyrallis
import wandb
import xminigrid
from flax import core,struct
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState 
from nn import ActorCriticRNN
from utils import Transition, calculate_gae, ppo_update_networks, rollout
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import GymAutoResetWrapper

from new_level_sampler import LevelSampler,make_level_generator,compute_max_returns,compute_score
import numpy as np
from enum import IntEnum
benchmark = xminigrid.load_benchmark("trivial-1m")
sample_random_level = make_level_generator(benchmark.num_rulesets())
pholder_level = sample_random_level(jax.random.PRNGKey(0))
print(pholder_level)