# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import os
from jax.profiler import TraceAnnotation
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
import nn
from nn import ActorCriticRNN
from utils import Transition, calculate_gae, ppo_update_networks, rollout
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper
import os

from new_level_sampler import LevelSampler,make_level_generator,compute_max_returns,compute_score
import numpy as np
from enum import IntEnum
import jax.profiler
from utils_ssp import HexagonalSSPSpace
from src.xminigrid.types import TimeStep, State, AgentState, EnvCarry, StepType
from jax import config
from jax import jit
# jax.config.update("jax_disable_jit", True)
class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1
# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)
########
Prioritization = Literal["rank", "topk"]

i_indices = jnp.arange(9)
j_indices = jnp.arange(9)
i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing='ij')

# flattening i_grid and j_grid to prepare for parallel processing.
i_grid_flat = i_grid.flatten()
j_grid_flat = j_grid.flatten()
up_x = i_grid_flat-8
up_y = j_grid_flat-4
right_x = j_grid_flat-4
right_y = -(i_grid_flat-8)
down_x = -(i_grid_flat-8)
down_y = -(j_grid_flat-4)
left_x = -(j_grid_flat-4)
left_y = i_grid_flat-8

class TrainState(TrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

@dataclass
class TrainConfig:
    project: str = "xminigrid"
    group: str = "default"
    name: str = "ssp_1B"
    env_id: str = "XLand-MiniGrid-R1-9x9"
    benchmark_id: str = "small-1m"
    img_obs: bool = False 
    # agent
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    # training
    enable_bf16: bool = False
    num_envs: int = 2
    num_steps_per_env: int = 4
    num_steps_per_update: int = 4
    update_epochs: int = 1
    num_minibatches: int = 1
    total_timesteps: int = 8
    lr: float = 0.001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_num_envs: int = 512
    eval_num_episodes: int = 10
    eval_seed: int = 42
    train_seed: int = 3
    checkpoint_path: Optional[str] = None
########
    replay_prob: float = 0.5
    staleness_coeff: float = 0.5
    minimum_fill_ratio: float = 1.0
    prioritization: Prioritization = "rank"
    duplicate_check: bool = False
    temperature: float =  0.99
    topk_k: int = 1
    score_function: str = "MaxMC"
########

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # splitting computation across all available devices
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_num_envs_per_device = self.eval_num_envs // num_devices
      
        assert self.num_envs % num_devices == 0
        self.num_meta_updates = round(
            self.total_timesteps_per_device / (self.num_envs_per_device * self.num_steps_per_env)
        )
        self.num_inner_updates = self.num_steps_per_env // self.num_steps_per_update
        assert self.num_steps_per_env % self.num_steps_per_update == 0
        print(f"Num devices: {num_devices}, Num meta updates: {self.num_meta_updates}")



def make_states(config: TrainConfig):
    # for learning rage scheduling
    def linear_schedule(count):
        total_inner_updates = config.num_minibatches * config.update_epochs * config.num_inner_updates
        frac = 1.0 - (count // total_inner_updates) / config.num_meta_updates
        return config.lr * frac
   
    # setup environment
    if "XLand" not in config.env_id:
        raise ValueError("Only meta-task environments are supported.")

    env, env_params = xminigrid.make(config.env_id)
    
    env_params = env_params.replace(view_size=9)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)

    # enabling image observations if needed
    if config.img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    # loading benchmark
    benchmark = xminigrid.load_benchmark(config.benchmark_id)

    # set up training state
    rng = jax.random.key(config.train_seed)
    rng, _rng = jax.random.split(rng)

    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=config.obs_emb_dim,
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
        dtype=jnp.bfloat16 if config.enable_bf16 else None,
    )
    # [batch_size, seq_len, ...]
    shapes = env.observation_shape(env_params)
    # hard code the initial obs_img shape, later it can be replaced
    init_obs = {
        "obs_img": jnp.zeros((config.num_envs_per_device, 1, 9,9,2),dtype=jnp.int32),
        "obs_dir": jnp.zeros((config.num_envs_per_device, 1, 4),dtype=jnp.int32),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)
        
    
    network_params = network.init(_rng, init_obs, init_hstate)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )

  
    sample_random_level = make_level_generator(benchmark.num_rulesets())
    pholder_level = sample_random_level(jax.random.PRNGKey(config.train_seed))
    levelsampler = LevelSampler(
        capacity=20480,
        replay_prob=config.replay_prob,
        staleness_coeff=config.staleness_coeff,
        minimum_fill_ratio=config.minimum_fill_ratio,
        prioritization=config.prioritization,
        prioritization_params={"temperature": config.temperature, "k": config.topk_k},
        duplicate_check=config.duplicate_check
    )
    sampler = levelsampler.initialize(pholder_level, {"max_return": -jnp.inf})
    pholder_level_batch = jax.tree_map(lambda x: jnp.array([x]).repeat(config.num_envs, axis=0), pholder_level)
    
########

    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch)

    return rng, env, env_params, benchmark, init_hstate, train_state


def make_train(
    env: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    config: TrainConfig,
):
    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
        init_hstate: jax.Array,
    ):
########        
        level_sampler = LevelSampler(
        capacity=20480,
        replay_prob=config.replay_prob,
        staleness_coeff=config.staleness_coeff,
        minimum_fill_ratio=config.minimum_fill_ratio,
        prioritization=config.prioritization,
        prioritization_params={"temperature": config.temperature, "k": config.topk_k},
        duplicate_check=config.duplicate_check
        )

        eval_hstate = init_hstate[0][None]       
        # META TRAIN LOOP
        def _meta_step(meta_state, _):
            rng, train_state = meta_state
            
            # INIT ENV
            rng, _rng1, _rng2 = jax.random.split(rng, num=3)
            ruleset_rng = jax.random.split(_rng1, num=config.num_envs_per_device)
            reset_rng = jax.random.split(_rng2, num=config.num_envs_per_device)
        
            def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):    
                sampler = train_state.sampler
                # jax.debug.print('episode1:{}',sampler['episode_count'])
                # jax.debug.print('levels1:{}',sampler['levels'])
                # jax.debug.print('scores1:{}',sampler['scores'])
                # jax.debug.print('size:{}',sampler['size'])
                sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, _rng1, config.num_envs_per_device)
                # jax.debug.print('episode2:{}',sampler['episode_count'])
                # jax.debug.print('sample_levels:{}',levels)
                # jax.debug.print('episode2:{}',sampler['episode_count'])
                # jax.debug.print('levels2:{}',sampler['levels'])
                # jax.debug.print('scores2:{}',sampler['scores'])
                # jax.debug.print('levels:{}',sampler['levels'])
                # jax.debug.print('scores:{}',sampler['scores'])
                # levels = jnp.array(levels)
    ########
                rulesets = jax.vmap(benchmark.get_ruleset)(levels) # change the random ruleset_rng to levels
                meta_env_params = env_params.replace(ruleset=rulesets)

                timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_rng)
                prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
                prev_reward = jnp.zeros(config.num_envs_per_device)

                # INNER TRAIN LOOP
                def _update_step(runner_state, _):
                    # COLLECT TRAJECTORIES
                    def _env_step(runner_state, _):
                        # jax.profiler.start_trace("/tmp/jax_trace")
                        
                        # start_time = time.time()
                        rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state
                        
                        agent_positions = prev_timestep.state.agent.position  # shape：[batch_size, 2]
                        agent_directions = prev_timestep.state.agent.direction.astype(int)  # shape: [batch_size]
                        jax.debug.print("dir shape:{x}",x = agent_directions)
                        @jit
                        def _is_in_bound(x,y):
                            return (x >= 0) & (x <= 8) & (y >= 0) & (y <= 8)
                        @jit   
                        def process_batch(batch,dir,pos):
                            

                            def case_0():
                                local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                                x = up_x + pos[0]
                                y = up_y + pos[1]
                                mask = _is_in_bound(x, y)

                                # 遍历每个位置，仅在满足条件的 (x, y) 位置上更新
                                def update_local_obs(i, obs):
                                    xi, yi = x[i], y[i]

                                    def set_update_value(obs):
                                        update_value = batch[xi - pos[0] + 8, yi - pos[1] + 4]
                                        return obs.at[xi, yi, :].set(update_value)

                                    # 使用 jax.lax.cond 进行条件更新
                                    obs = jax.lax.cond(
                                        mask[i],           # 条件为 True 时更新
                                        set_update_value,   # 满足条件时的更新函数
                                        lambda obs: obs,    # 不满足条件时保持不变
                                        obs                 # 传递的数组
                                    )
                                    return obs

                                local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                                
                                return local_obs

                            def case_1():
                                local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                                x = right_x + pos[0]
                                y = right_y + pos[1]
                                mask = _is_in_bound(x, y)

                                # 遍历每个位置，仅在满足条件的 (x, y) 位置上更新
                                def update_local_obs(i, obs):
                                    xi, yi = x[i], y[i]

                                    def set_update_value(obs):
                                        update_value = batch[8 + pos[1] - yi, xi + 4 - pos[0]]
                                        return obs.at[xi, yi, :].set(update_value)

                                    # 使用 jax.lax.cond 进行条件更新
                                    obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                    return obs

                                local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                                
                                return local_obs

                            def case_2():
                                local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                                x = down_x + pos[0]
                                y = down_y + pos[1]
                                mask = _is_in_bound(x, y)

                                # 遍历每个位置，仅在满足条件的 (x, y) 位置上更新
                                def update_local_obs(i, obs):
                                    xi, yi = x[i], y[i]

                                    def set_update_value(obs):
                                        update_value = batch[8 + pos[0] - xi, 4 + pos[1] - yi]
                                        return obs.at[xi, yi, :].set(update_value)

                                    obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                    return obs

                                local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                                
                                return local_obs

                            def case_3():
                                local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                                x = left_x + pos[0]
                                y = left_y + pos[1]
                                mask = _is_in_bound(x, y)

                                # 遍历每个位置，仅在满足条件的 (x, y) 位置上更新
                                def update_local_obs(i, obs):
                                    xi, yi = x[i], y[i]

                                    def set_update_value(obs):
                                        update_value = batch[8 + yi - pos[1], 4 - xi + pos[0]]
                                        return obs.at[xi, yi, :].set(update_value)

                                    obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                    return obs

                                local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                                
                                return local_obs
                            local_obs_final = jax.lax.switch(
                                dir,
                                [case_0, case_1, case_2, case_3]
                            )
                            return local_obs_final
                        # parallelly transform local observation to global one
                        all_batches_label_obs = jax.vmap(process_batch)(
                            prev_timestep.observation["img"], 
                            agent_directions,
                            agent_positions
                        )
                        # replace the local observation by global observation for later ssp use
                        prev_timestep.observation['img'] = all_batches_label_obs

                        rng, _rng = jax.random.split(rng)
                        dist, value, hstate = train_state.apply_fn(
                            train_state.params,
                            {
                                # [batch_size, seq_len=1, ...]
                                # "obs_img": prev_timestep.observation["img"][:, None],
                                "obs_img": prev_timestep.observation['img'][:, None],
                                "obs_dir": prev_timestep.observation["direction"][:, None],
                                "prev_action": prev_action[:, None],
                                "prev_reward": prev_reward[:, None],
                            },
                            prev_hstate,
                            
                        )
                        
                        action, log_prob = dist.sample_and_log_prob(seed=_rng)
                        # squeeze seq_len where possible
                        action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

                        # STEP ENV
                        timestep = jax.vmap(env.step, in_axes=0)(meta_env_params, prev_timestep, action)

                        transition = Transition(
                            # ATTENTION: done is always false, as we optimize for entire meta-rollout
                            done=jnp.zeros_like(timestep.last()),
                            action=action,
                            value=value,
                            reward=timestep.reward,
                            log_prob=log_prob,
                            obs=prev_timestep.observation["img"],
                            dir=prev_timestep.observation["direction"],
                            
                            prev_action=prev_action,
                            prev_reward=prev_reward,
                        )
                        
                        runner_state = (rng, train_state, timestep, action, timestep.reward, hstate)
                        # end_time = time.time() 
                        # print(f"_env_step took {end_time - start_time:.4f} seconds")
                        return runner_state, transition

                    initial_hstate = runner_state[-1]
                    # transitions: [seq_len, batch_size, ...]
                    runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps_per_update)

                    # CALCULATE ADVANTAGE
                    rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
                    # hard coding here, later can be changed
                    def _is_in_bound(x,y):
                        return (x >= 0) & (x <= 8) & (y >= 0) & (y <= 8)
                    @jit   
                    def process_batch(batch,dir,pos):
                        
                        # hard coding here, later can be changed
                        def case_0():
                            local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                            x = up_x + pos[0]
                            y = up_y + pos[1]
                            mask = _is_in_bound(x, y)

                            # loop through every pixel and only update in valid (x,y) position
                            def update_local_obs(i, obs):
                                xi, yi = x[i], y[i]

                                def set_update_value(obs):
                                    update_value = batch[xi - pos[0] + 8, yi - pos[1] + 4]
                                    return obs.at[xi, yi, :].set(update_value)

                               
                                obs = jax.lax.cond(
                                    mask[i],           
                                    set_update_value,  
                                    lambda obs: obs,    
                                    obs                
                                )
                                return obs

                        
                            local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                            
                            return local_obs

                        def case_1():
                            local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                            x = right_x + pos[0]
                            y = right_y + pos[1]
                            mask = _is_in_bound(x, y)

                            # loop through every pixel and only update in valid (x,y) position
                            def update_local_obs(i, obs):
                                xi, yi = x[i], y[i]

                                def set_update_value(obs):
                                    update_value = batch[8 + pos[1] - yi, xi + 4 - pos[0]]
                                    return obs.at[xi, yi, :].set(update_value)

                                
                                obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                return obs

                            local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                            
                            return local_obs

                        def case_2():
                            local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                            x = down_x + pos[0]
                            y = down_y + pos[1]
                            mask = _is_in_bound(x, y)

                            # loop through every pixel and only update in valid (x,y) position
                            def update_local_obs(i, obs):
                                xi, yi = x[i], y[i]

                                def set_update_value(obs):
                                    update_value = batch[8 + pos[0] - xi, 4 + pos[1] - yi]
                                    return obs.at[xi, yi, :].set(update_value)

                                obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                return obs

                            local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                            
                            return local_obs

                        def case_3():
                            local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                            x = left_x + pos[0]
                            y = left_y + pos[1]
                            mask = _is_in_bound(x, y)

                           # loop through every pixel and only update in valid (x,y) position
                            def update_local_obs(i, obs):
                                xi, yi = x[i], y[i]

                                def set_update_value(obs):
                                    update_value = batch[8 + yi - pos[1], 4 - xi + pos[0]]
                                    return obs.at[xi, yi, :].set(update_value)

                                obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                return obs

                            local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                            
                            return local_obs
                        local_obs_final = jax.lax.switch(
                            dir,
                            [case_0, case_1, case_2, case_3]
                        )
                        return local_obs_final
                    # parallel transformation from local to global
                    all_batches_label_obs_for_update = jax.vmap(process_batch)(
                        timestep.observation["img"], 
                        timestep.state.agent.direction.astype(int) ,
                        timestep.state.agent.position
                    )
                    # also need to transform local to global representation when update the network
                    timestep.observation['img'] = all_batches_label_obs_for_update
                    # calculate value of the last step for bootstrapping
                    _, last_val, _ = train_state.apply_fn(
                        train_state.params,
                        {
                            "obs_img": timestep.observation["img"][:, None],
                            "obs_dir": timestep.observation["direction"][:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                        },
                        hstate,
                        
                    )
                    advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)
    ########                
                    sampler = train_state.sampler
                    max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(transitions.done, transitions.reward))
                    # scores = compute_score(config.score_function, transitions.done, transitions.value, max_returns, advantages)
                    
                    scores = jnp.mean(jnp.abs(advantages),axis=0)
                    # jax.debug.print('shape_of_scores:{}',scores.shape)
                    # jax.debug.print('scores_inner_max:{}',scores.max())
                    sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})

                    # UPDATE NETWORK
                    def _update_epoch(update_state, _):
                        def _update_minbatch(train_state, batch_info):
                            init_hstate, transitions, advantages, targets = batch_info
                            new_train_state, update_info = ppo_update_networks(
                                train_state=train_state,
                                transitions=transitions,
                                init_hstate=init_hstate.squeeze(1),
                                advantages=advantages,
                                targets=targets,
                                clip_eps=config.clip_eps,
                                vf_coef=config.vf_coef,
                                ent_coef=config.ent_coef,
                            )

                            return new_train_state, update_info
                        rng, train_state, init_hstate, transitions, advantages, targets = update_state

                        # MINIBATCHES PREPARATION
                        rng, _rng = jax.random.split(rng)
                        permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                        # [seq_len, batch_size, ...]
                        batch = (init_hstate, transitions, advantages, targets)
                        # [batch_size, seq_len, ...], as our model assumes
                        batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                        shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                        # [num_minibatches, minibatch_size, ...]
                        minibatches = jtu.tree_map(
                            lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                        )
                    
                        train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                        update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
                        return update_state, update_info

                    # hstate shape: [seq_len=None, batch_size, num_layers, hidden_dim]
                    update_state = (rng, train_state, initial_hstate[None, :], transitions, advantages, targets)
                
                    update_state, loss_info= jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
                # WARN: do not forget to get updated params
                    rng, train_state = update_state[:2]
    ########                
                    train_state = train_state.replace(
                    sampler=sampler,)
    ########                
                    # averaging over minibatches then over epochs
                    loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
                    runner_state = (rng, train_state, timestep, prev_action, prev_reward, hstate)
                    return runner_state, loss_info
                # on each meta-update we reset rnn hidden to init_hstate
                runner_state = (rng, train_state, timestep, prev_action, prev_reward, init_hstate)

                runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_inner_updates)
            # WARN: do not forget to get updated params
                
                rng, train_state = runner_state[:2]
                # jax.debug.print('episode3:{}',train_state.sampler['episode_count'])
                # jax.debug.print('levels3:{}',sampler['levels'])
                # jax.debug.print('scores3:{}',train_state.sampler['scores'])
                # jax.debug.print('levels2:{}',train_state.sampler['levels'])
                # jax.debug.print('scores2:{}',train_state.sampler['scores'])
                train_state = train_state.replace(
                    
                    update_state=UpdateState.REPLAY,
                    num_replay_updates=train_state.num_replay_updates + 1,
                    replay_last_level_batch=levels,
                    )
                # jax.debug.print('episode4:{}',sampler['episode_count'])
                # jax.debug.print('levels4:{}',sampler['levels'])
                # jax.debug.print('scores4:{}',sampler['scores'])
                return rng, train_state ,loss_info
            def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
                
                sampler = train_state.sampler
                # jax.debug.print('size:{}',sampler['size'])
                sample_random_level = make_level_generator(benchmark.num_rulesets())
                new_levels = jax.vmap(sample_random_level)(jax.random.split(_rng1, config.num_envs_per_device))
                rulesets = jax.vmap(benchmark.get_ruleset)(new_levels) # change the random ruleset_rng to levels
                meta_env_params = env_params.replace(ruleset=rulesets)

                timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_rng)
                prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32
                )
                prev_reward = jnp.zeros(config.num_envs_per_device)

                # INNER TRAIN LOOP
                def _update_step(runner_state, _):
                    # COLLECT TRAJECTORIES
                    def _env_step(runner_state, _):
                        # jax.profiler.start_trace("/tmp/jax_trace")
                        # start_time = time.time()
                        rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state
                        # jax.debug.print("before:{x}",x=prev_timestep.observation["img"].squeeze())
                        
                        # jax.debug.print("rule:{x}",x=prev_timestep.state.rule_encoding)
                        # jax.debug.print("goal:{x}",x=prev_timestep.state.goal_encoding)
                        # jax.debug.print("obs:{x}",x=prev_timestep.observation["img"])
                        agent_positions = prev_timestep.state.agent.position  # shape:[batch_size, 2]
                        agent_directions = prev_timestep.state.agent.direction.astype(int)  # shape: [batch_size]
                        jax.debug.print("dir shape:{x}",x = agent_directions)
                        
                        def _is_in_bound(x,y):
                            return (x >= 0) & (x <= 8) & (y >= 0) & (y <= 8)
                        @jit
                        def process_batch(batch,dir,pos):
                            
                            def case_0():
                                local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                                x = up_x + pos[0]
                                y = up_y + pos[1]
                                mask = _is_in_bound(x, y)

                               
                                def update_local_obs(i, obs):
                                    xi, yi = x[i], y[i]

                                    def set_update_value(obs):
                                        update_value = batch[xi - pos[0] + 8, yi - pos[1] + 4]
                                        return obs.at[xi, yi, :].set(update_value)

                                    
                                    obs = jax.lax.cond(
                                        mask[i],           
                                        set_update_value,   
                                        lambda obs: obs,    
                                        obs                 
                                    )
                                    return obs

                               
                                local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                                
                                return local_obs

                            def case_1():
                                local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                                x = right_x + pos[0]
                                y = right_y + pos[1]
                                mask = _is_in_bound(x, y)

                               
                                def update_local_obs(i, obs):
                                    xi, yi = x[i], y[i]

                                    def set_update_value(obs):
                                        update_value = batch[8 + pos[1] - yi, xi + 4 - pos[0]]
                                        return obs.at[xi, yi, :].set(update_value)

                                    # 使用 jax.lax.cond 进行条件更新
                                    obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                    return obs

                                local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                                
                                return local_obs

                            def case_2():
                                local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                                x = down_x + pos[0]
                                y = down_y + pos[1]
                                mask = _is_in_bound(x, y)

                                
                                def update_local_obs(i, obs):
                                    xi, yi = x[i], y[i]

                                    def set_update_value(obs):
                                        update_value = batch[8 + pos[0] - xi, 4 + pos[1] - yi]
                                        return obs.at[xi, yi, :].set(update_value)

                                    obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                    return obs

                                local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                                
                                return local_obs

                            def case_3():
                                local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                                x = left_x + pos[0]
                                y = left_y + pos[1]
                                mask = _is_in_bound(x, y)

                                def update_local_obs(i, obs):
                                    xi, yi = x[i], y[i]

                                    def set_update_value(obs):
                                        update_value = batch[8 + yi - pos[1], 4 - xi + pos[0]]
                                        return obs.at[xi, yi, :].set(update_value)

                                    obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                    return obs

                                local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                                
                                return local_obs
                            local_obs_final = jax.lax.switch(
                                dir,
                                [case_0, case_1, case_2, case_3]
                            )
                            return local_obs_final
                            
                        
                        all_batches_label_obs = jax.vmap(process_batch)(
                            prev_timestep.observation["img"], 
                            agent_directions,
                            agent_positions
                        )

                        prev_timestep.observation['img'] = all_batches_label_obs
                        
                        # SELECT ACTION
                        rng, _rng = jax.random.split(rng)
                        dist, value, hstate = train_state.apply_fn(
                            train_state.params,
                            {
                                # [batch_size, seq_len=1, ...]
                                "obs_img": prev_timestep.observation['img'][:, None],
                                
                                "obs_dir": prev_timestep.observation["direction"][:, None],
                                "prev_action": prev_action[:, None],
                                "prev_reward": prev_reward[:, None],
                            },
                            prev_hstate,
                            
                        )
                        action, log_prob = dist.sample_and_log_prob(seed=_rng)
                        # squeeze seq_len where possible
                        action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)
                       
                        
                        # STEP ENV
                        timestep = jax.vmap(env.step, in_axes=0)(meta_env_params, prev_timestep, action)
                     
                        transition = Transition(
                            # ATTENTION: done is always false, as we optimize for entire meta-rollout
                            done=jnp.zeros_like(timestep.last()),
                            action=action,
                            value=value,
                            reward=timestep.reward,
                            log_prob=log_prob,
                            obs=prev_timestep.observation["img"],
                            dir=prev_timestep.observation["direction"],
                            prev_action=prev_action,
                            prev_reward=prev_reward,
                        )
                        runner_state = (rng, train_state, timestep, action, timestep.reward, hstate)
                        # end_time = time.time()  
                        # print(f"_env_step took {end_time - start_time:.4f} seconds")
                        # jax.profiler.stop_trace()  
                        return runner_state, transition

                    initial_hstate = runner_state[-1]
                    # transitions: [seq_len, batch_size, ...]
                    runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps_per_update)

                    # CALCULATE ADVANTAGE
                    rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
                    def _is_in_bound(x,y):
                        return (x >= 0) & (x <= 8) & (y >= 0) & (y <= 8)
                    @jit   
                    def process_batch(batch,dir,pos):
                        

                        def case_0():
                            local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                            x = up_x + pos[0]
                            y = up_y + pos[1]
                            mask = _is_in_bound(x, y)

                            
                            def update_local_obs(i, obs):
                                xi, yi = x[i], y[i]

                                def set_update_value(obs):
                                    update_value = batch[xi - pos[0] + 8, yi - pos[1] + 4]
                                    return obs.at[xi, yi, :].set(update_value)

                                
                                obs = jax.lax.cond(
                                    mask[i],           
                                    set_update_value,   
                                    lambda obs: obs,    
                                    obs                 
                                )
                                return obs
                            
                            local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                            
                            return local_obs

                        def case_1():
                            local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                            x = right_x + pos[0]
                            y = right_y + pos[1]
                            mask = _is_in_bound(x, y)

                            def update_local_obs(i, obs):
                                xi, yi = x[i], y[i]

                                def set_update_value(obs):
                                    update_value = batch[8 + pos[1] - yi, xi + 4 - pos[0]]
                                    return obs.at[xi, yi, :].set(update_value)

                                obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                return obs

                            local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                            
                            return local_obs

                        def case_2():
                            local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                            x = down_x + pos[0]
                            y = down_y + pos[1]
                            mask = _is_in_bound(x, y)

                            def update_local_obs(i, obs):
                                xi, yi = x[i], y[i]

                                def set_update_value(obs):
                                    update_value = batch[8 + pos[0] - xi, 4 + pos[1] - yi]
                                    return obs.at[xi, yi, :].set(update_value)

                                obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                return obs

                            local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                            
                            return local_obs

                        def case_3():
                            local_obs = jnp.zeros((9, 9, 2), dtype=jnp.uint8)
                            x = left_x + pos[0]
                            y = left_y + pos[1]
                            mask = _is_in_bound(x, y)

                            def update_local_obs(i, obs):
                                xi, yi = x[i], y[i]

                                def set_update_value(obs):
                                    update_value = batch[8 + yi - pos[1], 4 - xi + pos[0]]
                                    return obs.at[xi, yi, :].set(update_value)

                                obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
                                return obs

                            local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
                            
                            return local_obs
                        local_obs_final = jax.lax.switch(
                            dir,
                            [case_0, case_1, case_2, case_3]
                        )
                        return local_obs_final
                    all_batches_label_obs_for_update = jax.vmap(process_batch)(
                        timestep.observation["img"], 
                        timestep.state.agent.direction.astype(int) ,
                        timestep.state.agent.position
                    )
                    
                    # also need to transform local to global representation when update the network
                    timestep.observation['img'] = all_batches_label_obs_for_update
                    
                    # calculate value of the last step for bootstrapping
                    _, last_val, _ = train_state.apply_fn(
                        train_state.params,
                        {
                            "obs_img": timestep.observation["img"][:, None],
                            "obs_dir": timestep.observation["direction"][:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                        },
                        hstate,
                        
                    )
                    advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)
    ########        
                    
                    # jax.debug.print('shape_of_ad:{}',advantages.shape)
                    sampler = train_state.sampler
                    max_returns = compute_max_returns(transitions.done, transitions.reward)
                    # scores = compute_score(config.score_function, transitions.done, transitions.value, max_returns, advantages)
                    scores = jnp.mean(jnp.abs(advantages),axis=0)
                    sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})
                    # jax.debug.print('levels_in_sampler_updated:{}',sampler['levels'])
                    

                    # UPDATE NETWORK
                    def _update_epoch(update_state, _):
                        def _update_minbatch(train_state, batch_info):
                            init_hstate, transitions, advantages, targets = batch_info
                            new_train_state, update_info = ppo_update_networks(
                                train_state=train_state,
                                transitions=transitions,
                                init_hstate=init_hstate.squeeze(1),
                                advantages=advantages,
                                targets=targets,
                                clip_eps=config.clip_eps,
                                vf_coef=config.vf_coef,
                                ent_coef=config.ent_coef,
                            )

                            return new_train_state, update_info
                        rng, train_state, init_hstate, transitions, advantages, targets = update_state

                        # MINIBATCHES PREPARATION
                        rng, _rng = jax.random.split(rng)
                        permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                        # [seq_len, batch_size, ...]
                        batch = (init_hstate, transitions, advantages, targets)
                        # [batch_size, seq_len, ...], as our model assumes
                        batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                        shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                        # [num_minibatches, minibatch_size, ...]
                        minibatches = jtu.tree_map(
                            lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                        )
                    
                        train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                        update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
                        return update_state, update_info

                    # hstate shape: [seq_len=None, batch_size, num_layers, hidden_dim]
                    update_state = (rng, train_state, initial_hstate[None, :], transitions, advantages, targets)
                
                    update_state, loss_info= jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
                # WARN: do not forget to get updated params
                    rng, train_state = update_state[:2]
    ########                
                    train_state = train_state.replace(
                    sampler=sampler)
    ########        
                    # jax.debug.print('sampler_size: {}', train_state.sampler['size'])
                    # jax.debug.print('sampler_levels:{}',train_state.sampler['levels'])
                    # jax.debug.print('sampler_scores:{}',train_state.sampler['scores'])          
                    # averaging over minibatches then over epochs
                    loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
                    runner_state = (rng, train_state, timestep, prev_action, prev_reward, hstate)
                    return runner_state, loss_info
                # on each meta-update we reset rnn hidden to init_hstate
                runner_state = (rng, train_state, timestep, prev_action, prev_reward, init_hstate)

                runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_inner_updates)
            # WARN: do not forget to get updated params
                rng, train_state = runner_state[:2]
                train_state = train_state.replace(
                    
                    update_state=UpdateState.DR,
                    num_dr_updates=train_state.num_dr_updates + 1,
                    dr_last_level_batch=new_levels,
                    )
                return rng, train_state,loss_info
            # sample rulesets for this meta update
            rng, rng_replay = jax.random.split(rng)
            branch = level_sampler.sample_replay_decision(train_state.sampler,rng_replay)
            rng, train_state,loss_info = jax.lax.cond(
            branch,  
            lambda _: on_replay_levels(rng, train_state),
            lambda _: on_new_levels(rng, train_state),
            operand=None  
            )

            # EVALUATE AGENT
            eval_ruleset_rng, eval_reset_rng = jax.random.split(jax.random.key(config.eval_seed))
            eval_ruleset_rng = jax.random.split(eval_ruleset_rng, num=config.eval_num_envs_per_device)
            eval_reset_rng = jax.random.split(eval_reset_rng, num=config.eval_num_envs_per_device)

            eval_ruleset = jax.vmap(benchmark.sample_ruleset)(eval_ruleset_rng)
            eval_env_params = env_params.replace(ruleset=eval_ruleset)

            eval_stats = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None))(
                eval_reset_rng,
                env,
                eval_env_params,
                train_state,
                eval_hstate,
                config.eval_num_episodes,
            )
            eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")

            # averaging over inner updates, adding evaluation metrics
            loss_info = jtu.tree_map(lambda x: x.mean(-1), loss_info)
            loss_info.update(
                {
                    "eval/returns_mean": eval_stats.reward.mean(0),
                    "eval/returns_median": jnp.median(eval_stats.reward),
                    "eval/lengths": eval_stats.length.mean(0),
                    "eval/lengths_20percentile": jnp.percentile(eval_stats.length, q=20),
                    "eval/returns_20percentile": jnp.percentile(eval_stats.reward, q=20),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                    "levels":train_state.sampler["levels"].mean(),
                    "scores":train_state.sampler["scores"].mean()
                }
            )
            meta_state = (rng, train_state)
            jax.debug.print('scores:{}',train_state.sampler["scores"])
            jax.debug.print('scores_mean:{}',train_state.sampler["scores"].mean())
            return meta_state, loss_info

        meta_state = (rng, train_state)

        meta_state, loss_info = jax.lax.scan(_meta_step, meta_state, None, config.num_meta_updates)
        levels = meta_state[1].sampler['levels']
        scores = meta_state[1].sampler['scores']
        timestamps=meta_state[1].sampler['timestamps']
        size=meta_state[1].sampler['size']
        episode_count=meta_state[1].sampler['episode_count']
        num_replay = meta_state[1].num_replay_updates
        num_dr = meta_state[1].num_dr_updates
        levels_dr = meta_state[1].dr_last_level_batch

        return {"state": meta_state[-1], "loss_info": loss_info,'scores':scores,'levels':levels,'timestamps':timestamps,'size':size,'episode_count':episode_count,'num_dr':num_dr,'num_replay':num_replay,'levels_dr':levels_dr} 

    return train


@pyrallis.wrap()
def train(config: TrainConfig):


    # logging to wandb

    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    # removing existing checkpoints if any
    if config.checkpoint_path is not None and os.path.exists(config.checkpoint_path):
        shutil.rmtree(config.checkpoint_path)

    rng, env, env_params, benchmark, init_hstate, train_state = make_states(config)
    # replicating args across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())

    print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, benchmark, config)
    train_fn = train_fn.lower(rng, train_state, init_hstate).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng, train_state, init_hstate))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Logginig...")
    loss_info = unreplicate(train_info["loss_info"])
    levels_info = train_info["levels"]

    scores_info = train_info["scores"]
    size_info = train_info["size"]
    timestamps_info = train_info["timestamps"]
    episodecount_info = train_info["episode_count"]
    num_dr_info = train_info['num_dr']
    num_replay_info = train_info['num_replay']
    levels_dr_info = train_info['levels_dr']
########    
    # logging.basicConfig(filename='/home/jiangnan/new_xlandmini/xland-minigrid/training/levels_scores.log', level=logging.INFO, 
    #                 format='%(asctime)s - %(message)s')
    # logging.info(f'Levels: {levels_info}')
    # logging.info(f'Scores: {scores_info}')
    # logging.info("finished logging")
    # print("levels_info:",levels_info)
    # print('levels_info:',levels_info)
    print('num_replay:',num_replay_info)
    print('num_dr:',num_dr_info)
    print('episode:',episodecount_info)
    # print('levels_dr:',levels_dr_info)
    # print('shape of levels_dr:',levels_dr_info.shape)
########    
    # wandb.log({"levels_info": levels_info.tolist()})
    # wandb.log({'socres_info':scores_info.tolist()})
    total_transitions = 0
    for i in range(config.num_meta_updates):
        total_transitions += config.num_steps_per_env * config.num_envs_per_device * jax.local_device_count()
        info = jtu.tree_map(lambda x: x[i].item(), loss_info)
        # levels = levels_info.tolist()
        # scores = scores_info.tolist()
        info["transitions"] = total_transitions
        # info["levels"] = levels
        # info["scores"] = scores
        
        wandb.log(info)
    
    

    run.summary["training_time"] = elapsed_time
    run.summary["steps_per_second"] = (config.total_timesteps_per_device * jax.local_device_count()) / elapsed_time

    if config.checkpoint_path is not None:
        checkpoint = {"config": asdict(config), "params": unreplicate(train_info)["state"].params}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(checkpoint)
        orbax_checkpointer.save(config.checkpoint_path, checkpoint, save_args=save_args)

    print("Final return: ", float(loss_info["eval/returns_mean"][-1]))
    run.finish()


if __name__ == "__main__":
    train()

