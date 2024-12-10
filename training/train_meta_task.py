# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
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

import xminigrid
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper

from new_level_sampler import LevelSampler,make_level_generator,compute_max_returns,compute_score
import numpy as np
from enum import IntEnum

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1
# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)
########
Prioritization = Literal["rank", "topk"]

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
    name: str = "topdown_cnn_final"
    env_id: str = "XLand-MiniGrid-R1-9x9"
    benchmark_id: str = "high-1m"
    img_obs: bool = False 
    # agent
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    rule_emb_dim: int = 8
    goal_emb_dim: int = 8
    # training
    enable_bf16: bool = False
    num_envs: int = 4096
    num_steps_per_env: int = 2048
    num_steps_per_update: int =  32
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 1_000_000_000
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
    # env_params = env_params.replace(view_size=17)
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
        rule_emb_dim=config.rule_emb_dim,
        goal_emb_dim=config.goal_emb_dim,
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
    goal_shape = benchmark.goals.shape
    rule_shape = benchmark.rules.shape

    init_obs = {
        "obs_img": jnp.zeros((config.num_envs_per_device, 1,9,9,2)),
        "obs_dir": jnp.zeros((config.num_envs_per_device, 1, shapes["direction"])),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
        "rule": jnp.zeros((config.num_envs_per_device, 1,rule_shape[1],rule_shape[2]),dtype=jnp.int32),
        "goal": jnp.zeros((config.num_envs_per_device, 1,goal_shape[1]),dtype=jnp.int32),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
     
        # get the function to generate random level_id from benchmark
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
        
        level_sampler = LevelSampler(
        capacity=20480,
        replay_prob=config.replay_prob,
        staleness_coeff=config.staleness_coeff,
        minimum_fill_ratio=config.minimum_fill_ratio,
        prioritization=config.prioritization,
        prioritization_params={"temperature": config.temperature, "k": config.topk_k},
        duplicate_check=config.duplicate_check
        )
        # sample_random_level = make_level_generator(benchmark.num_rulesets())
        # pholder_level = sample_random_level(jax.random.PRNGKey(0))
        # sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})

       
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

                rulesets = jax.vmap(benchmark.get_ruleset)(levels) # change the random ruleset_rng to levels
                meta_env_params = env_params.replace(ruleset=rulesets)

                timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_rng)
                prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
                prev_reward = jnp.zeros(config.num_envs_per_device)

                # INNER TRAIN LOOP
                def _update_step(runner_state, _):
                    # COLLECT TRAJECTORIES
                    def _env_step(runner_state, _):
                        rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state

                        # SELECT ACTION
                        rng, _rng = jax.random.split(rng)
                        dist, value, hstate = train_state.apply_fn(
                            train_state.params,
                            {
                                # [batch_size, seq_len=1, ...]
                                # "obs_img": prev_timestep.observation["img"][:, None],
                                "obs_img": prev_timestep.state.grid[:, None],
                                "obs_dir": prev_timestep.observation["direction"][:, None],
                                "prev_action": prev_action[:, None],
                                "prev_reward": prev_reward[:, None],
                                "rule": prev_timestep.state.rule_encoding[:, None],
                                "goal": prev_timestep.state.goal_encoding[:, None]
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
                            obs=prev_timestep.state.grid,
                            dir=prev_timestep.observation["direction"],
                            rule=prev_timestep.state.rule_encoding,
                            goal=prev_timestep.state.goal_encoding,
                            prev_action=prev_action,
                            prev_reward=prev_reward,
                        )
                        runner_state = (rng, train_state, timestep, action, timestep.reward, hstate)
                        return runner_state, transition

                    initial_hstate = runner_state[-1]
                    # transitions: [seq_len, batch_size, ...]
                    runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps_per_update)
                    # CALCULATE ADVANTAGE
                    rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
                    # calculate value of the last step for bootstrapping
                    _, last_val, _ = train_state.apply_fn(
                        train_state.params,
                        {
                            "obs_img": timestep.state.grid[:, None],
                            "obs_dir": timestep.observation["direction"][:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                            "rule": timestep.state.rule_encoding[:, None],
                            "goal": timestep.state.goal_encoding[:, None]
                        },
                        hstate,
                    )
                    advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)

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
                prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
                prev_reward = jnp.zeros(config.num_envs_per_device)

                # INNER TRAIN LOOP
                def _update_step(runner_state, _):
                    # COLLECT TRAJECTORIES
                    def _env_step(runner_state, _):
                        rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state

                        # SELECT ACTION
                        rng, _rng = jax.random.split(rng)
                        dist, value, hstate = train_state.apply_fn(
                            train_state.params,
                           {
                            # [batch_size, seq_len=1, ...]
                            # "obs_img": prev_timestep.observation["img"][:, None],
                            "obs_img": prev_timestep.state.grid[:, None],
                            "obs_dir": prev_timestep.observation["direction"][:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                            "rule": prev_timestep.state.rule_encoding[:, None],
                            "goal": prev_timestep.state.goal_encoding[:, None]
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
                        obs=prev_timestep.state.grid,
                        dir=prev_timestep.observation["direction"],
                        rule=prev_timestep.state.rule_encoding,
                        goal=prev_timestep.state.goal_encoding,
                        prev_action=prev_action,
                        prev_reward=prev_reward,
                    )
                        runner_state = (rng, train_state, timestep, action, timestep.reward, hstate)
                        return runner_state, transition

                    initial_hstate = runner_state[-1]
                    # transitions: [seq_len, batch_size, ...]
                    runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps_per_update)
                    # CALCULATE ADVANTAGE
                    rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
                    # calculate value of the last step for bootstrapping
                    _, last_val, _ = train_state.apply_fn(
                        train_state.params,
                        {
                            "obs_img": timestep.state.grid[:, None],
                            "obs_dir": timestep.observation["direction"][:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                            "rule": timestep.state.rule_encoding[:, None],
                            "goal": timestep.state.goal_encoding[:, None]
                        },
                        hstate,
                    )
                    advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)      
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
            branch,  # 这里的 branch 是一个布尔值
            lambda _: on_replay_levels(rng, train_state),
            lambda _: on_new_levels(rng, train_state),
            operand=None  # 传递给 lambda 的占位符参数
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
            idx = jnp.arange(level_sampler.capacity) < train_state.sampler["size"]  
            mean_score = (train_state.sampler["scores"] * idx).sum() / idx.sum()  

            weights = level_sampler.level_weights(train_state.sampler)  
            weighted_score = (train_state.sampler["scores"] * weights).sum()  
            loss_info.update(
                {
                    "eval/returns_mean": eval_stats.reward.mean(0),
                    "eval/returns_median": jnp.median(eval_stats.reward),
                    "eval/lengths": eval_stats.length.mean(0),
                    "eval/lengths_20percentile": jnp.percentile(eval_stats.length, q=20),
                    "eval/returns_20percentile": jnp.percentile(eval_stats.reward, q=20),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                    "levels":train_state.sampler["levels"].mean(),
                    "mean_scores":mean_score,
                    "max_scores":train_state.sampler["scores"].max(),
                    "weighted_scores":weighted_score
                    
                }
            )
            meta_state = (rng, train_state)
            jax.debug.print('scores:{}',train_state.sampler["scores"])
            jax.debug.print('scores_mean:{}',mean_score)
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
    episodecount_info = train_info["episode_count"]
    num_dr_info = train_info['num_dr']
    num_replay_info = train_info['num_replay']
    print('num_replay:',num_replay_info)
    print('num_dr:',num_dr_info)
    print('episode:',episodecount_info) 
    wandb.log({"levels_info": levels_info.tolist()})
    wandb.log({'socres_info':scores_info.tolist()})
    total_transitions = 0
    for i in range(config.num_meta_updates):
        total_transitions += config.num_steps_per_env * config.num_envs_per_device * jax.local_device_count()
        info = jtu.tree_map(lambda x: x[i].item(), loss_info)
        info["transitions"] = total_transitions
        
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