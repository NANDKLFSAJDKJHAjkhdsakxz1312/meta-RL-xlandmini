# Model adapted from minigrid baselines:
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import math
from typing import Optional, TypedDict

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from flax.typing import Dtype
import numpy as np
import jax
import jax.numpy as jnp
from jax.profiler import TraceAnnotation
from xminigrid.core.constants import NUM_COLORS, NUM_TILES

import sys
import os
from xminigrid.core.rules import NUM_RULES
from xminigrid.core.goals import NUM_GOALS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.xminigrid.types  import TimeStep
import nengo_spa as spa
from src.xminigrid.types import TimeStep, State, AgentState, EnvCarry, StepType
import itertools

class GRU(nn.Module):
    hidden_dim: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        seq_len, input_dim = xs.shape
        
        # this init might not be optimal, for example bias for reset gate should be -1 (for now ok)
        Wi = self.param(
            "Wi",
            glorot_normal(in_axis=1, out_axis=0),
            (self.hidden_dim * 3, input_dim),
            self.param_dtype,
        )
        Wh = self.param(
            "Wh",
            orthogonal(column_axis=0),
            (self.hidden_dim * 3, self.hidden_dim),
            self.param_dtype,
        )
        bi = self.param("bi", zeros_init(), (self.hidden_dim * 3,), self.param_dtype)
        bn = self.param("bn", zeros_init(), (self.hidden_dim,), self.param_dtype)

        def _step_fn(h, x):
            igates = jnp.split(Wi @ x + bi, 3)
            hgates = jnp.split(Wh @ h, 3)
            reset = nn.sigmoid(igates[0] + hgates[0])
            update = nn.sigmoid(igates[1] + hgates[1])
            new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
            next_h = (1 - update) * new + update * h
            return next_h, next_h

        # cast to the computation dtype
        xs, init_state, Wi, Wh, bi, bn = promote_dtype(
            xs, init_state, Wi, Wh, bi, bn, dtype=self.dtype
        )
        last_state, all_states = jax.lax.scan(_step_fn, init=init_state, xs=xs)
        return all_states, last_state


class RNNModel(nn.Module):
    hidden_dim: int
    num_layers: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        # xs: [seq_len, input_dim]
        # init_state: [num_layers, hidden_dim]
        outs, states = [], []
        for layer in range(self.num_layers):
            xs, state = GRU(self.hidden_dim, self.dtype, self.param_dtype)(
                xs, init_state[layer]
            )
            outs.append(xs)
            states.append(state)
        # sum outputs from all layers, kinda like in ResNet
        return jnp.array(outs).sum(0), jnp.array(states)


BatchedRNNModel = nn.vmap(
    RNNModel,
    variable_axes={"params": None},
    split_rngs={"params": False},
    axis_name="batch",
)


class EmbeddingEncoder(nn.Module):
    emb_dim: int = 16
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, img):
        entity_emb = nn.Embed(NUM_TILES, self.emb_dim, self.dtype, self.param_dtype)
        color_emb = nn.Embed(NUM_COLORS, self.emb_dim, self.dtype, self.param_dtype)
        # [..., channels]
        img_emb = jnp.concatenate(
            [
                entity_emb(img[..., 0]),
                color_emb(img[..., 1]),
            ],
            axis=-1,
        )
        return img_emb

class RuleEncoder(nn.Module):
    emb_dim: int = 16
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, rules):
        rule_id_emb = nn.Embed(NUM_RULES, self.emb_dim, self.dtype, self.param_dtype)
        rule_tile_emb = nn.Embed(NUM_TILES, self.emb_dim, self.dtype, self.param_dtype)
        rule_color_emb = nn.Embed(NUM_COLORS, self.emb_dim, self.dtype, self.param_dtype)

        # [..., channels]
        rule_emb = jnp.concatenate(
            [
                rule_id_emb(rules[..., 0]),
                rule_tile_emb(rules[..., 1]),
                rule_color_emb(rules[..., 2]),
                rule_tile_emb(rules[..., 3]),
                rule_color_emb(rules[..., 4]),
                rule_tile_emb(rules[..., 5]),
                rule_color_emb(rules[..., 6]),
            ],
            axis=-1,
        )
        B, S = rules.shape[:2]
        rule_emb = rule_emb.reshape(B, S, -1)
        return rule_emb
    
class GoalEncoder(nn.Module):
    emb_dim: int = 16
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, rules):
        goal_id_emb = nn.Embed(NUM_GOALS, self.emb_dim, self.dtype, self.param_dtype)
        goal_tile_emb = nn.Embed(NUM_TILES, self.emb_dim, self.dtype, self.param_dtype)
        goal_color_emb = nn.Embed(NUM_COLORS, self.emb_dim, self.dtype, self.param_dtype)

        # [..., channels]
        goal_emb = jnp.concatenate(
            [
                goal_id_emb(rules[..., 0]),
                goal_tile_emb(rules[..., 1]),
                goal_color_emb(rules[..., 2]),
                goal_tile_emb(rules[..., 3]),
                goal_color_emb(rules[..., 4]),
            ],
            axis=-1,
        )
        return goal_emb


class AfterSSPEncoder(nn.Module):
    after_ssp_dim: int
    after_ssp_dim2: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        # 第一个 Dense 层
        x = nn.Dense(
            features=self.after_ssp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(inputs)
        x = nn.relu(x)  # 激活函数

        # 第二个 Dense 层
        x = nn.Dense(
            features=self.after_ssp_dim2,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.relu(x)  # 激活函数

        return x

class ActorCriticInput(TypedDict):
    obs_img: jax.Array
    # obs_img_cnn: jax.Array
    obs_dir: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array


class ActorCriticRNN(nn.Module):
    num_actions: int
    rule_emb_dim: int = 64
    goal_emb_dim: int = 16
    after_ssp_dim: int = 256
    after_ssp_dim2: int = 256
    # lstm_hidden_dim: int = 256
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    img_obs: bool = False
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self, inputs: ActorCriticInput, hidden: jax.Array
    ) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        # B ,S,_= inputs["obs_img"].shape[:]
        

        
        
        # encoder from https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        if self.img_obs:
            
            # img_encoder = ssp_encoder(inputs,ssp_dim=1015,length_scale=5,env_grid_size=9,timestep=timestep,rng=jax.random.PRNGKey(0))
            img_encoder = nn.Sequential(
                [
                    nn.Conv(
                        16,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                ]
            )
        else:
            # img_encoder = nn.Sequential(
            #     [
            #         # For small dims nn.Embed is extremely slow in bf16, so we leave everything in default dtypes
            #         EmbeddingEncoder(emb_dim=self.obs_emb_dim),
            #         nn.Conv(
            #             16,
            #             (2, 2),
            #             padding="VALID",
            #             kernel_init=orthogonal(math.sqrt(2)),
            #             dtype=self.dtype,
            #             param_dtype=self.param_dtype,
            #         ),
            #         nn.relu,
            #         nn.Conv(
            #             32,
            #             (2, 2),
            #             padding="VALID",
            #             kernel_init=orthogonal(math.sqrt(2)),
            #             dtype=self.dtype,
            #             param_dtype=self.param_dtype,
            #         ),
            #         nn.relu,
            #         nn.Conv(
            #             64,
            #             (2, 2),
            #             padding="VALID",
            #             kernel_init=orthogonal(math.sqrt(2)),
            #             dtype=self.dtype,
            #             param_dtype=self.param_dtype,
            #         ),
            #         nn.relu,
            #     ]
            # )
            cnn_encoder = nn.Sequential(
                [
                    # For small dims nn.Embed is extremely slow in bf16, so we leave everything in default dtypes
                    EmbeddingEncoder(emb_dim=self.obs_emb_dim),
                    nn.Conv(
                        16,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        64,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                ]
            )
            
            after_ssp_encoder = AfterSSPEncoder(
            after_ssp_dim=self.after_ssp_dim,
            after_ssp_dim2=self.after_ssp_dim2,
            dtype=self.dtype,
            param_dtype=self.param_dtype
            )

        
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)
        direction_encoder = nn.Dense(
            self.action_emb_dim, dtype=self.dtype, param_dtype=self.param_dtype
        )
        
        
        rnn_core = BatchedRNNModel(
            self.rnn_hidden_dim,
            self.rnn_num_layers,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        actor = nn.Sequential(
            [
                nn.Dense(
                    self.head_hidden_dim,
                    kernel_init=orthogonal(2),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.tanh,
                nn.Dense(
                    self.num_actions,
                    kernel_init=orthogonal(0.01),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
            ]
        )
        critic = nn.Sequential(
            [
                nn.Dense(
                    self.head_hidden_dim,
                    kernel_init=orthogonal(2),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.tanh,
                nn.Dense(
                    1,
                    kernel_init=orthogonal(1.0),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
            ]
        )
        # obs_emb = img_encoder(inputs["obs_img"].astype(jnp.int32)).reshape(B, S, -1)

        # obs_emb = img_encoder(inputs["obs_img"].astype(jnp.int32)).reshape(B, S, -1)  .reshape(B, S, -1)
       

        # cnn_emb = cnn_encoder(inputs['obs_img_cnn']).reshape(B, S, -1)
        # obs_emb = jnp.concatenate(
        #     [obs_emb,cnn_emb], axis=-1
        # )
        
        # # 添加 LSTM
        # obs_emb = obs_emb.reshape(obs_emb.shape[0], 1, -1)  # 添加伪序列维度
        # # 在构建 nn.scan 时传递 LSTMCell 的隐藏维度
        # lstm = nn.scan(
        #     nn.LSTMCell(features=self.lstm_hidden_dim),  # 指定隐藏维度
        #     variable_broadcast='params',
        #     split_rngs={'params': False},
        # )(name="lstm")

        # lstm_state = nn.LSTMCell.initialize_carry(RNG, (obs_emb.shape[0],), self.lstm_hidden_dim)
        # obs_emb, _ = lstm(lstm_state, obs_emb)
        

       
        # obs_emb = jnp.repeat(obs_emb[:, jnp.newaxis, :], 1, axis=1) 
        # jax.debug.print("obs_emb: {img}", img=obs_emb)
        # breakpoint()
        # 使用编码器处理输入
        obs_emb = after_ssp_encoder(inputs["obs_img"])
        dir_emb = nn.relu(direction_encoder(inputs["obs_dir"]))
        
        act_emb = nn.relu(action_encoder(inputs["prev_action"]))
        rule_encoder = RuleEncoder(self.rule_emb_dim)
        
        
        goal_encoder = GoalEncoder(self.goal_emb_dim)
        rule_emb = rule_encoder(inputs["rule"])
        goal_emb = goal_encoder(inputs["goal"])

        
        # breakpoint()
        # [batch_size, seq_len, hidden_dim + 2 * act_emb_dim + 1]
        out = jnp.concatenate(
            [obs_emb, dir_emb, act_emb, inputs["prev_reward"][..., None],rule_emb,goal_emb], axis=-1
        )


        
        # core networks
        out, new_hidden = rnn_core(out, hidden)
    
        # casting to full precision for the loss, as softmax/log_softmax
        # (inside Categorical) is not stable in bf16
        logits = actor(out).astype(jnp.float32)
        dist = distrax.Categorical(logits=logits)
        values = critic(out)
        return dist, jnp.squeeze(values, axis=-1), new_hidden 

    def initialize_carry(self, batch_size):
        return jnp.zeros(
            (batch_size, self.rnn_num_layers, self.rnn_hidden_dim), dtype=self.dtype
        )

