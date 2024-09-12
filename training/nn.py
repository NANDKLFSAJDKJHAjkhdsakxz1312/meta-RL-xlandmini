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


from xminigrid.core.constants import NUM_COLORS, NUM_TILES
from utils_ssp import HexagonalSSPSpace
import sys
import os

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


class ActorCriticInput(TypedDict):
    obs_img: jax.Array
    obs_dir: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array


class ActorCriticRNN(nn.Module):
    num_actions: int
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
        
        
        B, S = inputs["obs_img"].shape[:2]
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
            
            img_encoder = return_ssp_encoder()
            
            
            # img_encoder = jax.vmap(
            #     ssp_encoder, 
            #     in_axes=(0, None, None, None, None, None)
            # )
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
        init_obs = {
                "obs_img": jnp.zeros((512, 1, 5,5,2)),
                "obs_dir": jnp.zeros((512, 1,1)),
                "prev_action": jnp.zeros((512, 1), dtype=jnp.int32),
                "prev_reward": jnp.zeros((512, 1)),
            }
        key = jax.random.PRNGKey(0)
        step_num = jnp.array(0)
        grid = jnp.zeros((9, 9))
        agent = AgentState()
        goal_encoding = jnp.zeros((10,))
        rule_encoding = jnp.zeros((10,))
        carry = EnvCarry()

        state = State(
            key=key,
            step_num=step_num,
            grid=grid,
            agent=agent,
            goal_encoding=goal_encoding,
            rule_encoding=rule_encoding,
            carry=carry
        )

        
        step_type = StepType.FIRST
        reward = jnp.array(1.0)
        discount = jnp.array(0.99)
        

        
        timestep = TimeStep(
            state=state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=init_obs
        )
        timestep = timestep
        # obs_emb = img_encoder(inputs["obs_img"].astype(jnp.int32)).reshape(B, S, -1)
        obs_emb = img_encoder(inputs['obs_img'], 1015, 5, 9, timestep, jax.random.PRNGKey(0)).astype(jnp.int32).reshape(B, S,-1)
        
        dir_emb = direction_encoder(inputs["obs_dir"])
        act_emb = action_encoder(inputs["prev_action"])
        
       
        # [batch_size, seq_len, hidden_dim + 2 * act_emb_dim + 1]
        out = jnp.concatenate(
            [obs_emb, dir_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1
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
local_obs_dic = {}
global_obs_dic = {}



def ssp_encoder(inputs, ssp_dim: int, length_scale: int, env_grid_size: int, timestep: TimeStep, rng=jax.random.PRNGKey(0)) -> jax.Array:
    B = inputs.shape[0]  # 获取批次大小
    
    # 初始化 SSP 空间
    ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=ssp_dim, length_scale=length_scale, 
                                  domain_bounds=jnp.array([[0, env_grid_size], [0, env_grid_size]]))
    
    x_coords, y_coords = jnp.meshgrid(jnp.arange(0, env_grid_size), jnp.arange(0, env_grid_size), indexing='ij')
    coords = jnp.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)
    ssp_grid = ssp_space.encode(coords)
    ssp_grid = ssp_grid.reshape((env_grid_size, env_grid_size, -1))
    
    def process_single_sample(input_sample):
        shape = input_sample.shape  # 获取该批次样本的形状
        
        local_obs_coords = []
        object_labels = []
        
        # 将 `inputs` 中的局部坐标和标签转化为 JAX 兼容的形式
        for idx in itertools.product(range(shape[1]), range(shape[2])):
            object_label = jnp.array(input_sample[0, idx[0], idx[1], :])
            object_labels.append(object_label)
            local_obs_coords.append((idx[0], idx[1]))
        
        local_obs_coords = jnp.array(local_obs_coords)  # 转化为 JAX 数组
        
        object_labels = jnp.array(object_labels)
        vocab = spa.Vocabulary(dimensions=ssp_dim, pointer_gen=rng)
        def ssp_label(object_label):
            vector = vocab.algebra.create_vector(ssp_dim, properties={"positive", "unitary"})
            vocab.add(f"{object_label}", vector)
            return vocab
        vocab = jax.vmap(ssp_label)(object_labels)
        # 处理全局坐标并生成对应的 SSP 编码
        def get_global_obs(local_obs):
            agent_position = timestep.state.agent.position
            direction = timestep.state.agent.direction
            def case_0():
                return jnp.array([local_obs[0] - 4 + agent_position[0], local_obs[1] - 2 + agent_position[1]])
            
            def case_1():
                global_coord = jnp.array([local_obs[1]-2 ,-(local_obs[0]-4)])
                return global_coord + agent_position
            
            def case_2():
                global_coord = jnp.array([-(local_obs[0]-4), -(local_obs[1]-2)])
                return global_coord + agent_position
            
            def case_3():
                global_coord = jnp.array([ -(local_obs[1]-2),local_obs[0]-4])
                return global_coord + agent_position

            return jax.lax.switch(direction, [case_0, case_1, case_2, case_3])
        

        global_obs_coords = jnp.array(jax.vmap(get_global_obs)(local_obs_coords))
        
        
        # 将全局观察字典转换为具体化的 SSP 表示
        global_env_ssp = jnp.zeros(ssp_dim)
        for global_coord, object_label in zip(global_obs_coords, object_labels):
            loc_ssp = ssp_grid[global_coord[0], global_coord[1]]
            label_ssp = vocab[object_label].v
            bund_vector = ssp_space.bind(label_ssp, loc_ssp)
            global_env_ssp += bund_vector.squeeze()

        return global_env_ssp

    # 使用 `vmap` 并行处理每个批次的样本
    global_env_ssp_batch = jax.vmap(process_single_sample)(inputs)
    
    return global_env_ssp_batch

def return_ssp_encoder():
    return ssp_encoder