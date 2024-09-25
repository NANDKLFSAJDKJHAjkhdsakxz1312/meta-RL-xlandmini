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
        
        
        B ,S,_,_,_= inputs["obs_img"].shape[:]
       
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
        obs_emb = img_encoder(inputs['obs_img']).reshape(B,S,-1)
        
        # obs_emb = jnp.repeat(obs_emb[:, jnp.newaxis, :], 1, axis=1) 
        # jax.debug.print("obs_emb: {img}", img=obs_emb)
        # breakpoint()
        dir_emb = direction_encoder(inputs["obs_dir"])
        
        act_emb = action_encoder(inputs["prev_action"])
        
        # breakpoint()
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

global_obs_dic = {}
from src.xminigrid.core.constants import Tiles,Colors

import jax
import jax.numpy as jnp

# def ssp_encoder(inputs, ssp_dim: int, length_scale: int, env_grid_size: int, rng_key) -> jnp.ndarray:
#     B, S, H, W, _ = inputs.shape  # 包含序列长度 S

#     NUM_TILES = len(Tiles.__annotations__)  # 替换为实际的类别数量
#     NUM_COLORS = len(Colors.__annotations__)  # 替换为实际的类别数量

#     NUM_CLASSES = NUM_TILES * NUM_COLORS

#     # 创建 SSP 空间
#     ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=ssp_dim, length_scale=length_scale,
#                                   domain_bounds=jnp.array([[0, env_grid_size], [0, env_grid_size]]))

#     # 生成坐标网格
#     x_coords, y_coords = jnp.meshgrid(jnp.arange(0, env_grid_size), jnp.arange(0, env_grid_size), indexing='ij')
#     coords = jnp.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)
#     ssp_grid = ssp_space.encode(coords)
#     ssp_grid = ssp_grid.reshape((env_grid_size, env_grid_size, -1))

#     # 创建随机向量作为类别向量
#     rng_keys = jax.random.split(rng_key, NUM_CLASSES)
#     label_vectors = jax.vmap(lambda key: jax.random.normal(key, (ssp_dim,)))(rng_keys)  # [NUM_CLASSES, ssp_dim]

#     # 创建类别映射数组
#     tile_color_to_class_index_array = -jnp.ones((NUM_TILES, NUM_COLORS), dtype=jnp.int32)
#     index = 0
#     for i in range(NUM_TILES):
#         for j in range(NUM_COLORS):
#             tile_color_to_class_index_array = tile_color_to_class_index_array.at[i, j].set(index)
#             index += 1
    
#     # 定义处理单个批次和单个时间步的函数
#     def process_single_time_step(single_time_step_inputs):
#         # 获取 tile 和 color 标签
#         tile_labels = single_time_step_inputs[:, :, 0].astype(jnp.int32)  # [H, W]
#         color_labels = single_time_step_inputs[:, :, 1].astype(jnp.int32)  # [H, W]

#         # 获取类别索引
#         class_indices = tile_color_to_class_index_array[tile_labels, color_labels]  # [H, W]

#         x_indices, y_indices = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')  # [H, W]

#         # 将数组展平
#         class_indices_flat = class_indices.reshape(-1)
#         x_indices_flat = x_indices.reshape(-1)
#         y_indices_flat = y_indices.reshape(-1)

#         label_ssp = label_vectors[class_indices_flat]  # [H*W, ssp_dim]
#         loc_ssp = ssp_grid[x_indices_flat, y_indices_flat]  # [H*W, ssp_dim]

#         # 绑定位置和类别向量
#         bund_vectors = ssp_space.bind(label_ssp, loc_ssp)  # [H*W, ssp_dim]

#         # 对所有绑定向量求和
#         env_ssp = jnp.sum(bund_vectors, axis=0)  # [ssp_dim]

#         return env_ssp

#     # 处理所有批次和时间步
#     global_env_ssp = jax.vmap(jax.vmap(process_single_time_step))(inputs)  # [B, S, ssp_dim]

#     return global_env_ssp

NUM_TILES = len(Tiles.__annotations__)  # 替换为实际的类别数量
NUM_COLORS = len(Colors.__annotations__)  # 替换为实际的类别数量

NUM_CLASSES = NUM_TILES * NUM_COLORS
ssp_dim = 1015
length_scale = 5
env_grid_size = 9
RNG = jax.random.PRNGKey(0)
# 创建 SSP 空间
ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=ssp_dim, length_scale=length_scale,
                                domain_bounds=jnp.array([[0, env_grid_size], [0, env_grid_size]]))

# 生成坐标网格
x_coords, y_coords = jnp.meshgrid(jnp.arange(0, env_grid_size), jnp.arange(0, env_grid_size), indexing='ij')
coords = jnp.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)
ssp_grid = ssp_space.encode(coords)

# ssp_grid = ssp_grid.reshape((env_grid_size, env_grid_size, -1))

# 创建随机向量作为类别向量
CLASS_LST = [f"TILE_{i}_COLOR_{j}" for i in range(NUM_TILES) for j in range(NUM_COLORS)]




vocab = spa.Vocabulary(dimensions=ssp_dim, pointer_gen=RNG)
for i, class_name in enumerate(CLASS_LST):
    vector = vocab.algebra.create_vector(ssp_dim, properties={"positive", "unitary"})
    vocab.add(f"{class_name}", vector)

vocab_vectors = jnp.array(vocab.vectors)

# rng_keys = jax.random.split(RNG, NUM_CLASSES)
# label_vectors = jax.vmap(lambda key: jax.random.normal(key, (ssp_dim,)))(rng_keys)  # [NUM_CLASSES, ssp_dim]

# 创建类别映射数组
tile_color_to_class_index_array = -jnp.ones((NUM_TILES, NUM_COLORS), dtype=jnp.int32)
index = 0
for i in range(NUM_TILES):
    for j in range(NUM_COLORS):
        tile_color_to_class_index_array = tile_color_to_class_index_array.at[i, j].set(index)
        index += 1


def ssp_encoder(inputs) -> jnp.ndarray:
    
    B, S, H, W, _ = inputs.shape  # 包含序列长度 S

    # 定义处理单个批次的函数，处理该批次内的所有时间步
    def process_single_batch(batch_inputs):
        # 初始化累加的 SSP 向量为全 0，只对一个 batch 初始化一次
        init_carry = jnp.zeros((ssp_grid.shape[1],))  # [ssp_dim]
        
        # 初始化计数器
        count_times_init = 0
        count_binds_init = 0
        
        # 定义处理单个时间步的函数
        def process_single_time_step(carry_and_counts, single_time_step_inputs):
            carry, count_times, count_binds = carry_and_counts

            # 获取 tile 和 color 标签
            count_times += 1  # 更新计数器
            tile_labels = single_time_step_inputs[:, :, 0].astype(jnp.int32)  # [H, W]
            color_labels = single_time_step_inputs[:, :, 1].astype(jnp.int32)  # [H, W]

            # 创建掩码来标识无效（-1）的位置
            valid_mask = (tile_labels != 200) & (color_labels != 200)

            # 只有在 tile_labels 和 color_labels 不为 -1 时才查找 class_indices
            class_indices = jnp.where(
                valid_mask,
                tile_color_to_class_index_array[tile_labels, color_labels],
                200  # 如果无效，设置为 -1，表示无效的 class_index
            )  # [H, W]
            
            # 将 class_indices 展平
            class_indices_flat = class_indices.reshape(-1)

            # 生成一维索引（将二维坐标转换为一维坐标）
            flat_indices = jnp.arange(H * W)  # 对应 (x, y) 的展平索引

            # 使用 scan 逐步处理每个 class_index
            def scan_fn(carry_and_counts, i):
                carry, count_binds = carry_and_counts
                class_index = class_indices_flat[i]
                
                def bind_fn(carry_and_counts):
                    carry, count_binds = carry_and_counts

                    # 绑定类别向量和位置向量
                    label_ssp = vocab_vectors[class_index]
                    loc_ssp = ssp_grid[i]                   # 直接使用一维索引 i 索引 ssp_grid
                   
                    
                    bund_vector = ssp_space.bind(label_ssp, loc_ssp) 
                    bund_vector = jnp.squeeze(bund_vector) 
                    
                    count_binds += 1
                    
                    
                    return (carry + bund_vector, count_binds)

                # 仅当 class_index 不为 -1 时进行绑定
                carry, count_binds = jax.lax.cond((class_index != 200) & (class_index != 19)&(class_index != 30), bind_fn, lambda carry_and_counts: carry_and_counts, (carry, count_binds))
                
                return (carry, count_binds ), None  # 返回 carry 作为结果，用于保存每个时间步的 SSP 向量

            # 对当前时间步内的所有有效 class_index 进行处理
            (carry, count_binds), _ = jax.lax.scan(scan_fn, (carry, count_binds), flat_indices)
            # carry = ssp_space.normalize(carry)
            
            return (carry, count_times, count_binds),carry # 返回每个时间步的 SSP 向量

        # 使用 scan 来处理该 batch 中的所有时间步
        final_carry_and_counts, ssp_vectors = jax.lax.scan(
            process_single_time_step, 
            (init_carry, count_times_init, count_binds_init), 
            batch_inputs
        )
        
        
        
        # 最终的累加向量
        # final_carry = ssp_space.normalize(final_carry)  # 可以在需要时进行最终的归一化

        # 返回每个时间步的 SSP 向量
        return ssp_vectors  # 返回 [S, ssp_dim]，而不是累加向量


    # 处理所有批次
    global_env_ssp = jax.vmap(process_single_batch)(inputs)  # [B, S, ssp_dim]
    
    
    return global_env_ssp



def return_ssp_encoder():
    return ssp_encoder
