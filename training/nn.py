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
        obs_emb = img_encoder(inputs['obs_img']).reshape(B, S, -1)
        
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


ssp_grid = ssp_grid.reshape((env_grid_size, env_grid_size, -1))
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

# def ssp_encoder(inputs) -> jnp.ndarray:
    
#     B, S, H, W, _ = inputs.shape  # 包含序列长度 S

#     # 定义处理单个批次的函数，处理该批次内的所有时间步
#     def process_single_batch(batch_inputs):
#         # 初始化累加的 SSP 向量为全 0，只对一个 batch 初始化一次
#         init_carry = jnp.zeros((ssp_grid.shape[1],))  # [ssp_dim]
        
#         # 初始化计数器
#         count_times_init = 0
#         count_binds_init = 0
        
#         # 定义处理单个时间步的函数
#         def process_single_time_step(carry_and_counts, single_time_step_inputs):
#             carry, count_times, count_binds = carry_and_counts

#             # 获取 tile 和 color 标签
#             count_times += 1  # 更新计数器
#             tile_labels = single_time_step_inputs[:, :, 0].astype(jnp.int32)  # [H, W]
#             color_labels = single_time_step_inputs[:, :, 1].astype(jnp.int32)  # [H, W]

#             # 创建掩码来标识无效（-1）的位置
#             valid_mask = (tile_labels != 200) & (color_labels != 200)

#             # 只有在 tile_labels 和 color_labels 不为 -1 时才查找 class_indices
#             class_indices = jnp.where(
#                 valid_mask,
#                 tile_color_to_class_index_array[tile_labels, color_labels],
#                 200  # 如果无效，设置为 -1，表示无效的 class_index
#             )  # [H, W]
            
#             # 将 class_indices 展平
#             class_indices_flat = class_indices.reshape(-1)

#             # 生成一维索引（将二维坐标转换为一维坐标）
#             flat_indices = jnp.arange(H * W)  # 对应 (x, y) 的展平索引

#             # 使用 scan 逐步处理每个 class_index
#             def scan_fn(carry_and_counts, i):
#                 carry, count_binds = carry_and_counts
#                 class_index = class_indices_flat[i]
                
#                 def bind_fn(carry_and_counts):
#                     carry, count_binds = carry_and_counts

#                     # 绑定类别向量和位置向量
#                     label_ssp = vocab_vectors[class_index]
#                     loc_ssp = ssp_grid[i]                   # 直接使用一维索引 i 索引 ssp_grid
                   
                    
#                     bund_vector = ssp_space.bind(label_ssp, loc_ssp) 
#                     bund_vector = jnp.squeeze(bund_vector) 
                    
#                     count_binds += 1
                    
                    
#                     return (carry + bund_vector, count_binds)

#                 # 仅当 class_index 不为 -1 时进行绑定
#                 carry, count_binds = jax.lax.cond((class_index == 37) | (class_index == 138) | (class_index ==40 ) | (class_index ==78 ), bind_fn, lambda carry_and_counts: carry_and_counts, (carry, count_binds))
                
#                 return (carry, count_binds ), None  # 返回 carry 作为结果，用于保存每个时间步的 SSP 向量

#             # 对当前时间步内的所有有效 class_index 进行处理
#             (carry, count_binds), _ = jax.lax.scan(scan_fn, (carry, count_binds), flat_indices)
#             # carry = ssp_space.normalize(carry)
            
#             return (carry, count_times, count_binds),carry # 返回每个时间步的 SSP 向量

#         # 使用 scan 来处理该 batch 中的所有时间步
#         final_carry_and_counts, ssp_vectors = jax.lax.scan(
#             process_single_time_step, 
#             (init_carry, count_times_init, count_binds_init), 
#             batch_inputs
#         )
        
        
        
#         # 最终的累加向量
#         # final_carry = ssp_space.normalize(final_carry)  # 可以在需要时进行最终的归一化

#         # 返回每个时间步的 SSP 向量
#         return ssp_vectors  # 返回 [S, ssp_dim]，而不是累加向量


#     # 处理所有批次
#     global_env_ssp = jax.vmap(process_single_batch)(inputs)  # [B, S, ssp_dim]
    
    
#     return global_env_ssp

# def ssp_encoder(inputs) -> jnp.ndarray:
#     B, S, H, W, _ = inputs.shape  # 包含序列长度 S
    
#     def process_single_batch(batch_inputs):
#         # 初始化累加的 SSP 向量为全 0
#         init_carry = jnp.zeros((ssp_grid.shape[2],))  # [ssp_dim]
        
#         # 定义处理单个时间步的函数
#         def process_single_time_step(carry, single_time_step_inputs):
#             # 获取 tile 和 color 标签
#             tile_labels = single_time_step_inputs[..., 0].astype(jnp.int32)  # [H, W]
#             color_labels = single_time_step_inputs[..., 1].astype(jnp.int32)  # [H, W]
#             # jax.debug.print("tile:{x}",x=tile_labels)
#             # jax.debug.print("color:{x}",x=color_labels)

#             # 创建掩码来标识有效位置
#             valid_mask = (tile_labels != 0) & (color_labels != 0) &(tile_labels!=1) &(tile_labels!=2)# [H, W]
#             # jax.debug.print("vec:{x}",x=vocab_vectors)
#             # jax.debug.print("sspgrid:{x}",x=ssp_grid.reshape(H, W, -1))
#             # jax.debug.print("mask:{x}",x=valid_mask[..., None])
            
#             # jax.debug.print("mask:{x}",x=vocab_vectors[idx])
#             # 获取所有有效位置的 class_indices（无效位置标记为 0 向量）
#             class_indices = tile_color_to_class_index_array[tile_labels, color_labels]
#             label_ssps = jnp.where(
#                 valid_mask[..., None], 
#                 vocab_vectors[class_indices],  # 有效位置的向量
#                 jnp.zeros((H, W, ssp_grid.shape[2]))  # 无效位置为零向量
#             )  # [H, W, ssp_dim]
#             # jax.debug.print("class_indices:{x}",x=class_indices)
#             # jax.debug.print("label_ssps:{x}",x=label_ssps)

#             # 从 ssp_grid 中获取位置向量
#             loc_ssps = jnp.where(
#                 valid_mask[..., None], 
#                 ssp_grid,  # 有效位置的坐标向量
#                 jnp.zeros((H, W, ssp_grid.shape[2]))  # 无效位置为零向量
#             )  # [H, W, ssp_dim]
#             # jax.debug.print("loc_ssps:{x}",x=loc_ssps)
#             # jax.debug.print("y:{x}",x=ssp_grid[11])
#             idx = tile_color_to_class_index_array[6,6]
#             x = ssp_space.bind(vocab_vectors[idx],ssp_grid[1,2])
#             # jax.debug.print("selectedvec_6,6:{y}",y=ssp_grid[1,2])
#             idx = tile_color_to_class_index_array[11,6]
#             # jax.debug.print("selectedvec_11,6:{y}",y=ssp_grid[6,3])

#             x+=ssp_space.bind(vocab_vectors[idx],ssp_grid[6,3])
#             idx = tile_color_to_class_index_array[3,1]
#             # jax.debug.print("selectedvec_3,1:{y}",y=ssp_grid[2,3])

#             x+=ssp_space.bind(vocab_vectors[idx],ssp_grid[2,3])
#             idx = tile_color_to_class_index_array[3,4]
#             # jax.debug.print("selectedvec_3,4:{y}",y=ssp_grid[2,4])

#             x+=ssp_space.bind(vocab_vectors[idx],ssp_grid[2,4])
            
            
#             # jax.debug.print("x:{y}",y=x.squeeze())
#             # 对标签向量和位置向量进行绑定操作
#             # binding_vectors = ssp_space.bind(label_ssps, loc_ssps)  # [H, W, ssp_dim]
#             binding_vectors = jnp.where(
#             valid_mask[..., None], 
#             ssp_space.bind(label_ssps, loc_ssps),  # 只对有效位置绑定
#             jnp.zeros_like(label_ssps)  # 无效位置保持为零
#             )
#             # jax.debug.print("bundvec:{x}",x=jnp.where(valid_mask[...,None],label_ssps,jnp.zeros_like(label_ssps)))
#             # for i in range(ssp_grid.shape[0]):
#             #     for j in range(ssp_grid.shape[1]):
#             #         # 获取每个位置的向量
#             #         vector =ssp_grid[i, j]
#             #         # 打印位置和对应的向量
#             #         jax.debug.print("位置 ({i}, {j}) 的向量: {x}", i=i, j=j, x=vector)
            
#             # 累加所有有效位置的绑定向量
#             carry = carry + binding_vectors.sum(axis=(0, 1))  # [ssp_dim]
        
#             # carry = carry + x.squeeze() # [ssp_dim]
#             return x.squeeze(),x.squeeze()  # 返回 carry 作为结果，用于保存每个时间步的 SSP 向量

#         # 使用 scan 来处理该 batch 中的所有时间步
#         final_carry, ssp_vectors = jax.lax.scan(
#             process_single_time_step, 
#             init_carry, 
#             batch_inputs
#         )
#         jax.debug.print("ssp_vectors:{y}",y=ssp_vectors)
#         return ssp_vectors  # 返回 [S, ssp_dim]，而不是累加向量

#     # 处理所有批次
#     global_env_ssp = jax.vmap(process_single_batch)(inputs)  # [B, S, ssp_dim]
#     jax.debug.print("global_env_ssp:{y}",y=global_env_ssp)

#     return global_env_ssp
# 

# 
def ssp_encoder(inputs) -> jnp.ndarray:
    B, S, H, W, _ = inputs.shape  # 包含序列长度 S
    
    def process_single_batch(batch_inputs):
        # 初始化累加的 SSP 向量为全 0
        init_carry = jnp.zeros((ssp_grid.shape[2],))  # [ssp_dim]
        
        # 定义处理单个时间步的函数
        def process_single_time_step(carry, single_time_step_inputs):
            # 获取 tile 和 color 标签
            tile_labels = single_time_step_inputs[..., 0].astype(jnp.int32)  # [H, W]
            color_labels = single_time_step_inputs[..., 1].astype(jnp.int32)  # [H, W]

            # 创建掩码来标识有效位置
            valid_mask = (tile_labels != 0) & (color_labels != 0) & (tile_labels != 1) & (tile_labels != 2)  # [H, W]

            # 获取 class_indices 和相应的标签向量
            class_indices = tile_color_to_class_index_array[tile_labels, color_labels]
            label_ssps = jnp.where(
                valid_mask[..., None], 
                vocab_vectors[class_indices],  # 有效位置的向量
                jnp.zeros((H, W, ssp_grid.shape[2]))  # 无效位置为零向量
            )  # [H, W, ssp_dim]

            # 获取位置向量
            loc_ssps = jnp.where(
                valid_mask[..., None], 
                ssp_grid,  # 有效位置的坐标向量
                jnp.zeros((H, W, ssp_grid.shape[2]))  # 无效位置为零向量
            )  # [H, W, ssp_dim]

            # 使用 vmap 将 bind 函数应用到每个有效位置
            # binding_vectors = jnp.zeros((H, W, ssp_grid.shape[2]))  # 初始化绑定向量矩阵
            
            # 遍历每个位置的标签和坐标，并在每个有效位置上执行绑定操作
            def bind_single_position(i, j):
                return jax.lax.cond(
                    valid_mask[i, j],
                    lambda: ssp_space.bind(label_ssps[i, j], loc_ssps[i, j]).squeeze(),
                    lambda: jnp.zeros(ssp_grid.shape[2])
                )

            # 应用 vmap 逐元素绑定
            binding_vectors = jax.vmap(lambda i: jax.vmap(lambda j: bind_single_position(i, j))(jnp.arange(W)))(jnp.arange(H))

            # 累加所有有效位置的绑定向量
            carry = carry + binding_vectors.sum(axis=(0, 1))  # [ssp_dim]

            return carry, carry  # 返回 carry 作为结果，用于保存每个时间步的 SSP 向量

        # 使用 scan 来处理该 batch 中的所有时间步
        final_carry, ssp_vectors = jax.lax.scan(
            process_single_time_step, 
            init_carry, 
            batch_inputs
        )
     
        return ssp_vectors  # 返回 [S, ssp_dim]，而不是累加向量

    # 处理所有批次
    global_env_ssp = jax.vmap(process_single_batch)(inputs)  # [B, S, ssp_dim]
   

    return global_env_ssp



def return_ssp_encoder():
    return ssp_encoder
