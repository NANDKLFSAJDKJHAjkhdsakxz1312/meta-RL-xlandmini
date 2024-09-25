from nn import ssp_encoder
import numpy as np
from flax import struct
from typing import Dict
import jax
import jax.numpy as jnp
import nn
from utils_ssp import HexagonalSSPSpace
import matplotlib.pyplot as plt

# enb_ssp = np.zeros(nn.ssp_dim)
# label_ssp = nn.vocab_vectors[138]
# loc_ssp = nn.ssp_grid[57]
# bund = nn.ssp_space.bind(label_ssp,loc_ssp)
# enb_ssp += bund
# bund2 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[80])
# enb_ssp += bund2
# bund3 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[79])
# enb_ssp += bund3
# bund4 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[78])
# enb_ssp += bund4
# bund5 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[77])
# enb_ssp += bund5
# bund6 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[76])
# enb_ssp += bund6
# bund7 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[75])
# enb_ssp += bund7
# bund8 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[74])
# enb_ssp += bund8
# bund9 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[73])
# enb_ssp += bund9
# bund10 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[72])
# enb_ssp += bund10
# bund11 = nn.ssp_space.bind(nn.vocab_vectors[30],nn.ssp_grid[71])
# enb_ssp += bund11
# inv_ssp = nn.ssp_space.invert(label_ssp)
# out  = nn.ssp_space.bind(enb_ssp,inv_ssp)
# sims = out@nn.ssp_grid.T
# sims_map = sims.reshape(9,9)
# pred_loc = np.array(np.unravel_index(np.argmax(sims_map), sims_map.shape)) 
# print(f'{57} predicted location: {tuple(pred_loc)}')

# plt.imshow(sims_map, extent=[0,9,9,0])
# plt.xticks([0,9])
# plt.yticks([0,9])
# plt.gca().xaxis.set_ticks_position('top')  #
# plt.gca().xaxis.set_label_position('top')  #
# plt.xlabel('X')
# plt.ylabel('Y')
# # plt.gca().invert_yaxis()
# plt.colorbar(label='Similarity')
# plt.show()
# breakpoint()
class prev_timestep(struct.PyTreeNode):
    observation: Dict[str, jnp.ndarray]

# 创建 prev_timestep 类的实例
prev_ts = prev_timestep(observation={})

prev_action = jnp.array([[2,2,2,1]]) 
action_index = prev_action[1].astype(int)
print(action_index)
data2 = [
    [[200, 200], [200, 200], [200, 200], [200, 200], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6]],
    [[200, 200], [200, 200], [200, 200], [200, 200], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[200, 200], [200, 200], [200, 200], [200, 200], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[200, 200], [200, 200], [200, 200], [200, 200], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[200, 200], [200, 200], [200, 200], [200, 200], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[200, 200], [200, 200], [200, 200], [200, 200], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200]],
    [[200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200]],
    [[200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200]],
]
data1 = [[[200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200], [200, 200]],
                  [[2, 6], [1, 7], [6, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [3, 1], [3, 4], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [11, 6], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6]]]
# 转换为 JAX 数组

x  = jnp.array([[data1,data2]])


# breakpoint()
prev_ts.observation["img"] = x
 # 使用 jax.numpy 数组
# all_batches_label_obs = 200*jnp.ones((1,4, 9, 9, 2),dtype=jnp.uint8)  # 使用 jax.numpy 数组

# def process_batch(batch_index, all_batches_label_obs):
#     batch = prev_ts.observation["img"][batch_index]
#     obs_shape = (4,9, 9, 2)
#     local_obs = 200*jnp.ones(obs_shape)

#     def process_pixel(idx, local_obs):
#         i = idx // batch.shape[2]
#         j = idx % batch.shape[2]
#         label = batch[i, j, :]  # 获取 label
#             # 获取 label

#         def is_within_bounds(x, y):
#             return (x >= 0) & (x < 9) & (y >= 0) & (y < 9)

#         def case_0(local_obs):
#             x = i - 8 + agent_position[batch_index][0]
#             y = j - 4 + agent_position[batch_index][1]
#             return jax.lax.cond(
#                 is_within_bounds(x, y),
#                 lambda obs: obs.at[x, y, :].set(label),
#                 lambda obs: obs, local_obs
#             )

#         def case_1(local_obs):
#             x = j - 4 + agent_position[batch_index][0]
#             y = -(i - 8) + agent_position[batch_index][1]
#             return jax.lax.cond(
#                 is_within_bounds(x, y),
#                 lambda obs: obs.at[x, y, :].set(label),
#                 lambda obs: obs, local_obs
#             )

#         def case_2(local_obs):
#             x = -(i - 8) + agent_position[batch_index][0]
#             y = -(j - 4) + agent_position[batch_index][1]
#             return jax.lax.cond(
#                 is_within_bounds(x, y),
#                 lambda obs: obs.at[x, y, :].set(label),
#                 lambda obs: obs, local_obs
#             )

#         def case_3(local_obs):
#             x = -(j - 4) + agent_position[batch_index][0]
#             y = i - 8 + agent_position[batch_index][1]
#             return jax.lax.cond(
#                 is_within_bounds(x, y),
#                 lambda obs: obs.at[x, y, :].set(label),
#                 lambda obs: obs, local_obs
#             )

#         branches = [case_0, case_1, case_2, case_3]

#         dir_index = prev_action[batch_index].astype(int)
        
#         # 根据 action_index 选择分支
#         local_obs = jax.lax.switch(dir_index, branches, local_obs)

#         return local_obs

#     num_pixels = batch.shape[1] * batch.shape[2]
#     local_obs = jax.lax.fori_loop(0, num_pixels, process_pixel, local_obs)

#     all_batches_label_obs = all_batches_label_obs.at[batch_index].set(local_obs)
#     return all_batches_label_obs

# num_batches = prev_ts.observation["img"].shape[0]

# all_batches_label_obs = jax.lax.fori_loop(0, num_batches, process_batch, all_batches_label_obs)

# print('before:', prev_ts.observation['img'])

# prev_ts.observation['img'] = all_batches_label_obs
# print('after:', prev_ts.observation['img'])
obs_emb = ssp_encoder(prev_ts.observation['img'])
print('shape:',obs_emb.shape)
print(obs_emb)

obs_emb = obs_emb[0][-1]
print('shape:',obs_emb.shape)
class_index = nn.tile_color_to_class_index_array[11,6]

label_ssp = nn.vocab_vectors[class_index]
    

inv_ssp = nn.ssp_space.invert(label_ssp)

# get similarity map of label with all locations by binding with inverse ssp 
out = nn.ssp_space.bind(obs_emb, inv_ssp)

sims = out @ nn.ssp_grid.reshape((-1, nn.ssp_dim)).T

# decode location = point with maximum similarity to label 
sims_map = sims.reshape((9,9))

# don't forget to remove shift from decoded location 
pred_loc = np.array(np.unravel_index(np.argmax(sims_map), sims_map.shape)) 
print(f'{class_index} predicted location: {tuple(pred_loc)}')

plt.imshow(sims_map, extent=[0,9,9,0])
plt.xticks([0,9])
plt.yticks([0,9])
plt.gca().xaxis.set_ticks_position('top')  #
plt.gca().xaxis.set_label_position('top')  #
plt.xlabel('X')
plt.ylabel('Y')
# plt.gca().invert_yaxis()
plt.colorbar(label='Similarity')
plt.show()
