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




data2 = [
    [[0, 0], [0, 0], [0, 0], [0, 0], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
]
data1 = [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                  [[2, 6], [1, 7], [6, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [3, 1], [3, 4], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [11, 6], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [2, 6]],
                  [[2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6]]]
# 转换为 JAX 数组

x  = jnp.array([[data1,data1,data1,data1,data1,data1,data1,data1,data1,data2,data1,data1]])


# breakpoint()
prev_ts.observation["img"] = x

obs_emb = ssp_encoder(prev_ts.observation['img'])


obs_emb = obs_emb[0][-1]

class_index = nn.tile_color_to_class_index_array[6,6]

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
