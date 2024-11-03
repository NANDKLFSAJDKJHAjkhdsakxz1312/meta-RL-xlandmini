import jax.numpy as jnp
import nn
import utils_ssp
import numpy as np

import matplotlib.pyplot as plt
y = jnp.zeros(1015,)
z = jnp.zeros(1015,)

# 将嵌套的布尔值结构直接复制为 Python 列表
mask = [
    [[False], [False], [False], [False], [False], [False], [False], [False], [False]],
    [[False], [False], [True], [False], [False], [False], [False], [False], [False]],
    [[False], [False], [False], [True], [True], [False], [False], [False], [False]],
    [[False], [False], [False], [False], [False], [False], [False], [False], [False]],
    [[False], [False], [False], [False], [False], [False], [False], [False], [False]],
    [[False], [False], [False], [False], [False], [False], [False], [False], [False]],
    [[False], [False], [False], [True], [False], [False], [False], [False], [False]],
    [[False], [False], [False], [False], [False], [False], [False], [False], [False]],
    [[False], [False], [False], [False], [False], [False], [False], [False], [False]]
]
tile = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 6, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 3, 3, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 11, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2]
]


color = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 7, 6, 7, 7, 7, 7, 7, 6],
    [6, 7, 7, 1, 4, 7, 7, 7, 6],
    [6, 7, 7, 7, 7, 7, 7, 7, 6],
    [6, 7, 7, 7, 7, 7, 7, 7, 6],
    [6, 7, 7, 7, 7, 7, 7, 7, 6],
    [6, 7, 7, 6, 7, 7, 7, 7, 6],
    [6, 7, 7, 7, 7, 7, 7, 7, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6]
]
ssp_space = utils_ssp.HexagonalSSPSpace(domain_dim=2, ssp_dim=1015, length_scale=5,
                                domain_bounds=jnp.array([[0, 9], [0, 9]]))

# 转换为 jnp 数组
tile_jnp = jnp.array(tile)
color_jnp = jnp.array(color)
tile_color_to_class_index_array = -jnp.ones((13, 12), dtype=jnp.int32)
index = 0
for i in range(13):
    for j in range(12):
        tile_color_to_class_index_array = tile_color_to_class_index_array.at[i, j].set(index)
        index += 1
# 转换为 jnp 数组
p = ssp_space.bind(y,z)
print(p)
mask_jnp = jnp.array(mask)
x = tile_jnp*mask_jnp

breakpoint()
class_indices = tile_color_to_class_index_array[tile_jnp,color_jnp]
label_ssps = jnp.where(
                mask_jnp, 
                nn.vocab_vectors[class_indices],  # 有效位置的向量
                jnp.zeros((9,9, 1015))  # 无效位置为零向量
            ) 
local_ssps = jnp.where(
                mask_jnp, 
                nn.ssp_grid,  # 有效位置的向量
                jnp.zeros((9,9, 1015))  # 无效位置为零向量
            ) 

# print('label:',label_ssps)
# print('loc:',local_ssps)
binding_vectors = jnp.where(
            mask_jnp, 
            ssp_space.bind(label_ssps, local_ssps),  # 只对有效位置绑定
            jnp.zeros_like(label_ssps)  # 无效位置保持为零
            )
            # jax.debug.print("bundvec:{x}",x=jnp.where(valid_mask[...,None],label_ssps,jnp.zeros_like(label_ssps)))
            # for i in range(nn.ssp_grid.shape[0]):
            #     for j in range(nn.ssp_grid.shape[1]):
            #         # 获取每个位置的向量
            #         vector =nn.ssp_grid[i, j]
            #         # 打印位置和对应的向量
            #         jax.debug.print("位置 ({i}, {j}) 的向量: {x}", i=i, j=j, x=vector)
            

idx = tile_color_to_class_index_array[6,6]
x = ssp_space.bind(nn.vocab_vectors[idx],nn.ssp_grid[1,2])
# jax.debug.print("selectedvec_6,6:{y}",y=nn.ssp_grid[1,2])
idx = tile_color_to_class_index_array[11,6]
# jax.debug.print("selectedvec_11,6:{y}",y=nn.ssp_grid[6,3])

x+=ssp_space.bind(nn.vocab_vectors[idx],nn.ssp_grid[6,3])
idx = tile_color_to_class_index_array[3,1]
# jax.debug.print("selectedvec_3,1:{y}",y=nn.ssp_grid[2,3])

x+=ssp_space.bind(nn.vocab_vectors[idx],nn.ssp_grid[2,3])
idx = tile_color_to_class_index_array[3,4]
# jax.debug.print("selectedvec_3,4:{y}",y=nn.ssp_grid[2,4])

x+=ssp_space.bind(nn.vocab_vectors[idx],nn.ssp_grid[2,4])
carry = binding_vectors.sum(axis=(0, 1))


class_index = tile_color_to_class_index_array[6,6]

label_ssp = nn.vocab_vectors[class_index]
    

inv_ssp = ssp_space.invert(label_ssp)

# get similarity map of label with all locations by binding with inverse ssp 
out = ssp_space.bind(carry, inv_ssp)

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
