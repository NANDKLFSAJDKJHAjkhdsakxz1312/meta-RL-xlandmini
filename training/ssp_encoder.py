from src.xminigrid.core.constants import Tiles,Colors
import jax
import jax.numpy as jnp

from utils_ssp import HexagonalSSPSpace
import nengo_spa as spa


NUM_TILES = len(Tiles.__annotations__)  
NUM_COLORS = len(Colors.__annotations__)  

NUM_CLASSES = NUM_TILES * NUM_COLORS
ssp_dim = 1015
length_scale = 5
env_grid_size = 9

RNG = jax.random.PRNGKey(1)
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
num_classes = len(vocab_vectors)  # 类别数量
num_positions = len(ssp_grid)  # 网格中的位置数量
ssp_dim = ssp_grid.shape[1]  # SSP 向量的维度

# nn模块中初始化pre_bind_check_table时
print("Calculating pre_bind_check_table...")
pre_bind_check_table = jnp.zeros((num_classes, num_positions, ssp_dim))
for i in range(num_classes):
    for j in range(num_positions):
        pre_bind_check_table = pre_bind_check_table.at[i, j].set(
            jnp.squeeze(ssp_space.bind(vocab_vectors[i], ssp_grid[j]))
        )
print("pre_bind_check_table calculated.")



# rng_keys = jax.random.split(RNG, NUM_CLASSES)
# label_vectors = jax.vmap(lambda key: jax.random.normal(key, (ssp_dim,)))(rng_keys)  # [NUM_CLASSES, ssp_dim]

# 创建类别映射数组
tile_color_to_class_index_array = -jnp.ones((NUM_TILES, NUM_COLORS), dtype=jnp.int32)
index = 0
for i in range(NUM_TILES):
    for j in range(NUM_COLORS):
        tile_color_to_class_index_array = tile_color_to_class_index_array.at[i, j].set(index)
        index += 1




def ssp_encoder(inputs,global_ssp):
    
    B, S, H, W, _ = inputs.shape  # Includes sequence length S
    
    def process_single_batch(batch_inputs,carry):
        # Initialize accumulated SSP vector as all zeros
        # init_carry = jnp.zeros((ssp_grid.shape[1],))  # [ssp_dim]

        # Define function to process a single time step
        def process_single_time_step(carry, single_time_step_inputs):
        
            # Retrieve tile and color labels
            tile_labels = single_time_step_inputs[..., 0].astype(jnp.int32)  # [H, W]
            color_labels = single_time_step_inputs[..., 1].astype(jnp.int32)  # [H, W]

       
            # Create a mask to identify valid positions
            valid_mask = jnp.logical_and(
                jnp.not_equal(tile_labels, 0),  # no empty tile
                jnp.logical_and(
                    jnp.not_equal(tile_labels, 1),  # no floor
                    jnp.not_equal(tile_labels, 2)  # no label 2
                )
            )
        
            # Get class_indices and corresponding label vectors
            class_indices = tile_color_to_class_index_array[tile_labels, color_labels]
        
            # Create position indices for valid positions
            position_indices = jnp.arange(H * W).reshape(H, W) 

       
            binding_vectors = jnp.where(
                valid_mask[..., None],
                pre_bind_check_table[class_indices, position_indices],
                jnp.zeros((H, W, ssp_grid.shape[1]))
            )
            



            # Accumulate binding vectors from all valid positions
            carry = carry + binding_vectors.sum(axis=(0, 1))  # [ssp_dim]

            return carry, carry  # Return carry as the result to store SSP vector for each time step

        
        final_carry, ssp_vectors = jax.lax.scan(
            process_single_time_step, 
            carry, 
            batch_inputs
        )
        # ssp_vectors = ssp_vectors.reshape((S, ssp_dim))     
        return final_carry,ssp_vectors  # Return [S, ssp_dim] instead of the accumulated vector
    new_global_ssp, ssp_vectors = jax.vmap(
        lambda batch, carry: process_single_batch(batch, carry)
    )(inputs, global_ssp)
    new_global_ssp = ssp_space.normalize_by_last_dim(new_global_ssp)
    ssp_vectors = ssp_space.normalize_by_last_dim(ssp_vectors)
    return new_global_ssp, ssp_vectors
    
    # global_env_ssp = jax.vmap(process_single_batch)(inputs)  # [B, S, ssp_dim]
    # global_env_ssp = ssp_space.normalize(global_env_ssp)
    # return global_env_ssp


ssp_encoder = jax.jit(ssp_encoder)

