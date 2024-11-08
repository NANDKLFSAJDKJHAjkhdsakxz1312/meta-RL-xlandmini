# test whether local to global transformation is correct or not
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import timeit
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
import os
from xminigrid.types import RuleSet
from xminigrid.benchmarks import Benchmark, load_benchmark, load_benchmark_from_path, load_bz2_pickle, DATA_PATH, NAME2HFFILENAME
from xminigrid.rendering.text_render import print_ruleset
# utils for the demonstation
from xminigrid.core.grid import room
from xminigrid.types import AgentState
from xminigrid.core.actions import take_action
from xminigrid.core.constants import Tiles, Colors, TILES_REGISTRY
from xminigrid.rendering.rgb_render import render

# rules and goals
from xminigrid.core.goals import check_goal, AgentNearGoal
from xminigrid.core.rules import check_rule, AgentNearRule


import xminigrid
i_indices = jnp.arange(9)
j_indices = jnp.arange(9)
i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing='ij')

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
goal = AgentNearGoal(tile=TILES_REGISTRY[Tiles.SQUARE, Colors.PURPLE])
rule = AgentNearRule(
    tile=TILES_REGISTRY[Tiles.BALL, Colors.YELLOW], 
    prod_tile=TILES_REGISTRY[Tiles.SQUARE, Colors.PURPLE],
)

ruleset = RuleSet(
    goal=goal.encode(),
    rules=rule.encode()[None, ...],
    init_tiles=jnp.array((
        TILES_REGISTRY[Tiles.BALL, Colors.YELLOW],
    ))
)
print_ruleset(ruleset)
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from xminigrid.wrappers import GymAutoResetWrapper
import imageio
# create rollout function
def build_rollout(env, env_params, num_steps):
    def rollout(rng):
        def _step_fn(carry, _):
            rng, timestep = carry
            rng, _rng = jax.random.split(rng)
            action = jax.random.randint(_rng, shape=(), minval=0, maxval=env.num_actions(env_params))
            
            timestep = env.step(env_params, timestep, action)
            
            return (rng, timestep), timestep
    
        rng, _rng = jax.random.split(rng)
        timestep = env.reset(env_params, _rng)
        
        rng, transitions = jax.lax.scan(_step_fn, (rng, timestep), None, length=num_steps)
        return transitions

    return rollout

# craete environment
env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9",view_size=9)
env_params = env_params.replace(ruleset=ruleset)
env = GymAutoResetWrapper(env)

# set up number of steps for testing
num_steps = 10 
rollout_fn = jax.jit(build_rollout(env, env_params, num_steps=num_steps))

# run rollout 
transitions = rollout_fn(jax.random.PRNGKey(0))


print("Transitions shapes: \n", jtu.tree_map(jnp.shape, transitions))
images = []
steps = []
for i in trange(10):
    timestep = jtu.tree_map(lambda x: x[i], transitions)
    steps.append(timestep)
    image = env.render(env_params, timestep)
    images.append(image)  


    image_path = os.path.join("/scratch/jiang/ssp_xland/meta-RL-xlandmini/training/", f"timestep_{i}.png")
    plt.figure(dpi=64)
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()  


output_path = "/scratch/jiang/ssp_xland/meta-RL-xlandmini/training/example_rollout.mp4"
imageio.mimsave(output_path, images, fps=1, format="mp4")

print(f"Video saved to {output_path}")

from jax import jit
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

def extract_fields(steps):
    observations = jnp.array([step.observation for step in steps])  
    directions = jnp.array([step.state.agent.direction.astype(int) for step in steps])  
    positions = jnp.array([step.state.agent.position for step in steps])  
    return observations, directions, positions


observations, directions, positions = extract_fields(steps)

all_batches_label_obs = jax.vmap(process_batch)(
    observations, 
    directions,
    positions
)


import os
import numpy as np
import matplotlib.pyplot as plt

# tile-color mapping to RGB
color_map = {
    (1, 7): [0, 0, 0],       # floor - black
    (2, 6): [128, 128, 128], # wall - gray
    (0, 0): [255, 255, 255], # empty - white
    (3, 5): [0,0 , 0]    # special object - black background
}

# visilize observation image
def visualize_observation(observation):
    height, width, _ = observation.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            obj_color_pair = tuple(observation[i, j].tolist()) 
            rgb_image[i, j] = color_map.get(obj_color_pair, [255, 255, 255])  # 默认为白色
    return rgb_image

output_dir = "/scratch/jiang/ssp_xland/meta-RL-xlandmini/training"
os.makedirs(output_dir, exist_ok=True)


for step_idx in range(10):
    # render image
    timestep = steps[step_idx]
    full_image = env.render(env_params, timestep) 

    # local and global arrays for current step
    local_obs = observations[step_idx]  
    global_obs = all_batches_label_obs[step_idx] 

    
    local_image = visualize_observation(local_obs)
    global_image = visualize_observation(global_obs)

    # create 3 sub-figures for each step 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(full_image)
    axes[0].set_title("Environment Render")
    axes[0].axis("off")

    axes[1].imshow(local_image)
    axes[1].set_title("Local Observation")
    axes[1].axis("off")

    axes[2].imshow(global_image)
    axes[2].set_title("Global Observation")
    axes[2].axis("off")

    
    for ax, obs in zip([axes[1], axes[2]], [local_obs, global_obs]):
        height, width = obs.shape[:2]
        
      
        for i in range(10):  
            ax.hlines(i - 0.5, -0.5, width - 0.5, color=(.3, .3, .3, .5), linewidth=1)
            ax.vlines(i - 0.5, -0.5, height - 0.5, color=(.3, .3, .3, .5), linewidth=1)
        
        
        for i in range(height):
            for j in range(width):
                if tuple(obs[i, j].tolist()) == (3, 5):  
                    circle = plt.Circle((j, i), 0.5, color='yellow', ec='black', linewidth=1.0)
                    ax.add_patch(circle)

    fig.savefig(os.path.join(output_dir, f"step_{step_idx}_visualization.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)  


## below test ssp whether works correctly
from nn import ssp_encoder
import numpy as np
from flax import struct
from typing import Dict
import jax
import jax.numpy as jnp
import nn
from utils_ssp import HexagonalSSPSpace
import matplotlib.pyplot as plt

all_batches_label_obs_expanded = all_batches_label_obs[None, ...]
obs_emb = ssp_encoder(all_batches_label_obs_expanded)

breakpoint()
obs_emb = obs_emb[0][-1]

class_index = nn.tile_color_to_class_index_array[3,5]

label_ssp = nn.vocab_vectors[class_index]
    

inv_ssp = nn.ssp_space.invert(label_ssp)

# get similarity map of label with all locations by binding with inverse ssp 
out = nn.ssp_space.bind(obs_emb, inv_ssp)

sims = out @ nn.ssp_grid.reshape((-1, nn.ssp_dim)).T

# decode location = point with maximum similarity to label 
sims_map = sims.reshape((9,9))

pred_loc = np.array(np.unravel_index(np.argmax(sims_map), sims_map.shape)) 
print(f'{class_index} predicted location: {tuple(pred_loc)}')


plt.imshow(sims_map, extent=[0,9,9,0])
plt.xticks([0,9])
plt.yticks([0,9])
plt.gca().xaxis.set_ticks_position('top')  #
plt.gca().xaxis.set_label_position('top')  #
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Similarity')


output_path = "/scratch/jiang/ssp_xland/meta-RL-xlandmini/training/ssp_result.png"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close()  
