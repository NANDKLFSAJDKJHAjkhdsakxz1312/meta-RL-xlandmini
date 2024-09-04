import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nengo_spa as spa
import itertools
from utils_ssp import SPSpace, SSPSpace, HexagonalSSPSpace

# Hyper parameters
SSP_DIM = 1015
RES_X, RES_Y = 9, 9
RNG = jax.random.PRNGKey(42)

# All objects and their main characteristics
TILE_LST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
COLOR_LST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

all_combinations = list(itertools.product(TILE_LST, COLOR_LST))

vocab = spa.Vocabulary(dimensions=SSP_DIM, pointer_gen=RNG)

# Add semantic pointers for each object to the vocabulary
for i, class_name in enumerate(all_combinations):
    class_name_str = f"Class_{class_name[0]}_{class_name[1]}"
    vector = vocab.algebra.create_vector(SSP_DIM, properties={"positive", "unitary"})
    vocab.add(class_name_str, vector)

ssp_space = HexagonalSSPSpace(
    domain_dim=2,
    ssp_dim=SSP_DIM,
    length_scale=5,
    domain_bounds=jnp.array([[0, 4], [0, 4]]),
)

# Generate xy coordinates using JAX
x_coords, y_coords = jnp.meshgrid(jnp.arange(5), jnp.arange(5), indexing="ij")
coords = jnp.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)
ssp_grid = ssp_space.encode(coords)
ssp_grid = ssp_grid.reshape((5, 5, -1))

print(f"Generated SSP grid: {ssp_grid.shape}")

# Init empty environment
global_env_ssp = jnp.zeros(SSP_DIM)


@jax.jit
def get_global_env_ssp(obj_locations):
    global global_env_ssp
    for key, loc in obj_locations.items():
        print(key.upper())

        # Get label SSP from vocabulary and location SSP from grid
        label_ssp = vocab[key.upper()].v

        loc_ssp = ssp_grid[loc[1], loc[0]]

        # Bind label SSP to its location SSP
        bound_vectors = ssp_space.bind(label_ssp, loc_ssp)

        # Add bound vectors to the global environment SSP representation
        global_env_ssp += bound_vectors.squeeze()
        global_env_ssp = ssp_space.normalize(global_env_ssp)
    return global_env_ssp


@jax.jit
def check_global_env_ssp(obj_locations):
    for key in obj_locations:
        label_ssp = vocab[key.upper()].v
        inv_ssp = ssp_space.invert(label_ssp)

        # Get similarity map of label with all locations by binding with inverse SSP
        out = ssp_space.bind(global_env_ssp, inv_ssp)
        sims = out @ ssp_grid.reshape((-1, SSP_DIM)).T

        # Decode location = point with maximum similarity to label
        sims_map = sims.reshape((5, 5))

        # Don't forget to remove shift from decoded location
        pred_loc = jnp.array(
            jnp.unravel_index(jnp.argmax(sims_map), sims_map.shape)
        ) - jnp.array([10, 10])
        print(f"{key.upper()} predicted location: {tuple(jnp.flip(pred_loc))}")

        plt.imshow(sims_map, extent=(0, 4, 0, 4))
        plt.xticks([-10, 0, 10])
        plt.yticks([-10, 0, 10])
        plt.show()


def create_observation_dict(observation):
    observation_dict = {}
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            tile_id = observation[i, j, 0]
            color_id = observation[i, j, 1]
            class_key = f"CLASS_{tile_id}_{color_id}"
            if class_key not in observation_dict:
                observation_dict[class_key] = []
            observation_dict[class_key].append((i, j))
    return observation_dict
