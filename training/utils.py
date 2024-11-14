# utilities for PPO training and evaluation
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from xminigrid.environment import Environment, EnvParams
from jax import jit

i_indices = jnp.arange(5)
j_indices = jnp.arange(5)
i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing='ij')

# flattening i_grid and j_grid to prepare for parallel processing.
i_grid_flat = i_grid.flatten()
j_grid_flat = j_grid.flatten()
up_x = i_grid_flat-4
up_y = j_grid_flat-2
right_x = j_grid_flat-2
right_y = -(i_grid_flat-4)
down_x = -(i_grid_flat-4)
down_y = -(j_grid_flat-2)
left_x = -(j_grid_flat-2)
left_y = i_grid_flat-4
@jit
def _is_in_bound(x,y):
    return (x >= 0) & (x <= 8) & (y >= 0) & (y <= 8)
@jit   
def process_batch(batch,dir,pos):
    

    def case_0():
        local_obs = jnp.zeros((9,9, 2), dtype=jnp.uint8)
        x = up_x + pos[0]
        y = up_y + pos[1]
        mask = _is_in_bound(x, y)

        # 遍历每个位置，仅在满足条件的 (x, y) 位置上更新
        def update_local_obs(i, obs):
            xi, yi = x[i], y[i]

            def set_update_value(obs):
                update_value = batch[xi - pos[0] + 8, yi - pos[1] + 4]
                return obs.at[xi, yi, :].set(update_value)

            # 使用 jax.lax.cond 进行条件更新
            obs = jax.lax.cond(
                mask[i],           # 条件为 True 时更新
                set_update_value,   # 满足条件时的更新函数
                lambda obs: obs,    # 不满足条件时保持不变
                obs                 # 传递的数组
            )
            return obs

        local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
        
        return local_obs

    def case_1():
        local_obs = jnp.zeros((9,9, 2), dtype=jnp.uint8)
        x = right_x + pos[0]
        y = right_y + pos[1]
        mask = _is_in_bound(x, y)

        # 遍历每个位置，仅在满足条件的 (x, y) 位置上更新
        def update_local_obs(i, obs):
            xi, yi = x[i], y[i]

            def set_update_value(obs):
                update_value = batch[8 + pos[1] - yi, xi + 4 - pos[0]]
                return obs.at[xi, yi, :].set(update_value)

            # 使用 jax.lax.cond 进行条件更新
            obs = jax.lax.cond(mask[i], set_update_value, lambda obs: obs, obs)
            return obs

        local_obs = jax.lax.fori_loop(0, len(x), update_local_obs, local_obs)
        
        return local_obs

    def case_2():
        local_obs = jnp.zeros((9,9, 2), dtype=jnp.uint8)
        x = down_x + pos[0]
        y = down_y + pos[1]
        mask = _is_in_bound(x, y)

        # 遍历每个位置，仅在满足条件的 (x, y) 位置上更新
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

        # 遍历每个位置，仅在满足条件的 (x, y) 位置上更新
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


# Training stuff
class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    # for obs
    obs: jax.Array
    # obs_cnn: jax.Array
    dir: jax.Array
    rule: jax.Array
    goal: jax.Array
    # for rnn policy
    prev_action: jax.Array
    prev_reward: jax.Array


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = (
            transition.reward
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # breakpoint()
    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "obs_img": transitions.obs,
                # "obs_img_cnn": transitions.obs_cnn,
                "obs_dir": transitions.dir,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
                "rule": transitions.rule,
                "goal": transitions.goal
            },
            init_hstate,
            
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(
            -clip_eps, clip_eps
        )
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean(
        (loss, vloss, aloss, entropy, grads), axis_name="devices"
    )
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(struct.PyTreeNode):
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    episodes: jax.Array = jnp.asarray(0)


def rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        # parallelly transform local observation to global one
        all_batches_label_obs = process_batch(
            timestep.observation["img"], 
            timestep.state.agent.direction.astype(int),
            timestep.state.agent.position
        )
  
        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "obs_img": all_batches_label_obs[None, None, ...],
                # "obs_img_cnn": timestep.observation["img"][None, None, ...],
                "obs_dir": timestep.observation["direction"][None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
                "rule": timestep.state.rule_encoding[None, None, ...],
                "goal": timestep.state.goal_encoding[None, None, ...],

            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
        )
        carry = (rng, stats, timestep, action, timestep.reward, hstate)
        return carry

    timestep = env.reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), timestep, prev_action, prev_reward, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]

def create_mask(obs):
    # Check if each position is `[0, 0]` (invalid) or not
    mask = jnp.any(obs != jnp.array([0, 0]), axis=-1)
    return mask