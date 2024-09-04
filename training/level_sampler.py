import jax
import jax.numpy as jnp
import numpy as np
from ordered_set import OrderedSet


class LevelSampler:
    def __init__(self, rng, total_levels, values, rewards):
        self.rng = rng
        self.prob_new_level = 0.5
        self.total_levels = total_levels
        self.seen_levels = OrderedSet()
        self.scores = {}
        self.timestamps = {}
        self.count = 0
        self.prob_c = []
        self.prob_s = []
        self.rou = 0.5
        self.gamma = 0.99
        self.lamda = 0.95
        self.temperature = 1.0
        self.values = values
        self.rewards = rewards
        self.exp_buffer = []

    def sample_replay_decision(self):
        rng, sub_rng = jax.random.split(self.rng)
        decision = jax.random.bernoulli(sub_rng, self.prob_new_level)
        self.count += 1
        return decision

    def sample_new_level(self):
        rng, sub_rng = jax.random.split(self.rng)
        unseen_levels = [
            l for l in range(self.total_levels) if l not in self.seen_levels
        ]
        new_level = jax.random.choice(sub_rng, unseen_levels)
        self.scores[new_level] = 0
        self.seen_levels.add(new_level)
        self.timestamps[new_level] = 0
        return new_level

    def sample_replay_level(self):
        rng, sub_rng = jax.random.split(self.rng)
        priorities = self.calculate_priorities()
        replay_level = jax.random.choice(
            sub_rng, list(self.seen_levels), p=priorities
        )  # 注意，这里要将 OrderedSet 转换为列表
        return replay_level

    def calculate_priorities(self):
        total_staleness = jnp.sum(self.count - jnp.array(self.timestamps))

        for i in range(len(self.seen_levels)):
            prob_c_i = (self.count - self.timestamps[i]) / total_staleness
            self.prob_c.append(prob_c_i)

        priorities = self.rou * self.prob_c + (1 - self.rou) * self.prob_s
        return priorities

    def calculate_td_error(self):
        td_errors = self.rewards + self.gamma * self.values[1:] - self.values[:-1]
        return td_errors

    def calculate_score(self):
        T = len(self.rewards)
        td_errors = self.calculate_td_error(self.rewards, self.values, self.gamma)
        gae = jnp.zeros(T)
        for t in range(T):
            gae_t = 0.0
            for k in range(t, T):
                gae_t += (self.gamma * self.lamda) ** (k - t) * td_errors[k]
            gae = gae.at[t].set(gae_t)

        score = jnp.mean(jnp.abs(gae))
        return score

    def rank_prioritization(self, scores):

        sorted_indices = jnp.argsort(-scores)
        # 创建一个与 scores 同大小的数组用于存储优先级
        ranks = jnp.zeros_like(scores)

        # 根据排序后的索引，设置 ranks 数组中的排名
        ranks = ranks.at[sorted_indices].set(jnp.arange(1, len(scores) + 1))

        # 使用排名的倒数作为优先级
        h = 1 / ranks

        # 应用温度参数
        h = h ** (1 / self.temperature)

        # 归一化，确保总和为1
        self.prob_s = h / jnp.sum(h)
        return self.prob_s

    def sample(self):
        decision = self.sample_replay_decision()
        unseen_levels = [
            l for l in range(self.total_levels) if l not in self.seen_levels
        ]
        if decision == 0 and len(unseen_levels) > 0:
            sampled_level = self.sample_new_level()
        else:
            sampled_level = self.sample_replay_level()
        score = self.calculate_score()
        timestamp = self.count
        self.scores[sampled_level] = score
        self.timestamps[sampled_level] = timestamp
        return sampled_level
