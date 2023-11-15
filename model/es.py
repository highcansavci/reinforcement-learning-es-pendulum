import gym
import numpy as np
from datetime import datetime

from config.config import Config
from model.ann import ANN
from multiprocessing.dummy import Pool

config_ = Config().config
env = gym.make(config_["env_name"])
input_dim = len(env.reset())
hidden_dim = int(config_["hidden_size"])
output_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
pool = Pool(4)


def evolution_strategy(
        f,
        population_size,
        sigma,
        lr,
        initial_params,
        num_iters):
    # assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)

    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        noises = np.random.randn(population_size, num_params)

        rewards = pool.map(f, [params + sigma * noises[j] for j in range(population_size)])
        rewards = np.array(rewards)

        mean_rewards = rewards.mean()
        std_rewards = rewards.std()
        if std_rewards == 0:
            # we can't apply the following equation
            print("Skipping")
            continue

        advantage = (rewards - mean_rewards) / std_rewards
        reward_per_iteration[t] = mean_rewards
        params = params + lr / (population_size * sigma) * np.dot(noises.T, advantage)

        # update the learning rate
        lr *= 0.992354

        print("Iter:", t, "Avg Reward: %.3f" % mean_rewards, "Max:", rewards.max(), "Duration:", (datetime.now() - t0))

    return params, reward_per_iteration


def reward_function(params, display=False):
    model = ANN(input_dim, hidden_dim, output_dim)
    model.set_params(params)

    # play one episode and return the total reward
    episode_reward = 0
    done = False
    env = gym.make(config_["env_name"])
    if display:
        env = gym.wrappers.RecordVideo(env, 'es_monitor')
    state = env.reset()

    action_max = env.action_space.high[0]

    while not done:
        if display:
            env.render()
        # get the action
        action = model.sample_action(state, action_max)

        # perform the action
        state, reward, done, _ = env.step(action)

        # update total reward
        episode_reward += reward

    return episode_reward
