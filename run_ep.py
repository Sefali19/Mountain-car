import numpy as np
from q_learning.discretize_state import discretize_state
from q_learning.choose_action import choose_action
from q_learning.update_q_table import update_q_table


def run_ep(env, num_episodes):
    num_buckets = (20, 20)  # 20 intervals for position and 20 for velocity
    num_actions = env.action_space.n  # 3 possible actions
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995  # Decay rate for exploration
    min_epsilon = 0.01
    Q = np.zeros(num_buckets + (num_actions,))
    successes = np.zeros(num_episodes)
    steps_per_episode = np.zeros(num_episodes)

    for episode in range(num_episodes):
        current_state = discretize_state(env.reset(), state_bounds, num_buckets)
        done = False
        steps = 0

        while not done:
            action = choose_action(current_state, Q, epsilon, num_actions)
            next_state_raw, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state_raw, state_bounds, num_buckets)
            update_q_table(Q, current_state, action, reward, next_state, gamma, alpha)
            current_state = next_state
            steps += 1

            if done and next_state_raw[0] >= 0.5:
                successes[episode] = 1

        steps_per_episode[episode] = steps

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    cumulative_successes = np.cumsum(successes)
    return cumulative_successes, steps_per_episode
