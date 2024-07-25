import numpy as np
import matplotlib.pyplot as plt
from q_learning.run_ep import run_ep

def aggregation(env):
    num_episodes = 1000
    num_runs = 10
    all_successes = np.zeros((num_runs, num_episodes))
    all_steps_per_episode = np.zeros((num_runs, num_episodes))

    for run in range(num_runs):
        cumulative_successes, steps_per_episode = run_ep(env, num_episodes)
        all_successes[run] = cumulative_successes
        all_steps_per_episode[run] = steps_per_episode
        print(f"Run {run + 1}/{num_runs} completed")

    avg_successes = np.mean(all_successes, axis=0)
    avg_steps_per_episode = np.mean(all_steps_per_episode, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(range(num_episodes), avg_successes)
    ax1.set_title('Averaged Cumulative Number of Successes')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Cumulative Successes')

    ax2.plot(range(num_episodes), avg_steps_per_episode)
    ax2.set_title('Averaged Number of Steps per Episode')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Steps per Episode')

    plt.tight_layout()
    plt.show()
