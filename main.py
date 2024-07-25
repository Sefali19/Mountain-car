import gym
from q_learning.random_episode import random_episode
from q_learning.run_ep import run_ep
from q_learning.aggregation import aggregation

def main():
    env = gym.make('MountainCar-v0', render_mode="human")
    env.reset()
    # random_episode(env)
    run_ep(env, num_episodes=600)
    # aggregation(env)
    env.close()

if __name__ == "__main__":
    main()
