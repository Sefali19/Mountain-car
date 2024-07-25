import numpy as np

def choose_action(state, Q, epsilon, num_actions):
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state])
