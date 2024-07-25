def discretize_state(state, state_bounds, num_buckets):
    discretized = list()
    for i in range(len(state)):
        scaling = (state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])
        new_state = int(round((num_buckets[i] - 1) * scaling))
        new_state = min(num_buckets[i] - 1, max(0, new_state))
        discretized.append(new_state)
    return tuple(discretized)
