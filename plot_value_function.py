import matplotlib.pyplot as plt

def plot_value_function(q, episode):
    value_function = np.max(q, axis=2)
    plt.imshow(value_function, origin='lower')
    plt.colorbar()
    plt.title(f"Value Function after {episode} episodes")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.show()
