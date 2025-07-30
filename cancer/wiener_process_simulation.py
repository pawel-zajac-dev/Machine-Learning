
import numpy as np
import matplotlib.pyplot as plt

class WienerProcess:
    def __init__(self, n_steps=1000, T=1.0, n_paths=5):
        self.n_steps = n_steps
        self.T = T
        self.dt = T / n_steps
        self.n_paths = n_paths

    def generate_paths(self):
        paths = np.zeros((self.n_steps + 1, self.n_paths))
        for i in range(1, self.n_steps + 1):
            # Normal increments N(0, dt)
            increments = np.random.normal(0, np.sqrt(self.dt), self.n_paths)
            paths[i] = paths[i - 1] + increments
        return paths

    def plot_paths(self, paths):
        t = np.linspace(0, self.T, self.n_steps + 1)
        plt.figure(figsize=(10, 6))
        for j in range(paths.shape[1]):
            plt.plot(t, paths[:, j], lw=1)
        plt.title('Wiener Process Simulation (Brownian Motion)')
        plt.xlabel('Time')
        plt.ylabel('Process Value')
        plt.grid(True)
        plt.show()

    def analyze_paths(self, paths):
        final_values = paths[-1]
        mean = np.mean(final_values)
        std_dev = np.std(final_values)
        print(f"Mean of final values: {mean:.4f}")
        print(f"Standard deviation of final values: {std_dev:.4f}")
        return mean, std_dev


if __name__ == "__main__":
    process = WienerProcess(n_steps=1000, T=1.0, n_paths=5)
    paths = process.generate_paths()
    process.plot_paths(paths)
    process.analyze_paths(paths)
