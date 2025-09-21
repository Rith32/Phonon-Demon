import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import linregress

class RandomWalkPhononDemon2D:
    def __init__(self, grid_size, N, R, steps):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int) # Number of lattice sites
        self.N = N
        self.R = R
        self.steps = steps
        self.demon = R
        self.position = (grid_size // 2, grid_size // 2) # Contrary to Random Pick case, the lattice is now defined to be two-dimensional
        self.demon_energy_record = [] # The demon begins with all energy 
        self.demon_path = []  

    def run(self, delta):  
        for _ in range(self.steps):
            x, y = self.position
            dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)]) # The demon moves randomly in accordance to one of these four selected movements
            x = (x + dx) % self.grid_size #Ensures that if a demon hits the boundaries of the lattice, it wraps around
            y = (y + dy) % self.grid_size
            self.position = (x, y)

            change = np.random.randint(-delta, delta + 1)

            if change > 0 and self.demon >= change:
                self.grid[x, y] += change
                self.demon -= change
            elif change < 0 and self.grid[x, y] >= abs(change):
                self.grid[x, y] += change
                self.demon -= change

            self.demon_energy_record.append(self.demon)
            self.demon_path.append((x, y))  

    # Creates a heatmap of energy propogation 

    def get_heatmap(self):
        return self.grid

    # Returns average demon energy

    def average_demon_energy(self):
        return np.mean(self.demon_energy_record)

    def plot_beta_vs_sweeps_with_mse(self):
        demon_energies = np.array(self.demon_energy_record)
        running_avg = np.cumsum(demon_energies) / (np.arange(1, len(demon_energies) + 1))
        beta_vals = 1.0 / running_avg  # Assuming k_B = 1

        beta_true = 1.0 / np.mean(demon_energies)
        mse = (beta_vals - beta_true) ** 2

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot of β
        ax1.plot(np.arange(len(beta_vals)), beta_vals, color='red', label='β (1/T)', linewidth=1)
        ax1.set_xlabel("Monte Carlo Sweep")
        ax1.set_ylabel("β = 1/T", color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        # An additional plot of convergence of β and Mean Square Error over time
        #ax2 = ax1.twinx()
        #ax2.plot(np.arange(len(mse)), mse, color='blue', linestyle='--', label='MSE', linewidth=1)
        #ax2.set_ylabel("Mean Square Error", color='blue')
        #ax2.tick_params(axis='y', labelcolor='blue')

        #plt.title("β Convergence and Mean Square Error Over Time")
        #fig.tight_layout()
        #plt.grid(True)
        #plt.savefig("beta_vs_sweeps_with_mse.png", dpi=300)
        #plt.show()

        #print(f"Final β (1/T): {beta_true:.5f}")


# Run the simulation
sim = RandomWalkPhononDemon2D(grid_size=10, N=100, R=1000, steps=100000)
sim.run(delta=50)

# Plot the heatmap
heatmap = sim.get_heatmap()
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.title("Heat Map of Phonon Energy Distribution")
plt.colorbar(label='Energy')
plt.tight_layout()
plt.show()

# Analyze demon energy
demon_energies = np.array(sim.demon_energy_record)

# Histogram and probability distribution
counts, bin_edges = np.histogram(demon_energies, bins=100, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Filter out zero entries for log
nonzero = counts > 0
log_probs = np.log(counts[nonzero])
E_vals = bin_centers[nonzero]

# Limit to Ed <= 150
max_energy_cutoff = 130
mask = E_vals <= max_energy_cutoff

E_fit = E_vals[mask]
logP_fit = log_probs[mask]

# Linear regression on limited range
slope, intercept, r_value, p_value, std_err = linregress(E_fit, logP_fit)

# Plot ln P(Ed) and linear fit within cutoff
plt.plot(E_vals, log_probs, label='ln P(Ed)', marker='o', linestyle='None', markersize=4)
plt.plot(E_fit, slope * E_fit + intercept, color='red', label=f'Fit: slope = {slope:.4f}')
plt.xlim(0, max_energy_cutoff)
plt.xlabel("Demon Energy Ed")
plt.ylabel("ln P(Ed)")
plt.title("Linear fit of Distribution of Demon Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Report values
sim.plot_beta_vs_sweeps_with_mse()

avg_Ed = sim.average_demon_energy()
print(f"Average Demon Energy ⟨Ed⟩: {avg_Ed:.2f}")
print(f"Slope of ln P(Ed) vs Ed: {slope:.4f}")
print(f"Estimated Temperature T ≈ {-1/slope:.2f} (in demon units)")
