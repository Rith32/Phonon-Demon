import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class PhononDemon:
    def __init__(self, num_sites=100, total_energy=1000):
        self.N = num_sites
        self.R = total_energy
        self.sites = np.zeros(self.N, dtype=int)
        self.demon_energy = 0
        self.demon_energy_sum = 0
        self.steps_recorded = 0
        self.demon_energy_record = []  
        self.initialize_random_distribution()

    def initialize_random_distribution(self):
        remaining = self.R
        while remaining > 0:
            i = np.random.randint(self.N)
            self.sites[i] += 1
            remaining -= 1

    def monte_carlo_step(self, delta=50):
        site = np.random.randint(self.N)
        energy_change = np.random.randint(-delta, delta + 1)  # Energy exhange range: [-delta, delta]

        if energy_change > 0:
            # Demon gives energy to site
            if self.demon_energy >= energy_change:
                self.sites[site] += energy_change
                self.demon_energy -= energy_change
        elif energy_change < 0:
            # Demon takes energy from site
            if self.sites[site] >= -energy_change:
                self.sites[site] += energy_change  
                self.demon_energy -= energy_change  

        self.demon_energy_sum += self.demon_energy
        self.demon_energy_record.append(self.demon_energy)
        self.steps_recorded += 1


    # Run Simulation

    def run_simulation(self, steps=100000):
        for _ in range(steps):
            self.monte_carlo_step(delta=50)

     # Reports of results including the number of particles, total energy, Monte Carlo Steps, and the average demon energy

    def report_results(self):
        avg_demon_energy = self.demon_energy_sum / self.steps_recorded
        print(f"N: {self.N}")
        print(f"R: {self.R}")
        print(f"Steps : {self.steps_recorded}")
        print(f"Average demon energy: {avg_demon_energy:.4f}")
       
# Create plot of histogram and apply exponential fit


    def plot_histogram(self):
       
        values, counts = np.unique(self.demon_energy_record, return_counts=True)
        probs = counts / np.sum(counts)

       
        def exp_func(E, A, T):
            return A * np.exp(-E / T)

       
        popt, _ = curve_fit(exp_func, values, probs, p0=[1.0, np.mean(self.demon_energy_record)])
        A_fit, T_fit = popt

        
        print("\nFit results:")
        print(f"  A (coefficient) = {A_fit:.5f}")
       

       
        plt.figure(figsize=(8, 5))
        plt.bar(values, probs, alpha=0.6, width=0.9, label='Simulation (histogram)')
        plt.plot(values, exp_func(values, *popt), 'r--',
                 label=f'Fit: $P(E) = {A_fit:.3f} \\, e^{{-E/{T_fit:.3f}}}$')
        plt.xlabel("Demon Energy")
        plt.ylabel("Probability")
        plt.title("Demon Energy Distribution with Exponential Fit")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.xlim(0,60)
        plt.savefig("exponential_fit_phonon.png", dpi=300)
        plt.show()

# Run the code
if __name__ == "__main__":
    sim = PhononDemon(num_sites=100, total_energy=1000)
    sim.run_simulation(steps=100000)
    sim.report_results()
    sim.plot_histogram()
