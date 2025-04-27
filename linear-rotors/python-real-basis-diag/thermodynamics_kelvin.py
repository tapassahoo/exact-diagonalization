import numpy as np
import matplotlib.pyplot as plt

# Set the plotting style globally for consistent aesthetics
plt.rcParams.update({
	"font.size": 14,  # Default font size for text
	"axes.labelsize": 16,  # Label size for axes
	"axes.titlesize": 18,  # Title size for axes
	"legend.fontsize": 12,  # Legend font size
	"xtick.labelsize": 12,  # X-axis tick label size
	"ytick.labelsize": 12,  # Y-axis tick label size
	"figure.figsize": (10, 8),  # Set figure size
	"axes.grid": True,  # Enable grid by default
	"grid.alpha": 0.3,  # Set grid transparency for better visual appeal
	"lines.markersize": 8,  # Marker size for plot points
	"lines.linewidth": 2,  # Line width for plot lines
	"legend.frameon": True,  # Make the legend background visible
	"legend.framealpha": 0.9,  # Set transparency of legend background
})

def compute_probabilities(energy_levels, temperature):
	"""
	Compute the probability distribution for each energy level at a specific temperature.

	Parameters:
	- energy_levels: Array of energy levels (in Kelvin).
	- temperature: The temperature at which probabilities are computed (in Kelvin).

	Returns:
	- probabilities: Probability distribution of energy levels at the given temperature.
	"""
	beta = 1 / temperature  # Inverse temperature (1/K)
	boltzmann_factors = np.exp(-beta * energy_levels)  # Boltzmann factor
	partition_function = np.sum(boltzmann_factors)  # Partition function Z
	probabilities = boltzmann_factors / partition_function  # Probability distribution
	return probabilities

def compute_average_energy(energy_levels, temperature):
	"""
	Compute the average energy at a specific temperature.

	Parameters:
	- energy_levels: Array of energy levels (in Kelvin).
	- temperature: The temperature at which average energy is computed (in Kelvin).

	Returns:
	- average_energy: The computed average energy at the given temperature.
	"""
	probabilities = compute_probabilities(energy_levels, temperature)
	average_energy = np.sum(energy_levels * probabilities)  # <E> = sum(E_i * P(E_i, T))
	return average_energy

def compute_heat_capacity(energy_levels, temperature):
	"""
	Compute the heat capacity C_V at a given temperature.

	Formula: C_V = 1/T^2 * (<E^2> - <E>^2)

	Parameters:
	- energy_levels: Array of energy levels (in Kelvin).
	- temperature: The temperature at which to compute the heat capacity (in Kelvin).

	Returns:
	- heat_capacity: Heat capacity C_V.
	"""
	probabilities = compute_probabilities(energy_levels, temperature)
	
	# Compute <E> (average energy)
	average_energy = compute_average_energy(energy_levels, probabilities)
	
	# Compute <E^2> (average of energy squared)
	average_energy_squared = compute_average_energy(energy_levels**2, probabilities)
	
	# Compute heat capacity C_V
	#heat_capacity = (1 / temperature**2) * (average_energy_squared - average_energy**2)
	heat_capacity = (average_energy_squared - average_energy**2)
	print(heat_capacity)
	return heat_capacity


def plot_probability_profile(energy_levels, fixed_temperature, threshold=0.01):
	"""
	Plot the probability profile at a fixed temperature, restricting x-axis to exclude very low probabilities.

	Parameters:
	- energy_levels: Array of energy levels (in Kelvin).
	- fixed_temperature: The fixed temperature at which to plot the probability distribution.
	- threshold: Minimum probability to display on the plot (default is 0.01).
	"""
	probabilities_at_fixed_temp = compute_probabilities(energy_levels, fixed_temperature)

	# Filter out the energy levels where the probability is very low
	valid_indices = np.where(probabilities_at_fixed_temp > threshold)[0]
	valid_energy_levels = energy_levels[valid_indices]
	valid_probabilities = probabilities_at_fixed_temp[valid_indices]

	# Create a plot for Probability vs Basis Index using a bar plot for clarity
	plt.figure(figsize=(10, 6))

	# Bar plot to show probabilities for each energy level
	bars = plt.bar(valid_indices, valid_probabilities, color='tab:blue', edgecolor='black')

	# Adding annotations for each bar
	for i, bar in enumerate(bars):
		plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.3f}', 
				 ha='center', va='bottom', fontsize=12, color='black')

	# Set labels and title
	plt.xlabel('Basis Index', fontsize=14)
	plt.ylabel('Probability', fontsize=14)
	plt.title(f'Probability Profile at T = {fixed_temperature} K', fontsize=16)

	# Customize grid and axis limits
	plt.grid(True, which='both', linestyle='--', alpha=0.6)  # Grid styling
	plt.ylim(0, 1)  # Set y-axis limit to make the probability distribution clear

	# Customize x-axis ticks and range
	tick_step = max(1, len(valid_energy_levels) // 10)  # Only show every 10th basis index
	plt.xticks(range(0, len(valid_energy_levels), tick_step), 
			   [f'Basis {i}' for i in range(0, len(valid_energy_levels), tick_step)])

	# Rotate x-axis labels for better readability (if needed)
	plt.xticks(rotation=45, ha='right')  # Rotate labels by 45 degrees to avoid overlap
	plt.tight_layout()

	# Show plot
	plt.show()


def plot_average_energy_vs_temperature(energy_levels, temperatures):
	"""
	Plot average energy vs temperature.

	Parameters:
	- energy_levels: Array of energy levels (in Kelvin).
	- temperatures: Array of temperatures (in Kelvin) for average energy vs temperature plot.
	"""
	average_energies = np.array([compute_average_energy(energy_levels, T) for T in temperatures])

	# Create a plot for Average Energy vs Temperature
	plt.figure(figsize=(10, 6))
	plt.plot(temperatures, average_energies, marker='o', linestyle='-', color='tab:green', label='Average Energy')
	plt.xlabel('Temperature (K)', fontsize=14)
	plt.ylabel('Average Energy (K)', fontsize=14)
	plt.title('Average Energy vs Temperature', fontsize=16)
	plt.legend()
	plt.grid(True, which='both', linestyle='--', alpha=0.6)  # Grid styling
	plt.tight_layout()
	plt.show()


def plot_heat_capacity_vs_temperature(energy_levels, temperature_range):
	"""
	Plot heat capacity C_V as a function of temperature.

	Parameters:
	- energy_levels: Array of energy levels (in Kelvin).
	- temperature_range: Range of temperatures (in Kelvin).
	"""
	heat_capacities = []
	
	for T in temperature_range:
		heat_capacity = compute_heat_capacity(energy_levels, T)
		heat_capacities.append(heat_capacity)
	
	# Plot heat capacity
	plt.figure(figsize=(10, 6))
	plt.plot(temperature_range, heat_capacities, color='tab:red', marker='o', linestyle='-', label='Heat Capacity C_V')
	plt.xlabel('Temperature (K)', fontsize=14)
	plt.ylabel('Heat Capacity (J/K)', fontsize=14)
	plt.title('Heat Capacity vs Temperature', fontsize=16)
	plt.grid(True, which='both', linestyle='--', alpha=0.6)
	plt.legend()
	plt.tight_layout()
	plt.show()
