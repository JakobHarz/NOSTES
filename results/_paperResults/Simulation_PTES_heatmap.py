import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import pandas as pd
from PTES_1 import ThermalModel as PTES_g
from PTES_n1 import ThermalModel as sto_n1
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import FuncFormatter
# Define the formatting function
def scientific_to_latex(x, pos):
    exponent = int(np.log10(x))
    coefficient = x / (10 ** exponent)
    return f'${coefficient:.1f} \\cdot 10^{{{exponent}}}$'
def latexify():  
 params_MPL_Tex = {  
             'text.usetex': True,  
             'font.family': 'serif',  
             # Use 10pt font in plots, to match 10pt font in document  
             "axes.labelsize": 16,  
             "font.size": 16,  
             # Make the legend/label fonts a little smaller  
             "legend.fontsize": 16,  
             "xtick.labelsize": 16,  
             "ytick.labelsize": 16  
           }  
 plt.rcParams.update(params_MPL_Tex)
latexify()

df = pd.read_csv('../../data/data_dietenbach.csv')


# Simulation parameters
sim_year = 1
N = 8760 * sim_year  # Number of discretization points (1 year with hourly resolution)
T = 3600 * 24 * 365 * sim_year  # Total simulation time in seconds
h = T / N  # Time step size in seconds

# Extract time (in hours) and temperature values
time_hours = np.arange(N)
T_amb = df['T_amb']

# Import Heat Demand
heat_demand = df['Qdot_load']

# Parameters
tl = 153.3
bl = 73.2
U_wall = 90
T_init = 60 + 273.15
T_bc_init = 13.5 + 273.15
T_min = 10 + 273.15

# Simulation setup
def simulate_storage(model_class, tl, bl, num_layers, distance, U_wall):
    model = model_class(tl=tl, bl=bl, iter=num_layers, distance=distance, U_wall=U_wall)
    x = model.state
    u = model.con
    par = model.par
    f = model.f

    # Initial conditions
    x0 = ca.DM([T_init] + [273.15 + 13.5] * num_layers)

    # Build a forward Euler integrator
    FE = ca.Function('x1', [x, u, par], [x + h * f(x, u, par)])

    # Simulation parameters
    Qdot_add = np.full(N, 0)
    con = ca.horzcat(Qdot_add, Qdot_add).T
    T_bc = np.full(N, T_bc_init)
    Qdot_load = np.zeros(N)
    c_el = np.zeros(N)
    COP = np.zeros(N)
    par = ca.horzcat(T_bc, Qdot_load, c_el, COP, T_amb).T

    # Simulate the system
    Xsim = FE.mapaccum(N)(x0, con, par)
    return Xsim


# Calculate ground truth (X_tilde) with 200 layers and 100m total distance
X_tilde = simulate_storage(PTES_g, tl, bl, num_layers=200, distance=0.5, U_wall=U_wall)[0, :].full().T

# Scenarios for layers 1 to 10 and total distances from 1 to 10 meters
layer_counts = range(1, 11)
total_distances = range(1, 11)

scenarios = []
for num_layers in layer_counts:
    for total_distance in total_distances:
        distance_per_layer = total_distance / num_layers
        model_class = sto_n1 if num_layers == 1 else PTES_g
        Xsim = simulate_storage(model_class, tl, bl, num_layers=num_layers, distance=distance_per_layer, U_wall=U_wall)
        endpoint = Xsim[0, :].full().T
        # error = np.linalg.norm(endpoint - X_tilde)
        # error = np.log10(error)  # Logarithmic error

        # Calculate Logarithmic RMSE
        error = np.log10(endpoint) - np.log10(X_tilde)
        squared_error = error ** 2
        mean_squared_error = np.mean(squared_error)
        rmse = np.sqrt(mean_squared_error) 

        scenarios.append({'N': num_layers, 'Total_Distance': total_distance, 'error': rmse})

# Extract unique N and Total_Distance values
N_values = sorted(set(scenario['N'] for scenario in scenarios))
Total_Distance_values = sorted(set(scenario['Total_Distance'] for scenario in scenarios))

# Create a 2D array to store error values
error_matrix = np.full((len(N_values), len(Total_Distance_values)), np.nan)

# Fill the error matrix
for scenario in scenarios:
    N_index = N_values.index(scenario['N'])
    Total_Distance_index = Total_Distance_values.index(scenario['Total_Distance'])
    if scenario['error'] < 1000:  # Adjust the threshold for normalized error
        error_matrix[N_index, Total_Distance_index] = scenario['error']

# Create a heatmap
plt.figure(figsize=(10, 8))
#plt.imshow(error_matrix, aspect='auto', cmap='viridis', origin='lower', interpolation='none')
plt.imshow(error_matrix, norm=LogNorm(vmin=1e-4, vmax=1e-1), cmap='viridis')

plt.colorbar(label='log(Error)')

# Set the ticks and labels
plt.xticks(ticks=np.arange(len(Total_Distance_values)), labels=Total_Distance_values)
plt.yticks(ticks=np.arange(len(N_values)), labels=N_values)
plt.xlabel('Total Distance (m)')
plt.ylabel('Number of Layers (N)')
plt.title('Normalized Error Values Compared to X_tilde')

# Show the plot
plt.show()

levels = np.logspace(-5, -2, num=15)  # 10 levels from 10^-3 to 10^-1

plt.figure(figsize=(16, 7.5))
plt.contourf(error_matrix, levels = levels, norm=LogNorm(vmin=1e-5, vmax=1e-2), cmap='viridis')
cbar = plt.colorbar(format=LogFormatter(base=10, labelOnlyBase=False))
cbar.set_label('RMLSE')
cbar.ax.yaxis.set_major_formatter(FuncFormatter(scientific_to_latex))

# plt.colorbar(label='$\log\epsilon$')
# Set the ticks and labels
plt.xticks(ticks=np.arange(len(Total_Distance_values)), labels=Total_Distance_values)
plt.yticks(ticks=np.arange(len(N_values)), labels=N_values)
plt.xlabel('Distance d (m)')
plt.ylabel('Number of Layers n (-)')

# Add a dot or x mark at (5 meters, 2 layers)
distance_index = Total_Distance_values.index(4)
layer_index = N_values.index(2)
plt.scatter(distance_index, layer_index, color='red', marker='x', s=30, label='5m, 2 layers')

# Add a legend
#plt.legend()
plt.tight_layout()
plt.savefig('heatmap.pdf')
plt.show()


# Additional simulations for specific cases
# Xsim_n0 = simulate_storage(sto_n0, tl, bl, 0, 100, U_wall)
# Xsim_n1 = simulate_storage(sto_n1, tl, bl, 1, 3, U_wall)
Xsim_n2 = simulate_storage(PTES_g, tl, bl, 2, 4, U_wall)
X_sim_tilde_500 = simulate_storage(PTES_g, tl, bl, 500, 0.2, U_wall)

error_n2 = np.log(Xsim_n2[0, :].full().T +1) - np.log(X_tilde +1)
squared_error_n2 = error_n2 ** 2
mean_squared_error_n2 = np.mean(squared_error_n2)
rmse_n2 = np.sqrt(mean_squared_error_n2)


# Plot temperature over time for specific cases
plt.figure(figsize=(9, 4))
plt.plot(np.arange(N) / (N / 365), X_sim_tilde_500[0, :].full().T - 273.15, label=r'$\tilde{T_s}$')
plt.plot(np.arange(N) / (N / 365), Xsim_n2[0, :].full().T - 273.15, label='$T_s(n=2, d=4)$')

plt.xlabel('Day')
plt.legend()
plt.ylabel('$T$ [°C]')
plt.tight_layout()
plt.savefig('temp_profile.pdf')
plt.show()
