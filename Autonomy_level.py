# %%
from typing import List, Union
from utility import Results
import numpy as np
import matplotlib.pyplot as plt

def latexify():  
 params_MPL_Tex = {  
             'text.usetex': True,  
             'font.family': 'serif',  
             # Use 10pt font in plots, to match 10pt font in document  
             "axes.labelsize": 10,  
             "font.size": 10,  
             # Make the legend/label fonts a little smaller  
             "legend.fontsize": 10,  
             "xtick.labelsize": 10,  
             "ytick.labelsize": 10,
             'text.latex.preamble': r'\usepackage{eurosym}'  # Include the eurosym package
           }  
 plt.rcParams.update(params_MPL_Tex)

latexify()


# Load average results
results = []
sce = 10
for i in range(sce):
    results_i = Results.fromFile(f'results/dietenbach_average_varyPrice_{i}.npz')
    results_i.printAll()
    results.append(results_i)

# result_0 = Results.fromFile('results/dietenbach_average_varyPrice_0.npz')
# result_1 = Results.fromFile('results/dietenbach_average_varyPrice_1.npz')
result_2 = Results.fromFile('results/dietenbach_average_varyPrice_2.npz')
result_3 = Results.fromFile('results/dietenbach_average_varyPrice_3.npz')

P_grid_values = []
J_total_values = []
P_load_values = []
P_hp_values = []

for result in results:
    P_grid = result['P_grid']
    J_fix = result['J_fix']
    J_running = result['J_running']
    P_hp = result['P_hp']

    P_grid_positive = P_grid[P_grid > 0]
    P_grid_sum = P_grid_positive.sum() * 4#* 4 # step
    J_running_sum = J_running.sum() * 4
    
    # Append the values to the lists
    P_grid_values.append(P_grid_sum)
    J_total_values.append(J_fix + J_running_sum)
    P_hp_values.append(P_hp.sum() *4)

P_load = [32 * 1e9] * (sce)
P_load_sum = np.array(P_load)
P_hp_sum = np.array(P_hp_values)


# Convert lists to Numpy arrays for plotting
P_grid_values = np.array(P_grid_values)
J_total_values = np.array(J_total_values)
P_load_values = np.array(P_load_values)
autonomy = (P_load_sum + P_hp_sum - P_grid_values) / (P_load_sum + P_hp_sum)

# %% Plotting
plt.figure(figsize=(6, 4))
plt.plot(autonomy, J_total_values / 1e6, marker='o', linestyle='-', color='b')
# plt.xscale('log')
plt.xlabel('Autonomy')
plt.ylabel('Total Cost for 30 years (M\euro)')
#plt.xlim([0.5, 1])
# plt.title('Comparison of J_fix + J_running against P_grid')
# plt.legend()
plt.grid(True)
plt.savefig('autonomy_level.pdf')
plt.show()


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(P_grid_values/1e6, J_total_values / 1e6, marker='o', linestyle='-', color='b')
plt.xlabel('$P_\mathrm{grid} $(MW)')
plt.ylabel('Total Cost (M EUR)')
plt.xscale('log')
# plt.title('Comparison of J_fix + J_running against P_grid')
plt.legend()
plt.grid(True)
plt.show()


# %% Plot the base scenario
import pandas as pd

# Load data
data = pd.read_csv('data/data_dietenbach.csv')
T_amb = data['T_amb'][::4]
T_hp = 86 + 273.15
COP = 0.5 * T_hp / (T_hp - T_amb)
P_hp_base = result_2['Qdot_load'] / COP
P_hp_base_sum = P_hp_base.sum() * 4
P_load_base = result_2['P_hh'].sum() * 4
power_usage = (P_load_base + P_hp_base_sum) * 30

# Calculate total_cost_base for different values of c_el
c_el_values = np.linspace(0.1, 1.0, 10)
total_cost_base = []

for c_el in c_el_values:
    total_cost = power_usage * c_el / 1000 # Calculate total cost in M EUR
    total_cost_base.append(total_cost)
total_cost_base = np.array(total_cost_base) + 10* 1e6
# Assuming J_total_values is already calculated
# J_total_values should be a list or array of the same length as c_el_values


# Plotting
plt.figure(figsize=(7, 4))
plt.plot(c_el_values, total_cost_base / 1e6, marker='o', linestyle='-', color='b', label='Only Heat Pump')
plt.plot(c_el_values, J_total_values / 1e6, marker='x', linestyle='--', color='r', label='Full Model')
plt.xlabel('Electricity Price (EUR/kWh)')
plt.ylabel('Total Cost for 30 years (M EUR)')
# plt.yscale('log')  # Set y-axis to logarithmic scale

# Highlight the value of 0.3 on the x-axis
highlight_x = 0.3
plt.axvline(x=highlight_x, color='g', linestyle='--', linewidth=1)

# Find the corresponding y-values for the highlight_x
highlight_y1 = np.interp(highlight_x, c_el_values, total_cost_base / 1e6)
highlight_y2 = np.interp(highlight_x, c_el_values, J_total_values / 1e6)

# Draw circles around the points at x=0.3
plt.scatter([highlight_x], [highlight_y1], color='g', s=100, facecolors='none', edgecolors='g')
plt.scatter([highlight_x], [highlight_y2], color='g', s=100, facecolors='none', edgecolors='g')

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('cost_comparison.pdf')
plt.show()

# %%
