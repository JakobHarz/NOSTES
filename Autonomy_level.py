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
    if i == 2:
        continue  # Skip results_2
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

P_load = [32 * 1e9] * (sce-1)
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
# %%
