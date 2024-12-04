# %%
from typing import List, Union
from utility_wind import Results

import numpy as np
import matplotlib.pyplot as plt

#%%
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
           }  
 plt.rcParams.update(params_MPL_Tex)
latexify()

# load average results
results_average = Results.fromFile('results/dietenbach_average_varyPrice_2.npz')
results_nowind = Results.fromFile('results/dietenbach_average_nowind.npz')
results_average.printSizings(comparewith=results_nowind)
results_average.printNLPStats(comparewith=results_nowind)
results_average.printCosts(comparewith=results_nowind)

# results_nowind.printSizings(comparewith=results_average)
# results_nowind.printNLPStats(comparewith=results_average)
# results_nowind.printCosts(comparewith=results_average)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# %% Make some plots
plt.figure(figsize=(8, 9))
mape_values = []

# Define the explicit labels for the state variables
state_labels = {
    0: r"$T_\mathrm{s1}$",
    1: r"$T_\mathrm{s2}$",
    2: r"$T_\mathrm{s3}$",
    3: r"$T_\mathrm{s4}$",
    4: r"$T_\mathrm{g1}$",
    5: r"$T_\mathrm{g2}$"
}

for ind_x in range(results_average['nx']):
    ubx, lbx = float(results_average['ubx'][ind_x]), float(results_average['lbx'][ind_x])
    if ind_x in state_labels:
        plt.subplot(2, 1, 1)
        # Change labels for the original results
        label_org = f"{state_labels[ind_x]}"
        plt.plot(results_average.timegrid / 24, results_average.X[:, ind_x] - 273.15, f'C{ind_x}--')
        plt.plot(results_nowind.timegrid / 24, results_nowind.X[:, ind_x] - 273.15, f'C{ind_x}-', label=label_org)
        plt.axhline(ubx - 273.15, color='grey', linestyle='--')
        plt.axhline(lbx - 273.15, color='grey', linestyle='--')
    else:
        plt.subplot(2, 1, 2)
        # Change labels for the original results
        label_org = f"SOC"
        plt.plot(results_average.timegrid / 24, results_average['X'][:, ind_x], f'C{ind_x}--')
        plt.plot(results_nowind.timegrid / 24, results_nowind['X'][:, ind_x], f'C{ind_x}-', label=label_org)
        plt.axhline(ubx, color='grey', linestyle='--')
        plt.axhline(lbx, color='grey', linestyle='--')

    # Calculate MAPE for the current state variable
    mape = mean_absolute_percentage_error(results_average.X[:, ind_x], results_nowind.X[:, ind_x])
    mape_values.append(mape)

plt.subplot(2, 1, 1)
plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), ncols=6)
plt.ylabel(r'$T$ [°C]')
plt.grid(alpha=0.25)
plt.subplot(2, 1, 2)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# Plot the battery state of charge
plt.figure(figsize=(15, 4.5))
plt.plot(results_average.timegrid, results_average.X[:, -1], label='SOC average', linestyle='--', alpha=0.3)
plt.plot(results_nowind.timegrid, results_nowind.X[:, -1], label='SOC normal', linestyle='-')
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# Figure of power with the different y-label
plt.figure(figsize=(7, 9))
list_plot = ['P_grid', 'P_re', 'P_hh', 'P_hp', 'P_bat']
ylabel = [r'$P_\mathrm{grid}$ [MW]', r'$P_\mathrm{RE}$ [MW]', r'$P_\mathrm{load}$ [MW]', r'$P_\mathrm{hp}$ [MW]', r'$P_\mathrm{bat}$ [MW]']
for i in range(len(list_plot)):
    plt.subplot(len(list_plot),1,i+1)
    plt.plot(results_average.timegrid / 24, results_average[list_plot[i]] / 1E6, linewidth=0.3)
    plt.ylabel(ylabel[i])
    plt.grid(alpha=0.25)
plt.xlabel('Day')
plt.tight_layout()
plt.savefig('power_result.pdf')
plt.show()

P_grid_values = []
J_total_values = []
P_load_values = []
P_hp_values = []

# Load average results
results = [results_average, results_nowind]

for result in results:
    P_grid = result['P_grid']
    J_fix = result['J_fix']
    J_running = result['J_running']
    P_hp = result['P_hp']

    P_grid_positive = P_grid[P_grid > 0]
    P_grid_sum = P_grid_positive.sum() *4
    J_running_sum = J_running.sum() * 4
    
    # Append the values to the lists
    P_grid_values.append(P_grid_sum)
    J_total_values.append(J_fix + J_running_sum)
    P_hp_values.append(P_hp.sum()*4)

P_load = [32 * 1e9] * 2
P_load_sum = np.array(P_load)
P_hp_sum = np.array(P_hp_values)


# Convert lists to Numpy arrays for plotting
P_grid_values = np.array(P_grid_values)
J_total_values = np.array(J_total_values)
P_load_values = np.array(P_load_values)
autonomy = (P_load_sum + P_hp_sum - P_grid_values) / (P_load_sum + P_hp_sum)

print(f"Autonomy: {autonomy}")
print(f"Total cost: {J_total_values / 1e6}")