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
             "axes.labelsize": 14,  
             "font.size": 14,  
             # Make the legend/label fonts a little smaller  
             "legend.fontsize": 14,  
             "xtick.labelsize": 14,  
             "ytick.labelsize": 14,
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
    P_grid_sum = P_grid_positive.sum() 
    J_running_sum = J_running.sum() 
    
    # Append the values to the lists
    P_grid_values.append(P_grid_sum)
    J_total_values.append(J_fix + J_running_sum)
    P_hp_values.append(P_hp.sum() )

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
T_amb = data['T_amb']
T_hp = 40 + 273.15
COP = 0.5 * T_hp / (T_hp - T_amb)
P_hp_base = result_2['Qdot_load'] / COP
P_hp_base_sum = P_hp_base.sum()
P_load_base = result_2['P_hh'].sum()
power_usage = (P_load_base + P_hp_base_sum) * 30


import matplotlib.dates as mdates
time_values = result_2['timegrid']
# Calculate weekly averages
def calculate_weekly_average(data, time_values):
    weeks = np.unique(time_values // (24 * 7))  # Find unique weeks
    weekly_avg = []
    for week in weeks:
        mask = (time_values // (24 * 7)) == week
        weekly_avg.append(np.mean(data[mask]))
    return weeks, np.array(weekly_avg)

# Calculate weekly averages for P_hp and P_hp_base
weeks, P_hp_weekly = calculate_weekly_average(P_hp, time_values)
_, P_hp_base_weekly = calculate_weekly_average(P_hp_base, time_values)

# Convert weeks to time format for plotting
time_week = np.append(weeks * 7, (weeks[-1] + 1) * 7)  # Convert weeks to days and add an extra element

# Plot the data using stairs
plt.figure(figsize=(9, 5))
plt.stairs(P_hp_weekly / 1e6, time_week, label='Full Model (MW, weekly average)', alpha=1)
plt.stairs(P_hp_base_weekly / 1e6, time_week, label='Only Heat Pump (MW, weekly average)', alpha=1, color='r')

# Formatting the x-axis to show abbreviated month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names

# Optionally set the locator to show ticks at the start of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
plt.gcf().autofmt_xdate()

plt.xlabel('Time')
plt.ylabel('$P_\mathrm{hp}$ (MW)')
plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.02))
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()



# Calculate total_cost_base for different values of c_el
c_el_values = np.linspace(0.1, 1.0, 10)
total_cost_base = []

for c_el in c_el_values:
    total_cost = power_usage * c_el / 1000 # Calculate total cost in M EUR
    total_cost_base.append(total_cost)
total_cost_base = np.array(total_cost_base) + 10* 1e6
# Assuming J_total_values is already calculated
# J_total_values should be a list or array of the same length as c_el_values




# # Plotting
# plt.figure(figsize=(7, 4))
# plt.plot(c_el_values, total_cost_base / 1e6, marker='o', linestyle='-', color='b', label='Only Heat Pump')
# plt.plot(c_el_values, J_total_values / 1e6, marker='x', linestyle='--', color='r', label='Full Model')
# plt.xlabel('Electricity Price (EUR/kWh)')
# plt.ylabel('Total Cost for 30 years (M EUR)')
# # plt.yscale('log')  # Set y-axis to logarithmic scale

# # Highlight the value of 0.3 on the x-axis
# highlight_x = 0.3
# plt.axvline(x=highlight_x, color='g', linestyle='--', linewidth=1)

# # Find the corresponding y-values for the highlight_x
# highlight_y1 = np.interp(highlight_x, c_el_values, total_cost_base / 1e6)
# highlight_y2 = np.interp(highlight_x, c_el_values, J_total_values / 1e6)

# # Draw circles around the points at x=0.3
# plt.scatter([highlight_x], [highlight_y1], color='g', s=100, facecolors='none', edgecolors='g')
# plt.scatter([highlight_x], [highlight_y2], color='g', s=100, facecolors='none', edgecolors='g')

# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.savefig('cost_comparison.pdf')
# plt.show()

# %%
# Plotting
plt.figure(figsize=(9, 5))
plt.plot(c_el_values, total_cost_base / 1e6, linestyle='-', color='C0', label='Only Heat Pump')
plt.plot(c_el_values, J_total_values / 1e6, marker='o', linestyle='-', color='C1', label='Full Model')
plt.xlabel('Electricity Price (\euro/kWh)')
plt.ylabel('Total Cost for 30 years (M\euro)')
plt.ylim([100, 1000])
# plt.yscale('log')  # Set y-axis to logarithmic scale

# Highlight the value of 0.3 on the x-axis
highlight_x = 0.3
plt.axvline(x=highlight_x, color='g', linestyle='--', linewidth=1)

# Find the corresponding y-values for the highlight_x
highlight_y1 = np.interp(highlight_x, c_el_values, total_cost_base / 1e6)
highlight_y2 = np.interp(highlight_x, c_el_values, J_total_values / 1e6)

# Draw circles around the points at x=0.3
plt.scatter([highlight_x], [highlight_y1], color='g', s=10, facecolors='none', edgecolors='g')
plt.scatter([highlight_x], [highlight_y2], color='g', s=10, facecolors='none', edgecolors='g')

# Annotate the plot with the level of autonomy for each scenario
for i, (x, y) in enumerate(zip(c_el_values, J_total_values / 1e6)):
    plt.text(x+0.005, y - 10, f'{autonomy[i]:.2f}%', fontsize=12, ha='left', va='top', color='black')

plt.legend()
plt.grid(True, alpha= 0.5)
plt.tight_layout()
plt.savefig('cost_comparison.pdf')
plt.show()
# %%
