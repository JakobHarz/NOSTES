from utility import Results
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# load results
results_average = Results.fromFile('../dietenbach_average.npz')
results_full = Results.fromFile('../dietenbach.npz')

# %% Compare Normal and Average Results

print(" --- Comparing Normal and Average Results --- ")
print("\n1. Relative Differences of Sizing Parameters")
for key in ['V_s','C_bat','C_hp', 'C_pv', 'C_wind']:
    # compare the values, compute relative difference wrt the normal results
    normal_value = results_full[key]
    average_value = results_average[key]
    relative_difference = (average_value - normal_value) / normal_value * 100
    print(f"{key}: {relative_difference:.2f}%")

# plot the differences in power flows
print('\n2. Relative Differences of Total Power Flows')
for key in ['P_grid', 'P_re', 'P_hh', 'P_hp', 'P_bat']:
    power_average = results_average[key]
    power_average_total = np.sum(np.abs(power_average))

    power_normal = results_full[key]
    power_normal_total = np.sum(np.abs(power_normal))
    diff_rel = (power_normal_total - power_average_total) / power_normal_total * 100

    # compute RMSE
    rmse = np.sqrt(np.mean((power_average - power_normal) ** 2))
    rel_rmse = rmse / (np.max(power_normal) - np.min(power_normal)) * 100
    print(f"{key}: {rel_rmse:.2f} %")

print('3. Relative Differences of Temperature Profiles')
for index,key in enumerate(results_average['statenames'][:-1]):  # Exclude the last state variable (usually SOC)
    temperatures_normal = results_full.X[:, index]
    temperatures_average = results_average.X[:, index]

    # rmse
    rmse = np.sqrt(np.mean((temperatures_average - temperatures_normal) ** 2))
    rel_rmse = rmse / (np.max(temperatures_normal) - np.min(temperatures_normal)) * 100
    print(f"{key}: {rel_rmse:.2f} %")
