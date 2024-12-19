from utility import Results
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# load results
results = Results.fromFile('results/dietenbach_average.npz')

# print the results
results.printAll()
# results_normal.printSizings(comparewith=results_average)
# results_normal.printNLPStats(comparewith=results_average)
# results_normal.printCosts(comparewith=results_average)

# Assuming time_values is a numpy array representing the time in hours
time_values = results['timegrid']
Qdot_load = results['Qdot_load']
P_load = results['P_hh']

# Calculate weekly averages
def calculate_weekly_average(data, time_values):
    weeks = np.unique(time_values // (24 * 7))  # Find unique weeks
    weekly_avg = []
    for week in weeks:
        mask = (time_values // (24 * 7)) == week
        weekly_avg.append(np.mean(data[mask]))
    return weeks, np.array(weekly_avg)

# Calculate weekly averages for Qdot_load and P_load
weeks, Qdot_load_weekly = calculate_weekly_average(Qdot_load, time_values)
_, P_load_weekly = calculate_weekly_average(P_load, time_values)

# Convert weeks to time format for plotting
time_week = np.append(weeks * 7, (weeks[-1] + 1) * 7)  # Convert weeks to days and add an extra element

# Plot the data using stairs
plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.stairs(P_load_weekly / 1e6, time_week, label='Electricity Demand (MW, weekly average)', alpha=1)
plt.stairs(Qdot_load_weekly / 1e6, time_week, label='Heat Demand (MW, weekly average)', alpha=1, color='r')


# Formatting the x-axis to show abbreviated month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names

# Optionally set the locator to show ticks at the start of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
plt.gcf().autofmt_xdate()

plt.ylabel('Power (MW)')
plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.02))
plt.grid(True, alpha=0.25)


# Plot the wind and solar power
P_pv = results['P_pv']
P_wind = results['P_wind']

# calculate weekly averages
_, P_pv_weekly = calculate_weekly_average(P_pv, time_values)
_, P_wind_weekly = calculate_weekly_average(P_wind, time_values)

# Plot the data using stairs
plt.subplot(1,2,2)
plt.stairs(P_pv_weekly / 1e6, time_week, label='PV Power (MW, weekly average)', alpha=1, color = 'r')
plt.stairs(P_wind_weekly / 1e6, time_week, label='Wind Power (MW, weekly average)', alpha=1)

# Formatting the x-axis to show abbreviated month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names

# Optionally set the locator to show ticks at the start of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
plt.gcf().autofmt_xdate()

plt.ylabel('Power (MW)')
plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.02))
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('renewables.pdf')
plt.show()

# %% Plot the Power Flows

# Figure of power with the different y-label
plt.figure(figsize=(7, 9))
list_plot = ['P_grid', 'P_re', 'P_hh', 'P_hp', 'P_bat']
ylabel = [r'$P_\mathrm{grid}$ [MW]', r'$P_\mathrm{RE}$ [MW]', r'$P_\mathrm{load}$ [MW]', r'$P_\mathrm{hp}$ [MW]', r'$P_\mathrm{bat}$ [MW]']
for i in range(len(list_plot)):
    plt.subplot(len(list_plot),1,i+1)
    plt.plot(results.timegrid / 24, results[list_plot[i]] / 1E6, linewidth=0.3)
    plt.ylabel(ylabel[i])
    plt.grid(alpha=0.25)
plt.xlabel('Day')
plt.tight_layout()
plt.show()


# %% Make some plots
plt.figure(figsize=(8, 9))
mape_values = []

# Define the explicit labels for the state variables
temperature_labels = results['statenames'][:-1]

for ind_x in range(results['nx']):
    ubx, lbx = float(results['ubx'][ind_x]), float(results['lbx'][ind_x])
    name = results['statenames'][ind_x]
    if name in temperature_labels:
        plt.subplot(2, 1, 1)
        # Change labels for the original results
        # label_org = f"{state_labels[ind_x]}"
        plt.plot(results.timegrid / 24, results.X[:, ind_x] - 273.15, f'C{ind_x}-', label = name, linewidth=0.5)
        plt.axhline(ubx - 273.15, color='grey', linestyle='--')
        plt.axhline(lbx - 273.15, color='grey', linestyle='--')
    else:
        plt.subplot(2, 1, 2)
        # Change labels for the original results
        label_org = f"SOC"
        plt.plot(results.timegrid / 24, results['X'][:, ind_x], f'C{ind_x}-',label=name, linewidth=0.5)
        plt.axhline(ubx, color='grey', linestyle='--')
        plt.axhline(lbx, color='grey', linestyle='--')


plt.subplot(2, 1, 1)
plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05))
plt.ylabel(r'$T$ [°C]')
plt.grid(alpha=0.25)

# Formatting the x-axis to show abbreviated month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names
# Optionally set the locator to show ticks at the start of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.subplot(2, 1, 2)
plt.legend()


# Formatting the x-axis to show abbreviated month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names
# Optionally set the locator to show ticks at the start of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
plt.gcf().autofmt_xdate()

plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
