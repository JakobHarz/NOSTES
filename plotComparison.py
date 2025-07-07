# %%
from typing import List, Union
from utility import Results
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

#%%
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
             "ytick.labelsize": 14  
           }  
 plt.rcParams.update(params_MPL_Tex)
latexify()

# load average results
results_average = Results.fromFile('results/dietenbach_average.npz')
results_normal = Results.fromFile('results/dietenbach.npz')
# results_normal.printSizings(comparewith=results_average)
# results_normal.printNLPStats(comparewith=results_average)
# results_normal.printCosts(comparewith=results_average)
results_normal.printAll(comparewith=results_average)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




# %%

# Assuming time_values is a numpy array representing the time in hours
time_values = results_average['timegrid']
Qdot_load = results_average['Qdot_load']
P_load = results_average['P_hh']

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
time_week = np.append(weeks * 7, (weeks[-1]) * 7)  # Convert weeks to days and add an extra element

# Plot the data using stairs
plt.figure(figsize=(9, 5))
plt.stairs(P_load_weekly / 1e6, time_week, label='Electricity Demand (weekly average)', alpha=1)
plt.stairs(Qdot_load_weekly / 1e6, time_week, label='Heat Demand (weekly average)', alpha=1, color='r')


# Formatting the x-axis to show abbreviated month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names

# Optionally set the locator to show ticks at the start of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
plt.gcf().autofmt_xdate()

plt.ylabel('Power (MW)')
plt.legend(loc='upper left', bbox_to_anchor=(0.25, 0.95))
plt.grid(True, alpha=0.25)
plt.tight_layout()
# plt.savefig('weekly_loads.pdf')
plt.show()


#%% Plot the wind and solar power   
P_pv = results_average['P_pv']
P_wind = results_average['P_wind']

# calculate weekly averages
_, P_pv_weekly = calculate_weekly_average(P_pv, time_values)
_, P_wind_weekly = calculate_weekly_average(P_wind, time_values)

# Plot the data using stairs
plt.figure(figsize=(9, 5))
plt.stairs(P_pv_weekly / 1e6, time_week, label='PV Power (weekly average)', alpha=1, color = 'r')
plt.stairs(P_wind_weekly / 1e6, time_week, label='Wind Power (weekly average)', alpha=1)

import datetime

# Function to calculate the middle of each month
def middle_of_month(month):
    first_day = datetime.datetime(2023, month, 1)
    next_month = datetime.datetime(2023, month + 1, 1)
    middle_day = first_day + (next_month - first_day) / 2
    
    
    return middle_day

# Set major locator to the start and end of each month
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

middle_dates = [middle_of_month(month) for month in range(1, 12)]
# append December
middle_dates.append(datetime.datetime(2023, 12, 16, 12, 0))
# just keep the middle dates on the 12 months as a list without the datetime
middle_dates = [date.day for date in middle_dates]

# Set minor locator to the middle of each month
plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=16))

# Set formatter to show abbreviated month names for minor ticks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Hide the minor tick marks but show the labels
plt.gca().tick_params(which='major', length=0)
plt.gca().tick_params(which='minor', labelsize=0)

plt.gca().set_xlim([-5, 370])

plt.ylabel('Power (MW)')
plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
plt.grid(True, alpha=0.25)
plt.tight_layout()
# plt.savefig('renewables.pdf')
plt.show()

# %%
# Print the storage efficiency
Qdot_hp_sum = results_average['Qdot_hp'].sum()

storage_eff = Qdot_load.sum() / Qdot_hp_sum
print(f"Storage efficiency: {storage_eff}")




# # figure with all powers
# plt.figure(figsize=(7,13))
# P_plot_max = np.max(results_average['U'])

# list_plot = ['P_grid', 'P_re', 'P_hh', 'P_hp', 'P_bat']
# # list_plot = [r'$P_\mathrm{grid}$', r'$P_\mathrm{re}$', r'$P_\mathrm{load}$', r'$P_\mathrm{hp}$', r'$P_\mathrm{bat}$']
# for i in range(len(list_plot)):
#     plt.subplot(len(list_plot),1,i+1)
#     plt.plot(results_average.timegrid/24, results_average[list_plot[i]]/1E6)
#     plt.ylabel('Power [MW]')
#     # plt.ylim([-P_plot_max/1E6*0.05, 1.05*P_plot_max/1E6])
#     plt.legend()
#     plt.grid(alpha=0.25)
# plt.tight_layout()
# plt.show()

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
# plt.savefig('power_result.pdf')
plt.show()

# Define the list of plots and labels
list_plot = ['P_grid', 'P_re', 'P_hh', 'P_hp', 'P_bat']
ylabel = [r'$P_\mathrm{grid}$ [MW]', r'$P_\mathrm{RE}$ [MW]', r'$P_\mathrm{load}$ [MW]', r'$P_\mathrm{hp}$ [MW]', r'$P_\mathrm{bat}$ [MW]']

# Plot only the first plot separately
plt.figure(figsize=(9, 5))
plt.plot(results_average.timegrid / 24, results_average[list_plot[0]] / 1E6, linewidth=0.3)
plt.ylabel(ylabel[0])

# Formatting the x-axis to show abbreviated month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names
# Optionally set the locator to show ticks at the start of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
plt.gcf().autofmt_xdate()
plt.grid(alpha=0.25)
plt.tight_layout()
# plt.savefig('grid_power.pdf')
plt.show()


P_grid = results_average['P_grid']
P_grid_pos = P_grid[P_grid > 0]
P_grid_pos_sum = P_grid_pos.sum()
P_grid_neg = P_grid[P_grid < 0]
P_grid_neg_sum = P_grid_neg.sum()
print(f"Grid Usage: {P_grid_pos_sum}")
print(f"Grid Feed-in: {P_grid_neg_sum}")



# # figure of controls
# plt.figure(figsize=(15,9))
# for ind_u in range(5):
#     plt.subplot(5,1,ind_u+1)
#     plt.plot(res.timegrid, res['U'][:,ind_u]/1E6, label=systemmodel.controlNames[ind_u])
#     plt.ylabel('Power [MW]')
#     plt.ylim([-P_plot_max/1E6*0.05, 1.05*P_plot_max/1E6])
#     plt.legend()
#     plt.grid(alpha=0.25)
# plt.tight_layout()
# plt.show()


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
        plt.plot(results_normal.timegrid / 24, results_normal.X[:, ind_x] - 273.15, f'C{ind_x}-', label=label_org)
        plt.axhline(ubx - 273.15, color='grey', linestyle='--')
        plt.axhline(lbx - 273.15, color='grey', linestyle='--')
    else:
        plt.subplot(2, 1, 2)
        # Change labels for the original results
        label_org = f"SOC"
        plt.plot(results_average.timegrid / 24, results_average['X'][:, ind_x], f'C{ind_x}--')
        plt.plot(results_normal.timegrid / 24, results_normal['X'][:, ind_x], f'C{ind_x}-', label=label_org)
        plt.axhline(ubx, color='grey', linestyle='--')
        plt.axhline(lbx, color='grey', linestyle='--')

    # Calculate MAPE for the current state variable
    mape = mean_absolute_percentage_error(results_average.X[:, ind_x], results_normal.X[:, ind_x])
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
plt.plot(results_normal.timegrid, results_normal.X[:, -1], label='SOC normal', linestyle='-')
plt.legend()
plt.grid(alpha=0.25)
plt.show()

mape_values = np.array(mape_values)
print(f"MAPE values: {mape_values}")
mape_all = mean_absolute_percentage_error(results_average.X[:, :-1], results_normal.X[:, :-1])
print(f"MAPE all: {mape_all}")


#%%
plt.figure(figsize=(9, 5))
mape_values = []

for ind_x in range(results_average['nx']):
    ubx, lbx = float(results_average['ubx'][ind_x]), float(results_average['lbx'][ind_x])
    if ind_x in state_labels:
        # Change labels for the original results
        label_org = f"{state_labels[ind_x]}"
        #plt.plot(results_average.timegrid / 24, results_average.X[:, ind_x] - 273.15, f'C{ind_x}--', linewidth=0.5)
        plt.plot(results_normal.timegrid / 24, results_normal.X[:, ind_x] - 273.15, f'C{ind_x}-', label=label_org, linewidth=0.5)
        plt.axhline(ubx - 273.15, color='grey', linestyle='--', linewidth=1)
        plt.axhline(lbx - 273.15, color='grey', linestyle='--', linewidth=1)

    # Calculate MAPE for the current state variable
    mape = mean_absolute_percentage_error(results_average.X[:, ind_x], results_normal.X[:, ind_x])
    mape_values.append(mape)

plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), ncols=6)
plt.ylabel(r'$T$ [°C]')
#plt.xlabel('Month')

# Formatting the x-axis to show abbreviated month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names
# Optionally set the locator to show ticks at the start of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
plt.gcf().autofmt_xdate()
plt.grid(alpha=0.25)
plt.tight_layout()
# plt.savefig('org.pdf')
plt.show()

#%% for graphical abstract
plt.figure(figsize=(12, 5))
mape_values = []

for ind_x in range(results_average['nx']):
    ubx, lbx = float(results_average['ubx'][ind_x]), float(results_average['lbx'][ind_x])
    if ind_x in state_labels:
        # Change labels for the original results
        #label_org = f"{state_labels[ind_x]}"
        plt.plot(results_average.timegrid / 24, results_average.X[:, ind_x] - 273.15, f'C{ind_x}')
        # plt.axhline(ubx - 273.15, color='grey', linestyle='--', linewidth=1)
        # plt.axhline(lbx - 273.15, color='grey', linestyle='--', linewidth=1)

#plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), ncols=6)
#plt.ylabel(r'$T$ [°C]')
#plt.xlabel('Month')

# Formatting the x-axis to show abbreviated month names
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names
# Optionally set the locator to show ticks at the start of each month
#plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
#plt.gcf().autofmt_xdate()
#plt.grid(alpha=0.25)

# Remove x and y axis numbers
# plt.gca().set_xticklabels([])
# plt.gca().set_yticklabels([])

# remove x and y axis
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)

# Keep x and y axis lines (spines)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

plt.tight_layout()
# plt.savefig('org.png')
plt.show()
#%%
# Define the list of state variables for storage dynamics
storage_state_labels = {
    0: r"$T_\mathrm{s1}$",
    1: r"$T_\mathrm{s2}$",
    2: r"$T_\mathrm{s3}$",
    3: r"$T_\mathrm{s4}$",
    4: r"$T_\mathrm{g1}$",
    5: r"$T_\mathrm{g2}$"
}

# Create a new plot for storage dynamics
fig, ax = plt.subplots(figsize=(9, 5))
mape_values = []

for ind_x in range(results_average['nx']):
    ubx, lbx = float(results_average['ubx'][ind_x]), float(results_average['lbx'][ind_x])
    if ind_x in storage_state_labels:
        # Change labels for the original results
        label_org = f"{storage_state_labels[ind_x]}"
        ax.plot(results_average.timegrid / 24, results_average.X[:, ind_x] - 273.15, f'C{ind_x}--', linewidth=0.5, alpha = 0.5)
        ax.plot(results_normal.timegrid / 24, results_normal.X[:, ind_x] - 273.15, f'C{ind_x}-', label=label_org, linewidth=0.5, alpha = 0.5)
        ax.axhline(ubx - 273.15, color='grey', linestyle='--', linewidth=1)
        ax.axhline(lbx - 273.15, color='grey', linestyle='--', linewidth=1)

    # Calculate MAPE for the current state variable
    mape = mean_absolute_percentage_error(results_average.X[:, ind_x], results_normal.X[:, ind_x])
    mape_values.append(mape)

ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.00), ncols=6)
ax.set_ylabel(r'$T$ [°C]')
# ax.set_xlabel('Month')
ax.grid(alpha=0.25)

# Formatting the x-axis to show abbreviated month names
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' for abbreviated month names

# Optionally set the locator to show ticks at the start of each month
ax.xaxis.set_major_locator(mdates.MonthLocator())

# Rotate the x-axis labels if needed for better readability
fig.autofmt_xdate()

plt.tight_layout()


#Comment out the zoom
axins = inset_axes(ax, width="30%", height="30%", loc='lower center', bbox_to_anchor=(0.3, 0.2, 1, 1), bbox_transform=ax.transAxes)
# Define the zoom region
start_day = 330  # Example start day for zoom
end_day = start_day + 2  # Example end day for zoom (one week)
start_hour = start_day * 24
end_hour = end_day * 24

# Plot the zoomed region in the inset
for ind_x in range(results_average['nx']):
    if ind_x in storage_state_labels:
        axins.plot(results_average.timegrid / 24, results_average.X[:, ind_x] - 273.15, f'C{ind_x}--', linewidth=0.5)
        axins.plot(results_normal.timegrid / 24, results_normal.X[:, ind_x] - 273.15, f'C{ind_x}-', linewidth=0.5)

# Zoom in on the desired x and y region in the inset
axins.set_xlim(start_day, end_day)
axins.set_ylim(70, 85)

# Remove ticks from the inset
axins.set_xticks(range(start_day, end_day + 1))
axins.set_xticklabels([f'{day}' for day in range(start_day, end_day + 1)])
axins.set_ylabel(r'$T$ [°C]')

axins.set_xlabel('Day')

# Remove the grid from the inset
#ax_inset.grid(False)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
# plt.savefig('org_avg.pdf')
plt.show()


# %%
