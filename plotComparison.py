# %%
from typing import List, Union
from utility import Results

import numpy as np
import matplotlib.pyplot as plt

# load average results
results_average = Results.fromFile('results/dietenbach_average.npz')
results_normal = Results.fromFile('results/dietenbach.npz')
# results_normal.printSizings(comparewith=results_average)
# results_normal.printNLPStats(comparewith=results_average)
results_normal.printAll(comparewith=results_average)

# %% Make some plots
plt.figure(figsize=(15,9))
for ind_x in range(results_average['nx']):
    ubx, lbx = float(results_average['ubx'][ind_x]), float(results_average['lbx'][ind_x])
    if results_average['statenames'][ind_x].startswith('T'):
        plt.subplot(2, 1, 1)
        plt.plot(results_average.timegrid, results_average.X[:, ind_x] - 273.15, f'C{ind_x}--', label=results_average['statenames'][ind_x])
        plt.plot(results_normal.timegrid, results_normal.X[:, ind_x] - 273.15, f'C{ind_x}-', label=results_average['statenames'][ind_x])
        plt.axhline(ubx - 273.15, color='grey', linestyle='--')
        plt.axhline(lbx - 273.15, color='grey', linestyle='--')
    else:
        plt.subplot(2,1,2)
        plt.plot(results_average.timegrid, results_average['X'][:, ind_x], label=results_average['statenames'][ind_x])
        plt.axhline(ubx, color='grey', linestyle='--')
        plt.axhline(lbx, color='grey', linestyle='--')
plt.subplot(2, 1, 1)
plt.legend()
plt.grid(alpha=0.25)
plt.subplot(2, 1, 2)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# # figure with all powers
# plt.figure(figsize=(15,13))
# P_plot_max = np.max(results['U'])
# 
# list_plot = ['P_grid', 'P_re', 'P_hh', 'P_hp', 'P_bat']
# for i in range(len(list_plot)):
#     plt.subplot(len(list_plot),1,i+1)
#     plt.plot(results.timegrid, results[list_plot[i]]/1E6, label=list_plot[i])
#     plt.ylabel('Power [MW]')
#     plt.ylim([-P_plot_max/1E6*0.05, 1.05*P_plot_max/1E6])
#     plt.legend()
#     plt.grid(alpha=0.25)
# plt.tight_layout()
# plt.show()



# figure of controls
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