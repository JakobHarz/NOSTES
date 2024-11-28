import numpy as np

from systemmodels.averagedstratstoragemodel import AveragedStratStorageModel
from systemmodels.systemModel import Data
from systemmodels.stratstoragemodel import Strat_storage_model
from buildNLP_average import AverageSTESNLP
import matplotlib.pyplot as plt


data = Data('data/data_dietenbach.csv')
systemmodel =  AveragedStratStorageModel(2, 2, 2, data = data)

systemmodel.lbp[:] = 0.01
systemmodel.ubp[:] = 100

# systemmodel.lbp[3] = 0
# systemmodel.ubp[3] = 0

# turn off battery
# systemmodel.lbu[1:3] = 0
# systemmodel.ubu[1:3] = 0


nlp = AverageSTESNLP(systemmodel, data, N = 365*6)
res = nlp.solve({'ipopt.max_iter': 1000,'ipopt.linear_solver': 'ma97'})


#%% Some printouts

print('Optimal Parameters')
for ind, name in enumerate(systemmodel.p_fix_names):
    print(f'{name}:\t{res["p_fix"][ind]}')

print('Investement Costs')
print(f'{res["J_fix"]:.2e} EUR')


print("NLP costs")
print(f'Fix:\t{res["NLP_costs"]["J_fix"]:.3e} EUR')
print(f'Run:\t{res["NLP_costs"]["J_running"]:.3e} EUR')
print(f'Reg:\t{1E6*res["NLP_costs"]["J_reg"]:.3e} EUR')

# %%

plt.figure(figsize=(15,9))
for ind_x in range(systemmodel.nx):
    ubx, lbx = float(systemmodel.ubx[ind_x]), float(systemmodel.lbx[ind_x])
    if systemmodel.stateNames[ind_x].startswith('T'):
        plt.subplot(2, 1, 1)
        plt.plot(res['time_grid'], res['X'][:,ind_x] - 273.15, label=systemmodel.stateNames[ind_x])
        plt.axhline(ubx - 273.15, color='grey', linestyle='--')
        plt.axhline(lbx - 273.15, color='grey', linestyle='--')
    else:
        plt.subplot(2,1,2)
        plt.plot(res['time_grid'], res['X'][:,ind_x], label=systemmodel.stateNames[ind_x])
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

# figure with all powers
plt.figure(figsize=(15,13))
P_plot_max = np.max(res['U'])

list_plot = ['P_grid', 'P_re', 'P_hh', 'P_hp', 'P_bat']
for i in range(len(list_plot)):
    plt.subplot(len(list_plot),1,i+1)
    plt.plot(res['time_grid'], res[list_plot[i]]/1E6, label=list_plot[i])
    plt.ylabel('Power [MW]')
    plt.ylim([-P_plot_max/1E6*0.05, 1.05*P_plot_max/1E6])
    plt.legend()
    plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()



# figure of controls
plt.figure(figsize=(15,9))
for ind_u in range(5):
    plt.subplot(5,1,ind_u+1)
    plt.plot(res['time_grid'], res['U'][:,ind_u]/1E6, label=systemmodel.controlNames[ind_u])
    plt.ylabel('Power [MW]')
    plt.ylim([-P_plot_max/1E6*0.05, 1.05*P_plot_max/1E6])
    plt.legend()
    plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
