from matplotlib import pyplot as plt

from systemmodels.averagedstratstoragemodel import AveragedStratStorageModel
from systemmodels.systemModel import Data
from nlpAverage import AverageSTESNLP
from utility import Constants

data = Data('data/data_dietenbach.csv')
constants = Constants()

for i in range(1):
    constants.price_sell = 5
    systemmodel =  AveragedStratStorageModel(4, 2, 2, data = data, constants=constants)

    systemmodel.lbp[:] = 0.01
    systemmodel.ubp[:] = 100

    # systemmodel.lbp[3] = 0
    # systemmodel.ubp[3] = 0

    # turn off battery
    # systemmodel.lbu[1:3] = 0
    # systemmodel.ubu[1:3] = 0

    nlp = AverageSTESNLP(systemmodel, data, N = 365*6)
    res = nlp.solve({'ipopt.max_iter': 1000,'ipopt.linear_solver': 'ma97'})

    res.save(f'results/dietenbach_average_varyPrice_{i}.npz')


# %% Make some plots
plt.figure(figsize=(15,9))
for ind_x in range(res['nx']):
    ubx, lbx = float(res['ubx'][ind_x]), float(res['lbx'][ind_x])
    if res['statenames'][ind_x].startswith('T'):
        plt.subplot(2, 1, 1)
        plt.plot(res.timegrid, res.X[:, ind_x] - 273.15, f'C{ind_x}--', label=res['statenames'][ind_x])
        plt.plot(res.timegrid, res.X[:, ind_x] - 273.15, f'C{ind_x}-', label=res['statenames'][ind_x])
        plt.axhline(ubx - 273.15, color='grey', linestyle='--')
        plt.axhline(lbx - 273.15, color='grey', linestyle='--')
    else:
        plt.subplot(2,1,2)
        plt.plot(res.timegrid, res['X'][:, ind_x], label=res['statenames'][ind_x])
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
