from systemmodels.averagedstratstoragemodel import AveragedStratStorageModel
from nlpAverage import AverageSTESNLP
from utility import Constants, Data

# the user has to provide data,
# The provided data file should have the following columns:
#         - T_amb: ambient temperature [K (!)]
#         - P_pv: power of the pv [W], of the installed capacity defined below
#         - P_wind: power of the wind [W], of the installed capacity defined below
#         - P_load: electric household power demand [W]
#         - Qdot_load: heat demand of the household [W]
data = Data('data/data_dietenbach.csv')

# default constants, can be overwritten by the user
constants = Constants()

# the user has to provide rough guesses for the sizes of the components,
# for PV and WIND these correspond to the installed capacity that generated the data.
constants.C_bat_default = 2E7  # Wh
constants.C_hp_default = 2E7  # W (thermal)
constants.C_wind_default = 7.14 * 5 * 1e6  # Wp
constants.C_pv_default = 34.69 * 1e6  # Wp (Use instead of 20% rather 18% efficiency)

# build the system model
systemmodel = AveragedStratStorageModel(2, 2, 2, data=data, constants=constants)

# build the NLP
nlp = AverageSTESNLP(systemmodel, data, N=365 * 6)

# solve the NLP and save the results
res = nlp.solve({'ipopt.max_iter': 1000, 'ipopt.linear_solver': 'ma27'})
res.save('results/dietenbach_average.npz')

print("\nOptimal Sizings:\n"+ "-"*20)
res.printSizings()
