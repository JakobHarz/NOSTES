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
constants.C_wind_default = 11.2 * 1e6  # Wp (use 5.6MW two wind turbine, including wake effect)
constants.C_pv_default = 18.56 * 1e6  # Wp (23.2% efficiency, yield 232Wp/m^2, use 80*1e3 modules of 600Wp)

### USER CAN CHANGE THE FOLLOWING CONSTANTS ###
# Different Electricity price scenarios
# for i in range(6):
#     constants.price_buy = 0.1 + 0.1 * i
#     systemmodel.lbp[3] = 0
#     systemmodel.ubp[3] = 0
    # turn off battery
    # systemmodel.lbu[1:3] = 0
    # systemmodel.ubu[1:3] = 0

# build the system model
systemmodel = AveragedStratStorageModel(4, 2, 2, data=data, constants=constants)

# build the NLP
nlp = AverageSTESNLP(systemmodel, data, N=365 * 24)

# solve the NLP and save the results
res = nlp.solve({'ipopt.max_iter': 1000, 'ipopt.linear_solver': 'ma27'})
res.save('results/dietenbach_average_test.npz')

print("\nOptimal Sizings:\n"+ "-"*20)
res.printSizings()
