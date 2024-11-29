from systemmodels.averagedstratstoragemodel import AveragedStratStorageModel
from systemmodels.systemModel import Data
from nlpAverage import AverageSTESNLP


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
print(f'Fix:\t{res["NLP_J_fix"]:.3e} EUR')
print(f'Run:\t{res["NLP_J_running"]:.3e} EUR')
print(f'Reg:\t{1E6*res["NLP_J_reg"]:.3e} EUR')

res.save('results/dietenbach_average.npz')

