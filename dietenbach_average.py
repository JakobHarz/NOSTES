from systemmodels.averagedstratstoragemodel import AveragedStratStorageModel
from systemmodels.systemModel import Data
from nlpAverage import AverageSTESNLP


data = Data('data/data_dietenbach.csv')
systemmodel =  AveragedStratStorageModel(4, 2, 2, data = data)

systemmodel.lbp[:] = 0.01
systemmodel.ubp[:] = 100

# systemmodel.lbp[3] = 0
# systemmodel.ubp[3] = 0

# turn off battery
# systemmodel.lbu[1:3] = 0
# systemmodel.ubu[1:3] = 0


nlp = AverageSTESNLP(systemmodel, data, N = 365*6)
res = nlp.solve({'ipopt.max_iter': 1000,'ipopt.linear_solver': 'ma97'})

res.save('results/dietenbach_average.npz')

