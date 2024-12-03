import numpy as np

from systemmodels.systemModel import Data
from systemmodels.stratstoragemodel import StratStorageModel
from nlp import STESNLP
import matplotlib.pyplot as plt
from utility import Constants

data = Data('data/data_dietenbach.csv')
constants = Constants()



# systemmodel =  SimpleStorageModel(data = data)
systemmodel =  StratStorageModel(4, 2, 2, data = data, constants=constants)


systemmodel.lbp[:] = 0.1
systemmodel.ubp[:] = 5

nlp = STESNLP(systemmodel, data, N = 365*6)
res = nlp.solve({'ipopt.max_iter': 2000,'ipopt.linear_solver': 'ma97'})


#%% Some printouts

res.save('results/dietenbach.npz')
res.printAll()
