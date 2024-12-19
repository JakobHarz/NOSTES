import numpy as np

from systemmodels.stratstoragemodel import StratStorageModel
from nlp import STESNLP
import matplotlib.pyplot as plt
from utility import Constants, Data

data = Data('data/data_dietenbach.csv')
constants = Constants()



# systemmodel =  SimpleStorageModel(data = data)
systemmodel =  StratStorageModel(4, 2, 2, data = data, constants=constants)
nlp = STESNLP(systemmodel, data, N = 365*24)
res = nlp.solve({'ipopt.max_iter': 2000,'ipopt.linear_solver': 'ma27'})


#%% Some printouts

res.save('results/dietenbach.npz')
res.printAll()
