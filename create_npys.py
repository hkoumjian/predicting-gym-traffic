import numpy as np

umass_data = np.genfromtxt('umass_data.csv', delimiter=',')
berkeley_data = np.genfromtxt('berkeley_data.csv', delimiter=',')

np.save("umass.npy",umass_data)
np.save("berkeley.npy",berkeley_data)
