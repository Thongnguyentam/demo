# ----------------------
# - read the input data:
'''
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
'''
# ---------------------
# - network.py example:
#import network

'''
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
'''

import numpy as np
import warnings 
import pickle

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

import pandas as pd

df = pd.read_excel ('covid1.xlsx')

training = df.values

X = []
Y = []
X_val = []
Y_val = []
for i in range(0, len(training)):
    if training[i][5] == 'Other':
        training[i][5] = 1
    else:
        if training[i][5] == 'Abroad':
            training[i][5] = 2
        else:
            training[i][5] = 3
    if training[i][6] == 'negative':
        training[i][6] = 0
    else:
        training[i][6] = 1
    if i < len(training)*0.8:
        X.append(training[i][0:6])
        Y.append(training[i][6])
    else:
        X_val.append(training[i][0:6])
        Y_val.append(training[i][6])

Y1 = []
for y in Y:
    Y1.append(vectorized_result(int(y)))
X1 = [np.reshape(x, (6, 1)) for x in X]
training_data = zip(X1, Y1)

X_val1 = [np.reshape(x, (6, 1)) for x in X_val]
Y_val1 = [vectorized_result(int(y)) for y in Y_val]
validation_data = zip(X_val1, Y_val1)

# ----------------------
# - network2.py example:

import network2


net = network2.Network([6, 4, 2], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 40, 10, 0.25, lmbda = 0.375,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True, monitor_training_accuracy = True)

pickle.dump(net,open('model.pkl','wb'))