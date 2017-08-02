__author__ = 'yuwenhao'

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        files = (sys.argv[1:])
    else:
        files = ['data/viz/mp_rew_200.pkl']
    data = []
    for file in files:
        data += joblib.load(file)
    x = []
    y = []
    for d in data:
        x.append(d[0])
        y.append(d[1])

    x=np.array(x)
    if x.shape[1] == 2:
        plt.scatter(x[:,0], x[:,1],c=y, alpha=0.4)
        plt.colorbar()
        plt.show()
    elif x.shape[1] == 1:
        plt.scatter(x, y, alpha=0.4)
        plt.show()