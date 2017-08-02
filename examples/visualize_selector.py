__author__ = 'yuwenhao'

from UPSelector import *
import joblib
import numpy as np
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) > 1:
        selector = joblib.load(sys.argv[1])
    else:
        raise ValueError

    mp_dim = selector.models[0]._fit_X.shape[1]
    if mp_dim == 1:
        coordx = []
        pred_data = []
        for i in np.arange(0, 1, 0.02):
            reps = []
            for rep in range(1):
                reps.append(selector.classify(np.array([[i]]), stoch=False))
            pred = np.mean(reps)
            coordx.append(i)
            pred_data.append(pred)
        plt.scatter(coordx, pred_data)
        plt.show()
    elif mp_dim == 2:
        coordx = []
        coordy = []
        pred_data = []
        for i in np.arange(0, 1, 0.02):
            for j in np.arange(0, 1, 0.02):
                reps = []
                for rep in range(1):
                    reps.append(selector.classify(np.array([[i, j]]), stoch=False))
                pred = np.mean(reps)
                coordx.append(i)
                coordy.append(j)
                pred_data.append(pred)

        plt.imshow(np.reshape(pred_data, (50, 50)))
        plt.colorbar()
        plt.show()
    else:
        print('visualization only support 1d and 2d model parameters!')