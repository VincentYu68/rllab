__author__ = 'yuwenhao'

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import sys
import csv

if __name__ == '__main__':
    if len(sys.argv) > 1:
        files_g1 = (sys.argv[1:])
    else:
        files_g1 = ['data/split_progress/cp_sd5_vanilla_1000.pkl']

    if len(sys.argv) > 2:
        files_g2 = (sys.argv[2:])
    else:
        files_g2 = ['data/split_progress/cp_sd5_gradsplit_1000.pkl']

    data_group1 = []
    for file in files_g1:
        reader = csv.reader(open(file, 'r'))
        title = reader.__next__()
        avg_rt_id = title.index('AverageReturn')
        returns = []
        for row in reader:
            returns.append(row[avg_rt_id])
        data_group1.append(returns)

    data_group2 = []
    for file in files_g2:
        reader = csv.reader(open(file, 'r'))
        title = reader.__next__()
        avg_rt_id = title.index('AverageReturn')
        returns = []
        for row in reader:
            returns.append(row[avg_rt_id])
        data_group2.append(returns)

    for rts in data_group1:
        plt.plot(rts)
    plt.show()