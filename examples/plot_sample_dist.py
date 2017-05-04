__author__ = 'yuwenhao'

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

DATA_DIRECTORY = 'data/local/experiment/hopper_cap_frictorso_mp_resample5_uniform'
OUT_DIRECTORY = '/resample_plots'
PREFIX='mr_buffer_'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_DIRECTORY = sys.argv[1]
    if len(sys.argv) > 2:
        OUT_DIRECTORY = sys.argv[2]
    if len(sys.argv) > 3:
        PREFIX = sys.argv[3]

    output_directory = DATA_DIRECTORY + OUT_DIRECTORY
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i in range(1000):
        datafilename = DATA_DIRECTORY + '/' + PREFIX + str(i) + '.txt'
        if os.path.exists(datafilename):
            data = np.loadtxt(datafilename)
            x = data[:, 0]
            y = data[:, 1]
            plt.clf()
            plt.scatter(x, y)
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            plt.xlabel('Model parameter 2')
            plt.ylabel('Model parameter 1')
            plt.savefig(output_directory+'/resamp_plot_'+str(i)+'.png')
