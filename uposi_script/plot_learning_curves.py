__author__ = 'yuwenhao'

import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP, MLP_SplitAct, MLP_MaskedSplit
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano.tensor as TT
import theano as T
import theano

import joblib
from rllab.misc.ext import iterate_minibatches_generic
import copy
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    prefix = 'data/icra2018/reacher_3models/'
    learning_curve_profiles = [
        ['pretrain_0.0.pkl', 'learning_curve_0.0.pkl', 0, 399, '100% shared'],
        ['pretrain_0.5.pkl', 'learning_curve_0.5.pkl', 0, 300, '50% shared'],
        ['pretrain_0.0.pkl', 'learning_curve_0.0.pkl', 1, 399, '0% shared'],
        ['pretrain_variance.pkl', 'learning_curve_variance.pkl', 1, 399, '50% shared v'],
        #['pretrain_variance.pkl', 'learning_curve_variance.pkl', 0, 399, '100% shared v'],
    ]

    x_range = np.arange(0, 400, 1)
    learning_curve_data = []
    for profile in learning_curve_profiles:
        pretraining_data = [float(i) for i in joblib.load(prefix+profile[0])]

        learning_curve = joblib.load(prefix + profile[1])
        finetune_data = np.mean(learning_curve[profile[2]], axis=0)[0:profile[3]]

        learning_curve_data.append(np.concatenate([pretraining_data, finetune_data])[-400:])


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(len(learning_curve_data)):
        ax.plot(x_range, learning_curve_data[i], linewidth=2, label=learning_curve_profiles[i][-1], alpha=1.0)

    plt.legend(bbox_to_anchor=(0.9, 0.3),
    bbox_transform=plt.gcf().transFigure, numpoints=1)

    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Average Return", fontsize=18)

    #plt.xlim([np.min(param_range) - 0.01, np.max(param_range) + 0.01])

    #plt.ylim([1.5, 2.3])
    #plt.ylim([0.0, 1.3])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(17)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(17)

    plt.show()
























