__author__ = 'yuwenhao'

import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP, MLP_SplitAct
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

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield np.array(inputs)[excerpt], np.array(targets)[excerpt]

def train(train_fn, X, Y, iter):
    X=np.array(X)
    Y=np.array(Y)
    for epoch in range(iter):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        '''for batch in iterate_minibatches(X, Y, 128, shuffle=True):
            inputs, targets = batch
            #train_err += train_fn(inputs, targets)
            train_err=0
            train_batches += 1'''
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for start_idx in range(0, len(X) - 32 + 1, 32):
            excerpt = indices[start_idx:start_idx + 32]
            input = np.array(X[excerpt])
            output = np.array(Y[excerpt])
            train_err += train_fn(input, output)
            train_batches += 1

        #print("aux training loss:\t\t{:.6f}".format(train_err / train_batches))

if __name__ == '__main__':
    np.random.seed(0)
    in_dim = 3+1
    out_dim = 3
    task = 1 # 0: low, 1: mid, 2: high
    random_split = True
    append = 'low'
    reps = 5
    if task == 1:
        append = 'mid'
    elif task == 2:
        append = 'high'
    if random_split:
        append += '_rand'
    epochs = 1
    hidden_size = (16,)
    network = MLP(
            input_shape=(in_dim,),
            output_dim=out_dim,
            hidden_sizes=hidden_size,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
        )

    out_var = TT.matrix('out_var')
    prediction = network._output
    loss = lasagne.objectives.squared_error(prediction, out_var)
    loss = loss.mean()
    params = network.get_params(trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.002)
    train_fn = T.function([network.input_layer.input_var, out_var], loss, updates=updates)
    loss = T.function([network.input_layer.input_var, out_var], loss)
    out = T.function([network.input_layer.input_var], prediction)
    init_param_value = np.copy(network.get_param_values())
    sanity = out([[1,2,3,0]])

    # synthetic task of shuffling and calculating
    # task1: input: x,y,z, output: y,z,x+y
    # task2: input: x,y,z, output: y,x,z+y
    X1=[]
    Y1=[]
    for i in range(10000):
        d = np.random.uniform(-1, 15, 3)
        X1.append(np.concatenate([d,[0]]))
        Y1.append(np.array([d[1], d[2], d[0]+d[1]]))

    X2=[]
    Y2=[]
    for i in range(10000):
        d = np.random.uniform(-1, 15, 3)
        X2.append(np.concatenate([d,[1]]))
        #Y2.append(np.random.random(3))
        if task == 0:
            Y2.append(np.array([d[1], d[2], d[0]+d[1]])) # low
        elif task == 1:
            Y2.append(np.array([d[1], d[0], d[0]+d[1]])) # mig
        elif task == 2:
            Y2.append(np.array([d[0], d[0]+d[2], d[2]])) # high

    X1_grads = []
    X2_grads = []
    both_grads = []
    net_weights = []
    for i in range(epochs):
        cur_param_val = np.copy(network.get_param_values())
        cur_param = copy.deepcopy(network.get_params())

        cp = []
        for param in cur_param:
            cp.append(np.copy(param.get_value()))
        net_weights.append(cp)

        train(train_fn, X1, Y1, 1)
        new_param = network.get_params()
        grad = []
        for j in range(len(new_param)):
            grad.append(new_param[j].get_value() - cur_param[j].get_value())
        X1_grads.append(grad)
        network.set_param_values(cur_param_val)

        train(train_fn, X2, Y2, 1)
        new_param = network.get_params()
        grad = []
        for j in range(len(new_param)):
            grad.append(new_param[j].get_value() - cur_param[j].get_value())
        X2_grads.append(grad)
        network.set_param_values(cur_param_val)

        train(train_fn, X1+X2, Y1+Y2, 1)
        new_param = network.get_params()
        grad = []
        for j in range(len(new_param)):
            grad.append(new_param[j].get_value() - cur_param[j].get_value())
        both_grads.append(grad)


    testX1 = []
    testY1 = []
    for i in range(5000):
        d = np.random.uniform(-1, 15, 3)
        testX1.append(np.concatenate([d,[0]]))
        testY1.append(np.array([d[1], d[2], d[0]+d[1]]))
    pred = out(testX1)
    print(np.linalg.norm(pred-testY1)/len(pred))

    split_counts = []
    for i in range(len(both_grads[0])):
        split_counts.append(np.zeros(both_grads[0][i].shape))

    for i in range(len(both_grads)):
        for k in range(len(both_grads[i])):
            region_gradients = []
            region_gradients.append(X1_grads[i][k])
            region_gradients.append(X2_grads[i][k])
            region_gradients = np.array(region_gradients)
            if not random_split:
                split_counts[k] += np.var(region_gradients, axis=0) * np.abs(net_weights[i][k])
            else:
                split_counts[k] += np.random.random(split_counts[k].shape)

    for j in range(len(split_counts)):
        plt.figure()
        plt.title(network.get_params()[j].name)
        if len(split_counts[j].shape) == 2:
            plt.imshow(split_counts[j])
            plt.colorbar()
        elif len(split_counts[j].shape) == 1:
            plt.plot(split_counts[j])

        plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/' + network.get_params()[j].name + '.png')

    # test the effect of splitting
    split_percentages = [0.3,0.4, 0.6, 0.8]#[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8]
    total_param_size = len(network.get_param_values())

    split_indices = []
    for p in range(int(len(split_counts)/2)):
        for col in range(split_counts[p*2].shape[1]):
            split_metric = np.mean(split_counts[p*2][:, col]) + split_counts[p*2+1][col]
            split_indices.append([[p, col], split_metric])
    split_indices.sort(key=lambda x:x[1], reverse=True)

    for i in range(int(len(split_counts))):
        split_counts[i] *= 0

    # augment the training set
    for i in range(len(X1)):
        X1[i] = np.concatenate([X1[i], [1, 0]])
    for i in range(len(X2)):
        X2[i] = np.concatenate([X2[i], [0, 1]])
    for i in range(len(testX1)):
        testX1[i] = np.concatenate([testX1[i], [1, 0]])

    pred_lsit = []
    # use the optimized network
    init_param_value = np.copy(network.get_param_values())
    sanity = out([[1,2,3,0]])
    for split_percentage in split_percentages:
        split_param_size = split_percentage * total_param_size
        current_split_size = 0
        split_layer_units = []
        for i in range(len(split_indices)):
            pm = split_indices[i][0][0]
            col = split_indices[i][0][1]
            split_counts[pm*2][:, col] = 1
            split_counts[pm*2+1][col] = 1
            current_split_size += split_counts[pm*2].shape[0]+1
            split_layer_units.append([pm, col])
            if current_split_size > int(split_param_size):
                break

        split_layer_units.sort(key=lambda x: (x[0], x[1]))

        network.set_param_values(init_param_value)
        split_network = MLP_SplitAct(
                input_shape=(in_dim+2,),
                output_dim=out_dim,
                hidden_sizes=hidden_size,
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=None,
                split_num=2,
                split_units=split_layer_units,
                init_net=network,
            )
        out_var = TT.matrix('out_var2')
        prediction = split_network._output
        loss_split = lasagne.objectives.squared_error(prediction, out_var)
        loss_split = loss_split.mean()
        params = split_network.get_params(trainable=True)
        updates_split = lasagne.updates.sgd(loss_split, params, learning_rate=0.002)
        train_fn_split = T.function([split_network.input_layer.input_var, out_var], loss_split, updates=updates_split)
        out = T.function([split_network.input_layer.input_var], prediction)
        split_init_param = np.copy(split_network.get_param_values())
        print(sanity, out([[1, 2, 3, 0, 1, 0]]))
        if (np.abs(out([[1, 2, 3, 0, 1, 0]]) - sanity) > 0.0001).any():
            print(split_network.get_params())
            print(network.get_params())
            print(network.get_params()[-1].get_value())
            print(split_network.get_params()[-1].get_value())
            print(split_network.get_params()[-3].get_value())
            print(split_network.get_params()[-5].get_value())
            abc

        avg_error = 0.0
        for rep in range(int(reps)):
            split_network.set_param_values(split_init_param)

            X1=[]
            Y1=[]
            for i in range(10000):
                d = np.random.uniform(-1, 15, 3)
                X1.append(np.concatenate([d,[0, 1, 0]]))
                Y1.append(np.array([d[1], d[2], d[0]+d[1]]))

            X2=[]
            Y2=[]
            for i in range(10000):
                d = np.random.uniform(-1, 15, 3)
                X2.append(np.concatenate([d,[1, 0, 1]]))
                #Y2.append(np.random.random(3))
                if task == 0:
                    Y2.append(np.array([d[1], d[2], d[0]+d[1]])) # low
                elif task == 1:
                    Y2.append(np.array([d[1], d[0], d[0]+d[1]])) # mig
                elif task == 2:
                    Y2.append(np.array([d[0], d[0]+d[2], d[2]])) # high

            train(train_fn_split, X1+X2, Y1+Y2, epochs)
            pred = out(testX1)
            pred_error = np.linalg.norm(pred-testY1)/len(pred)
            avg_error += pred_error
        pred_lsit.append(avg_error / reps)

    plt.figure()
    plt.plot(split_percentages, pred_lsit)
    plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/split_performance.png')

