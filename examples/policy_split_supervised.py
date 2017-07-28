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
import os


def train(train_fn, X, Y, iter, batch = 32, shuffle=True):
    X=np.array(X)
    Y=np.array(Y)
    for epoch in range(iter):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(X) - batch + 1, batch):
            excerpt = indices[start_idx:start_idx + batch]
            input = np.array(X[excerpt])
            output = np.array(Y[excerpt])
            train_err += train_fn(input, output)
            train_batches += 1
        #print("aux training loss:\t\t{:.6f}".format(train_err / train_batches))

# default task is to reverse the order
# difficulty [0, dim-1] is the number of operations being mutated
def sample_tasks(dim, difficulties, seed = None):
    if seed is not None:
        np.random.seed(seed)
    tasks = []
    default_task = []
    for i in range(dim):
        default_task.append([0, dim-1-i])
    tasks.append(default_task)
    for difficulty in difficulties:
        default = copy.deepcopy(default_task)
        unmutated_lsit = np.arange(dim).tolist()
        for mutation in range(int(difficulty)):
            mutate_target = unmutated_lsit[np.random.randint(len(unmutated_lsit))]
            unmutated_lsit.remove(mutate_target)
            type = np.random.randint(4)
            idx1 = np.random.randint(dim)
            idx2 = idx1
            while idx2 == idx1:
                idx2 = np.random.randint(dim)
            if type == 0: # swap order
                default[mutate_target] = [0, idx1]
            else: # four operations for the two numbers
                default[mutate_target] = [type, idx1, idx2]
        tasks.append(default)
    return tasks


def synthesize_data(dim, size, tasks, split = False, seed = None):
    # synthetic task of shuffling and calculating
    if seed is not None:
        np.random.seed(seed)
    Xs = []
    Ys = []

    per_task_size = size / len(tasks)

    for i, task in enumerate(tasks):
        X=[]
        Y=[]
        for _ in range(int(per_task_size)):
            input = np.random.uniform(-1, 10, dim)
            input = np.concatenate([input, [i*1.0/(len(tasks)-1)]])
            if split:
                split_vec = [0] * len(tasks)
                split_vec[i] = 1
                input = np.concatenate([input, split_vec])
            output = np.zeros(dim)
            for idx, subtask in enumerate(task):
                if subtask[0] == 0:
                    output[idx] = input[subtask[1]]
                elif subtask[0] == 1:
                    output[idx] = input(subtask[1]) + input(subtask[2])
                elif subtask[0] == 2:
                    output[idx] = input(subtask[1]) - input(subtask[2])
                elif subtask[0] == 3:
                    output[idx] = input(subtask[1]) * input(subtask[2])
            X.append(input)
            Y.append(output)
        Xs.append(X)
        Ys.append(Y)
    return Xs, Ys

def test(out_fn, X, Y):
    pred = out_fn(X)
    return np.mean((pred-Y)**2)

if __name__ == '__main__':
    dim = 5
    in_dim = dim+1
    out_dim = dim
    difficulties = [0]
    random_split = False
    prioritized_split = False
    append = str(difficulties)
    reps = 10
    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'
    epochs = 40
    hidden_size = (64,16,8)

    test_num = 10
    performances = []

    if not os.path.exists('data/trained/gradient_temp/supervised_split_' + append):
        os.makedirs('data/trained/gradient_temp/supervised_split_' + append)

    for testit in range(test_num):
        print('======== Start Test ', testit, ' ========')
        np.random.seed(testit*3)

        tasks = sample_tasks(dim, difficulties)

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
        train_fn = T.function([network.input_layer.input_var, out_var], loss, updates=updates, allow_input_downcast=True)
        ls = TT.mean((prediction - out_var)**2)
        grad = T.grad(ls, params, disconnected_inputs='warn')
        grad_fn = T.function([network.input_layer.input_var, out_var], grad, allow_input_downcast=True)
        loss_fn = T.function([network.input_layer.input_var, out_var], loss, allow_input_downcast=True)
        out = T.function([network.input_layer.input_var], prediction, allow_input_downcast=True)


        Xs, Ys = synthesize_data(dim, 1000, tasks)
        train(train_fn, np.concatenate(Xs), np.concatenate(Ys), 20)
        print('------- initial training complete ---------------')

        init_param_value = np.copy(network.get_param_values())

        Xs, Ys = synthesize_data(dim, 10000, tasks)
        task_grads = []
        for i in range(len(Xs)):
            task_grads.append([])
        net_weights = []
        for i in range(epochs):
            #network.set_param_values(init_param_value)

            cur_param_val = np.copy(network.get_param_values())
            cur_param = copy.deepcopy(network.get_params())

            cp = []
            for param in cur_param:
                cp.append(np.copy(param.get_value()))
            net_weights.append(cp)

            for j in range(len(Xs)):
                train(train_fn, Xs[j], Ys[j], 1)
                new_param = network.get_params()
                grad = []
                for k in range(len(new_param)):
                    grad.append(new_param[k].get_value() - cur_param[k].get_value())
                task_grads[j].append(grad)
                network.set_param_values(cur_param_val)

            train(train_fn, np.concatenate(Xs), np.concatenate(Ys), 1)
        print('------- collected gradient info -------------')

        testXs, testYs = synthesize_data(dim, 10000, tasks)
        pred = out(np.concatenate(testXs))
        print(np.linalg.norm(pred-np.concatenate(testYs))/len(pred))

        split_counts = []
        for i in range(len(task_grads[0][0])):
            split_counts.append(np.zeros(task_grads[0][0][i].shape))

        for i in range(len(task_grads[0])):
            for k in range(len(task_grads[0][i])):
                region_gradients = []
                for region in range(len(task_grads)):
                    region_gradients.append(task_grads[region][i][k])
                region_gradients = np.array(region_gradients)
                if not random_split:
                    split_counts[k] += np.var(region_gradients, axis=0) * np.abs(net_weights[i][k]) + 100 * (len(task_grads[0][i])-k)
                elif prioritized_split:
                    split_counts[k] += np.random.random(split_counts[k].shape) * (len(task_grads[0][i])-k)
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
        split_percentages = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]#, 0.5, 0.8, 1.0]
        total_param_size = len(network.get_param_values())

        split_indices = []
        for p in range(int(len(split_counts)/2)):
            for col in range(split_counts[p*2].shape[1]):
                split_metric = np.mean(split_counts[p*2][:, col]) + split_counts[p*2+1][col]
                split_indices.append([[p, col], split_metric])
        split_indices.sort(key=lambda x:x[1], reverse=True)

        for i in range(int(len(split_counts))):
            split_counts[i] *= 0

        pred_list = []
        # use the optimized network
        init_param_value = np.copy(network.get_param_values())
        sanity = out([np.concatenate([np.arange(dim), [0]])])
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

            #print(split_layer_units)
            network.set_param_values(init_param_value)
            if split_param_size != 0:
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
            else:
                split_network = copy.deepcopy(network)
            out_var = TT.matrix('out_var2')
            prediction = split_network._output
            loss_split = lasagne.objectives.squared_error(prediction, out_var)
            loss_split = loss_split.mean()
            params_split = split_network.get_params(trainable=True)
            updates_split = lasagne.updates.sgd(loss_split, params_split, learning_rate=0.001)
            train_fn_split = T.function([split_network.input_layer.input_var, out_var], loss_split, updates=updates_split, allow_input_downcast=True)
            out = T.function([split_network.input_layer.input_var], prediction, allow_input_downcast=True)
            gradsplit = T.grad(loss_split, params_split, disconnected_inputs='warn')
            gradsplit_fn = T.function([split_network.input_layer.input_var, out_var], gradsplit, allow_input_downcast=True)
            losssplit_fn = T.function([split_network.input_layer.input_var, out_var], loss_split, allow_input_downcast=True)
            split_init_param = np.copy(split_network.get_param_values())
            if split_param_size != 0:
                print(split_percentage, sanity, out([np.concatenate([np.arange(dim), [0, 1], [0]*len(difficulties)])]))
            avg_error = 0.0

            for rep in range(int(reps)):
                split_network.set_param_values(split_init_param)

                Xs, Ys = synthesize_data(dim, 20000, tasks, split_param_size != 0)
                train(train_fn_split, np.concatenate(Xs), np.concatenate(Ys), 20, batch = 32, shuffle=True)

                testXs, testYs = synthesize_data(dim, 10000, tasks, split_param_size != 0)

                avg_error += test(out, np.concatenate(testXs), np.concatenate(testYs))
            pred_list.append(avg_error / reps)
        performances.append(pred_list)

    np.savetxt('data/trained/gradient_temp/supervised_split_' + append + '/performance.txt', performances)
    plt.figure()
    plt.plot(split_percentages, np.mean(performances, axis=0))
    plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/split_performance.png')



