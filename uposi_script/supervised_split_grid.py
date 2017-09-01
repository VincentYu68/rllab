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
from policy_split_supervised_edgewise import test, synthesize_data, sample_tasks, train

if __name__ == '__main__':
    dim = 20
    in_dim = dim+1
    out_dim = dim
    task_num = 3
    random_split = False
    prioritized_split = False
    append = 'edgewise_grid'+str(dim)
    reps = 1
    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'
    init_epochs = 70
    batch_size = 10000
    epochs = 30
    test_epochs = 300
    hidden_size = (64, 64)
    append += str(batch_size) + ':_' + str(init_epochs) + '_' + str(epochs) + '_' + str(test_epochs)+'_' + str(hidden_size)


    task_similarities = [0, 10, 20]
    split_percentages = [0.0, 0.5, 1.0] # 2.0 means using mean + 1 std as the threshold
    test_num = 3
    performances = []

    if not os.path.exists('data/trained/gradient_temp/supervised_split_' + append):
        os.makedirs('data/trained/gradient_temp/supervised_split_' + append)

    for similarity in task_similarities:
        performances.append([])
        for testit in range(test_num):
            seed = testit*3+1
            np.random.seed(seed)

            difficulties = [similarity] * task_num

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
            updates = lasagne.updates.adam(loss, params, learning_rate=0.002)
            train_fn = T.function([network.input_layer.input_var, out_var], loss, updates=updates, allow_input_downcast=True)
            ls = TT.mean((prediction - out_var)**2)
            grad = T.grad(ls, params, disconnected_inputs='warn')
            grad_fn = T.function([network.input_layer.input_var, out_var], grad, allow_input_downcast=True)
            loss_fn = T.function([network.input_layer.input_var, out_var], loss, allow_input_downcast=True)
            out = T.function([network.input_layer.input_var], prediction, allow_input_downcast=True)

            Xs, Ys = synthesize_data(dim, batch_size, tasks, seed = seed)
            train(train_fn, np.concatenate(Xs), np.concatenate(Ys), init_epochs)
            print('------- initial training complete ---------------')

            init_param_value = np.copy(network.get_param_values())

            #Xs, Ys = synthesize_data(dim, 2000, tasks)
            task_grads = []
            for i in range(len(Xs)):
                task_grads.append([])
            net_weight_values = []
            for i in range(epochs):
                net_weight_values.append(network.get_param_values())
                train(train_fn, np.concatenate(Xs), np.concatenate(Ys), 1)
            for i in range(epochs):
                network.set_param_values(net_weight_values[i])
                for j in range(len(Xs)):
                    grad = grad_fn(Xs[j], Ys[j])
                    task_grads[j].append(grad)
            print('------- collected gradient info -------------')

            testXs, testYs = synthesize_data(dim, 2000, tasks)
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
                        split_counts[k] += np.var(region_gradients, axis=0)# * np.abs(net_weights[i][k]) # + 100 * (len(task_grads[0][i])-k)
                    elif prioritized_split:
                        split_counts[k] += np.random.random(split_counts[k].shape) * (len(task_grads[0][i])-k)
                    else:
                        split_counts[k] += np.random.random(split_counts[k].shape)

            # organize the metric into each edges and sort them
            split_metrics = []
            metrics_list = []
            for k in range(len(task_grads[0][0])):
                for index, value in np.ndenumerate(split_counts[k]):
                    split_metrics.append([k, index, value])
                    metrics_list.append(value)
            split_metrics.sort(key=lambda x:x[2], reverse=True)
            print('Metric statistics (min/max/mean/std): ', np.min(metrics_list), np.max(metrics_list), np.mean(metrics_list), np.std(metrics_list))

            # test the effect of splitting
            total_param_size = len(network.get_param_values())

            pred_list = []
            # use the optimized network
            init_param_value = np.copy(network.get_param_values())
            sanity = out([np.concatenate([np.arange(dim), [0]])])

            for split_id, split_percentage in enumerate(split_percentages):
                split_param_size = int(split_percentage * total_param_size)
                current_split_size = 0
                masks = []
                for k in range(len(task_grads[0][0])):
                    masks.append(np.zeros(split_counts[k].shape))

                if split_percentage < 1.0:
                    for i in range(split_param_size):
                        masks[split_metrics[i][0]][split_metrics[i][1]] = 1
                else:
                    threshold = np.mean(metrics_list)# + np.std(metrics_list)
                    size= 0
                    for i in range(len(split_metrics)):
                        if split_metrics[i][2] < threshold:
                            break
                        else:
                            masks[split_metrics[i][0]][split_metrics[i][1]] = 1
                            size += 1
                    print('split size: ', size)

                network.set_param_values(init_param_value)
                if split_param_size != 0:
                    split_network = MLP_MaskedSplit(
                            input_shape=(in_dim+len(difficulties)+1,),
                            output_dim=out_dim,
                            hidden_sizes=hidden_size,
                            hidden_nonlinearity=NL.tanh,
                            output_nonlinearity=None,
                            split_num=len(difficulties)+1,
                            split_masks=masks,
                            init_net=network,
                        )
                else:
                    split_network = copy.deepcopy(network)
                    split_network.set_param_values(init_param_value)
                print('Network parameter size: ', total_param_size, len(split_network.get_param_values()))
                out_var = TT.matrix('out_var2')
                prediction = split_network._output
                loss_split = lasagne.objectives.squared_error(prediction, out_var)
                loss_split = loss_split.mean()
                params_split = split_network.get_params(trainable=True)
                updates_split = lasagne.updates.adam(loss_split, params_split, learning_rate=0.002)
                train_fn_split = T.function([split_network.input_layer.input_var, out_var], loss_split, updates=updates_split, allow_input_downcast=True)
                out = T.function([split_network.input_layer.input_var], prediction, allow_input_downcast=True)
                gradsplit = T.grad(loss_split, params_split, disconnected_inputs='warn')
                gradsplit_fn = T.function([split_network.input_layer.input_var, out_var], gradsplit, allow_input_downcast=True)
                losssplit_fn = T.function([split_network.input_layer.input_var, out_var], loss_split, allow_input_downcast=True)
                split_init_param = np.copy(split_network.get_param_values())
                if split_param_size != 0:
                    print(split_percentage, sanity, out([np.concatenate([np.arange(dim), [0, 1], [0]*len(difficulties)])]))
                avg_error = 0.0
                avg_learning_curve = []

                for rep in range(int(reps)):
                    split_network.set_param_values(split_init_param)

                    Xs, Ys = synthesize_data(dim, batch_size, tasks, split_param_size != 0, seed = seed)
                    losses = train(train_fn_split, np.concatenate(Xs), np.concatenate(Ys), test_epochs, batch = 32, shuffle=True)

                    #testXs, testYs = synthesize_data(dim, 10000, tasks, split_param_size != 0)

                    avg_error += test(out, np.concatenate(Xs), np.concatenate(Ys))
                    avg_learning_curve.append(losses)

                performances[-1].append(avg_error / reps)
                print(split_percentage, avg_error / reps)

    np.savetxt('data/trained/gradient_temp/supervised_split_' + append + '/performance.txt', performances)
    plt.imshow(np.reshape(performances, (len(task_similarities), len(split_percentages))), extent=[0, 1, 1, 0])
    plt.colorbar()
    plt.xlabel('Task Similarities')
    plt.ylabel('Split Percentages')
    plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/performances.png')

    plt.close('all')
