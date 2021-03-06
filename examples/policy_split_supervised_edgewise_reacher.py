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


def train(train_fn, X, Y, iter, batch = 32, shuffle=True, out = None, testX = None, testY = None):
    X=np.array(X)
    Y=np.array(Y)
    losses = []
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
        if out is not None:
            losses.append(test(out, testX, testY))
        else:
            losses.append(train_err / train_batches)
        #print("aux training loss:\t\t{:.6f}".format(train_err / train_batches))
    return losses

def test(out_fn, X, Y):
    pred = out_fn(X)
    exp_id = np.random.randint(len(X))
    #print('example test: ', X[exp_id], pred[exp_id], Y[exp_id])
    return np.mean(((pred-Y)**2))

def process_data(allX, allY):
    newAllX = []
    scaledAllY = []
    for i, Xs in enumerate(allX):
        newXs = []
        for X in Xs:
            newX = np.concatenate([X, [i]])
            newXs.append(newX)
        newXs = np.array(newXs)
        newAllX.append(newXs)

        scale = 3.14
        #if i == 0:
        #    scale = 1
        scaledAllY.append(np.array(allY[i]) / scale)
    return newAllX, scaledAllY

def augment_split_vec(allXs):
    newAllXs = []
    task_num = len(allX)
    for i, Xs in enumerate(allXs):
        newXs = []
        for X in Xs:
            split_vec = np.zeros(task_num)
            split_vec[i] = 1
            newX = np.concatenate([X, split_vec])
            newXs.append(newX)
        newXs = np.array(newXs)
        newAllXs.append(newXs)
    return newAllXs

if __name__ == '__main__':
    dataset_directory = 'data/trained/supervised_reacher/'
    dataset_names = ['reacher_task1.pkl', 'reacher_task2.pkl']#, 'reacher_task3.pkl']

    allX=[]
    allY=[]
    for dataset_name in dataset_names:
        X, Y = joblib.load(dataset_directory+dataset_name)
        allX.append(X)
        allY.append(Y)
    procXs, procYs = process_data(allX, allY)

    trainig_percent = 0.9

    trainingXs = []
    trainingYs = []
    testingXs = []
    testingYs = []
    total_training_size = 0
    for i in range(len(procXs)):
        trainingXs.append(procXs[i][0:int(trainig_percent * len(procXs[i]))])
        trainingYs.append(procYs[i][0:int(trainig_percent * len(procXs[i]))])
        testingXs.append(procXs[i][int(trainig_percent * len(procXs[i])):])
        testingYs.append(procYs[i][int(trainig_percent * len(procXs[i])):])
        total_training_size += int(trainig_percent * len(procXs[i]))
    trainingXs_aug = augment_split_vec(trainingXs)
    testingXs_aug = augment_split_vec(testingXs)

    print('training data: ', len(trainingXs), total_training_size)

    in_dim = len(trainingXs[0][0])
    out_dim = len(trainingYs[0][0])
    random_split = False
    prioritized_split = False
    append = 'edgewise_reacher'
    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'
    init_epochs = 30
    epochs = 40
    test_epochs = 300
    hidden_size = (128, 64)
    append += str(total_training_size) + ':_' + str(init_epochs) + '_' + str(epochs) +'_' + str(hidden_size)

    split_percentages = [0.0, 0.05, 0.5, 0.7, 1.0] # 2.0 means using mean as the threshold

    load_init_policy = False
    load_split_data = False

    performances = []
    learning_curves = []
    for i in range(len(split_percentages)):
        learning_curves.append([])

    if not os.path.exists('data/trained/gradient_temp/supervised_split_' + append):
        os.makedirs('data/trained/gradient_temp/supervised_split_' + append)

    average_metric_list = []

    print('======== Start Test ========')

    network = MLP(
            input_shape=(in_dim,),
            output_dim=out_dim,
            hidden_sizes=hidden_size,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
        )
    if load_init_policy:
        network = joblib.load('data/trained/gradient_temp/supervised_split_' + append + '/init_network.pkl')

    out_var = TT.matrix('out_var')
    prediction = network._output
    loss = lasagne.objectives.squared_error(prediction, out_var)
    loss = loss.mean()
    params = network.get_params(trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.0005)
    train_fn = T.function([network.input_layer.input_var, out_var], loss, updates=updates, allow_input_downcast=True)
    ls = TT.mean((prediction - out_var)**2)
    grad = T.grad(ls, params, disconnected_inputs='warn')
    grad_fn = T.function([network.input_layer.input_var, out_var], grad, allow_input_downcast=True)
    loss_fn = T.function([network.input_layer.input_var, out_var], loss, allow_input_downcast=True)
    out = T.function([network.input_layer.input_var], prediction, allow_input_downcast=True)

    if not load_init_policy:
        losses=train(train_fn, np.concatenate(trainingXs), np.concatenate(trainingYs), init_epochs)
        joblib.dump(network, 'data/trained/gradient_temp/supervised_split_' + append + '/init_network.pkl', compress=True)
    print('------- initial training complete ---------------')

    init_param_value = np.copy(network.get_param_values())

    #Xs, Ys = synthesize_data(dim, 2000, tasks)
    task_grads = []
    for i in range(len(trainingXs)):
        task_grads.append([])
    if not load_split_data:
        net_weight_values = []
        for i in range(epochs):
            net_weight_values.append(network.get_param_values())
            train(train_fn, np.concatenate(trainingXs), np.concatenate(trainingYs), 1)
        joblib.dump(net_weight_values, 'data/trained/gradient_temp/supervised_split_' + append + '/net_weight_values.pkl', compress=True)
    else:
        net_weight_values = joblib.load('data/trained/gradient_temp/supervised_split_' + append + '/net_weight_values.pkl')
    for i in range(epochs):
        network.set_param_values(net_weight_values[i])
        for j in range(len(trainingXs)):
            grad = grad_fn(trainingXs[j], trainingYs[j])
            task_grads[j].append(grad)
    print('------- collected gradient info -------------')

    pred = out(np.concatenate(testingXs))

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

    for j in range(len(split_counts)):
        plt.figure()
        plt.title(network.get_params()[j].name)
        if len(split_counts[j].shape) == 2:
            plt.imshow(split_counts[j])
            plt.colorbar()
        elif len(split_counts[j].shape) == 1:
            plt.plot(split_counts[j])

        plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/' + network.get_params()[j].name + '.png')

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

    for split_id, split_percentage in enumerate(split_percentages):
        split_param_size = int(split_percentage * total_param_size)
        current_split_size = 0
        masks = []
        for k in range(len(task_grads[0][0])):
            masks.append(np.zeros(split_counts[k].shape))

        if split_percentage <= 1.01:
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
            print('split size: ', size*1.0/total_param_size)

        network.set_param_values(init_param_value)
        if split_param_size != 0:
            split_network = MLP_MaskedSplit(
                    input_shape=(in_dim+len(trainingXs),),
                    output_dim=out_dim,
                    hidden_sizes=hidden_size,
                    hidden_nonlinearity=NL.tanh,
                    output_nonlinearity=None,
                    split_num=len(trainingXs),
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
        updates_split = lasagne.updates.adam(loss_split, params_split, learning_rate=0.0005)
        train_fn_split = T.function([split_network.input_layer.input_var, out_var], loss_split, updates=updates_split, allow_input_downcast=True)
        out = T.function([split_network.input_layer.input_var], prediction, allow_input_downcast=True)
        gradsplit = T.grad(loss_split, params_split, disconnected_inputs='warn')
        gradsplit_fn = T.function([split_network.input_layer.input_var, out_var], gradsplit, allow_input_downcast=True)
        losssplit_fn = T.function([split_network.input_layer.input_var, out_var], loss_split, allow_input_downcast=True)
        split_init_param = np.copy(split_network.get_param_values())
        avg_error = 0.0
        avg_learning_curve = []

        split_network.set_param_values(split_init_param)

        if split_param_size == 0:
            init_error = test(out, np.concatenate(testingXs), np.concatenate(testingYs))
        else:
            init_error = test(out, np.concatenate(testingXs_aug), np.concatenate(testingYs))
        print('init error: ', split_percentage, init_error)

        if split_param_size == 0:
            losses = train(train_fn_split, np.concatenate(trainingXs), np.concatenate(trainingYs), int(test_epochs), batch = 32, shuffle=True, out=out, testX=np.concatenate(testingXs), testY=np.concatenate(testingYs))
            avg_error += test(out, np.concatenate(testingXs), np.concatenate(testingYs))
        else:
            losses = train(train_fn_split, np.concatenate(trainingXs_aug), np.concatenate(trainingYs), int(test_epochs), batch = 32, shuffle=True, out=out, testX=np.concatenate(testingXs_aug), testY=np.concatenate(testingYs))
            avg_error += test(out, np.concatenate(testingXs_aug), np.concatenate(testingYs))

        avg_learning_curve.append(losses)

        pred_list.append(avg_error)
        print(split_percentage, avg_error)
        avg_learning_curve = np.mean(avg_learning_curve, axis=0)

        learning_curves[split_id].append(avg_learning_curve)

        avg_learning_curve = []
        for lc in range(len(learning_curves)):
            avg_learning_curve.append(np.mean(learning_curves[lc], axis=0))
        plt.figure()
        for lc in range(len(split_percentages)):
            plt.plot(avg_learning_curve[lc], label=str(split_percentages[lc]))
        plt.legend(bbox_to_anchor=(0.3, 0.3),
        bbox_transform=plt.gcf().transFigure, numpoints=1)
        plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/split_learning_curves.png')
    performances.append(pred_list)

    np.savetxt('data/trained/gradient_temp/supervised_split_' + append + '/performance.txt', performances)
    plt.figure()
    plt.plot(split_percentages, np.mean(performances, axis=0))
    plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/split_performance.png')

    avg_learning_curve = []
    for i in range(len(learning_curves)):
        avg_learning_curve.append(np.mean(learning_curves[i], axis=0))
    plt.figure()
    for i in range(len(split_percentages)):
        plt.plot(avg_learning_curve[i], label=str(split_percentages[i]))
    plt.legend(bbox_to_anchor=(0.3, 0.3),
    bbox_transform=plt.gcf().transFigure, numpoints=1)
    plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/split_learning_curves.png')

    plt.close('all')



