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

def KLDiv(m1, s1, m2, s2):
    return np.log(s2)-np.log(s1) + (s1**2 + (m1-m2)**2)/(2*(s2**2)) - 0.5

def symKLDiv(m1, s1, m2, s2):
    s1[s1==0] += 1e-10
    s2[s2==0] += 1e-10
    return KLDiv(m1, s1, m2, s2) + KLDiv(m2, s2, m1, s1)

def train(train_fn, X, Y, iter, batch = 32, shuffle=True):
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
        losses.append(train_err / train_batches)
        print(epoch, " aux training loss:\t\t{:.6f}".format(train_err / train_batches))
    return losses

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
            type = 1#np.random.randint(1)
            idx1 = np.random.randint(dim)
            while idx1 == mutate_target:
                idx1 = np.random.randint(dim)
            idx2 = idx1
            while idx2 == idx1:
                idx2 = np.random.randint(dim)
            if type == 1: # swap order
                cur_id=default[mutate_target][1]
                default[mutate_target] = [0, default[idx1][1]]
                default[idx1] = [0, cur_id]
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

    per_task_size = [size / len(tasks)] * len(tasks)
    #per_task_size = [size /3.0, size/3.0*2.0]

    for i, task in enumerate(tasks):
        X=[]
        Y=[]
        for _ in range(int(per_task_size[i])):
            input = np.random.uniform(-10, 10, dim)
            input = np.concatenate([input, [i*1.0/(len(tasks)-1)+0.5]])

            if split:
                split_vec = [0] * len(tasks)
                split_vec[i] = 1
                input = np.concatenate([input, split_vec])
            output = np.zeros(dim)
            exec_task = copy.deepcopy(tasks[i])
            #if _ > 0.5 * (int(per_task_size)):
            #    exec_task = copy.deepcopy(tasks[i-1])
            for idx, subtask in enumerate(exec_task):
                if subtask[0] == 0:
                    output[idx] = input[subtask[1]]
                elif subtask[0] == 1:
                    output[idx] = input[subtask[1]] + input[subtask[2]]
                elif subtask[0] == 2:
                    output[idx] = input[subtask[1]] - input[subtask[2]]
                #elif subtask[0] == 3:
                #    output[idx] = input[subtask[1]] * input[subtask[2]]
            X.append(input)
            Y.append(output)
        Xs.append(X)
        Ys.append(Y)
    return Xs, Ys

def test(out_fn, X, Y):
    pred = out_fn(X)
    return np.mean((pred-Y)**2)

if __name__ == '__main__':
    dim = 16
    in_dim = dim+1
    out_dim = dim
    difficulties = [8, 8]
    cross_task_kldivergence = False
    use_grad_variance = True
    random_split = False
    prioritized_split = False
    reverse_metric_order = True
    append = 'edgewise_test_unscaled_output_'+str(dim)+':'+str(difficulties)

    reps = 1
    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'
    if cross_task_kldivergence:
        append += '_crosstaskKL'
    if use_grad_variance:
        append += '_variance'
    if reverse_metric_order:
        append += '_metricl2s'

    init_epochs = 80
    batch_size = 10000
    epochs = 20
    test_epochs = 100
    hidden_size = (64, 32)

    append += str(batch_size) + ':_' + str(init_epochs) + '_' + str(epochs) + '_' + str(test_epochs)+'_' + str(hidden_size)


    #split_percentages = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]
    split_percentages = [0.2, 0.5, 0.8, 0.999999] # 2.0 means using mean + 1 std as the threshold
    test_num = 3
    performances = []
    learning_curves = []
    for i in range(len(split_percentages)):
        learning_curves.append([])

    if not os.path.exists('data/trained/gradient_temp/supervised_split_' + append):
        os.makedirs('data/trained/gradient_temp/supervised_split_' + append)

    average_metric_list = []

    for testit in range(test_num):
        print('======== Start Test ', testit, ' ========')
        seed = testit*3+1

        np.random.seed(seed)

        tasks = sample_tasks(dim, difficulties)
        print(tasks)

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
        updates = lasagne.updates.adam(loss, params, learning_rate=0.001)
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
        task_grad_samples = []
        for i in range(len(Xs)):
            task_grads.append([])
            task_grad_samples.append([])

        total_grads = []
        net_weight_values = []
        #for i in range(epochs):
        #    net_weight_values.append(network.get_param_values())
        #    train(train_fn, np.concatenate(Xs), np.concatenate(Ys), 1)
        epochs=1
        for i in range(epochs):
            #network.set_param_values(net_weight_values[i])
            for j in range(len(Xs)):
                grad = grad_fn(Xs[j], Ys[j])
                task_grads[j].append(grad)

                for k in range(100):
                    indices = np.arange(len(Xs[j]))
                    np.random.shuffle(indices)
                    grad = grad_fn(np.array(Xs[j])[indices[0:1000]], np.array(Ys[j])[indices[0:1000]])
                    task_grad_samples[j].append(grad)

            for j in range(100):
                allX = np.concatenate(Xs)
                allY = np.concatenate(Ys)
                indices = np.arange(len(allX))
                np.random.shuffle(indices)
                input = np.array(allX[indices[0:1000]])
                output = np.array(allY[indices[0:1000]])
                grad = grad_fn(input, output)
                total_grads.append(grad)
        print('------- collected gradient info -------------')

        testXs, testYs = synthesize_data(dim, 2000, tasks)
        pred = out(np.concatenate(testXs))
        print(np.linalg.norm(pred-np.concatenate(testYs))/len(pred))

        split_counts = []
        weight_variances = []
        weight_means = []

        task_grad_stds = []
        task_grad_means = []
        for i in range(len(task_grads)):
            task_grad_stds.append([])
            task_grad_means.append([])

        for i in range(len(task_grads[0][0])):
            split_counts.append(np.zeros(task_grads[0][0][i].shape))
            weight_variances.append(np.zeros(task_grads[0][0][i].shape))
            weight_means.append(np.zeros(task_grads[0][0][i].shape))

        for i in range(len(task_grad_samples)):
            for j in range(len(task_grad_samples[i][0])):
                one_param = []
                for k in range(len(task_grad_samples[i])):
                    one_param.append(np.asarray(task_grad_samples[i][k][j]))
                task_grad_means[i].append(np.mean(one_param, axis=0))
                task_grad_stds[i].append(np.std(one_param, axis=0))



        for i in range(len(task_grads[0])):
            for k in range(len(task_grads[0][i])):
                region_gradients = []
                for region in range(len(task_grads)):
                    region_gradients.append(np.asarray(task_grads[region][i][k]))
                region_gradients = np.array(region_gradients)
                if not random_split:
                    split_counts[k] += np.var(region_gradients, axis=0)# * np.abs(net_weights[i][k]) # + 100 * (len(task_grads[0][i])-k)
                elif prioritized_split:
                    split_counts[k] += np.random.random(split_counts[k].shape) * (len(task_grads[0][i])-k)
                else:
                    split_counts[k] += np.random.random(split_counts[k].shape)

                one_grad = []
                for g in range(len(total_grads)):
                    one_grad.append(np.asarray(total_grads[g][k]))
                weight_variances[k] += np.var(one_grad, axis=0)
                weight_means[k] += np.mean(one_grad, axis=0)


        for j in range(len(split_counts)):
            plt.figure()
            plt.title(network.get_params()[j].name)
            if len(split_counts[j].shape) == 2:
                plt.imshow(split_counts[j])
                plt.colorbar()
            elif len(split_counts[j].shape) == 1:
                plt.plot(split_counts[j])
            plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/' + network.get_params()[j].name + '_task_variance.png')

        for j in range(len(weight_variances)):
            plt.figure()
            plt.title(network.get_params()[j].name)
            if len(weight_variances[j].shape) == 2:
                plt.imshow(weight_variances[j])
                plt.colorbar()
            elif len(weight_variances[j].shape) == 1:
                plt.plot(weight_variances[j])
            plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/' + network.get_params()[j].name + '_variances.png')

        for j in range(len(network.get_params())):
            plt.figure()
            plt.title(network.get_params()[j].name)
            if len(network.get_params()[j].get_value().shape) == 2:
                plt.imshow(network.get_params()[j].get_value())
                plt.colorbar()
            elif len(network.get_params()[j].get_value().shape) == 1:
                plt.plot(network.get_params()[j].get_value())

            plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/' + network.get_params()[j].name + '.png')

        for j in range(len(weight_variances)):
            plt.figure()
            plt.title(network.get_params()[j].name)
            if len(weight_variances[j].shape) == 2:
                plt.imshow(weight_variances[j])
                plt.colorbar()
            elif len(weight_variances[j].shape) == 1:
                plt.plot(weight_variances[j])
            plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/' + network.get_params()[j].name + '_variances.png')

        # organize the metric into each edges and sort them
        split_metrics = []
        metrics_list = []
        variance_list = []
        for k in range(len(task_grads[0][0])):
            for index, value in np.ndenumerate(split_counts[k]):
                split_metrics.append([k, index, value, weight_variances[k][index]])
                metrics_list.append(value)
                variance_list.append(weight_variances[k][index])
        if use_grad_variance:
            split_metrics.sort(key=lambda x:x[3], reverse=reverse_metric_order)
        else:
            split_metrics.sort(key=lambda x: x[2], reverse=reverse_metric_order)



        '''################## test the effect of adding noise to the network ###################
        max_var = np.max(variance_list)
        original_param = np.copy(network.get_param_values())
        sensitivity_test = []
        sensitivity_list = []
        testXs, testYs = synthesize_data(dim, 10000, tasks, False)
        for param in network.get_params():
            sensitivity_test.append(np.zeros(param.get_value().shape))
            for id, val in np.ndenumerate(param.get_value()):
                cur_value = np.copy(param.get_value())
                cur_value[id] += np.random.normal(0, np.sqrt(max_var))*10
                param.set_value(cur_value)
                sens = test(out, np.concatenate(testXs), np.concatenate(testYs))
                sensitivity_test[-1][id] = sens
                sensitivity_list.append(sens)
                network.set_param_values(original_param)
        for j in range(len(sensitivity_test)):
            plt.figure()
            plt.title(network.get_params()[j].name)
            if len(sensitivity_test[j].shape) == 2:
                plt.imshow(sensitivity_test[j])
                plt.colorbar()
            elif len(sensitivity_test[j].shape) == 1:
                plt.plot(sensitivity_test[j])
            plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/' + network.get_params()[j].name + '_sensitivity.png')
            print('finish', j)


        max_var = np.max(variance_list)

        # test the usage of the various weights
        testXs, testYs = synthesize_data(dim, 10000, tasks, False)
        print('param size: ', len(split_metrics))
        print('original error: ', test(out, np.concatenate(testXs), np.concatenate(testYs)))
        original_param = np.copy(network.get_param_values())

        split_metrics.sort(key=lambda x: x[3], reverse=False)
        sensitivity_list.sort()

        sorted_weight = np.sort(np.abs(network.get_param_values()))
        print(len(sorted_weight), len(split_metrics))
        error_list = []
        for i in range(10):
            thres1 = split_metrics[int((i*0.1)*len(split_metrics))][3]
            thres2 = split_metrics[int(((i+1)*0.1)*len(split_metrics))-1][3]

            #thres1 = sorted_weight[int((i*0.1)*len(sorted_weight))]
            #thres2 = sorted_weight[int(((i+1) * 0.1) * len(sorted_weight))-1]

            #thres1 = sensitivity_list[int((i*0.1)*len(sorted_weight))]
            #thres2 = sensitivity_list[int(((i+1) * 0.1) * len(sorted_weight))-1]
            num = 0
            for lid, pm in enumerate(network.get_params()):
                val = pm.get_value()
                for index, value in np.ndenumerate(val):
                    #if np.abs(val)[index] < thres2 and np.abs(val[index]) >= thres1:
                    if weight_variances[lid][index] >= thres1 and weight_variances[lid][index] < thres2:
                    #if sensitivity_test[lid][index] >= thres1 and sensitivity_test[lid][index] < thres2:
                        val[index] += np.random.normal(0, np.sqrt(max_var))
                        num += 1
                pm.set_value(val)
            error_val = test(out, np.concatenate(testXs), np.concatenate(testYs))
            print('error mask: ',i , error_val, num)
            error_list.append(error_val)
            network.set_param_values(original_param)
        plt.figure()
        plt.plot(error_list)
        plt.savefig('data/trained/gradient_temp/supervised_split_' + append + '/sensitivity_error_variancebased.png')

        #print('param diff', np.linalg.norm(original_param-network.get_param_values()))

        abc
        ###########################################################################################
        '''


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

            if cross_task_kldivergence:
                '''split_size = 0
                search_ratio = 0.00
                search_delta = 0.001
                cur_threshold = 0.0
                variance_list.sort()
                while split_size < split_param_size:
                    split_size = 0
                    cur_threshold = variance_list[int(search_ratio * total_param_size)]
                    for metric_id in range(int(search_ratio * total_param_size)):
                        if split_metrics[metric_id][3] <= cur_threshold:
                            split_size += 1
                    search_ratio += search_delta
                    if search_ratio > 0.999:
                        break
                print('search_ratio ', search_ratio)
                for metric_id in range(int(search_ratio * total_param_size)):
                    if split_metrics[metric_id][3] <= cur_threshold:
                        masks[split_metrics[metric_id][0]][split_metrics[metric_id][1]] = 1'''
                variance_list.sort()
                var_threshold = variance_list[int(0.15*total_param_size)]
                split_size = 0
                for metric_id in range(len(split_metrics)):
                    if split_metrics[metric_id][3] <= var_threshold:
                        masks[split_metrics[metric_id][0]][split_metrics[metric_id][1]] = 1
                        split_size += 1
                    if split_size >= split_param_size:
                        break
            else:
                if split_percentage < 1.0:
                    start = 0#int((split_percentage-0.5)*total_param_size)
                    num = 0
                    for i in range(start, split_param_size):
                        masks[split_metrics[i][0]][split_metrics[i][1]] = 1
                        num += 1
                    print('NUM: ', num)
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
            split_param_size+=1
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
            updates_split = lasagne.updates.adam(loss_split, params_split, learning_rate=0.001)
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

                testXs, testYs = synthesize_data(dim, 10000, tasks, split_param_size != 0)

                avg_error += test(out, np.concatenate(testXs), np.concatenate(testYs))
                avg_learning_curve.append(losses)

            pred_list.append(avg_error / reps)
            print(split_percentage, avg_error / reps)
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



