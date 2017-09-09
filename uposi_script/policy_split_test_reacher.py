__author__ = 'yuwenhao'

from  rllab.uposi.policy_split_rl_evaluation import *

if __name__ == '__main__':
    num_parallel = 8

    hidden_size = (64, 32)
    batch_size = 40000
    pathlength = 500

    random_split = False
    prioritized_split = False
    adaptive_sample = False

    use_param_variance = 1
    reverse_metric = True

    initialize_epochs = 95
    grad_epochs = 5
    test_epochs = 100
    append = 'reacher_3modelsexp1_alivepenalty_tasksplit_taskinput_6432net_sd1_vanbaseline_splitstd_accumgrad_%dk_%d_%d_unweighted'%(batch_size/1000, initialize_epochs, grad_epochs)


    if use_param_variance == 1:
        append += '_param_variance'
    if reverse_metric:
        append += '_reverse_metric'

    env_name = "DartReacher3d-v1"
    task_size = 3

    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'

    load_init_policy = True
    load_split_data = True

    alternate_update = False
    accumulate_gradient = True

    imbalance_sample = False
    sample_ratio = [0.1, 0.9]

    if alternate_update:
        append += '_alternate_update'
    if accumulate_gradient:
        append += '_accumulate_gradient'

    split_percentages = [0.5]

    perform_evaluation(num_parallel, hidden_size,
                       batch_size,
                       pathlength,
                       random_split,
                       prioritized_split,
                       adaptive_sample,
                       initialize_epochs,
                       grad_epochs,
                       test_epochs,
                       append,
                       task_size,
                       load_init_policy,
                       load_split_data,
                       alternate_update,
                       accumulate_gradient,
                       imbalance_sample,
                       sample_ratio,
                       split_percentages,
                       env_name,
                       seed=1,
                       test_num=1,
                       use_param_variance=use_param_variance,
                       param_variance_batch=1000,
                       reverse_metric=reverse_metric)
