__author__ = 'yuwenhao'

from  rllab.uposi.policy_split_rl_evaluation import *

if __name__ == '__main__':
    num_parallel = 7

    hidden_size = (64, 32)
    batch_size = 20000
    pathlength = 1000

    random_split = False
    prioritized_split = False
    adaptive_sample = False

    initialize_epochs = 90
    grad_epochs = 10
    test_epochs = 300
    seed=2
    append = 'hopper_friction_6432net_sd%d_splitstd_accumgrad_specbaseline_%dk_%d_%d_unweighted'%(seed,batch_size/1000, initialize_epochs, grad_epochs)

    env_name = "DartHopper-v1"
    task_size = 2

    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'
    if adaptive_sample:
        append += '_adaptive_sample'

    load_init_policy = False
    load_split_data = False

    alternate_update = False
    accumulate_gradient = True

    imbalance_sample = False
    sample_ratio = [0.1, 0.9]

    if alternate_update:
        append += '_alternate_update'
    if accumulate_gradient:
        append += '_accumulate_gradient'

    split_percentages = [0.0, 0.5]

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
                       seed=seed,
                       test_num=3,
                       use_param_variance=0)
