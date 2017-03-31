import argparse

import joblib

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        default='',
                        help='path to the snapshot file')
    parser.add_argument('--env', type=str,
                        default='',
                        help='env to simulate')
    parser.add_argument('--policy', type=str,
                        default='',
                        help='policy to use')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')

    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    if len(args.file) == 0:
        env = normalize(GymEnv(args.env))
        policy = joblib.load(args.policy)
    else:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']


    '''param_bf = policy.get_params(trainable=True)
    print(param_bf)
    w1 = param_bf[4].get_value(borrow=True)
    b1 = param_bf[5].get_value(borrow=True)
    w2 = param_bf[10].get_value(borrow=True)
    b2 = param_bf[11].get_value(borrow=True)


    param_bf[4].set_value(w2)
    param_bf[5].set_value(b2)
    param_bf[10].set_value(w1)
    param_bf[11].set_value(b1)

    tw1 = param_bf[8].get_value(borrow=True)
    tb1 = param_bf[9].get_value(borrow=True)

    param_bf[12].set_value(tw1)
    param_bf[13].set_value(tb1)'''


    #flatten_param = flatten_tensors([param.get_value(borrow=True) for param in param_bf])

    #policy.set_param_values(flatten_param, trainable=True)

    print('=========')
    while True:
        path = rollout(env, policy, max_path_length=args.max_path_length,
                       animated=True, speedup=args.speedup)
        if not query_yes_no('Continue simulation?'):
            break
