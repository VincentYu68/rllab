from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_rbf_policy import GaussianRBFPolicy
from rllab.policies.gaussian_hmlp_policy import GaussianHMLPPolicy
from rllab.policies.gaussian_hlc_policy import GaussianHLCPolicy

import numpy as np
import joblib
import time
import os.path

cur_exp_name = 'Walker3d_llc'
dual_exp_name = 'Walker3d_hlc'

def hlc2llc(hlc, llc):
    # copy parameter from separate controller to integrated controller
    int_pol_param = llc._mean_network.get_params()
    hlc_param = hlc._mean_network.get_params()

    for int_param in int_pol_param:
        for param in hlc_param:
            if param.name == int_param.name:
                int_param.set_value(param.get_value(borrow=True))


def run_task(*_):
    env = normalize(GymEnv("DartWalker3d-v1", record_video=False))

    policy = GaussianHMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64,32),
        subnet_split1=[5, 6, 7, 8, 9, 10, 23, 24, 25, 26, 27, 28],
        subnet_split2=[11, 12, 13, 14, 15, 16, 29, 30, 31, 32, 33, 34],
        sub_out_dim=6,
        option_dim=4,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo2 = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=env.horizon,
        n_itr=10,
        discount=0.99,
        step_size=0.01,
        epopt_epsilon = 1.0,
        epopt_after_iter = 0,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    for i in range(100):
        algo2.current_itr = 0
        algo2.train()
        llc_signal_file = 'data/local/experiment/'+cur_exp_name+'/signalfile.txt'
        f = open(llc_signal_file, 'w')
        f.write(str(i))
        f.close()

        hlc_signal_file = 'data/local/experiment/'+dual_exp_name+'/signalfile.txt'
        hlc_policy_file = 'data/local/experiment/'+dual_exp_name+'/policy.pkl'
        while True:
            if os.path.isfile(hlc_signal_file):
                f = open(hlc_signal_file, 'r')
                signal = int(f.read())
                f.close()
                print(signal, i)
                if signal == i:
                    dual_policy = joblib.load(hlc_policy_file)
                    hlc2llc(dual_policy, policy)
                    break
            time.sleep(20) # sleep for a minute before check again


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=6,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    exp_name=cur_exp_name
    # plot=True
)
