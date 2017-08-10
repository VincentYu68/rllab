__author__ = 'yuwenhao'

import gym
from scipy.optimize import minimize
import numpy as np
import joblib
import math

class IK_Task:
    def __init__(self, dart_world):
        self.dart_world = dart_world

    def set_target(self, target):
        self.target = target

    def objective(self, q):
        self.dart_world.skeletons[-1].q = q

        fingertip = np.array([0.0, -0.25, 0.0])
        vec = self.dart_world.skeletons[-1].bodynodes[2].to_world(fingertip) - self.target

        return vec.dot(vec)

    def grad(self, q):
        self.dart_world.skeletons[-1].q = q
        fingertip = np.array([0.0, -0.25, 0.0])
        jac = self.dart_world.skeletons[-1].bodynodes[2].linear_jacobian(offset=fingertip, full=True)
        vec = self.dart_world.skeletons[-1].bodynodes[2].to_world(fingertip) - self.target
        return vec.dot(jac)

if __name__ == '__main__':
    data_set_name = 'reacher_task1.pkl'
    data_set_path = 'data/trained/supervised_reacher/' + data_set_name

    data_set_size = 10000

    env = gym.make('DartReacher3d-v1')
    dart_world = env.env.dart_world
    env.env.disableViewer=False

    joint_limits = []
    for i in range(len(dart_world.skeletons[-1].joints)):
        for j in range(dart_world.skeletons[-1].joints[i].num_dofs()):
            joint_limits.append([dart_world.skeletons[-1].joints[i].position_lower_limit(j), dart_world.skeletons[-1].joints[i].position_upper_limit(j)])

    joint_limits_tp = tuple(joint_limits)
    joint_limits = np.array(joint_limits)

    iktask = IK_Task(dart_world)

    X = []
    Y = []
    while len(X) <  data_set_size:
        target = np.random.uniform(0.05, 0.25, 3)
        iktask.set_target(target)

        x0 = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1], len(joint_limits))
        res = minimize(iktask.objective, x0, method='L-BFGS-B', jac=iktask.grad, bounds=joint_limits_tp)

        valid = True
        for num in res.x:
            if math.isnan(num):
                valid = False
                break
        if res.fun > 0.001:
            valid = False

        if valid:
            X.append(np.concatenate([x0, target]))
            Y.append(res.x)

    joblib.dump([X, Y], data_set_path, compress=True)

