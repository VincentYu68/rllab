import pickle
import numpy as np
import matplotlib.pyplot as plt



data = []
for i in range(140, 149):
    data += pickle.load(open('data/local/experiment/hopper_restfoot_seed6_seg_target1/mp_rew_' + str(i) + '.pkl', 'rb'))


x = []
y = []
for d in data:
    x.append(d[0])
    y.append(d[1])

x = np.array(x)

plt.scatter(x[:,0], x[:, 1], c = y, alpha=0.3)
plt.colorbar()
plt.show()
