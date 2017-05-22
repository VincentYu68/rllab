import pickle
import numpy as np
import matplotlib.pyplot as plt

data = []
for i in range(480, 499):
    data += pickle.load(open('data/local/experiment/hopper_footmass_0005/mp_rew_' + str(i) + '.pkl', 'rb'))

x = []
y = []
for d in data:
    x.append(d[0])
    y.append(d[1])

plane_search = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

min_std = 10000
best_sep_plane = -1

for sep_plane in plane_search:

    xl = []
    yl = []
    xr = []
    yr = []

    for i in range(len(x)):
        if x[i] < sep_plane:
            xl.append(x[i])
            yl.append(y[i])
        else:
            xr.append(x[i])
            yr.append(y[i])

    if np.abs(np.std(yl) + np.std(yr)) < min_std:
        min_std = np.abs(np.std(yl) + np.std(yr))
        best_sep_plane = sep_plane
    print(np.std(yl), np.std(yr))

xl = []
yl = []
xr = []
yr = []

for i in range(len(x)):
    if x[i] < best_sep_plane:
        xl.append(x[i])
        yl.append(y[i])
    else:
        xr.append(x[i])
        yr.append(y[i])

plt.scatter(xl, yl)
plt.scatter(xr, yr)
plt.show()