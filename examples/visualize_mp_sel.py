import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file = (sys.argv[1])
    else:
        assert(0)

    pol = joblib.load(file)

    classes = []

    sample_density = 40
    test_step = 1.0 / sample_density
    param_list1 = np.arange(0.0, 1.0 + test_step, test_step)
    param_list2 = np.arange(0.0, 1.0 + test_step, test_step)

    for param1 in param_list1:
        for param2 in param_list2:
            input = np.concatenate([np.zeros(11), np.array([param1, param2])])
            weights=(pol._f_blendweight([input]))
            classes.append(np.random.choice(4, 1, p=weights[0][0]))

    plt.imshow(np.reshape(classes, (len(param_list1), len(param_list2))), extent=[0, 1, 1, 0])
    plt.colorbar()
    plt.xlabel('Model parameter 2')
    plt.ylabel('Model parameter 1')
    plt.show()