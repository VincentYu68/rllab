from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import joblib


if __name__ == '__main__':
    dim = 10
    in_dim = dim + 1
    out_dim = dim
    task_num = 3
    random_split = False
    prioritized_split = False
    append = 'edgewise_grid' + str(dim)
    reps = 1
    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'
    init_epochs = 70
    batch_size = 15000
    epochs = 30
    test_epochs = 200
    hidden_size = (64, 32)
    append += str(batch_size) + ':_' + str(init_epochs) + '_' + str(epochs) + '_' + str(test_epochs) + '_' + str(
        hidden_size)

    task_similarities = [0, 2, 4, 6, 8, 10]
    split_percentages = [0.000001, 0.2, 0.4, 0.6, 0.8, 0.999]  # 2.0 means using mean + 1 std as the threshold


    performances = np.array(joblib.load('data/trained/gradient_temp/supervised_split_' + append + '/performance.pkl'))
    performances = performances.reshape((performances.shape[0], performances.shape[2]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(task_similarities, split_percentages)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, performances, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.show()