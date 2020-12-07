import numpy as np
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_decision_surface(clas, X, Y):
    """Plot a decision surface for 2 classes. """
    # step size in the mesh
    h = .02
    # Create color maps
    cmap_light = ListedColormap(['lightgreen', 'lightcoral'])
    cmap_bold = ListedColormap(['green','red'])
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X['X1'].min() - 1, X['X1'].max() + 1
    y_min, y_max = X['X2'].min() - 1, X['X2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clas.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

    # Plot also the training points
    plt.scatter(X['X1'], X['X2'], c=Y, cmap=cmap_bold, s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Classification")
    plt.show()

def plot_decision_surface_knn(knn, X, Y, voronoi=False):
    """Plot a decision surface for 2 classes, optionally
    overlaying the voronoi diagram. """
    # step size in the mesh
    h = .02
    # Create color maps
    cmap_light = ListedColormap(['lightgreen', 'lightcoral'])
    cmap_bold = ListedColormap(['green','red'])
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X['X1'].min() - 1, X['X1'].max() + 1
    y_min, y_max = X['X2'].min() - 1, X['X2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

    if voronoi:
        vor = Voronoi(X)
        voronoi_plot_2d(vor, show_points=False, ax=ax)
    # Plot also the training points
    plt.scatter(X['X1'], X['X2'], c=Y, cmap=cmap_bold, s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("k-NN Classification")
    plt.show()