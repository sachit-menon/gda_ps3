import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def scatter_3d(X, color, show=True):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)
    if show: plt.show()
    return fig, ax

def scatter2d(data: torch.Tensor, color = None, fig=None, ax=None, show: bool =True):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    
    ax.scatter(data.T[0], data.T[1], c=color, cmap=plt.cm.Spectral)
    if show: plt.show()
    return fig, ax

from matplotlib import offsetbox
# Scale and visualize the embedding vectors
def plot_embedding(X, y, data, title=None, zoom=1., fig=None, ax=None, reverse_cmap=False):
    X = np.array(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    # fig = plt.figure()
    # ax = plt.subplot(111)
    
    # for i in range(X.shape[0]):
    #     plt.text(X[i, 0], X[i, 1], str(y[i]),
    #             color=plt.cm.Set1(y[i]), #y[i] / 10.
    #             fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            cm = plt.cm.gray if reverse_cmap else plt.cm.gray_r
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(data.images[i], cmap=cm, zoom=zoom),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        ax.set_title(title)
    return fig, ax