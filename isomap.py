import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.graph import graph_shortest_path


class Isomap():
    def __init__(self, data, n_components: int, n_neighbors: int): # epsilon: float
        self.data = data
        self.n_components = n_components
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        
        self.dist_matrix = self.get_dist_matrix(data, self.n_neighbors)
        # self.dist_matrix = self.mean_centering(self.dist_matrix)
        
    @staticmethod
    def get_dist_matrix(data, n_neighbors): # epsilon
        dist_matrix = torch.cdist(data, data)
        # dist_matrix[dist_matrix >= epsilon] = 0
        
        nn_dist_matrix = torch.zeros_like(dist_matrix)
        sort_distances = torch.argsort(dist_matrix, axis=1)
        sort_distances = sort_distances[:, 1:n_neighbors+1]
        for i,j in enumerate(sort_distances):
            nn_dist_matrix[i,j] = dist_matrix[i,j]
        return nn_dist_matrix
        
        # return dist_matrix
    
    @staticmethod
    def mean_centering(data):
        row_mean = data.mean(dim=0)
        col_mean = data.mean(dim=1)
        total_mean = data.mean()
        return data - row_mean - col_mean + total_mean
        
    def mds(self, dist_matrix, n_components:int):
        dist_matrix = self.mean_centering(dist_matrix)
        eigenvals, eigenvecs = torch.eig(dist_matrix, eigenvectors=True)
        eigenvals = eigenvals[:,0]
        
        sorted_eigenvals, sort_order = torch.sort(eigenvals, descending=True)
        sorted_eigenvecs = eigenvecs[:, sort_order]
        
        return sorted_eigenvecs[:,:n_components]
    
    def get_projection(self):
        shortest_paths_dist_matrix = graph_shortest_path(self.dist_matrix, directed=False)
        shortest_paths_dist_matrix = -0.5 * (shortest_paths_dist_matrix ** 2)
        shortest_paths_dist_matrix = torch.tensor(shortest_paths_dist_matrix)
        return self.mds(shortest_paths_dist_matrix, self.n_components)
        

def plot_s_curve(X, color, show=True):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)
    if show: plt.show()
    return fig, ax


from sklearn import datasets

n_points = 1000
X, color = datasets.make_s_curve(n_points, random_state=0)
# plot_s_curve(X, color)

data = torch.tensor(X)

n_neighbors = 10
n_components = 2
epsilon=0.1

iso = Isomap(data, n_components, n_neighbors)

proj = iso.get_projection()
plt.scatter(proj.T[0], proj.T[1], c=color)
print('done')