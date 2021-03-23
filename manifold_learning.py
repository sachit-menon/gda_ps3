import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.utils.graph import graph_shortest_path
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

from plotting_helpers import *

class ManifoldProjection():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_dist_matrix(data, n_neighbors, return_sort: bool = False): # epsilon
        dist_matrix = torch.cdist(data, data)
        # dist_matrix[dist_matrix >= epsilon] = 0
        
        nn_dist_matrix = torch.zeros_like(dist_matrix)
        sorted_indices = torch.argsort(dist_matrix, axis=1)
        sorted_indices = sorted_indices[:, 1:n_neighbors+1]
        for i,j in enumerate(sorted_indices):
            nn_dist_matrix[i,j] = dist_matrix[i,j]
        if return_sort:
            return nn_dist_matrix, sorted_indices
        return nn_dist_matrix
    
    @staticmethod
    def mean_centering(data):
        row_mean = data.mean(dim=0)
        col_mean = data.mean(dim=1)
        total_mean = data.mean()
        return data - row_mean - col_mean + total_mean
    
    @staticmethod
    def get_sorted_eigenvalues_eigenvectors(input_matrix: torch.Tensor, descending: bool=False):
        eigenvals, eigenvecs = torch.eig(input_matrix, eigenvectors=True)
        eigenvals = eigenvals[:,0]
        
        sorted_eigenvals, sort_order = torch.sort(eigenvals, descending=descending)
        sorted_eigenvecs = eigenvecs[:, sort_order]
        return sorted_eigenvals, sorted_eigenvecs
        
    
class MDS(ManifoldProjection):
    def __init__(self, n_components: int, n_neighbors: int): # epsilon: float
        
        self.n_components = n_components
        # self.epsilon = epsilon
        self.n_neighbors = n_neighbors
    
    def mds(self, dist_matrix, n_components:int):
        dist_matrix = self.mean_centering(dist_matrix)
        _, sorted_eigenvecs = self.get_sorted_eigenvalues_eigenvectors(dist_matrix, descending=True)
        return sorted_eigenvecs[:,:n_components]
    
    def get_projection(self, data):
        self.dist_matrix = self.get_dist_matrix(data, self.n_neighbors)
        return self.mds(self.dist_matrix, self.n_components)
        
    
class Isomap(MDS):
    def __init__(self, n_components: int, n_neighbors: int): # epsilon: float
        
        self.n_components = n_components
        # self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        

    
    def get_projection(self, data):
        self.dist_matrix = self.get_dist_matrix(data, self.n_neighbors)
        shortest_paths_dist_matrix = graph_shortest_path(self.dist_matrix, directed=False)
        shortest_paths_dist_matrix = -0.5 * (shortest_paths_dist_matrix ** 2)
        shortest_paths_dist_matrix = torch.tensor(shortest_paths_dist_matrix)
        return self.mds(shortest_paths_dist_matrix, self.n_components)
       
       
class LLE(ManifoldProjection):
    def __init__(self, n_components: int, n_neighbors: int) -> None:
                
        self.n_components = n_components
        # self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        super().__init__()
        
    def get_projection(self, data: torch.Tensor):
        self.dist_matrix, sort_indices = self.get_dist_matrix(data, self.n_neighbors, return_sort=True)
        
        n = data.shape[0]
        W = torch.zeros((n, n), dtype=torch.double)
        for i in range(n):
            xi = data[i, :]
            xi_neighbor_indices = sort_indices[i, :]
            neighbor_values = data[xi_neighbor_indices, :] - xi
            local_covariance = neighbor_values @ neighbor_values.T # reversed? 
            solved_w_matrix = torch.pinverse(local_covariance)
            W[i, xi_neighbor_indices] = (torch.sum(solved_w_matrix, dim=1)/torch.sum(solved_w_matrix)).double()
        
        m_half = (torch.eye(n) - W)
        M = m_half.T @ m_half
        
        _, sorted_eigenvecs = self.get_sorted_eigenvalues_eigenvectors(M)
        return sorted_eigenvecs[:,1:self.n_components+1]
        
class LaplacianEigenmap(ManifoldProjection):
    def __init__(self, n_components: int, sigma: float=1, nearest_neighbor_m: int=10, graph_method: str = 'gaussian'):
        self.n_components = n_components
        self.sigma = sigma
        self.graph_method = graph_method
        self.nearest_neighbor_m = nearest_neighbor_m
        pass
    
    @staticmethod
    def gaussian_kernel(weight_matrix: torch.Tensor, sigma: float):
        return torch.exp(-weight_matrix)/(2*sigma**2)
    
    def form_weight_matrix(self, points):
        if self.graph_method == 'gaussian':
            W = torch.cdist(points, points, p=2).pow(2)
            W = self.gaussian_kernel(W, self.sigma)
            return W
        elif self.graph_method == 'kNN':
            # nbrs = NearestNeighbors(n_neighbors=self.nearest_neighbor_m, algorithm='ball_tree').fit(self.points)
            # nbrs
            W = torch.tensor(kneighbors_graph(points, n_neighbors=self.nearest_neighbor_m).toarray())
            return W
        else:
            raise NotImplementedError
    
    def form_random_walk_laplacian(self, W: torch.Tensor):
        # D = torch.zeros_like(W)
        diag = W.sum(dim=1)
        D = torch.diag_embed(diag)
        # D.fill_diagonal_(diag)
        return torch.inverse(D) @ (D - W)
    
    def get_projection(self, data):
        W = self.form_weight_matrix(data)
        L_rw = self.form_random_walk_laplacian(W)
        _, sorted_eigenvecs = self.get_sorted_eigenvalues_eigenvectors(L_rw, descending=False)
        return sorted_eigenvecs[:,1:self.n_components+1]

from sklearn import datasets
def run_all_algorithms_3d(data: torch.tensor, color=None, experiment_name=None):
    n_neighbors = 10
    n_components = 2
    epsilon=0.1
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,5), constrained_layout=True)

    title = 'Manifold Learning Comparison'
    if experiment_name: title += ' - ' + experiment_name
    fig.suptitle(title)
    
    mds = MDS(n_components, n_neighbors)
    proj = mds.get_projection(data)
    fig, ax = scatter2d(proj, color=color, fig=fig, ax=ax1, show=False)
    ax1.set_title('MDS Projection')
    # fig.suptitle('MDS Projection')
    # plt.show()

    iso = Isomap(n_components, n_neighbors)
    proj = iso.get_projection(data)
    fig, ax = scatter2d(proj, color=color, fig=fig, ax=ax2, show=False)
    ax2.set_title('Isomap Projection')
    # fig.suptitle('Isomap Projection')
    # plt.show()

    lle = LLE(n_components, n_neighbors)
    proj = lle.get_projection(data)
    fig, ax = scatter2d(proj, color=color, fig=fig, ax=ax3, show=False)
    ax3.set_title('LLE Projection')
    # fig.suptitle('LLE Projection')
    # plt.show()

    laple = LaplacianEigenmap(n_components, sigma=0., nearest_neighbor_m=n_neighbors, graph_method='kNN')
    proj = laple.get_projection(data)
    fig, ax = scatter2d(proj, color=color, fig=fig, ax=ax4, show=False)
    ax4.set_title('Laplacian Eigenmap Projection')
    
    
    # fig.suptitle('Laplacian Eigenmap Projection')
    # plt.show()

    # sigma = 1.

    # laple = LaplacianEigenmap(n_components, sigma=sigma)
    # proj = laple.get_projection(data)
    # fig, ax = scatter2d(proj, color=color, show=False)
    # fig.suptitle('Laplacian Eigenmap Projection')
    # plt.show()

    print('done')
    
def run_s_curve():
    n_points = 1000
    X, color = datasets.make_s_curve(n_points, random_state=0)
    scatter_3d(X, color)

    data = torch.tensor(X)
    
    run_all_algorithms_3d(data, color, experiment_name='S Curve')
    
def run_swiss_roll():
    n_points = 1000
    X, color = datasets.make_swiss_roll(n_points, random_state=0)
    scatter_3d(X, color)

    data = torch.tensor(X)
    
    run_all_algorithms_3d(data, color, experiment_name='Swiss Roll')
    
def run_digits():
    digits = datasets.load_digits(n_class=6)
    X = digits.data
    y = digits.target
    n_samples, n_features = X.shape
    data = torch.tensor(X)
    
    n_neighbors = 30
    n_components = 2
    
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,5), constrained_layout=True)

    # fig.suptitle('Manifold Learning Comparison - Digits')
    
    mds = MDS(n_components, n_neighbors)
    proj = mds.get_projection(data)
    fig, ax = plot_embedding(proj, y, digits, title='MDS Projection')
    plt.show()

    iso = Isomap(n_components, n_neighbors)
    proj = iso.get_projection(data)
    fig, ax = plot_embedding(proj, y, digits, title='Isomap Projection')
    plt.show()

    lle = LLE(n_components, n_neighbors)
    proj = lle.get_projection(data)
    fig, ax = plot_embedding(proj, y, digits, title='LLE Projection')
    plt.show()

    laple = LaplacianEigenmap(n_components, sigma=0., nearest_neighbor_m=n_neighbors, graph_method='kNN')
    proj = laple.get_projection(data)
    fig, ax = plot_embedding(proj, y, digits, title='Laplacian Eigenmap Projection')
    plt.show()

    # sigma = 1.

    # laple = LaplacianEigenmap(n_components, sigma=sigma)
    # proj = laple.get_projection(data)
    # fig, ax = plot_embedding(proj, y, digits, title='Laplacian Eigenmap Projection')
    # plt.show()
    
def run_faces():
    faces = datasets.fetch_olivetti_faces()
    X = faces.data
    y = faces.target
    
    X = X[y < 6]
    y = y[y < 6]
    
    n_samples, n_features = X.shape
    n_neighbors = 30
    n_components = 2
    data = torch.tensor(X)
    
    
    
    mds = MDS(n_components, n_neighbors)
    proj = mds.get_projection(data)
    fig, ax = plot_embedding(proj, y, faces, title='MDS Projection', reverse_cmap=True)
    plt.show()

    iso = Isomap(n_components, n_neighbors)
    proj = iso.get_projection(data)
    fig, ax = plot_embedding(proj, y, faces, title='Isomap Projection', reverse_cmap=True)
    plt.show()

    lle = LLE(n_components, n_neighbors)
    proj = lle.get_projection(data)
    fig, ax = plot_embedding(proj, y, faces, title='LLE Projection', reverse_cmap=True)
    plt.show()

    laple = LaplacianEigenmap(n_components, sigma=0., nearest_neighbor_m=n_neighbors, graph_method='kNN')
    proj = laple.get_projection(data)
    fig, ax = plot_embedding(proj, y, faces, title='Laplacian Eigenmap Projection', reverse_cmap=True)
    plt.show()

    sigma = 1.

    laple = LaplacianEigenmap(n_components, sigma=sigma)
    proj = laple.get_projection(data)
    fig, ax = plot_embedding(proj, y, faces, title='Laplacian Eigenmap Projection', reverse_cmap=True)
    plt.show()
    
# run_s_curve()

# run_swiss_roll()

run_digits()

# run_faces()
