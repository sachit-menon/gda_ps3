import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def sturm_algorithm(points: torch.Tensor, dims: int):
    iteration_num = 0
    current_sigma = points[0]
    
    while iteration_num <= 1000:
        old_sigma = current_sigma
        iteration_num += 1
        new_point = points[iteration_num]
        current_sigma = sturm_iteration(current_sigma, new_point, iteration_num)
        distance_moved = torch.dist(old_sigma, current_sigma)
        if distance_moved < 0.05:
            return current_sigma
    return current_sigma

def sturm_iteration(sigma: torch.Tensor, pt: torch.Tensor, n: int, dims: int):
    path = compute_geodesic(sigma, pt, dims)
    updated_sigma = traverse_geodesic_path(path, 1/n)
    return updated_sigma

class GeodesicPath():
    '''
    Geodesic path constructed by stitching together paths. 
    '''
    def __init__(self, *paths):
        self.paths = paths
        self.path_length = self.get_path_length()
        
    def get_path_length(self):
        length = 0
        for path in self.paths:
            length += path.length
        return length
    
    def traverse_path_length(self, length):
        for path in self.paths:
            length -= path.length
            if length < 0:
                return path
        return path
    
    def traverse_path_fraction(self, fraction):
        '''
        Returns the point found `fraction' of the way along the geodesic path.
        '''
        traversal_length = self.path_length * fraction
        return self.traverse_path_length(self, traversal_length)
    
class Tree():
    def __init__(self, edges=list(), edge_lengths=list()) -> None:
        self.edges = edges
        self.edge_lengths = edge_lengths
        
        
    def contract_split(self, edge_num):
        edges1 = self.edges[:edge_num]
        edge_lengths_1 = self.edge_lengths[:edge_num]
        if edge_num == len(self.edges):
            # trivial split
            return Tree(edges1, edge_lengths_1), None
        edges2 = self.edges[edge_num+1:]
        edge_lengths_2 = self.edge_lengths[edge_num+1:]
        return Tree(edges1, edge_lengths_1), Tree(edges2, edge_lengths_2)


class Path():
    '''
    Straight path in Euclidean space from one point to another. 
    '''
    def __init__(self, pt1: torch.Tensor, pt2: torch.Tensor) -> None:
        self.pt1 = pt1 
        self.pt2 = pt2
        self.length = torch.dist(pt1, pt2, p=2)

def compute_geodesic(points1: torch.Tensor, points2: torch.Tensor, orthants=(0)):
    '''
    Returns the geodesic path from pt1 to pt2 
    '''
    paths = []
    for i, orthant in enumerate(orthants):
        paths.append(Path(points1[i], points2[i]))
    return GeodesicPath(*paths)

def traverse_geodesic_path(path, fraction: float):
    '''
    Returns the point found `fraction' of the way along the geodesic path specified by `path'.
    '''
    return path.traverse_path_fraction(fraction)




