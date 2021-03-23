import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from typing import Callable
from plotting_helpers import *


def sample_uniform_sphere(n_dims: int=2, n_points: int=100):
    points = torch.randn((n_points, n_dims))
    points = F.normalize(points)
    return points

def euclidean_distance(z, pts):
    return ((z-pts)**2).sum()

def spherical_gradient_descent(objective_function: Callable, z: torch.Tensor, points: torch.Tensor, lr: float = 0.001, n_steps: int=100, return_losses:bool=False):
    def step():
        loss = objective_function(z, points)
        if return_losses: losses.append(loss.item())
            
        print(loss)
        
        loss.backward()
        opt.step()
        z.data = F.normalize(z.data)
        # with torch.no_grad():
        
        opt.zero_grad()
    # points = sample_uniform_sphere()
    # points = points[points[:,0] > 0]
    opt = SGD([z], lr=lr)
    if return_losses: losses = []
    # n_steps=100
    for step_num in range(n_steps):
        step()
    
    if return_losses: return z, np.array(losses)
    return z

def run_uniform_demo():
    pts = sample_uniform_sphere(n_points=1000)
    pts = pts[pts[:,0] > 0]
    
    z = torch.randn_like(pts[:1,:])
    z = F.normalize(z)
    z = torch.nn.Parameter(z, requires_grad=True)
    
    z = spherical_gradient_descent(euclidean_distance, z, pts, n_steps=100)
    fig, ax = scatter2d(pts, show=False)
    ax.scatter(z.data[:,0], z.data[:,1], color='red', s=100)
    plt.show()
    # scatter2d(torch.cat([points, z.data]))
    # ax.scatter(z.data[:,0], z.data[:,1])
    
    
    # fig, ax = scatter2d(points, show=False)
    # ax.scatter(z.data[:,0], z.data[:,1], color='red', s=100)
    # plt.show()
    

### hyperbolic stuff

def hyperbolic_ip(v1 : torch.Tensor, v2 : torch.Tensor, keepdim=False):
    ip = v1 * v2
    ip.narrow(-1, 0, 1)
    ip = ip*(-1)
    ipsum = torch.sum(ip, dim=-1, keepdim=keepdim)
    return ipsum

def hyperbolic_natural_grad(v : torch.Tensor):
    v_prime = v.grad
    v_prime.narrow(-1, 0, 1)
    v_prime = v_prime*(-1)
    return v_prime

def hyperbolic_projection(v : torch.Tensor, v_prime : torch.Tensor):
    projection_correction = hyperbolic_ip(v.data, v_prime, keepdim=True) * v.data
    return v_prime + projection_correction

def hyperbolic_exponential_mapping(v1 : torch.Tensor, v2 : torch.Tensor):
    hyperbolic_v2_norm = torch.sqrt(hyperbolic_ip(v2, v2, keepdim=True))
    cosh_term = torch.cosh(hyperbolic_v2_norm)*v1
    
    hyperbolic_normalized_v2 = v2 / hyperbolic_v2_norm
    sinh_term = torch.sinh(hyperbolic_v2_norm) * hyperbolic_normalized_v2
    
    return cosh_term + sinh_term

def hyperbolic_gradient_descent(objective_function: Callable, z: torch.Tensor, points: torch.Tensor, lr: float = 0.001, n_steps: int=100):
    def step():
        natural_grad = hyperbolic_natural_grad(z)
        projected_natural_grad = hyperbolic_projection(z, natural_grad)
        projected_natural_grad = projected_natural_grad*(-lr)
        exp_map = hyperbolic_exponential_mapping(z.data, projected_natural_grad)
        z.data.copy_(exp_map)
    # n_steps=100
    for step_num in range(n_steps):
        step()
    return z


### mds

def mds_objective(z, pts):
    true_dist_matrix = torch.cdist(pts, pts)
    proposed_dist_matrix = torch.cdist(z, z)
    return euclidean_distance(true_dist_matrix, proposed_dist_matrix)
    
    

def spherical_mds(n_dims:int=1):
    # pts = sample_uniform_sphere(n_points=1000)
    # pts = pts[pts[:,0] > 0]
    
    # z = torch.randn_like(pts)[:,:n_dims]
    # z = F.normalize(z)
    # z = torch.nn.Parameter(z, requires_grad=True)
    
    # z = spherical_gradient_descent(mds_objective, z, pts, lr=0.0001, n_steps=100)
    pts = sample_uniform_sphere(n_dims=3, n_points=1000)
    pts = pts[pts[:,0] > 0]

    z = torch.randn_like(pts)[:,:2]
    z = F.normalize(z)
    z = torch.nn.Parameter(z, requires_grad=True)

    z, losses = spherical_gradient_descent(mds_objective, z, pts, lr=0.0001, n_steps=1000, return_losses=True)
    return losses
    
    
    
    



