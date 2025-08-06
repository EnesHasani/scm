import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Dict
from torch.distributions.laplace import Laplace
import torch.distributions as dist
import torch.nn.functional as F
from playground_scm.utils import torch_random_choice

class ToModule(nn.Module):
    def __init__(self, func):
        super(ToModule, self).__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

def activation_sampling(nonlins: str):
    """
    Sample once and return a fixed activation function f such that
    every call f(x) is deterministic—and for the ‘sophisticated’ variants
    we clamp outputs to [-1000,1000].
    """

    # stateless activations
    def identity(x): return x

    if nonlins == "tanh":
        return torch.tanh
    
    elif nonlins == "relu":
        return torch.relu

    elif nonlins == "sin":
        return torch.sin

    elif nonlins == "id":
        return identity

    elif nonlins == 'mixed':
        return torch_random_choice([torch.tanh, torch.relu, identity, torch.sin])


def gaussian_sampling(shape: tuple, std: float = None) -> Callable[[], torch.Tensor]:
    """
    Generates a function that samples additive noise from a normal distribution.
    """
    def sample_noise():
        return torch.normal(0, std, shape)
    
    return sample_noise


class MakeStructuralEquations(nn.Module):
    """
    A PyTorch module that defines a structural equation for a node in a causal graph 
    based on its parents. The model linearly combines the parent values using a linear 
    layer and applies a randomly selected non-linear activation function.
    The additive noise is only sampled once and added to the output.


    :param parents: List of names of the parent variables for this node.
    :param possible_activations: Optional list of activation functions to sample from.
                                  If not provided, defaults to [square, ReLU, tanh].

    """ 

    def __init__(self, 
                 parents: List[str], 
                 samples_shape: tuple,
                 noise_std: float,
                 noise_dist: str,
                 nonlins: str,
                 max_hidden_layers: int,
                 nonlin_type: str
                ) -> None:
        super().__init__()
        self.parents: List[str] = parents
        
        if len(parents) > 0:
            self.layers: nn.Linear = nn.Linear(len(parents), 1, bias=False) 
        else:
            self.layers = None
            
        self.activation: Callable[[torch.Tensor], torch.Tensor] = activation_sampling(nonlins=nonlins)
        self.samples_shape: tuple = samples_shape
        self.nonlins: str = nonlins
        self.nonlin_type: str = nonlin_type
        
        if noise_dist == 'gaussian':
            self.additive_noise: torch.Tensor = gaussian_sampling(shape=samples_shape, std=noise_std)()
        

    def forward(self, **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the structural equation.

        :param kwargs: Keyword arguments where each key is a parent name and each value 
                       is a scalar or tensor representing the parent's value.

        :raises KeyError: If any required parent variable is missing from kwargs.

        :return: Transformed tensor after applying the learned linear combination 
                 and the sampled non-linear activation.
        """
        if len(self.parents) == 0:
           output = self.additive_noise
        else:
            parent_values = [kwargs[parent] for parent in self.parents]
            parent_tensor = torch.stack(parent_values, dim=-1)
            
            with torch.no_grad():
                if self.nonlin_type == 'pre':
                    output = self.layers(parent_tensor).squeeze(-1)
                    output = self.activation(output)
                    output += self.additive_noise
                elif self.nonlin_type == 'post':
                    output = self.layers(parent_tensor).squeeze(-1)
                    output += self.additive_noise
                    output = self.activation(output)
                    
        return output