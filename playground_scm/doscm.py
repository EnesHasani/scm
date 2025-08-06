import torch
import networkx as nx
import random
import numpy as np
from playground_scm.generators import SCMGenerator
from playground_scm.MakeStructuralEquations import MakeStructuralEquations, gaussian_sampling
from playground_scm.utils import torch_random_choice
import pickle as pkl
from copy import deepcopy

def get_batch(
    batch_size: int,
    seq_len: int, 
    num_features: int,
    hyperparameters: int,
    device: str = 'cpu',
    num_outputs: int = 1,
    num_treatments: int = 1,    
    epoch=None,
    return_SCM = False,
    **kwargs,
):

    class DoSCM(torch.nn.Module):
        def __init__(self, hyperparameters):
            super(DoSCM, self).__init__()

            with torch.no_grad():
                for key in hyperparameters:
                    setattr(self, key, hyperparameters[key])

            self.batch_size = batch_size
            self.num_samples = seq_len
            self.samples_shape = (self.batch_size, self.num_samples)
            self.num_features = round(num_features)
            self.num_nodes = self.num_features + round(self.num_unobserved) + num_outputs + num_treatments
            edge_prob_min = 1 / (self.num_features + 1)
            self.edge_prob = np.random.uniform(edge_prob_min, 1)
        
        def forward(self):  

            gen = SCMGenerator(all_functions={'nonlinear': MakeStructuralEquations}, 
                               seed=self.seed, 
                               samples_shape=self.samples_shape, 
                               noise_std=self.noise_std, 
                               noise_dist=self.noise_dist,
                               nonlins=self.nonlins,
                               max_hidden_layers=self.max_hidden_layers,
                               nonlin_type=self.nonlin_type)
            
            # generate graph
            if self.graph is None:
                self.graph = gen.create_graph_from_nodes(num_nodes=self.num_nodes, p=self.edge_prob)

            if self.exo_dist == 'gaussian':
                exo_distribution = gaussian_sampling(self.samples_shape, self.exo_std)

            # initialize SCM
            self.scm = gen.create_scm_from_graph(self.graph, 
                                                 possible_functions=["nonlinear"], 
                                                 exo_distribution=exo_distribution,
                                                 exo_distribution_kwargs={},
                                                 zero_one_treatment=self.zero_one_treatment
                                                ) 

            if self.t_idx is None: # sample treatment as variable with descendent(s)
                t_cand = [var for var in list(self.graph.nodes) if self.graph.out_degree(var) > 0]         
                self.scm.t_key = torch_random_choice(t_cand)
            elif len(list(self.graph.edges)) == 0: # if no edges
                self.scm.t_key = torch_random_choice(list(self.graph.nodes))
            else:
                for var in list(self.graph.nodes): # predefined treatment
                    if str(self.t_idx) in var:
                        self.scm.t_key = var

            # sample outcome as descendent of treatment
            if len(list(self.graph.edges)) == 0:
                self.scm.y_key = torch_random_choice(list(set(self.graph.nodes)-set([self.scm.t_key])))
            elif self.y_idx is None:
                t_desc = list(nx.descendants(self.graph, self.scm.t_key))
                self.scm.y_key = torch_random_choice(t_desc)
            else:
                for var in list(self.graph.nodes):
                    if str(self.y_idx) in var:
                        self.scm.y_key = var

            # sample observational dataset, get values of exogenous and endogenous variables
            endo_obs, exo_obs = self.scm.get_next_sample(graph=self.graph, binarize=True)
            sample_obs = endo_obs | exo_obs

            # sample counterfactual dataset by setting treatment and holding exogenous terms/noise constant
            t_cntf = deepcopy(sample_obs[self.scm.t_key])
            for b in range(self.batch_size):
                t_obs = deepcopy(sample_obs[self.scm.t_key][b])
                t_cntf[b] = torch.where(t_obs == self.scm.t1s[b], self.scm.t2s[b], self.scm.t1s[b])

            if 'X' in self.scm.t_key:  # if enogenous variable replace functional mechanism with do(t=t')
                self.scm.do_interventions([(self.scm.t_key, (lambda: t_cntf, {}))])
            else:  # if exogenous variable, change the value of the variable directly
                exo_obs[self.scm.t_key] = t_cntf

            endo_int, exo_int = self.scm.get_next_sample(exogenous_vars=exo_obs, graph=self.graph)
            sample_int = endo_int | exo_int

            # sample random covariate set
            if self.x_idcs is None:
                X_cand = set(self.graph.nodes) - set([self.scm.y_key, self.scm.t_key])
                self.x_keys = [self.scm.t_key] + list(np.random.choice(list(X_cand), size=self.num_features, replace=False))
            else: # pre-defined covariates
                self.x_keys = [self.scm.t_key]
                for var in list(self.graph.nodes):
                    if int(var[-1]) in self.x_idcs:
                        self.x_keys.append(var)

            x_obs = torch.stack([sample_obs[key] for key in self.x_keys]).permute(-1, 1, 0)
            x_int = torch.stack([sample_int[key] for key in self.x_keys]).permute(-1, 1, 0)

            if self.zero_one_treatment:
                x_obs[:, :, 0] = self.scm.get_zero_one_treatment(x_obs[:, :, 0])
                x_int[:, :, 0] = self.scm.get_zero_one_treatment(x_int[:, :, 0])

            y_obs, y_int = sample_obs[self.scm.y_key].T.unsqueeze(-1), sample_int[self.scm.y_key].T.unsqueeze(-1)

            return x_obs, y_obs, x_int, y_int

    do_scm = DoSCM(hyperparameters).to(device)
    x_obs, y_obs, x_int, y_int = do_scm.forward()

    if return_SCM:
        return x_obs.detach(), y_obs.detach(), y_int.detach(), x_int.detach(), do_scm
    else:
        return x_obs.detach(), y_obs.detach(), y_int.detach(), x_int.detach()
