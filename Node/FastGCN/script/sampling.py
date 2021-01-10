"""
"""


import numpy as np
import scipy.sparse as sp

from .utils import sparse_matrix_to_tensor
from scipy.sparse.linalg import norm as sparse_norm


class Sampler(object):

    def __init__(self, input_dim, sampler_dims, device):
        """
        """

        self.device = device
        self.input_dim = input_dim
        self.sampler_dims = sampler_dims
        self.num_layers = len(sampler_dims)

        return

    def sampling(self, X, adjacency, batch_nodes):
        """
        """

        self.X = X
        self.N = X.shape[0]
        self.adjacency = adjacency
        self.__init_probability(adjacency)

        sampled_adjacency = []
        sampled_nodes = batch_nodes

        for layer in range(self.num_layers - 1, -1, -1):
            nodes, adjacency = self.__single_layer_sampling(
                sampled_nodes, self.sampler_dims[layer])

            adjacency = sparse_matrix_to_tensor(adjacency, self.device)
            sampled_adjacency.append(adjacency)
            sampled_nodes = nodes

        sampled_X = self.X[sampled_nodes]
        sampled_adjacency.reverse()
        return sampled_X, sampled_adjacency

    def __init_probability(self, adjacency):
        """
        """

        norm_adjacency = sparse_norm(adjacency, axis=0)
        self.probability = norm_adjacency / np.sum(norm_adjacency)

        return

    def __single_layer_sampling(self, nodes, sampler_dim):
        """
        """

        adjacency = self.adjacency[nodes, :]
        neighbors = np.nonzero(np.sum(adjacency, axis=0))[1]
        probability = self.probability[neighbors]
        probability = probability / np.sum(probability)

        sampled = np.random.choice(
            np.arange(np.size(neighbors)),
            size=sampler_dim,
            replace=True,
            p=probability
        )

        sampled_nodes = neighbors[sampled]
        sampled_adjacency = adjacency[:, sampled_nodes]
        sampled_probability = probability[sampled]
        sampled_adjacency = sampled_adjacency.dot(sp.diags(
            1.0 / (sampled_probability * sampler_dim)
        ))

        return sampled_nodes, sampled_adjacency
