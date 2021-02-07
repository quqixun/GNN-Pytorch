"""
"""

import torch
import numpy as np


class GRAND(object):
    """
    """

    def __init__(self, **grand_params):
        """
        """

        self.S = grand_params['S']
        self.K = grand_params['K']
        self.T = grand_params['T']
        self.L = grand_params['L']
        self.D = grand_params['D']

        return

    def random_propagate(self, adjacency, X, train=False):
        """
        """

        device = X.device
        num_nodes = X.size(0)

        if train:
            # DropNode
            drop_nodes = torch.FloatTensor(np.ones(num_nodes) * self.D)
            masks = torch.bernoulli(1 - drop_nodes).unsqueeze(1)
            X = masks.to(device) * X * (1 / (1 - self.D))

        # Mixed-order propagation
        Xc = X
        # for _ in range(self.K):
            

        return

    def consistency_loss(self):
        """
        """

        return
