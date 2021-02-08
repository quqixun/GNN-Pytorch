"""
"""

import copy
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
        S = self.S if train else 1

        Xs = []
        for _ in range(S):
            Xcp = copy.deepcopy(X)
            # DropNode
            if train:
                drop_nodes = torch.FloatTensor(np.ones(num_nodes) * self.D)
                masks = torch.bernoulli(1 - drop_nodes).unsqueeze(1)
                Xcp = masks.to(device) * Xcp * (1 / (1 - self.D))

            # Mixed-order propagation
            Xc = copy.deepcopy(Xcp)
            for _ in range(self.K):
                Xcp = torch.spmm(adjacency, Xcp).detach()
                Xc += Xcp
            Xc = (Xc / (self.K + 1)).detach()
            Xs.append(Xc)

        return Xs

    def consistency_loss(self, pred):
        """
        """

        pred = torch.cat(pred, dim=0)
        pred_exp = torch.exp(pred)
        pred_avg = torch.mean(pred_exp, dim=(0, 1), keepdim=True)

        pred_pow = torch.pow(pred_avg, 1.0 / self.T)
        pred_sharp = pred_pow / torch.sum(pred_pow)
        pred_sharp = pred_sharp.detach()
        consist_loss = self.L * (pred_exp - pred_sharp).pow(2).mean()

        return consist_loss
