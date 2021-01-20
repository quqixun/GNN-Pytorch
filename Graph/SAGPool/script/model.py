"""
"""


import torch
import torch.nn as nn

from .utils import *
from .layers import *


class SAGPoolG(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 keep_ratio, dropout, use_bias):
        """
        """

        super(SAGPoolG, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.gcn1 = GraphConvolution(input_dim, hidden_dim, use_bias)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim, use_bias)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim, use_bias)

        self.sagpool = SelfAttentionPooling(hidden_dim * 3, keep_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        return

    def forward(self, data):
        """
        """

        X, graph, batch = data.x, data.edge_index, data.batch
        adjacency = generate_adjacency(X, graph)
        adjacency = normalize_adjacency(adjacency)
        adjacency = adjacency.to(X.device)

        gcn1 = self.act(self.gcn1(adjacency, X))
        gcn2 = self.act(self.gcn2(adjacency, gcn1))
        gcn3 = self.act(self.gcn3(adjacency, gcn2))
        features = torch.cat([gcn1, gcn2, gcn3], dim=1)

        mask_X, mask_adjacency, mask_batch = \
            self.sagpool(features, adjacency, batch)

        readout = torch.cat([
            global_avg_pool(mask_X, mask_batch),
            global_max_pool(mask_X, mask_batch),
        ], dim=1)
        output = self.mlp(readout)

        return output


class SAGPoolH(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 keep_ratio, dropout, use_bias):
        """
        """

        super(SAGPoolH, self).__init__()

        self.act = nn.ReLU(inplace=True)

        self.gcn1 = GraphConvolution(input_dim, hidden_dim, use_bias)
        self.sagpool1 = SelfAttentionPooling(hidden_dim, keep_ratio)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim, use_bias)
        self.sagpool2 = SelfAttentionPooling(hidden_dim, keep_ratio)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim, use_bias)
        self.sagpool3 = SelfAttentionPooling(hidden_dim, keep_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        return

    def __readout(self, X, batch):
        """
        """

        readout = torch.cat([
            global_avg_pool(X, batch),
            global_max_pool(X, batch)
        ], dim=1)

        return readout

    def forward(self, data):
        """
        """

        X, graph, batch = data.x, data.edge_index, data.batch
        adjacency = generate_adjacency(X, graph)
        adjacency = normalize_adjacency(adjacency)
        adjacency = adjacency.to(X.device)

        X = self.act(self.gcn1(adjacency, X))
        X, adjacency, batch = self.sagpool1(X, adjacency, batch)
        readout1 = self.__readout(X, batch)

        X = self.act(self.gcn2(adjacency, X))
        X, adjacency, batch = self.sagpool2(X, adjacency, batch)
        readout2 = self.__readout(X, batch)

        X = self.act(self.gcn3(adjacency, X))
        X, adjacency, batch = self.sagpool3(X, adjacency, batch)
        readout3 = self.__readout(X, batch)

        readout = readout1 + readout2 + readout3
        logits = self.mlp(readout)
        return logits
