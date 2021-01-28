"""MinCutPool模型结构
"""


import torch
import torch.nn as nn

from .utils import *
from .layers import *
from math import ceil
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch


class MinCutPoolModel(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, hidden_dim, avg_nodes,
                 aggregate='sum', dropout=0.0, use_bias=True):
        """
        """

        super(MinCutPool, self).__init__()

        self.conv1 = GraphConvolution(input_dim, hidden_dim, use_bias)
        self.conv2 = DenseGraphConvolution(hidden_dim, hidden_dim, aggregate, use_bias)
        self.conv3 = DenseGraphConvolution(hidden_dim, hidden_dim, aggregate, use_bias)

        self.act = nn.ReLU(inplace=True)
        self.pool1 = nn.Linear(hidden_dim, ceil(0.5 * avg_nodes))
        self.pool2 = nn.Linear(hidden_dim, ceil(0.25 * avg_nodes))

        self.mlp = nn.Sequential(

        )

        return

    def forward(self, data):
        """
        """

        # 获得输入数据
        # Na -> 所有图包含节点的总数
        # Ne -> 所有图包含边的总数
        # F  -> 节点特征数量
        # X:     [Na, F], 所有图包含的节点的特征
        # batch: [Na],    指明每个节点所属的图
        # graph: [2, Ne], 所有图包含的边
        X, graph, batch = data.x, data.edge_index, data.batch

        # 将输入数据处理成batch格式
        # b    -> batch size
        # Nmax -> 包含最多节点的图的节点数量
        # F    -> 节点特征数量
        # X:         [b, Nmax, F],    每张图包含的节点的特征
        # batch:     [b, Nmax],       每张图中的有效节点索引
        # adjacency: [b, Nmax, Nmax], 每张图的邻接矩阵
        adjacency = to_dense_adj(graph, batch)
        X, batch = to_dense_batch(X, batch)

        # batch中对每张图的邻接矩阵做正则化
        norm_adjacency = []
        for adj, b in zip(adjacency, batch):
            norm_adj = normalize_adjacency(adj, b)
            norm_adj = norm_adj.unsqueeze(0)
            norm_adjacency.append(norm_adj)
        adjacency = torch.cat(norm_adjacency)

        x1 = self.act(self.conv1(adjacency, X))
        cluster1 = self.pool1(x1)
        print(conv1.size(), cluster1.size())

        # conv2 = self.conv2(adjacency, conv1)

        return
