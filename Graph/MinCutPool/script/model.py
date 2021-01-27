"""MinCutPool模型结构
"""


import torch
import torch.nn as nn

from .utils import *
from .layers import *


class MinCutPool(nn.Module):
    """
    """

    def __init__(self):
        """
        """

        super(MinCutPool, self).__init__()

        self.conv1 = GraphConvolution()
        self.conv2 = DenseGraphConvolution()

        return

    def forward(self, data):
        """
        """

        # 获得输入数据
        X, graph, batch = data.x, data.edge_index, data.batch

        # 计算初始邻接矩阵
        adjacency = generate_adjacency(X, graph)
        adjacency = normalize_adjacency(adjacency)
        adjacency = adjacency.to(X.device)

        print(X.size(), adjacency.size(), batch.size())

        return
