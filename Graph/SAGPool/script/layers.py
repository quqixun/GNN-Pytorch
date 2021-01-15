"""
"""


import torch
import torch.nn as nn

from torch_scatter import scatter_max, scatter_mean


# ----------------------------------------------------------------------------
# 图卷积层


class GraphConvolution(nn.Module):
    """图卷积层
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积层

            Inputs:
            -------
            input_dim: int, 输入特征维度
            output_dim: int, 输出特征维度
            use_bias: boolean, 是否使用偏置

        """

        super(GraphConvolution, self).__init__()

        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重和偏置
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, adjacency, X):
        """图卷积层前馈

            Inputs:
            -------
            adjacency: tensor in shape [num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [num_nodes, input_dim], 节点特征

            Output:
            -------
            output: tensor in shape [num_nodes, output_dim], 输出

        """

        support = torch.mm(X, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


# ----------------------------------------------------------------------------
# 读出操作


def global_max_pool(X, graph_indicator):
    """
    """

    num_graphs = graph_indicator.max().item() + 1
    max_pool = scatter_max(X, graph_indicator, dim=0, dim_size=num_graphs)

    return max_pool


def global_avg_pool(X, graph_indicator):
    """
    """

    num_graphs = graph_indicator.max().item() + 1
    avg_pool = scatter_mean(X, graph_indicator, dim=0, dim_size=num_graphs)

    return avg_pool


# ----------------------------------------------------------------------------
# 自注意力机制池化层
