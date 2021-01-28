"""MinCutPool网络相关层函数
"""


import torch
import torch.nn as nn


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

        weight = torch.cat([self.weight.unsqueeze(0)] * X.size(0))
        support = torch.bmm(X, weight)
        output = torch.bmm(adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


# ----------------------------------------------------------------------------
# DenseGraphConv层


class DenseGraphConvolution(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, aggregate='sum', use_bias=True):
        """
        """

        super(DenseGraphConvolution, self).__init__()

        assert aggregate in ['sum', 'max', 'mean'], 'unknown aggregate'
        self.aggregate = aggregate

        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.linear2 = nn.Linear(input_dim, output_dim, bias=use_bias)

        return

    def forward(self, adjacency, X):
        """
        """

        # self.aggregate == 'sum'
        output = self.linear1(torch.matmul(adjacency, X))

        if self.aggregate == 'max':
            output = output.max(dim=-1, keepdim=True)[0]
        elif self.aggregate == 'mean':
            output = output / output.sum(dim=-1, keepdim=True)
        else:
            pass

        output = output + self.linear2(X)
        print(output.size())

        return output


# ----------------------------------------------------------------------------
# MinCutPooling层


class MinCutPooling(nn.Module):
    """
    """

    def __init__(self):
        """
        """

        super(MinCutPooling, self).__init__()

        return

    def forward(self):
        """
        """

        return
